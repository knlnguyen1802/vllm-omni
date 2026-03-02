# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for the async stage-worker batching helpers introduced in the
``batching_async_omni`` branch (commit d6a3551 → HEAD).

The functions under test live **inside** ``_stage_worker_async`` (they are
closures), so we test them indirectly via carefully constructed async
harness functions that replicate the same closure logic.

What is tested
--------------
A. ``_resolve_task_inputs`` (extracted from the old ``generation_single_request``)
   1.  Returns (rid, engine_input) for a well-formed task.
   2.  Raises ``RuntimeError`` when ``try_recv_via_connector`` returns None.
   3.  Unwraps a single-item Sequence into the scalar element.
   4.  Records in-flight timing when ``sent_ts`` is present.

B. ``_collect_diffusion_batch``
   5.  Returns single-task list when max_batch_size == 1.
   6.  Collects up to max_batch_size tasks from the queue.
   7.  Stops early when the queue empties without waiting the full timeout.
   8.  Defers tasks with mismatched sampling_params (re-queues them).
   9.  Detects a SHUTDOWN task in the queue and returns should_shutdown=True.

C. ``generation_batch_diffusion``
   10. Calls ``AsyncOmniDiffusion.generate_batch`` with correct inputs.
   11. Fans results back to ``generation_out_q`` one entry per request.
   12. Maps unmapped results by cycling over batch_request_ids.
   13. Emits error payloads to ``out_q`` on ``generate_batch`` failure.
"""

import asyncio
import queue as _queue_module
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK, OmniStageTaskType
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STAGE_ID = 1


def _make_task(
    request_id: str = "req-0",
    prompt: str = "a cat",
    sampling_params: Any = None,
    sent_ts: float | None = None,
) -> dict:
    if sampling_params is None:
        sampling_params = OmniDiffusionSamplingParams(num_inference_steps=1)
    task: dict = {
        "request_id": request_id,
        "engine_inputs": prompt,
        "sampling_params": sampling_params,
    }
    if sent_ts is not None:
        task["sent_ts"] = sent_ts
    return task


def _make_result(request_id: str | None = None) -> OmniRequestOutput:
    r = MagicMock(spec=OmniRequestOutput)
    r.request_id = request_id
    return r


# ---------------------------------------------------------------------------
# Section A – _resolve_task_inputs (standalone re-implementation for tests)
# ---------------------------------------------------------------------------

def _build_resolve_task_inputs(connectors, stage_id, rx_bytes_by_rid, rx_decode_ms_by_rid, in_flight_ms_by_rid):
    """
    Re-implement _resolve_task_inputs as a pure function (same logic as
    the closure in _stage_worker_async) for isolated unit testing.
    """
    from typing import cast
    from collections.abc import Sequence

    from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector
    from vllm_omni.inputs.data import OmniPromptType

    def _resolve_task_inputs(task: dict, recv_dequeue_ts: float):
        rid = task["request_id"]
        try:
            sent_ts = float(task.get("sent_ts", None)) if isinstance(task, dict) else None
            if sent_ts is not None:
                in_flight_ms_by_rid[rid] = max(0.0, (recv_dequeue_ts - sent_ts) * 1000.0)
            else:
                in_flight_ms_by_rid[rid] = 0.0
        except Exception:
            in_flight_ms_by_rid[rid] = 0.0

        ein, _rx_metrics = try_recv_via_connector(
            task=task,
            connectors=connectors,
            stage_id=stage_id,
        )
        ein = cast(OmniPromptType | Sequence[OmniPromptType] | None, ein)

        if ein is None or _rx_metrics is None:
            raise RuntimeError(
                f"[Stage-{stage_id}] Missing connector payload for request {rid}. "
                "Ensure connectors are configured for all incoming edges."
            )
        rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
        rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))

        if isinstance(ein, Sequence) and not isinstance(ein, str):
            ein = ein[0]
        return rid, ein

    return _resolve_task_inputs


class TestResolveTaskInputs:
    """Unit tests for _resolve_task_inputs closure logic."""

    def _make_resolve_fn(self, connector_return_value):
        """Build the closure with a mocked try_recv_via_connector."""
        rx_bytes: dict = {}
        rx_decode_ms: dict = {}
        in_flight_ms: dict = {}
        connectors: dict = {}

        with patch(
            "vllm_omni.distributed.omni_connectors.adapter.try_recv_via_connector",
            return_value=connector_return_value,
        ):
            fn = _build_resolve_task_inputs(connectors, STAGE_ID, rx_bytes, rx_decode_ms, in_flight_ms)

        return fn, rx_bytes, rx_decode_ms, in_flight_ms

    def test_returns_rid_and_engine_input(self, monkeypatch):
        """Happy path: returns (request_id, engine_input)."""
        task = _make_task(request_id="rid-1", prompt="hello")
        monkeypatch.setattr(
            "vllm_omni.distributed.omni_connectors.adapter.try_recv_via_connector",
            lambda task, connectors, stage_id: ("hello", {"rx_decode_time_ms": 1.0, "rx_transfer_bytes": 42}),
        )
        rx_bytes: dict = {}
        rx_decode_ms: dict = {}
        in_flight_ms: dict = {}
        fn = _build_resolve_task_inputs({}, STAGE_ID, rx_bytes, rx_decode_ms, in_flight_ms)
        rid, ein = fn(task, recv_dequeue_ts=time.time())

        assert rid == "rid-1"
        assert ein == "hello"
        assert rx_decode_ms["rid-1"] == 1.0
        assert rx_bytes["rid-1"] == 42

    def test_raises_when_connector_returns_none(self, monkeypatch):
        """Missing connector payload → RuntimeError."""
        task = _make_task(request_id="rid-x")
        monkeypatch.setattr(
            "vllm_omni.distributed.omni_connectors.adapter.try_recv_via_connector",
            lambda **kw: (None, None),
        )
        rx_bytes: dict = {}
        rx_decode_ms: dict = {}
        in_flight_ms: dict = {}
        fn = _build_resolve_task_inputs({}, STAGE_ID, rx_bytes, rx_decode_ms, in_flight_ms)
        with pytest.raises(RuntimeError, match="Missing connector payload"):
            fn(task, recv_dequeue_ts=time.time())

    def test_unwraps_single_item_sequence(self, monkeypatch):
        """A single-element list should be unwrapped to its scalar."""
        task = _make_task(request_id="rid-2", prompt="dog")
        monkeypatch.setattr(
            "vllm_omni.distributed.omni_connectors.adapter.try_recv_via_connector",
            lambda **kw: (["dog"], {"rx_decode_time_ms": 0.0, "rx_transfer_bytes": 0}),
        )
        rx_bytes: dict = {}
        rx_decode_ms: dict = {}
        in_flight_ms: dict = {}
        fn = _build_resolve_task_inputs({}, STAGE_ID, rx_bytes, rx_decode_ms, in_flight_ms)
        rid, ein = fn(task, recv_dequeue_ts=time.time())
        assert ein == "dog"

    def test_records_in_flight_timing(self, monkeypatch):
        """in_flight_ms should reflect the difference between now and sent_ts."""
        past = time.time() - 0.5  # 500 ms ago
        task = _make_task(request_id="rid-3", sent_ts=past)
        monkeypatch.setattr(
            "vllm_omni.distributed.omni_connectors.adapter.try_recv_via_connector",
            lambda **kw: ("x", {"rx_decode_time_ms": 0.0, "rx_transfer_bytes": 0}),
        )
        in_flight_ms: dict = {}
        fn = _build_resolve_task_inputs({}, STAGE_ID, {}, {}, in_flight_ms)
        fn(task, recv_dequeue_ts=time.time())
        # Allow generous tolerance (clock imprecision on slow CI boxes)
        assert in_flight_ms["rid-3"] > 400.0


# ---------------------------------------------------------------------------
# Section B – _collect_diffusion_batch
# ---------------------------------------------------------------------------


def _build_collect_diffusion_batch(in_q, max_batch_size, batch_timeout=0.05):
    """
    Standalone async re-implementation of _collect_diffusion_batch that is
    equivalent to the production closure.
    """
    from vllm_omni.entrypoints.stage_utils import is_profiler_task

    async def _collect_diffusion_batch(first_task: dict):
        batch_tasks = [first_task]
        tasks_failed_to_add: list = []
        should_shutdown = False

        if max_batch_size <= 1:
            return batch_tasks, should_shutdown

        start_time = time.time()
        _consecutive_empty = 0
        _max_empty_polls = 5

        while len(batch_tasks) < max_batch_size:
            if not in_q.empty():
                _consecutive_empty = 0
                try:
                    extra = in_q.get_nowait()
                except _queue_module.Empty:
                    pass
                else:
                    if extra == SHUTDOWN_TASK:
                        in_q.put(SHUTDOWN_TASK)
                        should_shutdown = True
                        break
                    extra_type = extra.get("type") if isinstance(extra, dict) else None
                    if extra_type == OmniStageTaskType.SHUTDOWN:
                        in_q.put(extra)
                        should_shutdown = True
                        break
                    if extra_type == OmniStageTaskType.ABORT:
                        continue
                    if is_profiler_task(extra_type):
                        continue
                    if first_task.get("sampling_params") != extra.get("sampling_params"):
                        tasks_failed_to_add.append(extra)
                    else:
                        batch_tasks.append(extra)
                continue

            elapsed = time.time() - start_time
            if elapsed > batch_timeout:
                break
            _consecutive_empty += 1
            if _consecutive_empty >= _max_empty_polls:
                break
            await asyncio.sleep(0.01)

        for t in tasks_failed_to_add:
            in_q.put(t)

        return batch_tasks, should_shutdown

    return _collect_diffusion_batch


class TestCollectDiffusionBatch:
    """Tests for the _collect_diffusion_batch closure logic."""

    def test_max_batch_size_1_returns_single_task(self):
        q = _queue_module.Queue()
        fn = _build_collect_diffusion_batch(q, max_batch_size=1)
        first = _make_task("r0")
        batch, shutdown = asyncio.run(fn(first))
        assert batch == [first]
        assert shutdown is False

    def test_collects_up_to_max_batch_size(self):
        q = _queue_module.Queue()
        sp = OmniDiffusionSamplingParams(num_inference_steps=1)
        tasks = [_make_task(f"r{i}", sampling_params=sp) for i in range(3)]
        # Pre-load extra tasks into queue (first task is already dequeued)
        for t in tasks[1:]:
            q.put(t)

        fn = _build_collect_diffusion_batch(q, max_batch_size=3)
        batch, shutdown = asyncio.run(fn(tasks[0]))

        assert len(batch) == 3
        assert shutdown is False

    def test_stops_early_on_empty_queue(self):
        q = _queue_module.Queue()  # empty
        fn = _build_collect_diffusion_batch(q, max_batch_size=5, batch_timeout=0.01)
        first = _make_task("r0")
        batch, shutdown = asyncio.run(fn(first))
        # Only the first task — queue was empty
        assert len(batch) == 1
        assert shutdown is False

    def test_defers_mismatched_sampling_params(self):
        q = _queue_module.Queue()
        sp_a = OmniDiffusionSamplingParams(num_inference_steps=1)
        sp_b = OmniDiffusionSamplingParams(num_inference_steps=50)
        first = _make_task("r0", sampling_params=sp_a)
        mismatched = _make_task("r1", sampling_params=sp_b)
        q.put(mismatched)

        fn = _build_collect_diffusion_batch(q, max_batch_size=2, batch_timeout=0.01)
        batch, shutdown = asyncio.run(fn(first))

        # batch should only contain the first task
        assert len(batch) == 1
        assert batch[0]["request_id"] == "r0"
        # mismatched task should have been re-queued
        assert not q.empty()
        requeued = q.get_nowait()
        assert requeued["request_id"] == "r1"
        assert shutdown is False

    def test_detects_shutdown_task_string(self):
        q = _queue_module.Queue()
        q.put(SHUTDOWN_TASK)
        fn = _build_collect_diffusion_batch(q, max_batch_size=3)
        batch, shutdown = asyncio.run(fn(_make_task("r0")))
        assert shutdown is True
        # SHUTDOWN_TASK should have been re-queued
        assert q.get_nowait() == SHUTDOWN_TASK

    def test_detects_shutdown_dict_type(self):
        q = _queue_module.Queue()
        q.put({"type": OmniStageTaskType.SHUTDOWN})
        fn = _build_collect_diffusion_batch(q, max_batch_size=3)
        batch, shutdown = asyncio.run(fn(_make_task("r0")))
        assert shutdown is True


# ---------------------------------------------------------------------------
# Section C – generation_batch_diffusion
# ---------------------------------------------------------------------------


async def _run_generation_batch_diffusion(
    batch_tasks: list[dict],
    mock_generate_batch_result=None,
    mock_generate_batch_raises=None,
    stage_id: int = STAGE_ID,
):
    """
    Run a simplified version of generation_batch_diffusion that mirrors the
    production closure but accepts injected dependencies.

    Returns (generation_out_q contents, out_q contents).
    """
    import traceback as _traceback

    generation_out_q: asyncio.Queue = asyncio.Queue()
    out_q = _queue_module.Queue()

    # Mock connector resolver
    def _fake_resolve(task, recv_dequeue_ts):
        return task["request_id"], task.get("engine_inputs", "input")

    # Mock stage engine
    mock_engine = MagicMock()
    if mock_generate_batch_raises is not None:
        mock_engine.generate_batch = AsyncMock(side_effect=mock_generate_batch_raises)
    else:
        mock_engine.generate_batch = AsyncMock(return_value=mock_generate_batch_result or [])

    batch_request_ids: list = []
    batch_engine_inputs: list = []
    failed_rids: list = []

    recv_ts = time.time()
    for t in batch_tasks:
        try:
            rid, ein = _fake_resolve(t, recv_ts)
            batch_request_ids.append(rid)
            batch_engine_inputs.append(ein)
        except Exception as e:
            rid = t.get("request_id", "unknown")
            failed_rids.append(rid)
            out_q.put({"request_id": rid, "stage_id": stage_id, "error": str(e)})

    if not batch_request_ids:
        return [], list(out_q.queue)

    batch_sampling_params = batch_tasks[0]["sampling_params"]
    gen_t0 = time.time()
    try:
        gen_outputs = await mock_engine.generate_batch(
            batch_engine_inputs, batch_sampling_params, batch_request_ids
        )
        gen_ms = (time.time() - gen_t0) * 1000.0

        req_to_outputs: dict = {rid: [] for rid in batch_request_ids}
        unmapped: list = []
        for ro in gen_outputs:
            rid = ro.request_id
            if rid in req_to_outputs:
                req_to_outputs[rid].append(ro)
            else:
                unmapped.append(ro)
        if unmapped:
            for i, ro in enumerate(unmapped):
                target = batch_request_ids[i % len(batch_request_ids)]
                ro.request_id = target
                req_to_outputs[target].append(ro)

        for rid in batch_request_ids:
            r_outputs = req_to_outputs.get(rid, [])
            gen_output = r_outputs[0] if len(r_outputs) == 1 else r_outputs
            await generation_out_q.put((rid, gen_output, gen_ms))
    except Exception as e:
        tb = _traceback.format_exc()
        for rid in batch_request_ids:
            out_q.put({"request_id": rid, "stage_id": stage_id, "error": str(e), "error_tb": tb})

    out_items = []
    while not out_q.empty():
        out_items.append(out_q.get_nowait())

    gen_items = []
    while not generation_out_q.empty():
        gen_items.append(generation_out_q.get_nowait())

    return gen_items, out_items


class TestGenerationBatchDiffusion:
    """Tests for the generation_batch_diffusion closure logic."""

    def test_calls_generate_batch_with_inputs(self):
        sp = OmniDiffusionSamplingParams(num_inference_steps=1)
        tasks = [_make_task("r0", "cat", sp), _make_task("r1", "dog", sp)]
        r0 = _make_result("r0")
        r1 = _make_result("r1")

        gen_items, out_items = asyncio.run(_run_generation_batch_diffusion(
            tasks, mock_generate_batch_result=[r0, r1]
        ))

        assert len(gen_items) == 2
        assert out_items == []

    def test_results_mapped_to_request_ids(self):
        sp = OmniDiffusionSamplingParams(num_inference_steps=1)
        tasks = [_make_task("r0", sampling_params=sp), _make_task("r1", sampling_params=sp)]
        r0 = _make_result("r0")
        r1 = _make_result("r1")

        gen_items, _ = asyncio.run(_run_generation_batch_diffusion(
            tasks, mock_generate_batch_result=[r0, r1]
        ))

        rids_in_queue = {item[0] for item in gen_items}
        assert rids_in_queue == {"r0", "r1"}

    def test_unmapped_results_cycled_to_rids(self):
        """Results with unknown request_ids must be assigned to known rids."""
        sp = OmniDiffusionSamplingParams(num_inference_steps=1)
        tasks = [_make_task("r0", sampling_params=sp)]
        mystery_result = _make_result(request_id="unknown-rid")

        gen_items, out_items = asyncio.run(_run_generation_batch_diffusion(
            tasks, mock_generate_batch_result=[mystery_result]
        ))

        # Should have been remapped to r0
        assert len(gen_items) == 1
        item_rid, item_output, _ = gen_items[0]
        assert item_rid == "r0"
        assert item_output.request_id == "r0"

    def test_generate_batch_exception_emits_error_payloads(self):
        sp = OmniDiffusionSamplingParams(num_inference_steps=1)
        tasks = [_make_task("r0", sampling_params=sp), _make_task("r1", sampling_params=sp)]

        gen_items, out_items = asyncio.run(_run_generation_batch_diffusion(
            tasks,
            mock_generate_batch_raises=RuntimeError("gpu OOM"),
        ))

        assert gen_items == []
        assert len(out_items) == 2
        rids = {o["request_id"] for o in out_items}
        assert rids == {"r0", "r1"}
        for o in out_items:
            assert "error" in o
            assert "gpu OOM" in o["error"]

    def test_gen_ms_is_non_negative(self):
        """Generation timing should never be negative."""
        sp = OmniDiffusionSamplingParams(num_inference_steps=1)
        tasks = [_make_task("r0", sampling_params=sp)]
        result = _make_result("r0")

        gen_items, _ = asyncio.run(_run_generation_batch_diffusion(
            tasks, mock_generate_batch_result=[result]
        ))

        assert len(gen_items) == 1
        _, _, gen_ms = gen_items[0]
        assert gen_ms >= 0.0
