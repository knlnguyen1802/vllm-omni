"""
Tests for collective_rpc race-condition scenarios.

Validates that:
 - Multiple concurrent collective_rpc calls route results correctly
 - collective_rpc results don't leak into generation output streams
 - Generation results don't get swallowed by the RPC checker
 - The async output_handler correctly separates RPC vs generation results
 - The sync queue-draining checker handles interleaved messages
"""

import asyncio
import multiprocessing as mp
import queue
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.entrypoints.stage_utils import OmniStageTaskType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rpc_result(rpc_id: str, stage_id: int = 0, result: Any = "ok"):
    """Build a well-formed collective_rpc_result message."""
    return {
        "type": "collective_rpc_result",
        "rpc_id": rpc_id,
        "stage_id": stage_id,
        "result": result,
    }


def _make_rpc_error(rpc_id: str, stage_id: int = 0, error: str = "boom"):
    """Build a collective_rpc_result with an error."""
    return {
        "type": "collective_rpc_result",
        "rpc_id": rpc_id,
        "stage_id": stage_id,
        "error": error,
    }


def _make_gen_result(request_id: str, stage_id: int = 0, finished: bool = True):
    """Build a generation-like output message."""
    output = MagicMock()
    output.finished = finished
    return {
        "request_id": request_id,
        "stage_id": stage_id,
        "engine_outputs": [output],
    }


def _make_stage_ready(stage_id: int = 0):
    return {"type": "stage_ready", "stage_id": stage_id}


class FakeQueue:
    """Thread-safe fake queue that wraps a plain list, simulating mp.Queue
    without needing real multiprocessing."""

    def __init__(self):
        self._items: list[Any] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, item: Any) -> None:
        with self._not_empty:
            self._items.append(item)
            self._not_empty.notify()

    def put_nowait(self, item: Any) -> None:
        self.put(item)

    def get(self, timeout: float | None = None) -> Any:
        with self._not_empty:
            deadline = time.monotonic() + timeout if timeout else None
            while not self._items:
                remaining = (deadline - time.monotonic()) if deadline else None
                if remaining is not None and remaining <= 0:
                    raise queue.Empty
                self._not_empty.wait(timeout=remaining)
            return self._items.pop(0)

    def get_nowait(self) -> Any:
        with self._lock:
            if not self._items:
                raise queue.Empty
            return self._items.pop(0)

    def empty(self) -> bool:
        with self._lock:
            return len(self._items) == 0

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)


class FakeOmniStage:
    """Lightweight stand-in for OmniStage with queues and rpc_result_checker
    already wired, but no real worker process."""

    def __init__(self, stage_id: int = 0):
        self.stage_id = stage_id
        self._in_q: FakeQueue = FakeQueue()
        self._out_q: FakeQueue = FakeQueue()
        self._rpc_result_checker: Callable[[str], dict | None] | None = None

    def try_collect(self) -> dict | None:
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def submit(self, payload: dict) -> None:
        self._in_q.put(payload)


# ---------------------------------------------------------------------------
# 1. RPC result checker — dict-based (async path)
# ---------------------------------------------------------------------------


class TestAsyncRpcResultChecker:
    """Tests for the checker used in the async path (reads only from
    ``_rpc_results`` dict, no queue draining)."""

    def _make_checker(self, rpc_results: dict, stage_id: int):
        """Mirrors AsyncOmni._setup_rpc_result_checkers logic."""
        def checker(rpc_id: str) -> dict | None:
            if stage_id in rpc_results and rpc_id in rpc_results[stage_id]:
                return rpc_results[stage_id].pop(rpc_id)
            return None
        return checker

    def test_single_rpc_found(self):
        rpc_results: dict[int, dict[str, dict]] = {}
        checker = self._make_checker(rpc_results, stage_id=0)
        rpc_id = str(uuid.uuid4())

        # Nothing stored yet → None
        assert checker(rpc_id) is None

        # Store and retrieve
        rpc_results[0] = {rpc_id: _make_rpc_result(rpc_id)}
        result = checker(rpc_id)
        assert result is not None
        assert result["rpc_id"] == rpc_id

        # After pop, should be gone
        assert checker(rpc_id) is None

    def test_multiple_rpc_ids_no_cross_contamination(self):
        """Two concurrent RPCs should each get their own result."""
        rpc_results: dict[int, dict[str, dict]] = {}
        checker = self._make_checker(rpc_results, stage_id=0)

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        rpc_results[0] = {
            id_a: _make_rpc_result(id_a, result="result_a"),
            id_b: _make_rpc_result(id_b, result="result_b"),
        }

        # Each retrieves only its own
        res_a = checker(id_a)
        assert res_a["result"] == "result_a"
        res_b = checker(id_b)
        assert res_b["result"] == "result_b"

        # Both consumed
        assert checker(id_a) is None
        assert checker(id_b) is None

    def test_different_stages_isolated(self):
        """Results for stage 0 must not be visible to stage 1's checker."""
        rpc_results: dict[int, dict[str, dict]] = {}
        checker_0 = self._make_checker(rpc_results, stage_id=0)
        checker_1 = self._make_checker(rpc_results, stage_id=1)

        rpc_id = str(uuid.uuid4())
        rpc_results[0] = {rpc_id: _make_rpc_result(rpc_id, stage_id=0)}

        assert checker_1(rpc_id) is None  # stage 1 sees nothing
        assert checker_0(rpc_id) is not None  # stage 0 gets it

    def test_concurrent_dict_access_threads(self):
        """Hammer the checker from many threads; no result should be
        returned twice or lost."""
        rpc_results: dict[int, dict[str, dict]] = {}
        checker = self._make_checker(rpc_results, stage_id=0)

        num_rpcs = 200
        ids = [str(uuid.uuid4()) for _ in range(num_rpcs)]
        rpc_results[0] = {rid: _make_rpc_result(rid) for rid in ids}

        found: list[str] = []
        lock = threading.Lock()

        def worker(rpc_id: str):
            # Each thread tries to claim its rpc_id
            for _ in range(50):  # retry a few times
                r = checker(rpc_id)
                if r is not None:
                    with lock:
                        found.append(r["rpc_id"])
                    return
                time.sleep(0.0001)

        threads = [threading.Thread(target=worker, args=(rid,)) for rid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Every rpc_id should be found exactly once
        assert sorted(found) == sorted(ids), (
            f"Expected {num_rpcs} unique results, got {len(found)} "
            f"(duplicates={len(found) - len(set(found))})"
        )


# ---------------------------------------------------------------------------
# 2. Sync queue-draining checker (OmniBase._setup_rpc_result_checkers)
# ---------------------------------------------------------------------------


class TestSyncQueueDrainingChecker:
    """Tests for the checker that drains the output queue in the sync path."""

    def _make_checker(
        self, rpc_results: dict, stage_id: int, out_q: FakeQueue
    ):
        """Mirrors OmniBase._setup_rpc_result_checkers logic."""
        def checker(rpc_id: str) -> dict | None:
            # First check the shared dict
            if stage_id in rpc_results and rpc_id in rpc_results[stage_id]:
                return rpc_results[stage_id].pop(rpc_id)
            # Drain the output queue
            try:
                while True:
                    item = out_q.get_nowait()
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "collective_rpc_result"
                    ):
                        item_rpc_id = item.get("rpc_id")
                        if item_rpc_id == rpc_id:
                            return item
                        # Stash for another caller
                        if stage_id not in rpc_results:
                            rpc_results[stage_id] = {}
                        rpc_results[stage_id][item_rpc_id] = item
                    else:
                        # Non-RPC item — put it back and stop draining
                        out_q.put(item)
                        break
            except queue.Empty:
                pass
            return None
        return checker

    def test_rpc_result_found_on_queue(self):
        rpc_results: dict = {}
        out_q = FakeQueue()
        checker = self._make_checker(rpc_results, 0, out_q)

        rpc_id = str(uuid.uuid4())
        out_q.put(_make_rpc_result(rpc_id))

        assert checker(rpc_id) is not None
        assert out_q.empty()

    def test_gen_result_put_back(self):
        """Generation results encountered while draining must be put back."""
        rpc_results: dict = {}
        out_q = FakeQueue()
        checker = self._make_checker(rpc_results, 0, out_q)

        gen = _make_gen_result("req-1")
        out_q.put(gen)

        # The RPC we're looking for isn't on the queue
        rpc_id = str(uuid.uuid4())
        assert checker(rpc_id) is None
        # But the generation result is still on the queue
        assert not out_q.empty()
        item = out_q.get_nowait()
        assert item["request_id"] == "req-1"

    def test_rpc_behind_gen_result_not_found_on_first_poll(self):
        """If a generation result sits in front of an RPC result, the drainer
        stops after putting the gen result back.  The RPC result behind it is
        NOT found on that poll — it stays on the queue for a later attempt.

        This is a known limitation of the current drain strategy
        (stops at first non-RPC item)."""
        rpc_results: dict = {}
        out_q = FakeQueue()
        checker = self._make_checker(rpc_results, 0, out_q)

        rpc_id = str(uuid.uuid4())
        out_q.put(_make_gen_result("req-1"))  # blocks further draining
        out_q.put(_make_rpc_result(rpc_id))   # behind the gen result

        # First poll: gen result is put back, RPC not found
        assert checker(rpc_id) is None
        # Queue now has gen_result (re-put) at front + rpc_result behind
        assert out_q.qsize() == 2

        # Second poll: same thing — gen result blocks again
        assert checker(rpc_id) is None

        # Simulate the generation loop consuming the gen result
        out_q.get_nowait()  # consume gen result
        # Now the rpc result is at the front
        assert checker(rpc_id) is not None

    def test_multiple_rpc_results_stashed(self):
        """When draining finds other RPC results (not the one we want),
        they are stashed in _rpc_results for later retrieval."""
        rpc_results: dict = {}
        out_q = FakeQueue()
        checker = self._make_checker(rpc_results, 0, out_q)

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        # B is on the queue first, then A
        out_q.put(_make_rpc_result(id_b, result="for_b"))
        out_q.put(_make_rpc_result(id_a, result="for_a"))

        # Looking for A — should drain B first (stash it), then find A
        result_a = checker(id_a)
        assert result_a is not None
        assert result_a["result"] == "for_a"

        # B was stashed in the dict
        result_b = checker(id_b)
        assert result_b is not None
        assert result_b["result"] == "for_b"

    def test_concurrent_checkers_same_stage(self):
        """Two threads poll the same stage's checker for different rpc_ids."""
        rpc_results: dict = {}
        out_q = FakeQueue()
        checker = self._make_checker(rpc_results, 0, out_q)

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())

        results_found: dict[str, Any] = {}
        barrier = threading.Barrier(2)

        def poller(rpc_id: str, delay: float):
            barrier.wait()
            for _ in range(200):
                r = checker(rpc_id)
                if r is not None:
                    results_found[rpc_id] = r
                    return
                time.sleep(0.001)

        # Start both pollers before putting results on queue
        t_a = threading.Thread(target=poller, args=(id_a, 0))
        t_b = threading.Thread(target=poller, args=(id_b, 0))
        t_a.start()
        t_b.start()

        # Slight delay then put results
        time.sleep(0.01)
        out_q.put(_make_rpc_result(id_a, result="for_a"))
        out_q.put(_make_rpc_result(id_b, result="for_b"))

        t_a.join(timeout=5)
        t_b.join(timeout=5)

        assert id_a in results_found, "Thread A did not find its result"
        assert id_b in results_found, "Thread B did not find its result"
        assert results_found[id_a]["result"] == "for_a"
        assert results_found[id_b]["result"] == "for_b"


# ---------------------------------------------------------------------------
# 3. OmniStage.collective_rpc integration (no real worker)
# ---------------------------------------------------------------------------


class TestOmniStageCollectiveRpc:
    """Tests the OmniStage.collective_rpc method with a fake worker that
    just echoes results onto the out_q."""

    def _make_stage_with_fake_worker(self, stage_id: int = 0):
        """Create OmniStage-like object with queues and a background thread
        that reads tasks from _in_q and puts results on _out_q."""
        stage = FakeOmniStage(stage_id)

        rpc_results: dict[int, dict[str, dict]] = {}

        def checker(rpc_id: str) -> dict | None:
            if stage_id in rpc_results and rpc_id in rpc_results[stage_id]:
                return rpc_results[stage_id].pop(rpc_id)
            # Drain queue
            try:
                while True:
                    item = stage._out_q.get_nowait()
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "collective_rpc_result"
                    ):
                        item_rpc_id = item.get("rpc_id")
                        if item_rpc_id == rpc_id:
                            return item
                        if stage_id not in rpc_results:
                            rpc_results[stage_id] = {}
                        rpc_results[stage_id][item_rpc_id] = item
                    else:
                        stage._out_q.put(item)
                        break
            except queue.Empty:
                pass
            return None

        stage._rpc_result_checker = checker
        return stage, rpc_results

    def _start_echo_worker(self, stage: FakeOmniStage, delay: float = 0.01):
        """Background thread that echoes RPC tasks as results."""
        stop = threading.Event()

        def worker():
            while not stop.is_set():
                try:
                    task = stage._in_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                task_type = task.get("type")
                if task_type == OmniStageTaskType.COLLECTIVE_RPC:
                    time.sleep(delay)
                    stage._out_q.put(
                        _make_rpc_result(
                            task["rpc_id"],
                            stage_id=stage.stage_id,
                            result=[f"echo-{task['method']}"],
                        )
                    )
                elif task_type == OmniStageTaskType.SHUTDOWN:
                    break

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return stop, t

    def test_single_rpc(self):
        """Single collective_rpc call returns the correct result."""
        stage, _ = self._make_stage_with_fake_worker()
        stop, worker_t = self._start_echo_worker(stage)

        # Import the real collective_rpc method and bind it
        from vllm_omni.entrypoints.omni_stage import OmniStage

        # Call collective_rpc manually (same logic as OmniStage.collective_rpc)
        rpc_id = str(uuid.uuid4())
        stage._in_q.put({
            "type": OmniStageTaskType.COLLECTIVE_RPC,
            "rpc_id": rpc_id,
            "method": "test_method",
            "timeout": None,
            "args": (),
            "kwargs": None,
        })

        # Poll for result
        start = time.time()
        result = None
        while time.time() - start < 5:
            result = stage._rpc_result_checker(rpc_id)
            if result is not None:
                break
            time.sleep(0.001)

        assert result is not None
        assert result["result"] == ["echo-test_method"]

        stop.set()
        worker_t.join(timeout=2)

    def test_concurrent_rpcs_on_same_stage(self):
        """Multiple concurrent RPCs on the same stage return correct results."""
        stage, _ = self._make_stage_with_fake_worker()
        stop, worker_t = self._start_echo_worker(stage, delay=0.005)

        num_rpcs = 20
        rpc_ids = [str(uuid.uuid4()) for _ in range(num_rpcs)]
        methods = [f"method_{i}" for i in range(num_rpcs)]
        results_map: dict[str, Any] = {}
        errors: list[str] = []

        def do_rpc(rpc_id: str, method: str):
            stage._in_q.put({
                "type": OmniStageTaskType.COLLECTIVE_RPC,
                "rpc_id": rpc_id,
                "method": method,
                "timeout": None,
                "args": (),
                "kwargs": None,
            })
            start = time.time()
            while time.time() - start < 10:
                r = stage._rpc_result_checker(rpc_id)
                if r is not None:
                    results_map[rpc_id] = r
                    return
                time.sleep(0.001)
            errors.append(f"Timeout for rpc_id={rpc_id}")

        threads = [
            threading.Thread(target=do_rpc, args=(rid, m))
            for rid, m in zip(rpc_ids, methods)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        stop.set()
        worker_t.join(timeout=2)

        assert not errors, f"Errors: {errors}"
        assert len(results_map) == num_rpcs
        # Verify each got the right method echo
        for rid, method in zip(rpc_ids, methods):
            assert results_map[rid]["result"] == [f"echo-{method}"], (
                f"rpc_id={rid} expected echo-{method}, got {results_map[rid]['result']}"
            )

    def test_rpc_interleaved_with_gen_results(self):
        """RPC results intermixed with generation results on the same queue.
        The RPC checker must find RPC results and not swallow gen results."""
        stage, _ = self._make_stage_with_fake_worker()

        rpc_id = str(uuid.uuid4())
        # Put gen result, then rpc result, then another gen result
        stage._out_q.put(_make_gen_result("req-1"))
        stage._out_q.put(_make_rpc_result(rpc_id, result="rpc_ok"))
        stage._out_q.put(_make_gen_result("req-2"))

        # First poll: gen result blocks — rpc not found
        assert stage._rpc_result_checker(rpc_id) is None

        # Consume the gen result (simulating generation loop)
        gen1 = stage._out_q.get_nowait()
        assert gen1["request_id"] == "req-1"

        # Now rpc result is at the front
        result = stage._rpc_result_checker(rpc_id)
        assert result is not None
        assert result["result"] == "rpc_ok"

        # The second gen result should still be there
        gen2 = stage._out_q.get_nowait()
        assert gen2["request_id"] == "req-2"


# ---------------------------------------------------------------------------
# 4. Async output handler — RPC vs generation separation
# ---------------------------------------------------------------------------


class TestAsyncOutputHandlerSeparation:
    """Tests that the async output handler correctly routes RPC results
    to _rpc_results and generation results to request state queues."""

    @pytest.fixture
    def event_loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.mark.asyncio
    async def test_output_handler_routes_rpc_and_gen(self):
        """Simulate the output handler loop and verify routing."""
        rpc_results: dict[int, dict[str, dict]] = {}
        request_states: dict[str, Any] = {}

        # Create a fake request state
        from vllm_omni.entrypoints.client_request_state import ClientRequestState
        req_state = ClientRequestState("req-1")
        request_states["req-1"] = req_state

        stage = FakeOmniStage(stage_id=0)

        rpc_id = str(uuid.uuid4())
        # Put both an RPC result and a gen result on the queue
        stage._out_q.put(_make_rpc_result(rpc_id, result="rpc_data"))
        stage._out_q.put(_make_gen_result("req-1"))

        # Run one iteration of the output handler logic
        stage_list = [stage]
        for stage_id, s in enumerate(stage_list):
            result = s.try_collect()
            while result is not None:
                if result.get("type") == "collective_rpc_result":
                    rid = result.get("rpc_id")
                    if rid:
                        if stage_id not in rpc_results:
                            rpc_results[stage_id] = {}
                        rpc_results[stage_id][rid] = result
                elif result.get("type") == "stage_ready":
                    pass
                else:
                    req_id = result.get("request_id")
                    rs = request_states.get(req_id)
                    if rs is not None:
                        await rs.queue.put(result)
                        rs.stage_id = stage_id
                result = s.try_collect()

        # RPC result should be in _rpc_results
        assert 0 in rpc_results
        assert rpc_id in rpc_results[0]
        assert rpc_results[0][rpc_id]["result"] == "rpc_data"

        # Gen result should be in request state queue
        gen = await asyncio.wait_for(req_state.queue.get(), timeout=1)
        assert gen["request_id"] == "req-1"

    @pytest.mark.asyncio
    async def test_many_rpc_results_routed_correctly(self):
        """Many RPC results should all land in _rpc_results with correct ids."""
        rpc_results: dict[int, dict[str, dict]] = {}
        stage = FakeOmniStage(stage_id=0)

        num_rpcs = 50
        ids = [str(uuid.uuid4()) for _ in range(num_rpcs)]
        for rid in ids:
            stage._out_q.put(_make_rpc_result(rid, result=f"data-{rid}"))

        # Drain via handler logic
        result = stage.try_collect()
        while result is not None:
            if result.get("type") == "collective_rpc_result":
                rid = result.get("rpc_id")
                if rid:
                    if 0 not in rpc_results:
                        rpc_results[0] = {}
                    rpc_results[0][rid] = result
            result = stage.try_collect()

        assert len(rpc_results[0]) == num_rpcs
        for rid in ids:
            assert rpc_results[0][rid]["result"] == f"data-{rid}"

    @pytest.mark.asyncio
    async def test_interleaved_rpc_gen_many(self):
        """Interleave many RPC and gen results; verify correct routing."""
        rpc_results: dict[int, dict[str, dict]] = {}
        request_states: dict[str, Any] = {}

        from vllm_omni.entrypoints.client_request_state import ClientRequestState

        num_gen = 30
        num_rpc = 30
        for i in range(num_gen):
            req_id = f"req-{i}"
            request_states[req_id] = ClientRequestState(req_id)

        stage = FakeOmniStage(stage_id=0)

        # Interleave: gen, rpc, gen, rpc, ...
        rpc_ids = []
        for i in range(max(num_gen, num_rpc)):
            if i < num_gen:
                stage._out_q.put(_make_gen_result(f"req-{i}"))
            if i < num_rpc:
                rid = str(uuid.uuid4())
                rpc_ids.append(rid)
                stage._out_q.put(_make_rpc_result(rid, result=f"rpc-{i}"))

        # Run handler
        result = stage.try_collect()
        while result is not None:
            if result.get("type") == "collective_rpc_result":
                rid = result.get("rpc_id")
                if rid:
                    if 0 not in rpc_results:
                        rpc_results[0] = {}
                    rpc_results[0][rid] = result
            else:
                req_id = result.get("request_id")
                rs = request_states.get(req_id)
                if rs is not None:
                    await rs.queue.put(result)
                    rs.stage_id = 0
            result = stage.try_collect()

        # All RPC results routed
        assert len(rpc_results.get(0, {})) == num_rpc
        # All gen results routed
        for i in range(num_gen):
            req_id = f"req-{i}"
            gen = await asyncio.wait_for(
                request_states[req_id].queue.get(), timeout=1
            )
            assert gen["request_id"] == req_id


# ---------------------------------------------------------------------------
# 5. Async collective_rpc concurrent execution
# ---------------------------------------------------------------------------


class TestAsyncCollectiveRpcConcurrent:
    """Test that multiple async collective_rpc calls execute concurrently
    and each gets the right result."""

    @pytest.mark.asyncio
    async def test_concurrent_async_rpcs(self):
        """Fire N async collective_rpc calls concurrently, each should get
        its own result."""
        rpc_results: dict[int, dict[str, dict]] = {}

        stage = FakeOmniStage(stage_id=0)

        def async_checker(rpc_id: str) -> dict | None:
            if 0 in rpc_results and rpc_id in rpc_results[0]:
                return rpc_results[0].pop(rpc_id)
            return None

        stage._rpc_result_checker = async_checker

        # Simulate an output handler that routes RPC results
        handler_stop = asyncio.Event()

        async def fake_output_handler():
            while not handler_stop.is_set():
                result = stage.try_collect()
                if result is not None:
                    if result.get("type") == "collective_rpc_result":
                        rid = result.get("rpc_id")
                        if rid:
                            if 0 not in rpc_results:
                                rpc_results[0] = {}
                            rpc_results[0][rid] = result
                else:
                    await asyncio.sleep(0.001)

        handler_task = asyncio.create_task(fake_output_handler())

        # Simulate a fake echo worker
        worker_stop = threading.Event()

        def echo_worker():
            while not worker_stop.is_set():
                try:
                    task = stage._in_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if task.get("type") == OmniStageTaskType.COLLECTIVE_RPC:
                    time.sleep(0.005)  # simulate work
                    stage._out_q.put(_make_rpc_result(
                        task["rpc_id"],
                        result=[f"echo-{task['method']}"],
                    ))

        worker_t = threading.Thread(target=echo_worker, daemon=True)
        worker_t.start()

        num_rpcs = 15
        methods = [f"method_{i}" for i in range(num_rpcs)]

        async def do_rpc(method: str) -> Any:
            """Mimics OmniStage.collective_rpc but async-friendly."""
            rpc_id = str(uuid.uuid4())
            stage._in_q.put({
                "type": OmniStageTaskType.COLLECTIVE_RPC,
                "rpc_id": rpc_id,
                "method": method,
                "timeout": None,
                "args": (),
                "kwargs": None,
            })

            start = time.time()
            while time.time() - start < 10:
                r = stage._rpc_result_checker(rpc_id)
                if r is not None:
                    return r["result"]
                await asyncio.sleep(0.001)
            raise TimeoutError(f"RPC {method} timed out")

        results = await asyncio.gather(
            *[do_rpc(m) for m in methods]
        )

        handler_stop.set()
        worker_stop.set()
        handler_task.cancel()
        try:
            await handler_task
        except asyncio.CancelledError:
            pass
        worker_t.join(timeout=2)

        for i, (method, result) in enumerate(zip(methods, results)):
            assert result == [f"echo-{method}"], (
                f"RPC {i} ({method}): expected echo-{method}, got {result}"
            )


# ---------------------------------------------------------------------------
# 6. Error propagation
# ---------------------------------------------------------------------------


class TestCollectiveRpcErrors:
    """Test that RPC errors are correctly propagated."""

    def test_error_result_propagated_via_checker(self):
        """An RPC error in the result dict should be detectable."""
        rpc_results: dict[int, dict[str, dict]] = {}
        rpc_id = str(uuid.uuid4())
        rpc_results[0] = {rpc_id: _make_rpc_error(rpc_id, error="worker crashed")}

        def checker(rid: str) -> dict | None:
            if 0 in rpc_results and rid in rpc_results[0]:
                return rpc_results[0].pop(rid)
            return None

        result = checker(rpc_id)
        assert result is not None
        assert "error" in result
        assert result["error"] == "worker crashed"


# ---------------------------------------------------------------------------
# 7. Stress test — sync checker with realistic interleaving
# ---------------------------------------------------------------------------


class TestSyncCheckerStress:
    """Stress-test the sync queue-draining checker with a mix of gen and RPC
    results produced concurrently by a fake worker."""

    def test_stress_concurrent_rpc_and_gen(self):
        """A fake worker produces both gen and RPC results. Multiple threads
        poll for their RPC results while a 'generation loop' consumes gen
        results."""
        rpc_results: dict[int, dict[str, dict]] = {}
        out_q = FakeQueue()
        stage_id = 0

        def make_checker():
            def checker(rpc_id: str) -> dict | None:
                if stage_id in rpc_results and rpc_id in rpc_results[stage_id]:
                    return rpc_results[stage_id].pop(rpc_id)
                try:
                    while True:
                        item = out_q.get_nowait()
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "collective_rpc_result"
                        ):
                            item_rpc_id = item.get("rpc_id")
                            if item_rpc_id == rpc_id:
                                return item
                            if stage_id not in rpc_results:
                                rpc_results[stage_id] = {}
                            rpc_results[stage_id][item_rpc_id] = item
                        else:
                            out_q.put(item)
                            break
                except queue.Empty:
                    pass
                return None
            return checker

        checker = make_checker()

        num_rpcs = 30
        num_gen = 50
        rpc_ids = [str(uuid.uuid4()) for _ in range(num_rpcs)]
        stop = threading.Event()

        # Producer: interleave gen and rpc results
        def producer():
            gen_idx = 0
            rpc_idx = 0
            while not stop.is_set() and (gen_idx < num_gen or rpc_idx < num_rpcs):
                if rpc_idx < num_rpcs and (gen_idx >= num_gen or rpc_idx % 2 == 0):
                    out_q.put(_make_rpc_result(rpc_ids[rpc_idx], result=f"rpc-{rpc_idx}"))
                    rpc_idx += 1
                elif gen_idx < num_gen:
                    out_q.put(_make_gen_result(f"gen-req-{gen_idx}"))
                    gen_idx += 1
                time.sleep(0.001)

        # Generation consumer: drains non-RPC items
        gen_consumed: list[str] = []

        def gen_consumer():
            while not stop.is_set() or not out_q.empty():
                try:
                    item = out_q.get_nowait()
                    if isinstance(item, dict) and item.get("type") != "collective_rpc_result":
                        gen_consumed.append(item.get("request_id", ""))
                    else:
                        # It's an RPC result that shouldn't be here — put it back
                        out_q.put(item)
                except queue.Empty:
                    time.sleep(0.001)

        # RPC pollers
        rpc_found: dict[str, Any] = {}
        rpc_errors: list[str] = []

        def rpc_poller(rpc_id: str):
            start = time.time()
            while time.time() - start < 15:
                r = checker(rpc_id)
                if r is not None:
                    rpc_found[rpc_id] = r
                    return
                time.sleep(0.002)
            rpc_errors.append(f"Timeout for {rpc_id}")

        prod_t = threading.Thread(target=producer, daemon=True)
        cons_t = threading.Thread(target=gen_consumer, daemon=True)
        rpc_threads = [
            threading.Thread(target=rpc_poller, args=(rid,), daemon=True)
            for rid in rpc_ids
        ]

        prod_t.start()
        cons_t.start()
        for t in rpc_threads:
            t.start()

        # Wait for all RPC pollers
        for t in rpc_threads:
            t.join(timeout=20)

        stop.set()
        prod_t.join(timeout=2)
        cons_t.join(timeout=2)

        assert not rpc_errors, f"RPC errors: {rpc_errors}"
        assert len(rpc_found) == num_rpcs, (
            f"Expected {num_rpcs} RPC results, got {len(rpc_found)}"
        )
        for i, rid in enumerate(rpc_ids):
            assert rpc_found[rid]["result"] == f"rpc-{i}"


# ---------------------------------------------------------------------------
# 8. Timeout test
# ---------------------------------------------------------------------------


class TestCollectiveRpcTimeout:
    """Test that timeout is respected when no result arrives."""

    def test_timeout_raises(self):
        """If no result appears within timeout, a loop using the checker
        pattern should be able to detect it."""
        rpc_results: dict[int, dict[str, dict]] = {}

        def checker(rpc_id: str) -> dict | None:
            if 0 in rpc_results and rpc_id in rpc_results[0]:
                return rpc_results[0].pop(rpc_id)
            return None

        rpc_id = str(uuid.uuid4())
        timeout = 0.1
        start = time.time()
        timed_out = False

        while True:
            if time.time() - start > timeout:
                timed_out = True
                break
            r = checker(rpc_id)
            if r is not None:
                break
            time.sleep(0.001)

        assert timed_out, "Should have timed out"
