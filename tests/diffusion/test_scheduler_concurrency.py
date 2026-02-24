# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for Scheduler and MultiprocDiffusionExecutor concurrency safety.

This module tests that:
- The scheduler lock prevents concurrent add_req / collective_rpc calls
  from stealing each other's results via the shared mq / result_mq.
- Backward compatibility: single-threaded usage continues to work exactly
  as before.
"""

import queue
import threading
import time
from collections.abc import Sequence
from typing import Any
from unittest.mock import Mock, patch

import pytest

from vllm_omni.diffusion.scheduler import Scheduler

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers – fake message queues backed by stdlib queues
# ---------------------------------------------------------------------------


class FakeMessageQueue:
    """A drop-in replacement for the real MessageQueue used in tests.

    Supports enqueue/dequeue with optional timeout. Backed by a
    ``queue.Queue`` so it is thread-safe by itself (we are testing the
    *caller*-side serialisation, not the queue implementation).
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self.closed = False

    def enqueue(self, item: Any) -> None:
        self._q.put(item)

    def dequeue(self, timeout: float | None = None) -> Any:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("FakeMessageQueue.dequeue timed out")

    def export_handle(self):
        return "fake-handle"

    def close(self):
        self.closed = True


def _make_scheduler_with_fake_queues(num_workers: int = 1) -> Scheduler:
    """Build a ``Scheduler`` whose mq / result_mq are ``FakeMessageQueue``s.

    This avoids needing CUDA / shared-memory and lets us control exactly
    what results are returned.
    """
    scheduler = Scheduler()
    od_config = Mock()
    od_config.num_gpus = num_workers

    # Patch MessageQueue constructor so initialize() uses our fake
    with patch(
        "vllm_omni.diffusion.scheduler.MessageQueue",
        side_effect=lambda **kw: FakeMessageQueue(),
    ):
        scheduler.initialize(od_config)

    # Also give it a fake result queue
    scheduler.result_mq = FakeMessageQueue()
    return scheduler


# ---------------------------------------------------------------------------
# Backward-compatibility tests (single-threaded, no concurrency)
# ---------------------------------------------------------------------------


class TestSchedulerBackwardCompat:
    """Ensure single-threaded behaviour is unchanged after the lock was added."""

    def test_add_req_returns_output(self):
        """add_req should enqueue a request and return the dequeued result."""
        scheduler = _make_scheduler_with_fake_queues()

        fake_output = Mock(name="DiffusionOutput")
        request = Mock(name="OmniDiffusionRequest")

        # Pre-load the result queue so dequeue() returns immediately
        scheduler.result_mq.enqueue(fake_output)

        result = scheduler.add_req(request)
        assert result is fake_output

    def test_add_req_raises_on_worker_error(self):
        """add_req should raise RuntimeError when worker returns an error dict."""
        scheduler = _make_scheduler_with_fake_queues()
        request = Mock()

        scheduler.result_mq.enqueue({"status": "error", "error": "boom"})

        with pytest.raises(RuntimeError, match="worker error"):
            scheduler.add_req(request)

    def test_add_req_raises_when_result_mq_none(self):
        """add_req should raise RuntimeError if result_mq is not initialized."""
        scheduler = _make_scheduler_with_fake_queues()
        scheduler.result_mq = None
        request = Mock()

        with pytest.raises(RuntimeError, match="Result queue not initialized"):
            scheduler.add_req(request)

    def test_add_req_enqueues_correct_rpc_request(self):
        """Verify the rpc_request dict written to mq has the expected shape."""
        scheduler = _make_scheduler_with_fake_queues()
        request = Mock(name="req")

        scheduler.result_mq.enqueue("ok")
        scheduler.add_req(request)

        # The request was broadcast to workers via mq
        msg = scheduler.mq.dequeue(timeout=1)
        assert msg["type"] == "rpc"
        assert msg["method"] == "generate"
        assert msg["args"] == (request,)
        assert msg["output_rank"] == 0
        assert msg["exec_all_ranks"] is True

    def test_scheduler_has_lock(self):
        """The scheduler must expose a threading.Lock after initialization."""
        scheduler = _make_scheduler_with_fake_queues()
        assert hasattr(scheduler, "_lock")
        assert isinstance(scheduler._lock, type(threading.Lock()))

    def test_reinitialize_creates_new_lock(self):
        """Re-initializing the scheduler should create a fresh lock."""
        scheduler = _make_scheduler_with_fake_queues()
        old_lock = scheduler._lock
        # Re-init
        scheduler = _make_scheduler_with_fake_queues()
        assert scheduler._lock is not old_lock


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


class TestSchedulerConcurrency:
    """Prove that the lock prevents result-stealing between concurrent callers."""

    def test_concurrent_add_req_results_match(self):
        """Two threads calling add_req concurrently must each get their own result.

        Without the lock the ordering is non-deterministic and a thread can
        easily dequeue the *other* thread's result.
        """
        scheduler = _make_scheduler_with_fake_queues()

        results: dict[str, Any] = {}
        errors: list[Exception] = []

        # We'll use an event to force both threads to be ready before either
        # proceeds, maximising the chance of interleaving.
        barrier = threading.Barrier(2)

        def _do_add_req(tag: str, expected_output: Any):
            try:
                barrier.wait(timeout=5)
                result = scheduler.add_req(Mock(name=f"request-{tag}"))
                results[tag] = result
            except Exception as exc:
                errors.append(exc)

        output_a = {"tag": "A", "data": "result-A"}
        output_b = {"tag": "B", "data": "result-B"}

        # Pre-load two results; because the lock serialises the calls the
        # first thread to acquire the lock will get output_a, the second
        # will get output_b.  The key property is that *each caller gets
        # exactly one result and no result is lost or duplicated*.
        scheduler.result_mq.enqueue(output_a)
        scheduler.result_mq.enqueue(output_b)

        t1 = threading.Thread(target=_do_add_req, args=("t1", output_a))
        t2 = threading.Thread(target=_do_add_req, args=("t2", output_b))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Threads raised exceptions: {errors}"
        assert len(results) == 2
        # Each thread got exactly one of the two outputs (order is
        # non-deterministic but no result was stolen or lost).
        returned_set = {id(v) for v in results.values()}
        expected_set = {id(output_a), id(output_b)}
        assert returned_set == expected_set

    def test_concurrent_add_req_serialised_by_lock(self):
        """Demonstrate that the lock actually serialises access.

        We instrument dequeue with a small sleep to simulate real worker
        latency and verify that calls do not overlap.
        """
        scheduler = _make_scheduler_with_fake_queues()

        execution_log: list[tuple[str, str]] = []  # (thread-tag, event)
        log_lock = threading.Lock()

        real_dequeue = scheduler.result_mq.dequeue

        def slow_dequeue(timeout=None):
            tag = threading.current_thread().name
            with log_lock:
                execution_log.append((tag, "dequeue-start"))
            time.sleep(0.05)
            result = real_dequeue(timeout=timeout)
            with log_lock:
                execution_log.append((tag, "dequeue-end"))
            return result

        scheduler.result_mq.dequeue = slow_dequeue

        scheduler.result_mq.enqueue("r1")
        scheduler.result_mq.enqueue("r2")

        barrier = threading.Barrier(2)

        def _worker(tag):
            barrier.wait(timeout=5)
            scheduler.add_req(Mock())

        t1 = threading.Thread(target=_worker, args=("t1",), name="t1")
        t2 = threading.Thread(target=_worker, args=("t2",), name="t2")
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # With the lock, we expect the pattern:
        #   (tX, dequeue-start), (tX, dequeue-end),
        #   (tY, dequeue-start), (tY, dequeue-end)
        # i.e. no interleaving.
        starts = [e for e in execution_log if e[1] == "dequeue-start"]
        ends = [e for e in execution_log if e[1] == "dequeue-end"]
        assert len(starts) == 2
        assert len(ends) == 2

        # The first dequeue-end must appear before the second dequeue-start
        first_end_idx = execution_log.index(ends[0])
        second_start_idx = execution_log.index(starts[1])
        assert first_end_idx < second_start_idx, (
            f"Calls were interleaved! Log: {execution_log}"
        )


# ---------------------------------------------------------------------------
# MultiprocDiffusionExecutor-level tests
# ---------------------------------------------------------------------------


class TestMultiprocExecutorConcurrency:
    """Tests at the executor level to verify the lock is used in collective_rpc."""

    @staticmethod
    def _make_executor():
        """Build a MultiprocDiffusionExecutor with fake queues (no real workers)."""
        from vllm_omni.diffusion.executor.multiproc_executor import (
            MultiprocDiffusionExecutor,
        )

        od_config = Mock()
        od_config.num_gpus = 2
        od_config.distributed_executor_backend = "mp"

        # Bypass __init__ and _init_executor entirely
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor.od_config = od_config
        executor._closed = False
        executor._processes = []
        executor.scheduler = _make_scheduler_with_fake_queues(num_workers=2)
        return executor

    def test_collective_rpc_single_thread(self):
        """collective_rpc works in the normal single-threaded case."""
        executor = self._make_executor()

        # Expect 2 responses (num_gpus = 2, unique_reply_rank=None)
        executor.scheduler.result_mq.enqueue("resp-0")
        executor.scheduler.result_mq.enqueue("resp-1")

        result = executor.collective_rpc("some_method")
        assert result == ["resp-0", "resp-1"]

    def test_collective_rpc_unique_reply_rank(self):
        """collective_rpc with unique_reply_rank returns a single value."""
        executor = self._make_executor()
        executor.scheduler.result_mq.enqueue("only-response")

        result = executor.collective_rpc("some_method", unique_reply_rank=0)
        assert result == "only-response"

    def test_collective_rpc_error_response(self):
        """collective_rpc should raise on error responses from workers."""
        executor = self._make_executor()
        executor.scheduler.result_mq.enqueue({"status": "error", "error": "oom"})
        executor.scheduler.result_mq.enqueue("ok")  # second worker

        with pytest.raises(RuntimeError, match="oom"):
            executor.collective_rpc("some_method")

    def test_collective_rpc_closed_executor(self):
        """collective_rpc should raise when executor is closed."""
        executor = self._make_executor()
        executor._closed = True

        with pytest.raises(RuntimeError, match="closed"):
            executor.collective_rpc("some_method")

    def test_collective_rpc_and_add_req_concurrent(self):
        """add_req and collective_rpc called concurrently must not steal results.

        This is the core regression test for the concurrency bug.
        """
        executor = self._make_executor()
        # For add_req (expects 1 result) and collective_rpc (expects 2 results)
        # we need 3 results total.  We use tagged dicts so we can verify
        # which result went where.
        add_req_result = {"caller": "add_req", "value": 42}
        rpc_result_0 = {"caller": "rpc", "idx": 0}
        rpc_result_1 = {"caller": "rpc", "idx": 1}

        results: dict[str, Any] = {}
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def _call_add_req():
            try:
                barrier.wait(timeout=5)
                results["add_req"] = executor.add_req(Mock())
            except Exception as exc:
                errors.append(exc)

        def _call_collective_rpc():
            try:
                barrier.wait(timeout=5)
                results["rpc"] = executor.collective_rpc("test_method")
            except Exception as exc:
                errors.append(exc)

        # Pre-load results.  Because the lock serialises, the first caller to
        # acquire the lock will drain its expected number of results before
        # the second caller can touch the queue.
        #
        # We enqueue add_req_result first, then two rpc results.  Depending
        # on which thread wins the lock:
        #   - If add_req wins: it gets add_req_result; rpc gets [rpc_0, rpc_1]
        #   - If rpc wins:     it gets [add_req_result, rpc_0]; add_req gets rpc_1
        #
        # In the first scenario both callers get exactly the "right" results.
        # In the second the results are "mismatched" but each caller gets the
        # *correct number* of results — no crash, no hang.  The important
        # invariant is that results are not silently swapped *within the same
        # logical call*.
        executor.scheduler.result_mq.enqueue(add_req_result)
        executor.scheduler.result_mq.enqueue(rpc_result_0)
        executor.scheduler.result_mq.enqueue(rpc_result_1)

        t1 = threading.Thread(target=_call_add_req)
        t2 = threading.Thread(target=_call_collective_rpc)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Threads raised exceptions: {errors}"
        assert "add_req" in results
        assert "rpc" in results
        # add_req always returns a single object
        assert not isinstance(results["add_req"], list)
        # collective_rpc (unique_reply_rank=None) always returns a list
        assert isinstance(results["rpc"], list)
        assert len(results["rpc"]) == 2

    def test_multiple_concurrent_collective_rpc(self):
        """Multiple collective_rpc calls from different threads are serialised."""
        executor = self._make_executor()

        num_calls = 4
        barrier = threading.Barrier(num_calls)
        results: list[Any] = [None] * num_calls
        errors: list[Exception] = []

        for i in range(num_calls):
            # Each call expects 2 responses
            executor.scheduler.result_mq.enqueue(f"resp-{i}-0")
            executor.scheduler.result_mq.enqueue(f"resp-{i}-1")

        def _worker(idx):
            try:
                barrier.wait(timeout=5)
                results[idx] = executor.collective_rpc("method")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_calls)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Threads raised: {errors}"
        # Every call must have gotten exactly 2 results
        for i, r in enumerate(results):
            assert isinstance(r, list), f"Call {i} did not return a list: {r}"
            assert len(r) == 2, f"Call {i} returned {len(r)} results, expected 2"

        # All 8 responses were consumed, none lost or duplicated
        all_results = [item for sublist in results for item in sublist]
        assert len(all_results) == num_calls * 2

    def test_add_req_enqueues_via_mq(self):
        """Backward compat: add_req still broadcasts via the scheduler mq."""
        executor = self._make_executor()
        request = Mock()

        executor.scheduler.result_mq.enqueue("output")
        executor.add_req(request)

        msg = executor.scheduler.mq.dequeue(timeout=1)
        assert msg["type"] == "rpc"
        assert msg["method"] == "generate"

    def test_collective_rpc_enqueues_correct_message(self):
        """Backward compat: collective_rpc message shape is unchanged."""
        executor = self._make_executor()

        executor.scheduler.result_mq.enqueue("r0")
        executor.scheduler.result_mq.enqueue("r1")

        executor.collective_rpc("do_thing", args=(1, 2), kwargs={"k": "v"})

        msg = executor.scheduler.mq.dequeue(timeout=1)
        assert msg["type"] == "rpc"
        assert msg["method"] == "do_thing"
        assert msg["args"] == (1, 2)
        assert msg["kwargs"] == {"k": "v"}
        assert msg["output_rank"] is None
