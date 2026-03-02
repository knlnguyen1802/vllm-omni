# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for the refactored ``Scheduler`` class.

The ``batching_async_omni`` branch replaced ZMQ + ``MessageQueue`` with
plain ``multiprocessing.Queue`` objects.  These tests verify the new
public surface without spawning real worker processes.

Covered scenarios
-----------------
1.  ``initialize`` creates the expected number of broadcast queues and one
    result queue.
2.  ``get_broadcast_queues`` returns the same list returned by initialize.
3.  ``get_result_queue`` returns the shared result queue.
4.  ``broadcast`` puts the message into every per-worker queue exactly once.
5.  ``broadcast`` sends *independent* copies so one worker's queue
    consumption does not affect another.
6.  Re-initializing a Scheduler calls ``close`` first (no resource leak).
7.  ``add_req`` broadcasts the correct RPC envelope and returns the value
    placed on the result queue by the (mocked) worker.
8.  ``add_req`` raises ``RuntimeError`` when the worker signals an error.
9.  ``close`` drains and closes all queues; double-close is safe.
"""

import multiprocessing as mp
import queue as queue_module

import pytest

from vllm_omni.diffusion.scheduler import Scheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_od_config(num_gpus: int = 2):
    """Return a lightweight OmniDiffusionConfig stand-in."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.num_gpus = num_gpus
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchedulerInitialize:
    """Tests for Scheduler.initialize."""

    def test_creates_correct_number_of_broadcast_queues(self):
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=3))
        assert len(s.broadcast_queues) == 3

    def test_creates_result_queue(self):
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=1))
        assert s.result_queue is not None
        assert isinstance(s.result_queue, mp.queues.Queue)

    def test_get_broadcast_queues_returns_same_list(self):
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=2))
        assert s.get_broadcast_queues() is s.broadcast_queues

    def test_get_result_queue_returns_same_object(self):
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=2))
        assert s.get_result_queue() is s.result_queue

    def test_reinitialize_closes_old_queues(self):
        """Re-initializing must not keep stale queues from the first init."""
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=2))
        old_queues = list(s.broadcast_queues)

        # Re-initialize with a different worker count
        s.initialize(_make_od_config(num_gpus=4))

        assert len(s.broadcast_queues) == 4
        # Old queue objects should NOT be in the new list
        for old_q in old_queues:
            assert old_q not in s.broadcast_queues


class TestSchedulerBroadcast:
    """Tests for Scheduler.broadcast."""

    def test_broadcast_delivers_to_all_workers(self):
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=3))

        msg = {"type": "rpc", "method": "ping"}
        s.broadcast(msg)

        for i, q in enumerate(s.broadcast_queues):
            received = q.get_nowait()
            assert received == msg, f"Worker {i} did not receive the message"

    def test_broadcast_delivers_independent_copies(self):
        """
        Each worker should be able to consume the message independently.
        Consuming from worker-0's queue must not empty worker-1's queue.
        """
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=2))

        msg = "hello"
        s.broadcast(msg)

        # Worker 0 consumes
        assert s.broadcast_queues[0].get_nowait() == msg
        # Worker 1's queue is still full
        assert s.broadcast_queues[1].get_nowait() == msg

    def test_broadcast_multiple_messages_ordered(self):
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=1))

        for i in range(5):
            s.broadcast(i)

        received = []
        q = s.broadcast_queues[0]
        # mp.Queue.empty() is unreliable (feeder thread lag); use get(timeout) instead.
        for _ in range(5):
            received.append(q.get(timeout=2))
        assert received == list(range(5))


class TestSchedulerAddReq:
    """Tests for Scheduler.add_req."""

    def _build_scheduler_with_mock_result(self, result_value, num_gpus=1):
        """Return a Scheduler whose result_queue already has a value."""
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=num_gpus))
        # Pre-load the result queue as if a worker replied
        s.result_queue.put(result_value)
        return s

    def _make_request(self):
        from unittest.mock import MagicMock
        from vllm_omni.diffusion.request import OmniDiffusionRequest

        req = MagicMock(spec=OmniDiffusionRequest)
        return req

    def test_add_req_returns_worker_result(self):
        from vllm_omni.diffusion.data import DiffusionOutput

        expected = DiffusionOutput()
        s = self._build_scheduler_with_mock_result(expected)
        req = self._make_request()

        result = s.add_req(req)

        # mp.Queue serialises objects via pickle, so identity (is) cannot be
        # used across the queue boundary. DiffusionOutput is a dataclass with
        # auto-generated __eq__, so value equality is the correct assertion.
        assert result == expected

    def test_add_req_broadcasts_rpc_envelope(self):
        from vllm_omni.diffusion.data import DiffusionOutput

        s = self._build_scheduler_with_mock_result(DiffusionOutput())
        req = self._make_request()
        s.add_req(req)

        # All broadcast queues should have received exactly one RPC envelope.
        # Use get(timeout) instead of get_nowait(): mp.Queue uses a background
        # feeder thread, so the item may not be readable immediately after put().
        for i, q in enumerate(s.broadcast_queues):
            msg = q.get(timeout=2)
            assert isinstance(msg, dict), f"Worker {i} did not get a dict"
            assert msg["type"] == "rpc"
            assert msg["method"] == "generate"
            assert msg["exec_all_ranks"] is True
            assert msg["args"][0] is req

    def test_add_req_raises_on_worker_error_signal(self):
        error_payload = {"status": "error", "error": "something went wrong"}
        s = self._build_scheduler_with_mock_result(error_payload)
        req = self._make_request()

        with pytest.raises(RuntimeError, match="worker error"):
            s.add_req(req)


class TestSchedulerClose:
    """Tests for Scheduler.close."""

    def test_close_clears_broadcast_queues(self):
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=2))
        s.close()
        assert s.broadcast_queues == []

    def test_close_clears_result_queue(self):
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=1))
        s.close()
        assert s.result_queue is None

    def test_double_close_is_safe(self):
        """Calling close twice should not raise."""
        s = Scheduler()
        s.initialize(_make_od_config(num_gpus=1))
        s.close()
        s.close()  # should not raise

    def test_close_on_uninitialized_scheduler_is_safe(self):
        """Closing before initialize should not raise."""
        s = Scheduler()
        s.close()  # should be a no-op
