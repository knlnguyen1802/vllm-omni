# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bulk abort + pause: empty request lists still pause AR EngineCore schedulers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine.messages import AbortRequestMessage, AbortResultMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeARClient:
    stage_type = "llm"
    final_output = True
    final_output_type = "text"

    def __init__(self) -> None:
        self.abort_calls: list[list[str]] = []
        self.pause_calls: list[dict[str, Any]] = []

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))

    async def pause_scheduler_async(self, **kwargs: Any) -> None:
        self.pause_calls.append(dict(kwargs))


class _FakeDiffusionClient(_FakeARClient):
    stage_type = "diffusion"
    final_output_type = "image"


@pytest.mark.asyncio
async def test_abort_requests_empty_without_pause_is_noop() -> None:
    client = _FakeARClient()
    pool = StagePool(0, [client])

    assert await pool.abort_requests([]) == []
    assert not client.abort_calls
    assert not client.pause_calls


@pytest.mark.asyncio
async def test_abort_requests_empty_with_pause_pauses_every_live_ar_replica() -> None:
    clients = [_FakeARClient(), _FakeARClient()]
    pool = StagePool(0, clients)

    assert await pool.abort_requests([], pause=True) == []
    for client in clients:
        assert not client.abort_calls
        assert client.pause_calls == [{"mode": "abort", "clear_cache": False}]


@pytest.mark.asyncio
async def test_abort_requests_pause_skips_diffusion_clients() -> None:
    client = _FakeDiffusionClient()
    pool = StagePool(0, [client])

    assert await pool.abort_requests([], pause=True) == []
    assert not client.abort_calls
    assert not client.pause_calls


@pytest.mark.asyncio
async def test_abort_requests_pause_pauses_only_affected_replicas() -> None:
    bound_client = _FakeARClient()
    idle_client = _FakeARClient()
    pool = StagePool(0, [bound_client, idle_client])
    pool._request_bindings["req-1"] = 0

    assert await pool.abort_requests(["req-1"], pause=True) == []
    assert bound_client.abort_calls == [["req-1"]]
    assert bound_client.pause_calls == [{"mode": "abort", "clear_cache": False}]
    assert not idle_client.abort_calls
    assert not idle_client.pause_calls


class _RecordingPool:
    def __init__(self, stage_id: int) -> None:
        self.stage_id = stage_id
        self.abort_calls: list[tuple[list[str], bool]] = []
        self.released: list[list[str]] = []

    async def abort_requests(self, request_ids: list[str], *, pause: bool = False) -> list[Any]:
        self.abort_calls.append((list(request_ids), pause))
        return []

    def release_bindings(self, request_ids: list[str]) -> None:
        self.released.append(list(request_ids))

    def get_bound_replica_id(self, _request_id: str) -> None:
        return None


def _bare_orchestrator(pools: list[_RecordingPool]) -> Orchestrator:
    orch = object.__new__(Orchestrator)
    orch.stage_pools = pools
    orch.request_states = {}
    orch._cfg_tracker = SimpleNamespace(
        get_parent_id=lambda _rid: None,
        is_companion_done=lambda _rid: True,
        cleanup_parent=lambda _rid: [],
        pop_pending_parent=lambda _pid: None,
    )
    orch.duplex_control_plane = None
    orch._pd_kv_params = {}
    orch._running_counter = None
    orch.output_async_queue = asyncio.Queue()
    orch.rpc_async_queue = asyncio.Queue()
    return orch


@pytest.mark.asyncio
async def test_orchestrator_empty_acknowledged_abort_with_pause_reaches_every_pool() -> None:
    pools = [_RecordingPool(0), _RecordingPool(1)]
    orch = _bare_orchestrator(pools)

    await orch._handle_abort(AbortRequestMessage(request_ids=[], rpc_id="rpc-pause", pause=True))

    for pool in pools:
        assert pool.abort_calls == [([], True)]
    result = orch.rpc_async_queue.get_nowait()
    assert isinstance(result, AbortResultMessage)
    assert result.rpc_id == "rpc-pause"
    assert result.success is True
    assert result.abort_outputs is None


@pytest.mark.asyncio
async def test_orchestrator_empty_abort_without_pause_stays_noop() -> None:
    pools = [_RecordingPool(0)]
    orch = _bare_orchestrator(pools)

    await orch._handle_abort(AbortRequestMessage(request_ids=[], rpc_id="rpc-noop"))

    assert pools[0].abort_calls == []
    result = orch.rpc_async_queue.get_nowait()
    assert isinstance(result, AbortResultMessage)
    assert result.success is True
