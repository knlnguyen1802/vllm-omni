"""Unit tests for AR EngineCore vs diffusion worker pause/sleep routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.diffusion.data import CuMemTag, OmniACK
from vllm_omni.entrypoints.async_omni import AsyncOmni


def _make_omni(*, stage_types: list[str]) -> AsyncOmni:
    omni = object.__new__(AsyncOmni)
    omni._name = "AsyncOmni"
    omni._pause_cond = asyncio.Condition()
    omni._paused = False
    omni._sleeping_tags = set()
    omni._level2_sleeping = False
    omni.event_resolver = SimpleNamespace(watch_task=lambda *a, **k: None, resolve=AsyncMock())
    omni._final_output_handler = lambda: None

    stage_clients = [SimpleNamespace(stage_type=stage_type) for stage_type in stage_types]
    omni.engine = SimpleNamespace(
        stage_clients=stage_clients,
        stage_vllm_configs=[None] * len(stage_types),
        pause_scheduler_async=AsyncMock(),
        resume_scheduler_async=AsyncMock(),
        sleep_async=AsyncMock(),
        wake_up_async=AsyncMock(),
    )
    omni.collective_rpc = AsyncMock(return_value=[])
    return omni


@pytest.mark.cpu
def test_split_stage_ids_by_type():
    omni = _make_omni(stage_types=["llm", "diffusion", "llm"])
    ar_ids, diff_ids = omni._split_stage_ids_by_type()
    assert ar_ids == [0, 2]
    assert diff_ids == [1]


@pytest.mark.cpu
def test_pause_generation_routes_ar_to_engine_core():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni.reset_prefix_cache = AsyncMock(return_value=True)
        omni.reset_mm_cache = AsyncMock()
        omni.reset_encoder_cache = AsyncMock()

        await omni.pause_generation(mode="abort", clear_cache=True)

        assert omni._paused is True
        omni.engine.pause_scheduler_async.assert_awaited_once_with(
            stage_ids=[0],
            mode="abort",
            clear_cache=True,
        )
        # Diffusion has no EngineCore pause path; only frontend gate applies.
        assert omni.engine.pause_scheduler_async.await_args.kwargs["stage_ids"] == [0]

    asyncio.run(run())


@pytest.mark.cpu
def test_resume_generation_resumes_ar_then_clears_frontend_pause():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni._paused = True

        await omni.resume_generation()

        omni.engine.resume_scheduler_async.assert_awaited_once_with(stage_ids=[0])
        assert omni._paused is False

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_routes_ar_to_engine_core_and_diffusion_to_worker_rpc():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        diffusion_ack = OmniACK(task_id="diff", status="SUCCESS", stage_id=1, rank=0)
        omni._sleep_diffusion = AsyncMock(return_value=[diffusion_ack])

        acks = await omni.sleep(stage_ids=[0, 1], level=1, mode="abort")

        omni.engine.sleep_async.assert_awaited_once_with(stage_ids=[0], level=1, mode="abort")
        omni._sleep_diffusion.assert_awaited_once_with([1], 1)
        assert {ack.stage_id for ack in acks} == {0, 1}
        assert any(ack.metadata.get("path") == "engine_core" for ack in acks if ack.stage_id == 0)
        assert CuMemTag.WEIGHTS.value in omni._sleeping_tags
        assert CuMemTag.KV_CACHE.value in omni._sleeping_tags

    asyncio.run(run())


@pytest.mark.cpu
def test_wake_up_routes_ar_to_engine_core_and_diffusion_to_worker_rpc():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni._sleeping_tags = {CuMemTag.WEIGHTS.value, CuMemTag.KV_CACHE.value}
        diffusion_ack = OmniACK(task_id="diff", status="SUCCESS", stage_id=1, rank=0)
        omni._wake_diffusion = AsyncMock(return_value=[diffusion_ack])

        acks = await omni.wake_up(stage_ids=[0, 1])

        omni.engine.wake_up_async.assert_awaited_once()
        assert omni.engine.wake_up_async.await_args.kwargs["stage_ids"] == [0]
        omni._wake_diffusion.assert_awaited_once()
        assert {ack.stage_id for ack in acks} == {0, 1}
        assert not omni._sleeping_tags

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_diffusion_only_skips_engine_core():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni._sleep_diffusion = AsyncMock(
            return_value=[OmniACK(task_id="d", status="SUCCESS", stage_id=0, rank=0)]
        )

        await omni.sleep(level=1)

        omni.engine.sleep_async.assert_not_awaited()
        omni._sleep_diffusion.assert_awaited_once_with([0], 1)

    asyncio.run(run())
