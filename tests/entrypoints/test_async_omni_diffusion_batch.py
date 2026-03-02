# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for AsyncOmniDiffusion.generate_batch.

Tests cover the new ``generate_batch`` method introduced in the
``batching_async_omni`` branch (on top of d6a3551):

1.  Auto-generates request_ids when none are supplied.
2.  Pads request_ids when fewer are supplied than prompts.
3.  Uses supplied request_ids as-is when count matches.
4.  Delegates to ``engine.step`` via the thread executor with a single
    ``OmniDiffusionRequest`` carrying all prompts.
5.  Stamps missing ``request_id`` on each result returned by the engine.
6.  Propagates ``lora_request`` into ``sampling_params``.
7.  Wraps engine exceptions in ``RuntimeError("Diffusion batch generation
    failed: ...")``.
8.  Sets ``guidance_scale_provided`` when ``guidance_scale`` is truthy.
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(request_id: str | None = None) -> OmniRequestOutput:
    r = MagicMock(spec=OmniRequestOutput)
    r.request_id = request_id
    return r


def _make_async_omni_diffusion(engine_step_return=None, engine_step_raises=None):
    """
    Build a minimal AsyncOmniDiffusion instance without real GPU or network
    access by bypassing ``__init__`` and wiring in mock internals.
    """
    from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

    obj = object.__new__(AsyncOmniDiffusion)

    # Minimal attributes expected by generate_batch
    mock_engine = MagicMock()
    if engine_step_raises is not None:

        def _raising_step(request):
            raise engine_step_raises

        mock_engine.step = _raising_step
    else:
        mock_engine.step = MagicMock(return_value=engine_step_return or [])

    obj.engine = mock_engine
    # Use a real single-thread executor so run_in_executor works
    obj._executor = ThreadPoolExecutor(max_workers=1)
    obj._closed = False
    return obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncOmniDiffusionBatchGenerate:
    """Tests for AsyncOmniDiffusion.generate_batch."""

    def _sampling_params(self, guidance_scale=0.0) -> OmniDiffusionSamplingParams:
        sp = OmniDiffusionSamplingParams(num_inference_steps=1)
        sp.guidance_scale = guidance_scale
        sp.guidance_scale_provided = False
        sp.lora_request = None
        return sp

    # ------------------------------------------------------------------
    # 1. Auto-generates request_ids
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_auto_generates_request_ids(self):
        prompts = ["a cat", "a dog"]
        results = [_make_result(request_id=None), _make_result(request_id=None)]
        obj = _make_async_omni_diffusion(engine_step_return=results)
        sp = self._sampling_params()

        outputs = await obj.generate_batch(prompts=prompts, sampling_params=sp)

        # engine.step was called once
        obj.engine.step.assert_called_once()
        call_arg = obj.engine.step.call_args[0][0]  # OmniDiffusionRequest

        assert len(call_arg.request_ids) == 2
        for rid in call_arg.request_ids:
            assert rid.startswith("diff-"), f"Unexpected id format: {rid}"

    # ------------------------------------------------------------------
    # 2. Pads request_ids when fewer than prompts
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_pads_request_ids(self):
        prompts = ["p1", "p2", "p3"]
        supplied_ids = ["my-id-0"]
        results = [_make_result() for _ in prompts]
        for i, r in enumerate(results):
            r.request_id = None
        obj = _make_async_omni_diffusion(engine_step_return=results)
        sp = self._sampling_params()

        await obj.generate_batch(prompts=prompts, sampling_params=sp, request_ids=supplied_ids)

        call_arg = obj.engine.step.call_args[0][0]
        assert call_arg.request_ids[0] == "my-id-0"
        # The other two should have been auto-generated
        assert call_arg.request_ids[1].startswith("diff-")
        assert call_arg.request_ids[2].startswith("diff-")
        assert len(call_arg.request_ids) == 3

    # ------------------------------------------------------------------
    # 3. Uses supplied request_ids when count matches
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_supplied_request_ids_used_as_is(self):
        prompts = ["p1", "p2"]
        supplied_ids = ["req-a", "req-b"]
        results = [_make_result("req-a"), _make_result("req-b")]
        obj = _make_async_omni_diffusion(engine_step_return=results)
        sp = self._sampling_params()

        outputs = await obj.generate_batch(prompts=prompts, sampling_params=sp, request_ids=supplied_ids)

        call_arg = obj.engine.step.call_args[0][0]
        assert call_arg.request_ids == ["req-a", "req-b"]
        assert len(outputs) == 2

    # ------------------------------------------------------------------
    # 4. Batches prompts into a single engine.step call
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_single_engine_step_call(self):
        prompts = ["x", "y", "z"]
        results = [_make_result(f"diff-{i}") for i in range(3)]
        obj = _make_async_omni_diffusion(engine_step_return=results)
        sp = self._sampling_params()

        await obj.generate_batch(prompts=prompts, sampling_params=sp)

        # Exactly one step call with all three prompts
        assert obj.engine.step.call_count == 1
        req = obj.engine.step.call_args[0][0]
        assert req.prompts == prompts

    # ------------------------------------------------------------------
    # 5. Stamps missing request_id on results
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stamps_missing_request_id(self):
        prompts = ["prompt1", "prompt2"]
        supplied_ids = ["id-0", "id-1"]
        # Engine returns results with no request_id set
        results = [_make_result(request_id=None), _make_result(request_id=None)]
        obj = _make_async_omni_diffusion(engine_step_return=results)
        sp = self._sampling_params()

        outputs = await obj.generate_batch(prompts=prompts, sampling_params=sp, request_ids=supplied_ids)

        assert outputs[0].request_id == "id-0"
        assert outputs[1].request_id == "id-1"

    # ------------------------------------------------------------------
    # 6. LoRA request is injected into sampling_params
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_lora_request_injected(self):
        prompts = ["p1"]
        results = [_make_result("r1")]
        obj = _make_async_omni_diffusion(engine_step_return=results)
        sp = self._sampling_params()
        lora = MagicMock()

        await obj.generate_batch(prompts=prompts, sampling_params=sp, lora_request=lora)

        assert sp.lora_request is lora

    # ------------------------------------------------------------------
    # 7. Engine exceptions are wrapped in RuntimeError
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_engine_exception_wrapped(self):
        obj = _make_async_omni_diffusion(engine_step_raises=ValueError("bad model"))
        sp = self._sampling_params()

        with pytest.raises(RuntimeError, match="Diffusion batch generation failed"):
            await obj.generate_batch(prompts=["p1"], sampling_params=sp)

    # ------------------------------------------------------------------
    # 8. guidance_scale_provided set when guidance_scale is truthy
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_guidance_scale_flag_set(self):
        results = [_make_result("r0")]
        obj = _make_async_omni_diffusion(engine_step_return=results)
        sp = self._sampling_params(guidance_scale=7.5)

        await obj.generate_batch(prompts=["a landscape"], sampling_params=sp)

        assert sp.guidance_scale_provided is True

    @pytest.mark.asyncio
    async def test_guidance_scale_flag_not_set_when_zero(self):
        results = [_make_result("r0")]
        obj = _make_async_omni_diffusion(engine_step_return=results)
        sp = self._sampling_params(guidance_scale=0.0)

        await obj.generate_batch(prompts=["a landscape"], sampling_params=sp)

        # Should remain False (falsy guidance_scale)
        assert sp.guidance_scale_provided is False
