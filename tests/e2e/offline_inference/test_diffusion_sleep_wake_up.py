# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import gc
import sys
import uuid
from pathlib import Path

import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.utils import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig


MODEL_NAME = "riverclouds/qwen_image_random"
PROMPT = "a photo of a cat sitting on a laptop keyboard"


def _sampling_params(seed: int = 42) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
        num_outputs_per_prompt=1,
    )


def _extract_images(output: OmniRequestOutput) -> list:
    if output.images:
        return output.images
    inner = getattr(output, "request_output", None)
    if inner is not None and hasattr(inner, "images") and inner.images:
        return inner.images
    return []


def _run_sync_roundtrip(tp_size: int) -> None:
    omni = None
    try:
        omni = Omni(
            model=MODEL_NAME,
            parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp_size),
        )

        before = omni.generate(PROMPT, _sampling_params(seed=11))[0]
        before_images = _extract_images(before)
        assert before.final_output_type == "image"
        assert len(before_images) >= 1

        # Trigger sleep/wake_up via control RPC (sync Omni path).
        omni.engine.collective_rpc(method="sleep", args=(1,))
        omni.engine.collective_rpc(method="wake_up", args=(None,))

        after = omni.generate(PROMPT, _sampling_params(seed=22))[0]
        after_images = _extract_images(after)
        assert after.final_output_type == "image"
        assert len(after_images) >= 1
        assert after_images[0].width == 256
        assert after_images[0].height == 256
    finally:
        if omni is not None:
            omni.close()
        cleanup_dist_env_and_memory()
        gc.collect()
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()


async def _collect_async_generate(omni: AsyncOmni, *, seed: int) -> OmniRequestOutput:
    last: OmniRequestOutput | None = None
    async for out in omni.generate(
        prompt={"prompt": PROMPT, "negative_prompt": "blurry"},
        request_id=f"sleep-wakeup-{seed}-{uuid.uuid4().hex[:8]}",
        sampling_params_list=[_sampling_params(seed=seed)],
    ):
        last = out

    if last is None:
        raise RuntimeError("No output received from AsyncOmni.generate")
    return last


async def _run_async_roundtrip(tp_size: int) -> None:
    omni = AsyncOmni(
        model=MODEL_NAME,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp_size),
    )
    try:
        before = await _collect_async_generate(omni, seed=33)
        before_images = _extract_images(before)
        assert before.final_output_type == "image"
        assert len(before_images) >= 1

        await omni.sleep(level=1)
        await omni.wake_up(tags=None)

        after = await _collect_async_generate(omni, seed=44)
        after_images = _extract_images(after)
        assert after.final_output_type == "image"
        assert len(after_images) >= 1
        assert after_images[0].width == 256
        assert after_images[0].height == 256
    finally:
        omni.shutdown()
        cleanup_dist_env_and_memory()
        gc.collect()
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 2})
@pytest.mark.parametrize("runtime", ["omni", "async_omni"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_diffusion_sleep_wake_up(runtime: str, tp_size: int) -> None:
    if current_omni_platform.is_rocm() or current_omni_platform.is_npu() or current_omni_platform.is_xpu():
        pytest.skip("qwen_image_random sleep/wake_up e2e is only enabled on CUDA.")

    if tp_size == 2 and (
        not current_omni_platform.is_available() or current_omni_platform.device_count() < 2
    ):
        pytest.skip("TP=2 sleep/wake_up test requires >= 2 devices.")

    if runtime == "omni":
        _run_sync_roundtrip(tp_size)
    else:
        asyncio.run(_run_async_roundtrip(tp_size))
