# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for AsyncOmni Qwen-Image generation flow (no Ray, no HTTP server)."""

from __future__ import annotations

import asyncio
import atexit
import os
import shutil
import uuid
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from tests.utils import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

MODEL_REPO = "tiny-random/Qwen-Image"
LOCAL_MODEL_PATH = Path(os.path.expanduser("~/models/tiny-random/Qwen-Image"))
CACHE_DIR = Path(os.path.expanduser("~/.cache/tiny-random/Qwen-Image"))

CUSTOM_PIPELINE_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.qwen_image_pipeline_with_logprob."
    "QwenImagePipelineWithLogProbForTest"
)
WORKER_EXTENSION_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.worker_extension."
    "vLLMOmniColocateWorkerExtensionForTest"
)


def ensure_model_available() -> str:
    """Ensure the model weights and tokenizer are available for testing.

    If missing locally, download from HF Hub, cache temporarily, and
    mark for deletion on process exit.
    """
    if LOCAL_MODEL_PATH.exists():
        print(f"\u2705 Using local model at {LOCAL_MODEL_PATH}")
        return str(LOCAL_MODEL_PATH)

    print(f"\u26a0\ufe0f Local model not found at {LOCAL_MODEL_PATH}. "
          f"Pulling from Hugging Face Hub...")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hf_model_path = snapshot_download(
        repo_id=MODEL_REPO,
        cache_dir=str(CACHE_DIR),
        local_files_only=False,
        resume_download=True,
    )

    print(f"\u2705 Downloaded model to cache: {hf_model_path}")

    def _cleanup():
        print(f"\U0001f9f9 Cleaning up downloaded model cache: {hf_model_path}")
        shutil.rmtree(hf_model_path, ignore_errors=True)

    atexit.register(_cleanup)
    return hf_model_path


MODEL = ensure_model_available()

_TOKENIZER_CACHE = None


def _get_tokenizer():
    """Lazy load tokenizer from the resolved model path."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(
            MODEL, trust_remote_code=True
        )
    return _TOKENIZER_CACHE


def _tokenize_prompt(text: str) -> list[int]:
    """Tokenize a text prompt into valid token IDs for the model."""
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer not found. ensure_model_available() should have "
            "downloaded the model; check that the repo exists on HF Hub."
        )
    messages = [{"role": "user", "content": text}]
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return token_ids


def _sampling_params(*, logprobs: bool = False, seed: int = 42) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=2,
        guidance_scale=0.0,
        height=256,
        width=256,
        seed=seed,
        extra_args={"logprobs": logprobs},
    )


async def _generate_once(
    engine: AsyncOmni,
    prompt: str | list[int] | dict,
    *,
    request_id: str,
    sampling_params: OmniDiffusionSamplingParams,
) -> OmniRequestOutput:
    # Convert text prompt to dict with tokenized prompt_ids
    if isinstance(prompt, str):
        prompt_ids = _tokenize_prompt(prompt)
        prompt = {"prompt_ids": prompt_ids}
    elif isinstance(prompt, list):
        prompt = {"prompt_ids": prompt}
    # else: assume it's already a dict with prompt_ids

    last_output = None
    async for output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=[sampling_params],
        output_modalities=["image"],
    ):
        last_output = output

    assert last_output is not None
    assert isinstance(last_output, OmniRequestOutput)
    return last_output


def _assert_valid_image_output(output: OmniRequestOutput) -> None:
    assert output.final_output_type == "image"
    assert output.images, "Expected at least one generated image"

    image = output.images[0]
    arr = np.asarray(image, dtype=np.float32) / 255.0

    assert arr.ndim == 3 and arr.shape[2] == 3, f"Expected HWC RGB image, got shape={arr.shape}"
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    assert 0.0 <= float(arr[0, 0, 0]) <= 1.0


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_async_omni_generate():
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
        )
        after.callback(engine.shutdown)

        output = await _generate_once(
            engine,
            "a beautiful sunset over the ocean with vibrant clouds",
            request_id=f"test_{uuid.uuid4().hex[:8]}",
            sampling_params=_sampling_params(logprobs=False, seed=42),
        )

        _assert_valid_image_output(output)


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_async_omni_generate_with_logprobs():
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
        )
        after.callback(engine.shutdown)

        output = await _generate_once(
            engine,
            "a futuristic city at night with neon lights",
            request_id=f"test_lp_{uuid.uuid4().hex[:8]}",
            sampling_params=_sampling_params(logprobs=True, seed=123),
        )

        _assert_valid_image_output(output)

        all_log_probs = output.custom_output.get("all_log_probs")
        assert all_log_probs is not None, "all_log_probs should be present when logprobs=True"
        assert hasattr(all_log_probs, "shape")
        assert all_log_probs.numel() > 0


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_async_omni_generate_concurrent():
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
        )
        after.callback(engine.shutdown)

        prompts = [
            "a beautiful sunset over the ocean with vibrant clouds",
            "a fluffy orange cat on a windowsill in sunlight",
            "a mountain landscape with snow and a frozen lake",
            "a futuristic city skyline with flying cars",
        ]

        tasks = [
            _generate_once(
                engine,
                prompt,
                request_id=f"concurrent_{i}_{uuid.uuid4().hex[:8]}",
                sampling_params=_sampling_params(logprobs=False, seed=100 + i),
            )
            for i, prompt in enumerate(prompts)
        ]

        outputs = await asyncio.gather(*tasks)

        assert len(outputs) == len(prompts)
        for output in outputs:
            _assert_valid_image_output(output)

