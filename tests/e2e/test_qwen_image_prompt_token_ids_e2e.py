# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test: generate with prompt_token_ids via AsyncOmni for QwenImage.

This test tokenizes a text prompt into token IDs, then calls
``AsyncOmni.generate()`` with ``prompt_token_ids`` to verify the
pre-tokenized path works end-to-end.

Usage (offline)::

    pytest tests/e2e/test_qwen_image_prompt_token_ids_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

pytestmark = [pytest.mark.core_model]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available — skipping GPU-only E2E test")


def _load_qwen_image_tokenizer(model: str) -> "PreTrainedTokenizer":
    """Load the Qwen2 tokenizer directly from the model repo.

    Uses the same ``tokenizer`` subfolder that ``QwenImagePipeline`` reads.
    """
    from transformers import AutoTokenizer

    # Check if *model* is a local path so we can pass local_files_only.
    local_files_only = os.path.exists(model)
    return AutoTokenizer.from_pretrained(
        model,
        subfolder="tokenizer",
        local_files_only=local_files_only,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestQwenImagePromptTokenIdsE2E:
    """End-to-end verification that prompt_token_ids flows correctly."""

    @pytest.fixture(scope="class")
    def engine(self) -> AsyncOmni:
        """Class-scoped engine — loaded once for all test methods."""
        _require_gpu()
        model = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
        print(f"\n[E2E] Initializing AsyncOmni(model={model!r}) ...")
        omni = AsyncOmni(
            model=model,
            log_stats=False,
            init_timeout=600,
        )
        yield omni
        print("\n[E2E] Shutting down AsyncOmni ...")
        omni.shutdown()

    @pytest.fixture(scope="class")
    def tokenizer(self) -> "PreTrainedTokenizer":
        """Load the tokenizer once for the test class."""
        model = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
        return _load_qwen_image_tokenizer(model)

    @pytest.fixture(scope="class")
    def tokenized_prompt(self, tokenizer: "PreTrainedTokenizer") -> list[list[int]]:
        """Tokenize 'A cup of coffee on the table' with the Qwen template."""
        text = "A cup of coffee on the table"
        # Apply the same template the pipeline uses internally.
        template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, "
            "texture, quantity, text, spatial relationships of the "
            "objects and background:<|im_end|>\n"
            "<|im_start|>user\n"
            "{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        formatted = template.format(text)
        encoded = tokenizer(formatted, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        print(f"\n[E2E] Tokenized prompt: {len(token_ids)} tokens")
        print(f"[E2E] First 10 token IDs: {token_ids[:10]}")
        return [token_ids]

    @pytest.mark.asyncio
    async def test_generate_with_prompt_token_ids_produces_valid_image(
        self,
        engine: AsyncOmni,
        tokenized_prompt: list[list[int]],
        tmp_path: Path,
    ) -> None:
        """Generate an image using pre-tokenized prompt_token_ids."""
        sampling_params = OmniDiffusionSamplingParams(
            height=256,
            width=256,
            num_inference_steps=8,
            num_outputs_per_prompt=1,
            seed=42,
            true_cfg_scale=4.0,
            guidance_scale=4.0,
        )

        prompt_dict = {
            "prompt_token_ids": tokenized_prompt[0],
        }

        print("\n[E2E] Calling engine.generate() with prompt_token_ids ...")
        outputs: list = []
        async for output in engine.generate(
            prompt=prompt_dict,
            request_id="e2e-token-ids-1",
            sampling_params_list=[sampling_params],
            output_modalities=["image"],
        ):
            outputs.append(output)

        assert len(outputs) > 0, "No outputs received from engine.generate()"
        result = outputs[-1]

        # Verify we got images
        req_out = getattr(result, "request_output", None)
        assert req_out is not None, "request_output missing"
        images = getattr(req_out, "images", [])
        assert len(images) > 0, "No images in request_output"

        # Save for manual inspection
        output_path = tmp_path / "qwen_image_prompt_token_ids_e2e.png"
        images[0].save(output_path)
        print(f"\n[E2E] Saved output image to: {output_path}")
        assert output_path.stat().st_size > 0, "Output image file is empty"

        # Verify it's a valid image
        from PIL import Image

        img = Image.open(output_path)
        assert img.size == (256, 256), f"Expected 256x256, got {img.size}"
        print(f"[E2E] Image size: {img.size}, mode: {img.mode}")

    @pytest.mark.asyncio
    async def test_generate_with_prompt_token_ids_vs_string_prompt_produces_same_shape(
        self,
        engine: AsyncOmni,
        tokenized_prompt: list[list[int]],
    ) -> None:
        """Both prompt_token_ids and string prompt should produce images."""
        torch.manual_seed(42)
        current_omni_platform.manual_seed(42)

        sampling_params = OmniDiffusionSamplingParams(
            height=256,
            width=256,
            num_inference_steps=8,
            num_outputs_per_prompt=1,
            seed=42,
            true_cfg_scale=4.0,
            guidance_scale=4.0,
        )

        # Path 1: prompt_token_ids
        prompt_ids_dict = {"prompt_token_ids": tokenized_prompt[0]}
        print("\n[E2E] Generating with prompt_token_ids ...")
        ids_outputs = []
        async for output in engine.generate(
            prompt=prompt_ids_dict,
            request_id="e2e-token-ids-2",
            sampling_params_list=[sampling_params],
            output_modalities=["image"],
        ):
            ids_outputs.append(output)

        ids_images = getattr(ids_outputs[-1].request_output, "images", [])
        assert len(ids_images) > 0, "No images from token_ids path"

        # Path 2: plain string prompt (same seed, should be deterministic)
        torch.manual_seed(42)
        current_omni_platform.manual_seed(42)

        prompt_str_dict = {"prompt": "A cup of coffee on the table"}
        print("\n[E2E] Generating with string prompt ...")
        str_outputs = []
        async for output in engine.generate(
            prompt=prompt_str_dict,
            request_id="e2e-token-ids-3",
            sampling_params_list=[sampling_params],
            output_modalities=["image"],
        ):
            str_outputs.append(output)

        str_images = getattr(str_outputs[-1].request_output, "images", [])
        assert len(str_images) > 0, "No images from string prompt path"

        # Both paths produce the same resolution
        assert ids_images[0].size == str_images[0].size, (
            f"Size mismatch: token_ids={ids_images[0].size} vs string={str_images[0].size}"
        )
        print(f"[E2E] Both paths produced {ids_images[0].size} images — OK")


# ---------------------------------------------------------------------------
# Standalone runner (python -m pytest or direct)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run as a standalone script for offline verification."""
    _require_gpu()

    async def _main() -> None:
        model = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
        print(f"Initializing AsyncOmni(model={model!r}) ...")
        engine = AsyncOmni(model=model, log_stats=False, init_timeout=600)

        try:
            tokenizer = _load_qwen_image_tokenizer(model)

            text = "A cup of coffee on the table"
            template = (
                "<|im_start|>system\n"
                "Describe the image by detailing the color, shape, size, "
                "texture, quantity, text, spatial relationships of the "
                "objects and background:<|im_end|>\n"
                "<|im_start|>user\n"
                "{}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            formatted = template.format(text)
            encoded = tokenizer(formatted, add_special_tokens=False)
            token_ids = encoded["input_ids"]
            print(f"Tokenized prompt: {len(token_ids)} tokens")

            sampling_params = OmniDiffusionSamplingParams(
                height=256,
                width=256,
                num_inference_steps=8,
                num_outputs_per_prompt=1,
                seed=42,
                true_cfg_scale=4.0,
                guidance_scale=4.0,
            )

            prompt_dict = {"prompt_token_ids": token_ids}
            print("Calling engine.generate() with prompt_token_ids ...")
            outputs = []
            async for output in engine.generate(
                prompt=prompt_dict,
                request_id="offline-e2e",
                sampling_params_list=[sampling_params],
                output_modalities=["image"],
            ):
                outputs.append(output)

            result = outputs[-1]
            images = getattr(result.request_output, "images", [])
            if images:
                output_path = Path("qwen_image_prompt_token_ids_output.png")
                images[0].save(output_path)
                print(f"Saved image to: {output_path} ({images[0].size})")
            else:
                print("ERROR: No images generated!")
        finally:
            engine.shutdown()

    asyncio.run(_main())
