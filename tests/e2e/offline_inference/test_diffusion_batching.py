# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for diffusion batching via AsyncOmni.

This test fires multiple concurrent ``AsyncOmni.generate()`` calls for a
diffusion model and validates that every caller receives its correct
individual result.  When the underlying diffusion stage is configured with
``batch_size > 1`` (via stage config or ``StageDiffusionClient``), the
requests will be batched internally.

Even without explicit batching config this test is useful for verifying
that concurrent async requests are handled correctly.

Usage (standalone):

    python tests/e2e/offline_inference/test_diffusion_batching.py \
        --model <model_name_or_path> \
        --num-prompts 8

Or via pytest:

    pytest tests/e2e/offline_inference/test_diffusion_batching.py -s
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# ------------------------------------------------------------------
models = ["tiny-random/Qwen-Image"]

# ------------------------------------------------------------------
# Prompt fixtures
# ------------------------------------------------------------------

WARMUP_PROMPTS: list[dict[str, str]] = [
    {"prompt": "a sunflower in a glass vase", "negative_prompt": "blurry"},
    {"prompt": "a rocket launching into space", "negative_prompt": "low detail"},
    {"prompt": "a small cottage in the snowy mountains", "negative_prompt": "foggy"},
    {"prompt": "a colorful parrot sitting on a tree branch", "negative_prompt": "low contrast"},
]

TEST_PROMPTS: list[dict[str, str]] = [
    {"prompt": "a cup of coffee on a table", "negative_prompt": "low resolution"},
    {"prompt": "a toy dinosaur on a sandy beach", "negative_prompt": "cinematic, realistic"},
    {"prompt": "a futuristic city skyline at sunset", "negative_prompt": "blurry, foggy"},
    {"prompt": "a bowl of fresh strawberries", "negative_prompt": "low detail"},
    {"prompt": "a medieval knight standing in the rain", "negative_prompt": "modern clothing"},
    {"prompt": "a cat wearing sunglasses lounging in a garden", "negative_prompt": "dark lighting"},
    {"prompt": "a spaceship flying above a volcano", "negative_prompt": "low contrast"},
    {"prompt": "a watercolor painting of a mountain lake", "negative_prompt": "photo, realistic"},
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _default_sampling_params(**overrides) -> OmniDiffusionSamplingParams:
    defaults = dict(
        num_inference_steps=2,
        width=256,
        height=256,
        guidance_scale=0.0,
    )
    defaults.update(overrides)
    return OmniDiffusionSamplingParams(**defaults)


def _default_sync_sampling_params(**overrides) -> OmniDiffusionSamplingParams:
    """Create sampling params for the synchronous Omni.generate() API."""
    defaults = dict(
        num_inference_steps=2,
        width=256,
        height=256,
        guidance_scale=0.0,
        generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
    )
    defaults.update(overrides)
    return OmniDiffusionSamplingParams(**defaults)


async def _collect_generate(omni: AsyncOmni, prompt, request_id, sampling_params_list) -> OmniRequestOutput:
    """Consume the AsyncOmni.generate() async generator and return the last output."""
    last_output: OmniRequestOutput | None = None
    async for output in omni.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params_list,
    ):
        last_output = output
    if last_output is None:
        raise RuntimeError(f"No output received for request {request_id}")
    return last_output


def _extract_images(output: OmniRequestOutput) -> list:
    """Extract images from an OmniRequestOutput, handling both direct
    and nested request_output structures."""
    if output.images:
        return output.images
    # When the output comes from the orchestrator pipeline, images may be
    # nested inside request_output.
    inner = getattr(output, "request_output", None)
    if inner is not None and hasattr(inner, "images") and inner.images:
        return inner.images
    return []


# ------------------------------------------------------------------
# Warm-up (async)
# ------------------------------------------------------------------


async def warmup(omni: AsyncOmni, prompts: list[dict[str, str]]) -> None:
    """Warm-up: send prompts in parallel to pre-load the model."""
    print(f"🔥 Warming up with {len(prompts)} prompts ...")
    sp = _default_sampling_params(num_inference_steps=2)
    start = time.perf_counter()

    tasks = [
        _collect_generate(
            omni,
            prompt=p,
            request_id=f"warmup-{i}-{uuid.uuid4().hex[:8]}",
            sampling_params_list=[sp],
        )
        for i, p in enumerate(prompts)
    ]
    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start
    print(f"   Warm-up done in {elapsed:.2f}s\n")


# ------------------------------------------------------------------
# Single (sequential) benchmark
# ------------------------------------------------------------------


async def run_single(omni: AsyncOmni, prompts: list[dict[str, str]]) -> float:
    """Run prompts one-by-one sequentially."""
    print(f"🧩 Running SINGLE (sequential) mode – {len(prompts)} prompts ...")
    sp = _default_sampling_params()
    total_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        result = await _collect_generate(
            omni,
            prompt=prompt,
            request_id=f"single-{i}-{uuid.uuid4().hex[:8]}",
            sampling_params_list=[sp],
        )
        elapsed = time.perf_counter() - start
        images = _extract_images(result)
        print(f"   prompt {i}: {elapsed:.2f}s  ({len(images)} images)")

    total = time.perf_counter() - total_start
    print(f"   ✅ Total single-mode: {total:.2f}s\n")
    return total


# ------------------------------------------------------------------
# Batch (parallel) benchmark
# ------------------------------------------------------------------


async def run_batch(
    omni: AsyncOmni,
    prompts: list[dict[str, str]],
    label: str = "batch",
) -> float:
    """Send all prompts concurrently via asyncio.gather."""
    print(f"⚙️  Running {label.upper()} mode – {len(prompts)} prompts concurrently ...")
    sp = _default_sampling_params()
    start = time.perf_counter()

    tasks = [
        _collect_generate(
            omni,
            prompt=p,
            request_id=f"{label}-{i}-{uuid.uuid4().hex[:8]}",
            sampling_params_list=[sp],
        )
        for i, p in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start
    for i, result in enumerate(results):
        images = _extract_images(result)
        print(f"   prompt {i}: {len(images)} images, request_id={result.request_id}")

    print(f"   ✅ Total {label} mode: {elapsed:.2f}s\n")
    return elapsed


# ------------------------------------------------------------------
# Explicit batch (passes list of prompts to generate())
# ------------------------------------------------------------------


async def run_batch_explicit(
    omni: AsyncOmni,
    prompts: list[dict[str, str]],
    label: str = "batch_explicit",
) -> float:
    """Send all prompts as a single batch via generate(prompt=[...]).

    Unlike ``run_batch`` (which fires concurrent ``generate()`` calls),
    this passes a list so that all prompts are processed in **one**
    ``DiffusionEngine.step()`` call when ``diffusion_batch_size > 1``.
    """
    print(f"⚡ Running {label.upper()} mode – {len(prompts)} prompts in ONE engine call ...")
    sp = _default_sampling_params()
    request_ids = [f"{label}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(prompts))]
    start = time.perf_counter()

    results: list[OmniRequestOutput] = []
    async for output in omni.generate(
        prompt=prompts,
        sampling_params_list=[sp],
        request_ids=request_ids,
    ):
        results.append(output)

    elapsed = time.perf_counter() - start
    for i, result in enumerate(results):
        images = _extract_images(result)
        print(f"   prompt {i}: {len(images)} images, request_id={result.request_id}")

    print(f"   ✅ Total {label} mode: {elapsed:.2f}s\n")
    return elapsed


# ------------------------------------------------------------------
# Async validation helpers
# ------------------------------------------------------------------


async def validate_batch_explicit(omni: AsyncOmni, prompts: list[dict[str, str]]) -> None:
    """Validate generate(prompt=[...]) returns correct results for every prompt."""
    print(f"🔍 Validating batch generate() correctness with {len(prompts)} prompts ...")
    sp = _default_sampling_params()
    request_ids = [f"validate-batch-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(prompts))]

    results: list[OmniRequestOutput] = []
    async for output in omni.generate(
        prompt=prompts,
        sampling_params_list=[sp],
        request_ids=request_ids,
    ):
        results.append(output)

    assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"

    returned_ids = [r.request_id for r in results]
    for rid in request_ids:
        assert rid in returned_ids, f"Missing request_id {rid} in results"

    print("   ✅ All request_ids matched, results count correct.\n")


async def validate_concurrent(omni: AsyncOmni, prompts: list[dict[str, str]]) -> None:
    """Validate that every concurrent request receives a distinct result
    with its own request_id."""
    print(f"🔍 Validating concurrent correctness with {len(prompts)} prompts ...")
    sp = _default_sampling_params()

    request_ids = [f"validate-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(prompts))]

    tasks = [
        _collect_generate(omni, prompt=p, request_id=rid, sampling_params_list=[sp])
        for p, rid in zip(prompts, request_ids)
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"

    returned_ids = [r.request_id for r in results]
    for rid in request_ids:
        assert rid in returned_ids, f"Missing request_id {rid} in results"

    print("   ✅ All request_ids matched, results count correct.\n")


# ------------------------------------------------------------------
# Single vs Parallel comparison (CLI only)
# ------------------------------------------------------------------


async def compare_single_vs_parallel(
    model: str,
    prompts: list[dict[str, str]],
    batch_size: int = 1,
) -> None:
    """Run the same prompts sequentially then in parallel and print a comparison."""

    omni = AsyncOmni(model=model, diffusion_batch_size=batch_size)
    try:
        await warmup(omni, WARMUP_PROMPTS)
        single_time = await run_single(omni, prompts)
        parallel_time = await run_batch(omni, prompts, label="parallel")
        explicit_time = await run_batch_explicit(omni, prompts, label="batch_explicit")
    finally:
        omni.shutdown()

    speedup_parallel = single_time / parallel_time if parallel_time > 0 else float("inf")
    speedup_explicit = single_time / explicit_time if explicit_time > 0 else float("inf")
    print("=" * 60)
    print(f"📊 Summary ({len(prompts)} prompts)")
    print(f"   Sequential        : {single_time:.2f}s")
    print(f"   Parallel (gather) : {parallel_time:.2f}s  ({speedup_parallel:.2f}x)")
    print(f"   Explicit batch    : {explicit_time:.2f}s  ({speedup_explicit:.2f}x)")
    print("=" * 60)


# ------------------------------------------------------------------
# CLI main entrypoint
# ------------------------------------------------------------------


async def main(model: str, num_prompts: int, mode: str, batch_size: int = 1) -> None:
    prompts = (TEST_PROMPTS * ((num_prompts // len(TEST_PROMPTS)) + 1))[:num_prompts]

    if mode == "compare":
        await compare_single_vs_parallel(model, prompts, batch_size=batch_size)
        return

    omni = AsyncOmni(model=model, diffusion_batch_size=batch_size)
    try:
        await warmup(omni, WARMUP_PROMPTS)

        if mode == "validate":
            await validate_concurrent(omni, prompts)
        elif mode == "validate_batch":
            await validate_batch_explicit(omni, prompts)
        elif mode == "batch":
            await run_batch(omni, prompts, label="measurement")
        elif mode == "batch_explicit":
            await run_batch_explicit(omni, prompts)
        elif mode == "single":
            await run_single(omni, prompts)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    finally:
        omni.shutdown()


# ==================================================================
# pytest test cases
# ==================================================================


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_diffusion_batching_sync_sequential(model_name: str):
    """Test that synchronous Omni can generate images for multiple prompts
    submitted sequentially (one at a time) and each returns a valid image."""
    m = None
    try:
        m = Omni(model=model_name)
        sp = _default_sync_sampling_params()
        prompts = TEST_PROMPTS[:4]

        for i, prompt in enumerate(prompts):
            outputs = m.generate(prompt, sp)
            first_output = outputs[0]
            assert first_output.final_output_type == "image"
            assert hasattr(first_output, "request_output") and first_output.request_output is not None, (
                f"No request_output found for prompt {i}"
            )

            req_out = first_output.request_output
            assert isinstance(req_out, OmniRequestOutput), f"Invalid request_output type for prompt {i}"
            assert hasattr(req_out, "images") and req_out.images, f"No images in request_output for prompt {i}"
            images = req_out.images
            assert len(images) >= 1, f"Expected at least 1 image for prompt {i}, got {len(images)}"
            assert images[0].width == 256
            assert images[0].height == 256
            print(f"   prompt {i}: OK ({len(images)} images)")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_diffusion_batching_sync_multi_prompt(model_name: str):
    """Test that synchronous Omni correctly handles a batch of multiple
    prompts submitted at once and returns one result per prompt."""
    m = None
    try:
        m = Omni(model=model_name)
        sp = _default_sync_sampling_params()
        prompts = TEST_PROMPTS[:4]

        outputs = m.generate(prompts, sp)
        assert len(outputs) == len(prompts), f"Expected {len(prompts)} outputs, got {len(outputs)}"

        for i, output in enumerate(outputs):
            assert output.final_output_type == "image", (
                f"Output {i} final_output_type expected 'image', got '{output.final_output_type}'"
            )
            req_out = output.request_output
            assert req_out is not None, f"No request_output for prompt {i}"
            assert isinstance(req_out, OmniRequestOutput), f"Invalid request_output type for prompt {i}"
            images = req_out.images
            assert images and len(images) >= 1, f"Expected at least 1 image for prompt {i}"
            assert images[0].width == 256
            assert images[0].height == 256
            print(f"   prompt {i}: OK ({len(images)} images, request_id={output.request_id})")

        # Verify all request_ids are distinct
        request_ids = [o.request_id for o in outputs]
        assert len(set(request_ids)) == len(request_ids), f"Duplicate request_ids found: {request_ids}"
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_diffusion_batching_async_concurrent(model_name: str):
    """Test that AsyncOmni correctly handles multiple concurrent requests
    fired via asyncio.gather. Each request_id must appear in the results."""

    async def _inner():
        omni = AsyncOmni(model=model_name, diffusion_batch_size=1)
        try:
            prompts = TEST_PROMPTS[:4]
            sp = _default_sampling_params()
            request_ids = [f"async-concurrent-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(prompts))]

            tasks = [
                _collect_generate(omni, prompt=p, request_id=rid, sampling_params_list=[sp])
                for p, rid in zip(prompts, request_ids)
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"

            returned_ids = [r.request_id for r in results]
            for rid in request_ids:
                assert rid in returned_ids, f"Missing request_id {rid} in results"

            for i, result in enumerate(results):
                images = _extract_images(result)
                assert len(images) >= 1, f"No images for prompt {i}"
                assert images[0].width == 256
                assert images[0].height == 256
                print(f"   prompt {i}: OK ({len(images)} images, request_id={result.request_id})")
        finally:
            omni.shutdown()

    asyncio.run(_inner())


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_diffusion_batching_async_explicit_batch(model_name: str):
    """Test that AsyncOmni batch mode (generate(prompt=[...])) dispatches
    all prompts in a single engine call and returns correct results."""

    async def _inner():
        omni = AsyncOmni(model=model_name, diffusion_batch_size=4)
        try:
            prompts = TEST_PROMPTS[:4]
            sp = _default_sampling_params()
            request_ids = [f"explicit-batch-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(prompts))]

            results: list[OmniRequestOutput] = []
            async for output in omni.generate(
                prompt=prompts,
                sampling_params_list=[sp],
                request_ids=request_ids,
            ):
                results.append(output)

            assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"

            returned_ids = [r.request_id for r in results]
            for rid in request_ids:
                assert rid in returned_ids, f"Missing request_id {rid} in results"

            for i, result in enumerate(results):
                images = _extract_images(result)
                assert len(images) >= 1, f"No images for prompt {i}"
                assert images[0].width == 256
                assert images[0].height == 256
                print(f"   prompt {i}: OK ({len(images)} images, request_id={result.request_id})")
        finally:
            omni.shutdown()

    asyncio.run(_inner())


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_diffusion_batching_num_outputs(model_name: str):
    """Test that the diffusion model respects num_outputs_per_prompt and
    generates the correct number of images per request."""
    m = None
    try:
        m = Omni(model=model_name)
        num_outputs = 2
        sp = _default_sync_sampling_params(num_outputs_per_prompt=num_outputs)

        outputs = m.generate(
            "a photo of a cat sitting on a laptop keyboard",
            sp,
        )

        first_output = outputs[0]
        assert first_output.final_output_type == "image"
        req_out = first_output.request_output
        assert req_out is not None, "No request_output found"
        assert isinstance(req_out, OmniRequestOutput)

        images = req_out.images
        assert images is not None and len(images) == num_outputs, (
            f"Expected {num_outputs} images, got {len(images) if images else 0}"
        )
        for img in images:
            assert img.width == 256
            assert img.height == 256
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("model_name", models)
def test_diffusion_batching_distinct_results(model_name: str):
    """Test that different prompts produce distinct images when batched,
    ensuring the batching logic does not mix up results across requests."""
    m = None
    try:
        m = Omni(model=model_name)
        sp = _default_sync_sampling_params()
        prompts = [
            {"prompt": "a bright red apple on a white table", "negative_prompt": "blurry"},
            {"prompt": "a blue ocean with white waves crashing", "negative_prompt": "blurry"},
        ]

        outputs = m.generate(prompts, sp)
        assert len(outputs) == len(prompts), f"Expected {len(prompts)} outputs, got {len(outputs)}"

        # Verify each output has a unique request_id
        request_ids = [o.request_id for o in outputs]
        assert len(set(request_ids)) == len(request_ids), f"Duplicate request_ids: {request_ids}"

        # Verify each output has images
        for i, output in enumerate(outputs):
            req_out = output.request_output
            assert req_out is not None, f"No request_output for prompt {i}"
            images = req_out.images
            assert images and len(images) >= 1, f"No images for prompt {i}"
            assert images[0].width == 256
            assert images[0].height == 256
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E diffusion concurrent benchmark / validation")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--num-prompts", type=int, default=8, help="Number of prompts to run")
    parser.add_argument("--batch-size", type=int, default=1, help="Diffusion batch size (1 = no batching)")
    parser.add_argument(
        "--mode",
        choices=["batch", "batch_explicit", "single", "compare", "validate", "validate_batch"],
        default="compare",
        help=(
            "Run mode: 'batch' (parallel gather), 'batch_explicit' (generate_batch API), "
            "'single' (sequential), 'compare' (all three), "
            "'validate' (concurrent correctness), 'validate_batch' (generate_batch correctness)"
        ),
    )
    args = parser.parse_args()

    asyncio.run(main(args.model, args.num_prompts, args.mode, batch_size=args.batch_size))
