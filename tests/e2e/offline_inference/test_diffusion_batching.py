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

Usage (standalone – not included in the regular CI test flow):

    python tests/e2e/offline_inference/test_diffusion_batching.py \
        --model <model_name_or_path> \
        --num-prompts 8

Or via pytest:

    pytest tests/e2e/offline_inference/test_diffusion_batching.py -s
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


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


# ------------------------------------------------------------------
# Warm-up
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
# Validation
# ------------------------------------------------------------------

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

    assert len(results) == len(prompts), (
        f"Expected {len(prompts)} results, got {len(results)}"
    )

    returned_ids = [r.request_id for r in results]
    for rid in request_ids:
        assert rid in returned_ids, f"Missing request_id {rid} in results"

    print("   ✅ All request_ids matched, results count correct.\n")


# ------------------------------------------------------------------
# Single vs Parallel comparison
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
    finally:
        omni.shutdown()

    speedup = single_time / parallel_time if parallel_time > 0 else float("inf")
    print("=" * 60)
    print(f"📊 Summary ({len(prompts)} prompts)")
    print(f"   Sequential : {single_time:.2f}s")
    print(f"   Parallel   : {parallel_time:.2f}s")
    print(f"   Speed-up   : {speedup:.2f}x")
    print("=" * 60)


# ------------------------------------------------------------------
# Main entrypoint
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
        elif mode == "batch":
            await run_batch(omni, prompts, label="measurement")
        elif mode == "single":
            await run_single(omni, prompts)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    finally:
        omni.shutdown()


# ------------------------------------------------------------------
# pytest entry point (for quick validation, not in CI flow)
# ------------------------------------------------------------------

def test_diffusion_batching_correctness():
    """Lightweight pytest-compatible smoke test.

    Instantiates AsyncOmni, fires 4 requests in parallel and checks
    that every request_id comes back.

    Skips automatically when the model is not available.
    """
    import os

    import pytest

    model = os.environ.get("DIFFUSION_TEST_MODEL")
    if not model:
        pytest.skip("Set DIFFUSION_TEST_MODEL env var to run this test")

    batch_size = int(os.environ.get("DIFFUSION_BATCH_SIZE", "1"))

    async def _inner():
        omni = AsyncOmni(model=model, diffusion_batch_size=batch_size)
        try:
            prompts = TEST_PROMPTS[:4]
            await validate_concurrent(omni, prompts)
        finally:
            omni.shutdown()

    asyncio.run(_inner())


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
        choices=["batch", "single", "compare", "validate"],
        default="compare",
        help="Run mode: 'batch' (parallel), 'single' (sequential), 'compare' (both), or 'validate' (correctness)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.model, args.num_prompts, args.mode, batch_size=args.batch_size))
