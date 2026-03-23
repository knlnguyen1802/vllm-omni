import asyncio
import time
import uuid

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


# ==========================================================
# 🔥 Warm-up (Batch Version)
# ==========================================================
async def warmup_batch(omni: AsyncOmni):
    """Warm up using batch generation to pre-load the model.

    AsyncOmni.generate() supports batch prompts by passing a list of
    prompts as the ``prompt`` argument.  When the diffusion stage receives
    a list it dispatches them in a single DiffusionEngine.step() call.
    """
    warmup_prompts = [
        {"prompt": "a sunflower in a glass vase", "negative_prompt": "blurry"},
        {"prompt": "a rocket launching into space", "negative_prompt": "low detail"},
        {"prompt": "a small cottage in the snowy mountains", "negative_prompt": "foggy"},
        {"prompt": "a colorful parrot sitting on a tree branch", "negative_prompt": "low contrast"},
        {"prompt": "a steaming bowl of ramen noodles", "negative_prompt": "dark lighting"},
        {"prompt": "a classic car on a desert road", "negative_prompt": "modern"},
        {"prompt": "a neon-lit street in Tokyo at night", "negative_prompt": "low resolution"},
        {"prompt": "a dragon flying above a castle", "negative_prompt": "low detail"},
    ]

    print(f"Running warm-up (batch mode) with {len(warmup_prompts)} requests...")
    start = time.time()

    request_id = f"warmup-{uuid.uuid4().hex[:8]}"
    async for _ in omni.generate(
        warmup_prompts,
        request_id=request_id,
        sampling_params_list=[
            OmniDiffusionSamplingParams(
                num_inference_steps=10,  # fewer steps for faster warmup
                width=256,
                height=256,
            )
        ],
    ):
        pass  # discard warmup outputs

    duration = time.time() - start
    print(f"Warm-up batch completed in {duration:.2f} seconds.\n")


# ==========================================================
# 🧩 Single Sequential Run
# ==========================================================
async def run_single(omni: AsyncOmni, prompts: list[dict[str, str]]):
    """Run prompts sequentially using the .generate() API and measure total time."""
    print("Running sequential single mode inference...")
    total_start = time.time()

    for i, prompt in enumerate(prompts):
        start = time.time()
        request_id = f"single-{i}"

        async for output in omni.generate(
            prompt,
            request_id=request_id,
            sampling_params_list=[
                OmniDiffusionSamplingParams(num_inference_steps=10, width=512, height=512)
            ],
        ):
            if output.images:
                image = output.images[0]
                image.save(f"single_{i}.jpg")

        print(f"Single mode prompt {i} took: {time.time() - start:.2f} seconds")

    total_time = time.time() - total_start
    print(f"\nTotal single-mode runtime for {len(prompts)} prompts: {total_time:.2f} seconds\n")


# ==========================================================
# ⚙️ Unified Batch Mode
# ==========================================================
async def run_batch(omni: AsyncOmni, prompts: list[dict[str, str]], label: str = "batch"):
    """Run image generation in a single batch asynchronously.

    AsyncOmni.generate() accepts a list of prompts and internally routes
    them through the batch diffusion path.  A single OmniRequestOutput is
    returned containing **all** generated images.
    """
    print(f"Running {label} mode with {len(prompts)} prompts in one batch...")
    start = time.time()

    request_id = f"{label}-{uuid.uuid4().hex[:8]}"

    # Batch mode yields ONE output with ALL images combined.
    async for output in omni.generate(
        prompts,
        request_id=request_id,
        sampling_params_list=[
            OmniDiffusionSamplingParams(num_inference_steps=10, width=512, height=512)
        ],
    ):
        if output.images:
            for i, image in enumerate(output.images):
                image.save(f"{label}_{i}.jpg")
            print(f"Saved {len(output.images)} images from batch output")

    duration = time.time() - start
    print(f"{label.capitalize()} batch mode took: {duration:.2f} seconds\n")


# ==========================================================
# 🚀 Main
# ==========================================================
async def main():
    omni = AsyncOmni(model="/data/n0090/Qwen-Image")

    test_prompts = [
        {"prompt": "a cup of coffee on a table", "negative_prompt": "low resolution"},
        {"prompt": "a toy dinosaur on a sandy beach", "negative_prompt": "cinematic, realistic"},
        {"prompt": "a futuristic city skyline at sunset", "negative_prompt": "blurry, foggy"},
        {"prompt": "a bowl of fresh strawberries", "negative_prompt": "low detail"},
        {"prompt": "a medieval knight standing in the rain", "negative_prompt": "modern clothing"},
        {"prompt": "a cat wearing sunglasses lounging in a garden", "negative_prompt": "dark lighting"},
        {"prompt": "a spaceship flying above a volcano", "negative_prompt": "low contrast"},
        {"prompt": "a watercolor painting of a mountain lake", "negative_prompt": "photo, realistic"},
    ]

    mode = "batch"  # options: "batch" or "single"

    # 🔥 Warm up (batch mode)
    await warmup_batch(omni)

    print("Running measurement...")

    if mode == "batch":
        await run_batch(omni, test_prompts, label="measurement")
    else:
        await run_single(omni, test_prompts)


if __name__ == "__main__":
    asyncio.run(main())
