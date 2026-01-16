#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example: Using RayActorExecutor for Diffusion

This example demonstrates how to use RayActorExecutor, which creates
a Ray actor that internally manages diffusion workers.

The pattern is similar to AsyncOmniDiffusion - simple and clean interface.
"""

import asyncio
import multiprocessing as mp

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor import RayActorExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest


async def main():
    """Main async function demonstrating RayActorExecutor usage."""
    print("=" * 60)
    print("RayActorExecutor Example")
    print("=" * 60)

    # Step 1: Create configuration
    print("\n1. Creating configuration...")
    config = OmniDiffusionConfig(
        model="/mnt/nvme3n1/n0090/Qwen-Image-Edit",  # Replace with your model path
        num_gpus=1,
        distributed_executor_backend="ray_actor",  # Use Ray actor executor
        # Optional Ray configuration
        ray_actor_cpus=1,
        ray_actor_gpus=1,
        # ray_actor_memory=32 * 1024 * 1024 * 1024,  # 32GB
        # ray_init_kwargs={"num_gpus": 1},
    )
    print(f"   Model: {config.model}")
    print(f"   Backend: {config.distributed_executor_backend}")

    # Step 2: Create engine (automatically creates Ray actor)
    print("\n2. Creating DiffusionEngine with RayActorExecutor...")
    engine = DiffusionEngine(config)
    print(f"   Engine created successfully")
    print(f"   Executor type: {type(engine.executor).__name__}")

    try:
        # Step 3: Check health
        print("\n3. Checking health...")
        engine.check_health()
        print("   ✓ Health check passed")

        # Step 4: Generate image
        print("\n4. Generating image...")
        requests = [
            OmniDiffusionRequest(
                prompt="A beautiful sunset over the ocean",
                height=512,
                width=512,
                num_inference_steps=20,
                num_outputs_per_prompt=1,
                request_id="req-1",
            )
        ]

        print(f"   Prompt: {requests[0].prompt}")
        print(f"   Size: {requests[0].width}x{requests[0].height}")
        print(f"   Steps: {requests[0].num_inference_steps}")

        # Execute request (this runs on Ray actor)
        result = engine.step(requests)

        if result and not result.error:
            print("   ✓ Generation completed successfully")
            if hasattr(result, "images") and result.images:
                print(f"   Generated {len(result.images)} image(s)")
        else:
            print(f"   ✗ Generation failed: {result.error if result else 'No result'}")

        # Step 5: Test RPC call
        print("\n5. Testing collective RPC...")
        # Example: Call a method on the worker
        try:
            # This calls the method on the worker inside the Ray actor
            rpc_result = engine.collective_rpc("get_rank")
            print(f"   RPC result: {rpc_result}")
        except Exception as e:
            print(f"   RPC call note: {e}")

        # Step 6: Multiple requests
        print("\n6. Generating multiple images...")
        multi_requests = [
            OmniDiffusionRequest(
                prompt="A serene mountain landscape",
                height=512,
                width=512,
                num_inference_steps=15,
                request_id="req-2",
            ),
            OmniDiffusionRequest(
                prompt="A bustling city at night",
                height=512,
                width=512,
                num_inference_steps=15,
                request_id="req-3",
            ),
        ]

        result = engine.step(multi_requests)
        if result:
            print(f"   ✓ Generated outputs for {len(multi_requests)} requests")

    finally:
        # Step 7: Cleanup
        print("\n7. Cleaning up...")
        engine.close()
        print("   ✓ Engine closed")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


async def simple_example():
    """Minimal example - just like AsyncOmniDiffusion."""
    # Create engine with Ray actor executor
    engine = DiffusionEngine(
        OmniDiffusionConfig(
            model="/mnt/nvme3n1/n0090/Qwen-Image-Edit",
            distributed_executor_backend="ray_actor",
        )
    )

    # Generate
    result = engine.step(
        [
            OmniDiffusionRequest(
                prompt="A beautiful sunset over the ocean",
                request_id="req-1",
            )
        ]
    )

    print(result)

    # Cleanup
    engine.close()


def synchronous_example():
    """Synchronous version - no async needed."""
    print("\n" + "=" * 60)
    print("Synchronous Example")
    print("=" * 60)

    # Create config
    config = OmniDiffusionConfig(
        model="/mnt/nvme3n1/n0090/Qwen-Image-Edit",
        distributed_executor_backend="ray_actor",
    )

    # Create engine
    engine = DiffusionEngine(config)

    try:
        # Generate
        result = engine.step(
            [
                OmniDiffusionRequest(
                    prompt="A futuristic cityscape",
                    height=512,
                    width=512,
                )
            ]
        )

        print(f"Result: {result}")

    finally:
        engine.close()


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()  # Safe even on Linux

    # Choose which example to run
    example = "full"  # "full", "simple", or "sync"

    if example == "full":
        # Run the full async example
        asyncio.run(main())
    elif example == "simple":
        # Run the minimal async example
        asyncio.run(simple_example())
    elif example == "sync":
        # Run synchronous example (no async)
        synchronous_example()
