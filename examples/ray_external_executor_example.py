#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example: Using RayExternalDiffusionExecutor

This example demonstrates how to use the Ray-based external executor
for diffusion models. The executor connects to Ray actors that manage
worker processes, allowing for flexible distributed execution.

Architecture:
    1. Create Ray actors (RayDiffusionWorkerActor) - one per GPU
    2. Each actor wraps a WorkerWrapperBase instance
    3. DiffusionEngine uses RayExternalDiffusionExecutor
    4. Executor forwards all calls to Ray actors
    5. Ray actors execute calls on their workers

Usage:
    python ray_external_executor_example.py
"""

import os
import sys

import ray

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.ray_external_executor import (
    RayDiffusionWorkerActor,
    RayExternalDiffusionExecutor,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest


def create_worker_actors(num_gpus: int, od_config: OmniDiffusionConfig) -> list:
    """Create and initialize Ray worker actors.

    Args:
        num_gpus: Number of GPUs to use
        od_config: Diffusion configuration

    Returns:
        List of initialized Ray actor handles
    """
    print(f"Creating {num_gpus} Ray worker actors...")

    worker_actors = []
    for rank in range(num_gpus):
        # Apply @ray.remote decorator to create actor class
        worker_actor_class = ray.remote(RayDiffusionWorkerActor)
        
        # Create actor with GPU resources
        actor = worker_actor_class.options(
            num_gpus=1,  # Each actor gets 1 GPU
            name=f"diffusion_worker_{rank}",  # Named actor for discovery
            max_concurrency=1,  # Process one request at a time
        ).remote(rank=rank, od_config=od_config)

        worker_actors.append(actor)
        print(f"  Created worker actor for rank {rank}")

    # Initialize all actors in parallel
    print("Initializing worker actors...")
    init_futures = [actor.initialize.remote() for actor in worker_actors]
    init_results = ray.get(init_futures)

    # Check initialization results
    for rank, result in enumerate(init_results):
        if result["status"] != "ready":
            raise RuntimeError(f"Worker {rank} failed to initialize: {result.get('error')}")
        print(f"  Worker {rank} initialized successfully")

    print("All worker actors ready!")
    return worker_actors


def main():
    """Main example function."""
    # Configuration
    num_gpus = 2
    model_path = "your-model-path"  # Replace with your model

    # Step 1: Initialize Ray
    print("=" * 60)
    print("Step 1: Initialize Ray")
    print("=" * 60)

    if not ray.is_initialized():
        ray.init(
            num_gpus=num_gpus,
            # Configure Ray as needed
            # runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}},
        )
        print(f"Ray initialized with {num_gpus} GPUs")
    else:
        print("Ray already initialized")

    try:
        # Step 2: Create configuration
        print("\n" + "=" * 60)
        print("Step 2: Create diffusion configuration")
        print("=" * 60)

        od_config = OmniDiffusionConfig(
            model=model_path,
            num_gpus=num_gpus,
            # Will be overridden when creating engine
            distributed_executor_backend="mp",
        )
        print(f"Configuration created for {num_gpus} GPUs")

        # Step 3: Create Ray worker actors
        print("\n" + "=" * 60)
        print("Step 3: Create and initialize Ray worker actors")
        print("=" * 60)

        worker_actors = create_worker_actors(num_gpus, od_config)

        # Step 4: Set up environment for executor to find actors
        print("\n" + "=" * 60)
        print("Step 4: Configure executor to use Ray actors")
        print("=" * 60)

        actor_names = [f"diffusion_worker_{i}" for i in range(num_gpus)]
        os.environ["DIFFUSION_WORKER_ACTOR_NAMES"] = ",".join(actor_names)
        print(f"Environment configured with actor names: {actor_names}")

        # Step 5: Create DiffusionEngine with Ray external executor
        print("\n" + "=" * 60)
        print("Step 5: Create DiffusionEngine with RayExternalDiffusionExecutor")
        print("=" * 60)

        # Override executor backend
        od_config.distributed_executor_backend = RayExternalDiffusionExecutor

        engine = DiffusionEngine(od_config)
        print("DiffusionEngine created successfully!")
        print(f"Executor type: {type(engine.executor).__name__}")

        # Step 6: Test the engine
        print("\n" + "=" * 60)
        print("Step 6: Test diffusion generation")
        print("=" * 60)

        requests = [
            OmniDiffusionRequest(
                prompt="a beautiful mountain landscape at sunset",
                height=512,
                width=512,
                num_inference_steps=20,
                num_outputs_per_prompt=1,
            )
        ]

        print(f"Submitting request: {requests[0].prompt}")
        output = engine.step(requests)

        if output:
            print("Generation completed successfully!")
            if hasattr(output, "images") and output.images:
                print(f"Generated {len(output.images)} image(s)")
        else:
            print("Generation returned no output")

        # Step 7: Health check
        print("\n" + "=" * 60)
        print("Step 7: Health check")
        print("=" * 60)

        try:
            engine.check_health()
            print("All workers are healthy!")
        except Exception as e:
            print(f"Health check failed: {e}")

        # Step 8: Test RPC call
        print("\n" + "=" * 60)
        print("Step 8: Test collective RPC")
        print("=" * 60)

        # Example RPC call to all workers
        results = engine.collective_rpc("get_rank")
        print(f"Worker ranks: {results}")

        # Example RPC call to single worker
        rank_0_result = engine.collective_rpc("get_rank", unique_reply_rank=0)
        print(f"Rank 0 returned: {rank_0_result}")

        # Step 9: Cleanup
        print("\n" + "=" * 60)
        print("Step 9: Cleanup")
        print("=" * 60)

        print("Closing engine (disconnecting from workers)...")
        engine.close()
        print("Engine closed")

        print("Shutting down worker actors...")
        # Terminate Ray actors
        shutdown_futures = [actor.shutdown.remote() for actor in worker_actors]
        ray.get(shutdown_futures)

        # Kill actors
        for actor in worker_actors:
            ray.kill(actor)

        print("Worker actors shut down")

        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # Optional: shutdown Ray
        # ray.shutdown()
        pass


if __name__ == "__main__":
    main()
