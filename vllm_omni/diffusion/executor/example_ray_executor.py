"""
Example script demonstrating RayActorExecutor and DiffusionWorkerActor.
"""

import asyncio
import sys

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.diffusion.executor.ray_actor_executor import RayActorExecutor


async def example_usage():
    config = OmniDiffusionConfig(
        model="/mnt/nvme3n1/n0090/Qwen-Image-Edit",
        num_gpus=1,
        executor_class=RayActorExecutor,
    )

    print("Creating AsyncOmniDiffusion with Ray executor...")
    async_diffusion = AsyncOmniDiffusion(model=config.model, od_config=config)

    try:
        print("Generating example image...")

        result = await async_diffusion.generate(
             prompt="A beautiful sunset over the ocean",
             request_id="req-1",
        )
        print(f"✓ Completed: {result} outputs")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        await async_diffusion.close()
        print("✓ AsyncOmniDiffusion shutdown complete")


if __name__ == "__main__":
    print("=" * 60)
    print("Ray Actor Executor Example")
    print("=" * 60)

    try:
        import ray
        print("✓ Ray is installed")
    except ImportError:
        print("✗ Please install Ray: pip install ray")
        sys.exit(1)

    try:
        asyncio.run(example_usage())
        print("✓ Example completed successfully!")
    except Exception as e:
        print(f"✗ Example failed: {e}")
        import traceback; traceback.print_exc()