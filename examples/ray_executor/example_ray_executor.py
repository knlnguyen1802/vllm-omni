"""
Example script demonstrating RayActorExecutor and DiffusionWorkerActor.
"""

import asyncio
import sys

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.diffusion.executor.ray_actor_executor import RayActorExecutor

async def example_usage():
    # Note: Replace the model path with your actual model path
    model_path = "/mnt/nvme3n1/n0090/Qwen-Image-Edit"  # Or use a local path like "/mnt/nvme3n1/n0090/Qwen-Image-Edit"
    
    config = OmniDiffusionConfig(
        model=model_path,
        num_gpus=1,
        distributed_executor_backend=RayActorExecutor,  # Use ray_actor backend instead of executor_class
    )

    print("Creating AsyncOmniDiffusion with Ray executor...")
    async_diffusion = AsyncOmniDiffusion(model=config.model, od_config=config)

    try:
        print("Generating example image...")

        result = await async_diffusion.generate(
             prompt="A beautiful sunset over the ocean",
             request_id="req-1",
             num_inference_steps=20,  # Add num_inference_steps for better clarity
             height=512,  # Add height
             width=512,  # Add width
        )
        
        if hasattr(result, 'images') and result.images:
            print(f"✓ Completed: Generated {len(result.images)} image(s)")
            # Optionally save the first image
            if result.images:
                output_path = "ray_executor_output.png"
                result.images[0].save(output_path)
                print(f"✓ Image saved to {output_path}")
        else:
            print(f"✓ Completed: {result}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        async_diffusion.close()
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