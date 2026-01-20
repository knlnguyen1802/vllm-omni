import asyncio
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, TransformerConfig
from vllm.transformers_utils.config import get_hf_file_to_dict

async def main():
    async_diffusion = AsyncOmniDiffusion(
        model="/mnt/nvme3n1/n0090/Qwen-Image-Edit",
        custom_pipeline_args={
            "pipeline_class": "custom_pipeline.CustomPipeline",
        },
    )
    # Run async inference
    results: DiffusionOutput = await async_diffusion.generate(
        prompt="A beautiful sunset over the ocean",
        request_id="req-1",
    )

    print("Generation result:", results)

if __name__ == "__main__":
    asyncio.run(main())
    
    

