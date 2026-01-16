import asyncio
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, TransformerConfig
from vllm.transformers_utils.config import get_hf_file_to_dict

async def main():
    async_diffusion = AsyncOmniDiffusion(
        model="/mnt/nvme3n1/n0090/Qwen-Image-Edit",
        worker_extension_cls="custom_pipeline.WorkerExtension",
    )

    # Optional: reinitialize pipeline
    async_diffusion.engine.collective_rpc(
        method="re_init_pipeline",
        args=()
    )

    # Run async inference
    results: DiffusionOutput = await async_diffusion.generate(
        prompt="A beautiful sunset over the ocean",
        request_id="req-1",
    )

    print("Generation result:", results)
    # generate 
    # output = async_diffusion.generate(...)
    # from custom_pipeline import CustomPipeline
    # od_config = OmniDiffusionConfig(
    #     model="/mnt/nvme3n1/n0090/Qwen-Image-Edit",
    # )
    # tf_config_dict = get_hf_file_to_dict(
    #     "transformer/config.json",
    #     "/mnt/nvme3n1/n0090/Qwen-Image-Edit",
    # )
    # od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
    # custom_pipeline = CustomPipeline(od_config=od_config)

if __name__ == "__main__":
    asyncio.run(main())
    
    

