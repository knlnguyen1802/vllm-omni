# from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

# def main():
#     async_diffusion = AsyncOmniDiffusion(model="/mnt/nvme3n1/n0090/Qwen-Image-Edit",num_gpus=2)

# if __name__ == "__main__":
#     import multiprocessing as mp
#     mp.freeze_support()  # safe even on Linux
#     main()
# import ray
# import torch

# @ray.remote(num_gpus=1)
# def train():
#     visible_devices = torch.cuda.device_count()
#     print("Visible GPUs:", visible_devices)
    
#     if visible_devices > 0:
#         # Always use local index 0 in Ray tasks
#         torch.cuda.set_device(1)
#         print("Using device:", torch.cuda.current_device())
#         print("Device name:", torch.cuda.get_device_name(0))
#     else:
#         print("No GPUs visible inside Ray task!")

# ray.init(address="auto")  # connect to cluster
# ray.get(train.remote())

import ray
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
import os
@ray.remote(num_gpus=2)
class DiffusionActor:
    def __init__(self, model_path="/mnt/nvme3n1/n0090/Qwen-Image-Edit"):
        print("Initializing AsyncOmniDiffusion model...")
        print(f"Os cuda visible devices is {os.environ["CUDA_VISIBLE_DEVICES"]}")
        print(f"Ray private resource {ray._private.state.available_resources_per_node()}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        # Create the diffusion model using the specified GPUs
        self.async_diffusion = AsyncOmniDiffusion(model=model_path, num_gpus=1)
        # print("Model initialized successfully!")

    def warmup(self):
        """Optional method for basic startup check."""
        return "AsyncOmniDiffusion model is ready."

def main():
    # Initialize Ray; if you were connecting to a Ray cluster, use ray.init(address="auto")
    ray.init()

    # Create a diffusion actor
    diffusion_actor = DiffusionActor.remote()

    # Run a simple test to confirm things work
    result = ray.get(diffusion_actor.warmup.remote())
    print(result)

    # You can add inference calls here later:
    # e.g., ray.get(diffusion_actor.generate.remote(prompt="a crystal castle under moonlight"))

    # Final note: Ray keeps the actor alive; program ends here if not using it interactively
    print("Done.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Safe for Windows
    main()
# import ray
# import os
# import torch

# # Optional: increase debug visibility
# ray.init(include_dashboard=False, ignore_reinit_error=True)

# print("=== Driver ===")
# print("Driver torch.cuda.device_count:", torch.cuda.device_count())
# print("Driver visible GPUs:", os.getenv("CUDA_VISIBLE_DEVICES"))
# print("Ray cluster resources:", ray._private.state.available_resources_per_node())

# # --- The key: unset CUDA_VISIBLE_DEVICES inside a GPU worker ---
# @ray.remote(num_gpus=1)
# def gpu_worker_forced_missing_env():
#     import os, torch

#     # Simulate Ray not populating CUDA_VISIBLE_DEVICES (or user unsets it)
#     if "CUDA_VISIBLE_DEVICES" in os.environ:
#         del os.environ["CUDA_VISIBLE_DEVICES"]

#     # Now check what PyTorch sees
#     print("\n=== Inside Ray GPU worker (no CUDA_VISIBLE_DEVICES env) ===")
#     print("os.environ['CUDA_VISIBLE_DEVICES']:", os.getenv("CUDA_VISIBLE_DEVICES"))
#     print("ray.get_gpu_ids():", ray.get_gpu_ids())
#     print("torch.cuda.device_count():", torch.cuda.device_count())

#     # Try to actually use the GPU driver directly
#     try:
#         x = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
#         msg = f"✅ Tensor placed on GPU device cuda:0: {x}"
#     except Exception as e:
#         msg = f"❌ Failed to use GPU explicitly: {e}"

#     return msg

# result = ray.get(gpu_worker_forced_missing_env.remote())
# # print("\nResult from worker:", result)