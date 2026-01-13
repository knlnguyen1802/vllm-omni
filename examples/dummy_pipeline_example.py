"""
Example of using the dummy pipeline mode.

This mode is useful for:
- Fast initialization without loading models upfront
- Testing the pipeline infrastructure without loading actual models
- Debugging distributed execution logic
- Quick validation of configuration changes
- CI/CD testing without GPU requirements
- Lazy model loading - start with dummy, load components separately as needed
"""

from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionParallelConfig
from vllm_omni.diffusion.worker.gpu_worker import GPUWorker

# Example 1: Basic dummy pipeline configuration
def example_basic_dummy_pipeline():
    """Create a basic dummy pipeline configuration."""
    config = OmniDiffusionConfig(
        model="dummy-model",
        use_dummy_pipeline=True,  # Enable dummy pipeline mode
        num_gpus=1,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(
            data_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )
    
    # Worker will initialize with dummy pipeline (no model loading)
    # worker = GPUWorker(local_rank=0, rank=0, od_config=config)
    print("Configuration created for dummy pipeline")
    print(f"use_dummy_pipeline: {config.use_dummy_pipeline}")
    return config


# Example 2: Dummy pipeline with parallel configuration
def example_dummy_pipeline_with_parallelism():
    """Create a dummy pipeline with parallel configuration for testing."""
    config = OmniDiffusionConfig(
        model="dummy-model",
        use_dummy_pipeline=True,
        num_gpus=2,
        parallel_config=DiffusionParallelConfig(
            data_parallel_size=2,
            tensor_parallel_size=1,
        ),
    )
    
    print("Multi-GPU dummy pipeline configuration")
    print(f"Number of GPUs: {config.num_gpus}")
    print(f"Data parallel size: {config.parallel_config.data_parallel_size}")
    return config


# Example 3: Switching between real and dummy pipeline
def example_conditional_dummy_pipeline(use_dummy: bool = False):
    """Toggle between real and dummy pipeline based on flag."""
    config = OmniDiffusionConfig(
        model="Qwen/Qwen3-Omni" if not use_dummy else "dummy-model",
        use_dummy_pipeline=use_dummy,
        num_gpus=1,
        dtype="bfloat16",
    )
    
    if use_dummy:
        print("Using dummy pipeline (fast initialization, no actual inference)")
    else:
        print("Using real pipeline (will load actual model)")
    
    return config


# Example 4: Granular initialization - load components separately
def example_granular_initialization():
    """Start with dummy pipeline and initialize components separately as needed."""
    # Start with dummy pipeline for fast initialization
    config = OmniDiffusionConfig(
        model="Qwen/Qwen3-Omni",  # Model name for later loading
        use_dummy_pipeline=True,  # Start with dummy
        num_gpus=1,
    )
    
    print("Step 1: Initialize with dummy pipeline (instant)")
    # worker = GPUWorker(local_rank=0, rank=0, od_config=config)
    # At this point, worker.pipeline is a DummyPipeline
    
    print("\nStep 2: When ready, initialize distributed environment")
    # worker.init_distributed_env()
    
    print("Step 3: Initialize model parallelism")
    # worker.init_model_parallel()
    
    print("Step 4: Load the actual model")
    # worker.load_model()
    
    print("Step 5: (Optional) Setup CPU offload if needed")
    # worker.setup_cpu_offload()
    
    print("Step 6: (Optional) Compile model with torch.compile")
    # worker.compile_model()
    
    print("Step 7: (Optional) Setup cache backend")
    # worker.setup_cache_backend()
    
    print("\nYou have full control over when each component is initialized!")
    print("This allows flexible initialization based on your application needs.")
    return config


# Example 5: Partial initialization - only what you need
def example_partial_initialization():
    """Load only specific components you need."""
    config = OmniDiffusionConfig(
        model="Qwen/Qwen3-Omni",
        use_dummy_pipeline=True,
        num_gpus=1,
        enforce_eager=True,  # Skip compilation
        cache_backend="none",  # No cache
    )
    
    print("Example: Initialize only distributed env and model, skip extras")
    # worker = GPUWorker(local_rank=0, rank=0, od_config=config)
    # worker.init_distributed_env()
    # worker.init_model_parallel()
    # worker.load_model()
    # # Skip: setup_cpu_offload, compile_model, setup_cache_backend
    
    print("Model loaded, extras skipped based on config!")
    return config


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic Dummy Pipeline")
    print("=" * 60)
    example_basic_dummy_pipeline()
    
    print("\n" + "=" * 60)
    print("Example 2: Dummy Pipeline with Parallelism")
    print("=" * 60)
    example_dummy_pipeline_with_parallelism()
    
    print("\n" + "=" * 60)
    print("Example 3: Conditional Pipeline (Dummy)")
    print("=" * 60)
    example_conditional_dummy_pipeline(use_dummy=True)
    
    print("\n" + "=" * 60)
    print("Example 3: Conditional Pipeline (Real)")
    print("=" * 60)
    example_conditional_dummy_pipeline(use_dummy=False)
    
    print("\n" + "=" * 60)
    print("Example 4: Granular Initialization")
    print("=" * 60)
    example_granular_initialization()
    
    print("\n" + "=" * 60)
    print("Example 5: Partial Initialization")
    print("=" * 60)
    example_partial_initialization()
