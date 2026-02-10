# Custom Pipeline Extension Guide

This guide demonstrates how to use the newly added features for extending vLLM-Omni's diffusion pipeline with custom functionality.

## Overview

Three main features enable custom pipeline extension:

1. **`WorkerWrapperBase`**: A wrapper class that enables dynamic worker extension with custom functionality
2. **`load_format`**: A parameter that controls how diffusion models are loaded, including support for custom pipelines
3. **`CustomPipelineWorkerExtension`**: An extension class that enables pipeline re-initialization with custom implementations

## Features

### WorkerWrapperBase

`WorkerWrapperBase` is a wrapper class that creates `DiffusionWorker` instances with optional extension support. It enables dynamic inheritance, allowing you to add custom methods and functionality to workers without modifying the base worker class.

**Key capabilities:**
- Dynamic worker class extension via `worker_extension_cls`
- Support for custom pipeline initialization via `custom_pipeline_args`
- Method delegation to underlying worker
- Attribute access forwarding

**Location:** `vllm_omni/diffusion/worker/diffusion_worker.py`

### load_format Parameter

The `load_format` parameter controls how diffusion models are loaded. It supports the following values:

- **`"default"`**: Standard model loading using the model registry (default behavior)
- **`"custom_pipeline"`**: Load a custom pipeline class specified by `custom_pipeline_name`
- **`"dummy"`**: Skip model loading (useful for testing or when pipeline will be initialized separately)

**Location:** `vllm_omni/diffusion/model_loader/diffusers_loader.py`

### CustomPipelineWorkerExtension

`CustomPipelineWorkerExtension` is a mixin class that extends `DiffusionWorker` with the ability to re-initialize the pipeline with a custom implementation.

**Key method:**
- `re_init_pipeline(custom_pipeline_args)`: Re-initializes the pipeline with custom arguments, properly cleaning up the old pipeline first

**Location:** `vllm_omni/diffusion/worker/diffusion_worker.py`

## Usage Example

### Step 1: Create a Custom Pipeline

Create a custom pipeline class that extends an existing pipeline. In this example, we extend `QwenImageEditPipeline` to add trajectory tracking:

```python
# custom_pipeline.py
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import QwenImageEditPipeline
import torch

class CustomPipeline(QwenImageEditPipeline):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)

    def forward(self, req, prompt=None, negative_prompt=None, **kwargs):
        # Call parent's forward to get normal output
        output = super().forward(req=req, prompt=prompt, negative_prompt=negative_prompt, **kwargs)
        
        # Add custom trajectory data
        actual_num_steps = req.sampling_params.num_inference_steps or kwargs.get('num_inference_steps', 50)
        output.trajectory_timesteps = torch.linspace(1000, 0, actual_num_steps, dtype=torch.float32)
        output.trajectory_latents = torch.randn(actual_num_steps, 1, 16, 64, 64, dtype=torch.float32)
        
        return output
```

### Step 2: Use the Custom Pipeline with Omni

Initialize the `Omni` engine with custom pipeline configuration:

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Initialize with custom pipeline
omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    diffusion_load_format="dummy",  # Skip initial loading
    custom_pipeline_args={
        "pipeline_class": "custom_pipeline.CustomPipeline"
    },
)

# Generate with the custom pipeline
outputs = omni.generate(
    {
        "prompt": "Edit this image",
        "multi_modal_data": {"image": input_image},
    },
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        true_cfg_scale=4.0,
    ),
)

# Access custom trajectory data
output = outputs[0].request_output[0]
print(f"Trajectory timesteps shape: {output.metrics['trajectory_timesteps'].shape}")
print(f"Trajectory latents shape: {output.latents.shape}")
```

### Step 3: Run the Example

The example provided in this directory demonstrates the complete workflow:

```bash
cd examples/offline_inference/custom_pipeline/image_to_image

# Run with custom pipeline
python image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image cherry_blossom.jpg \
    --prompt "Let this mascot dance under the moon, surrounded by floating stars" \
    --output output_image_edit.png \
    --num_inference_steps 10
```

## Advanced Usage

### Custom Worker Extension

You can create custom worker extensions to add new methods beyond pipeline re-initialization:

```python
from typing import Any

class MyCustomExtension:
    def custom_method(self):
        """Your custom worker method."""
        return "custom_result"
    
    def another_method(self, data: Any):
        """Another custom method."""
        # Access worker internals via self
        return self.model_runner.some_operation(data)

# Use with Omni (internal API)
from vllm_omni.diffusion.worker.diffusion_worker import WorkerWrapperBase

wrapper = WorkerWrapperBase(
    gpu_id=0,
    od_config=od_config,
    worker_extension_cls=MyCustomExtension,
)

# Now you can call custom methods
result = wrapper.worker.custom_method()
```

### Combining Extensions with Custom Pipeline

You can combine custom worker extensions with custom pipelines:

```python
omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    diffusion_load_format="dummy",
    custom_pipeline_args={
        "pipeline_class": "custom_pipeline.CustomPipeline"
    },
    # Note: worker_extension_cls is an internal parameter
    # CustomPipelineWorkerExtension is automatically applied when custom_pipeline_args is provided
)
```

## Implementation Details

### How It Works

1. **Initialization Flow:**
   - `Omni` creates a diffusion executor with `custom_pipeline_args`
   - The executor spawns worker processes using `WorkerProc.worker_main`
   - `WorkerProc` creates a `WorkerWrapperBase` with the extension configuration
   - `WorkerWrapperBase._prepare_worker_class()` dynamically creates an extended worker class
   - If `custom_pipeline_args` is provided, `CustomPipelineWorkerExtension` is added to the worker
   - After worker initialization, `re_init_pipeline()` is called to load the custom pipeline

2. **Custom Pipeline Loading:**
   - `DiffusionModelRunner.load_model()` is called with `load_format="custom_pipeline"`
   - The loader uses `resolve_obj_by_qualname()` to import the custom pipeline class
   - The custom pipeline is instantiated and replaces the default pipeline

3. **Method Resolution:**
   - Methods called on `WorkerWrapperBase` are delegated to the wrapped worker via `execute_method()` or `__getattr__()`
   - The worker instance has methods from both `DiffusionWorker` and any extension classes
   - Custom pipeline methods override parent pipeline methods via standard Python inheritance

## Best Practices

1. **Pipeline Inheritance**: Always extend an existing pipeline class (e.g., `QwenImageEditPipeline`) to maintain compatibility

2. **Module Import Path**: Use fully qualified module paths for `pipeline_class` (e.g., `"my_module.CustomPipeline"`)

3. **Output Compatibility**: Ensure your custom pipeline returns `DiffusionOutput` objects compatible with the vLLM-Omni framework

4. **Resource Cleanup**: The framework automatically cleans up the old pipeline when re-initializing, but be mindful of additional resources your custom pipeline might allocate

5. **Testing**: Use `diffusion_load_format="dummy"` during development to skip initial model loading and test your custom pipeline in isolation

## Files in This Example

- `custom_pipeline.py`: Custom pipeline implementation that extends `QwenImageEditPipeline`
- `image_edit.py`: Complete example demonstrating custom pipeline usage
- `run.sh`: Shell script to run the example
- `README.md`: This documentation file

## See Also

- Unit tests: `tests/diffusion/test_worker_wrapper_base.py`
- Worker implementation: `vllm_omni/diffusion/worker/diffusion_worker.py`
- Model loader: `vllm_omni/diffusion/model_loader/diffusers_loader.py`
- Pipeline registry: `vllm_omni/diffusion/registry.py`
