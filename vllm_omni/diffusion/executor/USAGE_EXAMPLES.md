# Using Custom Executors - Quick Examples

## Example 1: Default Executor (No Configuration Needed)

```python
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

# Create config without executor_class - uses MultiProcDiffusionExecutor by default
config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    model_name_or_path="Qwen/Qwen3-Omni",
    num_gpus=2,
)

engine = DiffusionEngine(config)
request = OmniDiffusionRequest(prompt="A beautiful landscape")
output = engine.step([request])
engine.close()
```

## Example 2: Custom Executor via String Path

```python
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig

# Specify executor by string path
config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    num_gpus=1,
    executor_class="my_company.inference.CustomExecutor",  # Full path
)

engine = DiffusionEngine(config)
# Engine will automatically use your CustomExecutor
```

## Example 2b: Using Shorthand Names

```python
# Use shorthand names for built-in executors
config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    num_gpus=2,
    executor_class="multiproc",  # Shorthand for MultiProcDiffusionExecutor
    # Also supported: "mp", "external", "http"
)

engine = DiffusionEngine(config)
```

## Example 3: Custom Executor via Class Import

```python
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig
from my_company.inference import CustomExecutor

# Specify executor by class reference
config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    num_gpus=1,
    executor_class=CustomExecutor,  # Direct class reference
)

engine = DiffusionEngine(config)
```

## Example 4: HTTP Remote Executor

```python
from vllm_omni.diffusion.executor import HTTPDiffusionExecutor
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig

# Use the built-in HTTP executor example (after implementing it)
config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    num_gpus=0,  # No local GPUs needed
    executor_class=HTTPDiffusionExecutor,
    # Add your server URL to config if needed
)

engine = DiffusionEngine(config)
```

## Example 5: Creating a Simple Custom Executor

```python
# File: my_executor.py
from vllm_omni.diffusion.executor import DiffusionExecutor
from vllm_omni.diffusion.data import DiffusionOutput
import requests

class RemoteAPIExecutor(DiffusionExecutor):
    """Send requests to a remote API."""
    
    def _init_executor(self):
        self.api_url = "https://api.example.com/v1/diffusion"
        self.api_key = "your-api-key"
    
    def collective_rpc(self, method, **kwargs):
        # For simplicity, just return success
        return [True]
    
    def execute_model(self, requests):
        # Send to remote API
        payload = {"requests": [r.to_dict() for r in requests]}
        response = requests.post(
            self.api_url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        
        # Convert response to DiffusionOutput
        data = response.json()
        return DiffusionOutput(
            output=data.get("images"),
            error=None,
            trajectory_latents=None,
            trajectory_timesteps=None,
        )
    
    def check_health(self):
        response = requests.get(f"{self.api_url}/health")
        if response.status_code != 200:
            raise RuntimeError("API is unhealthy")
    
    def shutdown(self):
        pass  # Nothing to clean up

# Usage:
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig

config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    executor_class="my_executor.RemoteAPIExecutor",  # Or: executor_class=RemoteAPIExecutor
)

engine = DiffusionEngine(config)
```

## Example 6: Conditional Executor Selection

```python
import os
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor import MultiProcDiffusionExecutor
from my_company import CloudExecutor

# Choose executor based on environment
if os.getenv("USE_CLOUD") == "true":
    executor = CloudExecutor
else:
    executor = MultiProcDiffusionExecutor

config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    num_gpus=2 if executor == MultiProcDiffusionExecutor else 0,
    executor_class=executor,
)

engine = DiffusionEngine(config)
```

## Example 7: External Executor with Custom Config

```python
# If your executor needs additional configuration,
# you can add custom fields to the config object

from dataclasses import dataclass
from vllm_omni.diffusion.data import OmniDiffusionConfig

# Create a custom config class
@dataclass
class MyConfig(OmniDiffusionConfig):
    remote_url: str = "http://localhost:8000"
    api_token: str = ""
    timeout: int = 300

# Use it
config = MyConfig(
    model_class_name="Qwen3Omni",
    executor_class="my_executor.RemoteExecutor",
    remote_url="https://my-inference-server.com",
    api_token="secret-token",
    timeout=600,
)

engine = DiffusionEngine(config)
# Your executor can access: config.remote_url, config.api_token, etc.
```

## Key Points

### ✅ Supported executor_class Values

1. **None** (default): Uses `MultiProcDiffusionExecutor`
2. **Shorthand strings**:
   - `"multiproc"` or `"mp"`: MultiProcDiffusionExecutor
   - `"external"`: ExternalDiffusionExecutor
   - `"http"`: HTTPDiffusionExecutor
3. **Full string path**: e.g., `"my_package.MyExecutor"`
4. **Class type**: e.g., `MyExecutor` (imported)

### ✅ Validation

The engine validates that:
- String paths resolve to valid classes
- Classes are subclasses of `DiffusionExecutor`
- Proper error messages for invalid configurations

### ✅ Error Handling

```python
# Invalid type
config = OmniDiffusionConfig(executor_class=123)  # TypeError

# Invalid string path
config = OmniDiffusionConfig(executor_class="nonexistent.Module")  # ValueError

# Not a DiffusionExecutor subclass
class BadExecutor:
    pass
config = OmniDiffusionConfig(executor_class=BadExecutor)  # TypeError
```

## Migration from Old Code

**Before (modifying source code):**
```python
# Had to modify DiffusionEngine._get_executor_class()
# Not flexible, required code changes
```

**After (configuration-based):**
```python
# Simply set executor_class in config
config = OmniDiffusionConfig(
    executor_class="my_executor.MyExecutor"
)
# No source code modifications needed!
```
