# Diffusion Executor Architecture

## Overview

The Diffusion Executor provides an abstraction layer between the `DiffusionEngine` and the worker processes that execute the actual model inference. This design enables flexible execution strategies while maintaining a consistent interface.

## Architecture

```
┌─────────────────────┐
│  DiffusionEngine    │  ← High-level API
└──────────┬──────────┘
           │
           │ uses
           ▼
┌─────────────────────┐
│  DiffusionExecutor  │  ← Abstract interface
└──────────┬──────────┘
           │
           │ implements
           ▼
┌─────────────────────────────────────────────┐
│  Concrete Executors                         │
├─────────────────────────────────────────────┤
│  • MultiProcDiffusionExecutor (default)     │
│  • ExternalDiffusionExecutor (template)     │
│  • HTTPDiffusionExecutor (example)          │
│  • Your custom executor                     │
└──────────┬──────────────────────────────────┘
           │
           │ manages
           ▼
┌─────────────────────┐
│  Worker Processes   │  ← Actual model execution
└─────────────────────┘
```

## Components

### 1. DiffusionExecutor (Abstract Base Class)

Located in: `vllm_omni/diffusion/executor/diffusion_executor.py`

The abstract base class that defines the interface all executors must implement:

- `_init_executor()` - Initialize executor-specific resources
- `collective_rpc()` - Execute RPC calls on workers
- `execute_model()` - Run model inference
- `check_health()` - Verify executor/worker health
- `shutdown()` - Clean up resources
- `sleep()` / `wake_up()` - Power management

### 2. MultiProcDiffusionExecutor (Default Implementation)

Located in: `vllm_omni/diffusion/executor/multiproc_executor.py`

The default executor that uses Python's multiprocessing to spawn worker processes:

**Features:**
- Spawns one worker process per GPU
- Uses shared memory message queues for communication
- Handles worker lifecycle management
- Implements graceful shutdown

**Communication Flow:**
1. Engine sends requests via executor
2. Executor broadcasts to workers via message queue
3. Workers process requests
4. Workers send results back via result queue
5. Executor collects and returns results

### 3. ExternalDiffusionExecutor (Template)

Located in: `vllm_omni/diffusion/executor/external_executor.py`

A template/example showing how to implement custom executors that forward calls to external systems.

## Creating a Custom Executor

### Step 1: Inherit from DiffusionExecutor

```python
from vllm_omni.diffusion.executor import DiffusionExecutor
from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

class MyCustomExecutor(DiffusionExecutor):
    """My custom executor implementation."""
    
    def _init_executor(self) -> None:
        """Initialize your custom resources."""
        # Set up connections, clients, etc.
        pass
```

### Step 2: Implement Required Methods

```python
    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ):
        """Forward RPC calls to your execution system."""
        # Implement your RPC logic
        pass
    
    def check_health(self) -> None:
        """Check if your system is healthy."""
        # Implement health checking
        pass
    
    def shutdown(self) -> None:
        """Clean up resources."""
        # Implement cleanup
        pass
```

### Step 3: (Optional) Override execute_model

If your system has a specialized inference API:

```python
    def execute_model(
        self, requests: list[OmniDiffusionRequest]
    ) -> DiffusionOutput | None:
        """Custom model execution logic."""
        # Use your system's native API
        result = self.my_client.generate(requests)
        return result
```

## Example Use Cases

### 1. Remote HTTP Inference Server

```python
class HTTPDiffusionExecutor(DiffusionExecutor):
    """Execute via remote HTTP API."""
    
    def _init_executor(self) -> None:
        import requests
        self.session = requests.Session()
        self.base_url = "http://inference-server:8000"
    
    def execute_model(self, requests):
        payload = {"requests": [r.to_dict() for r in requests]}
        response = self.session.post(
            f"{self.base_url}/v1/diffusion/generate",
            json=payload,
            timeout=300
        )
        return DiffusionOutput.from_dict(response.json())
```

### 2. Cloud Service Integration

```python
class CloudExecutor(DiffusionExecutor):
    """Execute via cloud inference service."""
    
    def _init_executor(self) -> None:
        from cloud_sdk import InferenceClient
        self.client = InferenceClient(
            api_key=self.od_config.cloud_api_key,
            region=self.od_config.cloud_region
        )
    
    def execute_model(self, requests):
        return self.client.diffusion.generate(requests)
```

### 3. Custom Distributed Backend

```python
class MPIExecutor(DiffusionExecutor):
    """Execute using MPI for distributed execution."""
    
    def _init_executor(self) -> None:
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        # Initialize MPI workers...
    
    def collective_rpc(self, method, **kwargs):
        # Use MPI communication
        self.comm.bcast((method, kwargs), root=0)
        results = self.comm.gather(None, root=0)
        return results
```

## Using a Custom Executor

### Configuration-Based (Recommended)

You can specify a custom executor via `OmniDiffusionConfig`:

**Option 1: Using String Path**
```python
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig

config = OmniDiffusionConfig(
    model_class_name="MyModel",
    executor_class="my_package.MyCustomExecutor",  # String path
    # ... other config
)

engine = DiffusionEngine(config)
```

**Option 2: Using Class Type**
```python
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig
from my_package import MyCustomExecutor

config = OmniDiffusionConfig(
    model_class_name="MyModel",
    executor_class=MyCustomExecutor,  # Direct class reference
    # ... other config
)

engine = DiffusionEngine(config)
```

**Option 3: Use Default Executor**
```python
config = OmniDiffusionConfig(
    model_class_name="MyModel",
    # executor_class is None by default -> uses MultiProcDiffusionExecutor
)
```

## RPC Interface

All calls from the engine to workers go through the `collective_rpc` method:

```python
# Execute model
result = executor.collective_rpc("generate", args=(requests,), unique_reply_rank=0)

# Call worker method
results = executor.collective_rpc("get_cache_stats")

# Sleep/wake workers
executor.collective_rpc("sleep", kwargs={"level": 1})
executor.collective_rpc("wake_up", kwargs={"tags": ["weights"]})
```

### Common Worker Methods

- `generate(requests)` - Execute model inference
- `sleep(level)` - Enter power-saving mode
- `wake_up(tags)` - Exit power-saving mode
- `shutdown()` - Terminate worker

## Benefits of the Executor Pattern

1. **Flexibility**: Easy to swap execution strategies without changing engine code
2. **Extensibility**: Users can implement custom executors for their specific needs
3. **Separation of Concerns**: Engine handles business logic, executor handles distribution
4. **Testability**: Can mock executor for testing without spinning up workers
5. **Compatibility**: Results are guaranteed to be compatible as long as the interface is followed

## Testing Your Executor

```python
def test_custom_executor():
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from my_package import MyCustomExecutor
    
    config = OmniDiffusionConfig(
        model_class_name="MyModel",
        num_gpus=2,
    )
    
    executor = MyCustomExecutor(config)
    
    try:
        # Test health check
        executor.check_health()
        
        # Test RPC
        result = executor.collective_rpc("test_method")
        assert result is not None
        
        # Test model execution
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        request = OmniDiffusionRequest(prompt="test")
        output = executor.execute_model([request])
        assert output is not None
        
    finally:
        executor.shutdown()
```

## Migration Guide

If you have existing code that directly accesses workers or schedulers:

### Before (Direct Access)
```python
# Old way - direct scheduler access
scheduler.add_req(requests)
scheduler.mq.enqueue(message)
```

### After (Via Executor)
```python
# New way - via executor
engine.executor.execute_model(requests)
engine.collective_rpc("method_name", args=(...))
```

## Future Enhancements

- [ ] Add Ray-based executor for Ray cluster support
- [ ] Add executor configuration via `OmniDiffusionConfig`
- [ ] Add async/non-blocking execution support
- [ ] Add executor plugin system for dynamic loading
- [ ] Add metrics and monitoring hooks
- [ ] Add request batching/routing strategies

## See Also

- [DiffusionExecutor API](diffusion_executor.py)
- [MultiProcDiffusionExecutor Implementation](multiproc_executor.py)
- [External Executor Template](external_executor.py)
- vLLM v1 Executor: `vllm/v1/executor/executor.py` (reference implementation)
