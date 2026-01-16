# RayActorExecutor Guide

## Overview

`RayActorExecutor` is a simple and efficient executor that creates a single Ray actor to manage diffusion workers internally. Unlike the external executor pattern, this executor owns the Ray actor's lifecycle and provides a clean, simple API.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DiffusionEngine                           │
│  - Request processing                                        │
│  - Pre/post processing                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ Uses
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 RayActorExecutor                             │
│  - Creates Ray actor                                         │
│  - Forwards all calls to actor                               │
│  - Manages actor lifecycle                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ Owns
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            DiffusionWorkerActor (Ray Actor)                  │
│  - Manages WorkerWrapperBase                                 │
│  - Executes on GPU node                                      │
│  - Handles RPC calls                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │ Contains
                       ▼
               WorkerWrapperBase
               (Diffusion Model)
```

## Key Features

✅ **Simple API** - Similar to AsyncOmniDiffusion pattern  
✅ **Single Ray Actor** - One actor manages all workers  
✅ **Automatic Lifecycle** - Executor creates and destroys actor  
✅ **GPU Management** - Actor runs on GPU node  
✅ **RPC Support** - Full collective_rpc support  
✅ **Error Handling** - Proper timeout and error handling  

## Quick Start

### Minimal Example

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest

# Create engine with Ray actor executor
engine = DiffusionEngine(
    OmniDiffusionConfig(
        model="your-model-path",
        distributed_executor_backend="ray_actor",
    )
)

# Generate
result = engine.step([
    OmniDiffusionRequest(
        prompt="A beautiful landscape",
        height=512,
        width=512,
    )
])

print(result)

# Cleanup
engine.close()
```

### Full Example with Async

```python
import asyncio
import multiprocessing as mp

async def main():
    engine = DiffusionEngine(
        OmniDiffusionConfig(
            model="your-model-path",
            distributed_executor_backend="ray_actor",
        )
    )
    
    try:
        # Check health
        engine.check_health()
        
        # Generate
        result = engine.step([
            OmniDiffusionRequest(
                prompt="A sunset over the ocean",
                request_id="req-1",
            )
        ])
        
        print(result)
        
    finally:
        engine.close()

if __name__ == "__main__":
    mp.freeze_support()
    asyncio.run(main())
```

## Configuration

### Basic Configuration

```python
config = OmniDiffusionConfig(
    model="your-model-path",
    num_gpus=1,
    distributed_executor_backend="ray_actor",
)
```

### Advanced Configuration

```python
config = OmniDiffusionConfig(
    model="your-model-path",
    num_gpus=2,
    distributed_executor_backend="ray_actor",
    
    # Ray actor resource allocation
    ray_actor_cpus=2,          # CPUs for the actor
    ray_actor_gpus=2,          # GPUs for the actor
    ray_actor_memory=32 * 1024 * 1024 * 1024,  # 32GB memory
    
    # Ray initialization kwargs
    ray_init_kwargs={
        "num_gpus": 2,
        "num_cpus": 4,
        "address": "auto",  # Connect to existing cluster
    },
)
```

## Usage Patterns

### Pattern 1: Simple Synchronous

```python
# Create engine
engine = DiffusionEngine(
    OmniDiffusionConfig(
        model="model-path",
        distributed_executor_backend="ray_actor",
    )
)

# Use it
result = engine.step([OmniDiffusionRequest(prompt="test")])

# Done
engine.close()
```

### Pattern 2: With Context Manager (Recommended)

```python
class DiffusionEngineContext:
    def __init__(self, config):
        self.config = config
        self.engine = None
    
    def __enter__(self):
        self.engine = DiffusionEngine(self.config)
        return self.engine
    
    def __exit__(self, *args):
        self.engine.close()

# Use it
config = OmniDiffusionConfig(
    model="model-path",
    distributed_executor_backend="ray_actor",
)

with DiffusionEngineContext(config) as engine:
    result = engine.step([OmniDiffusionRequest(prompt="test")])
```

### Pattern 3: Multiple Requests

```python
engine = DiffusionEngine(config)

# Batch requests
requests = [
    OmniDiffusionRequest(prompt="landscape", request_id="req-1"),
    OmniDiffusionRequest(prompt="portrait", request_id="req-2"),
    OmniDiffusionRequest(prompt="abstract art", request_id="req-3"),
]

# Process all at once
results = engine.step(requests)

# Results is a list of outputs
for i, result in enumerate(results):
    print(f"Request {i}: {result}")

engine.close()
```

### Pattern 4: With Health Monitoring

```python
engine = DiffusionEngine(config)

try:
    # Periodic health check
    engine.check_health()
    
    # Generate
    result = engine.step(requests)
    
    # Check again
    engine.check_health()
    
except RuntimeError as e:
    print(f"Engine unhealthy: {e}")
    # Handle error
    
finally:
    engine.close()
```

## API Reference

### RayActorExecutor

```python
class RayActorExecutor(DiffusionExecutor):
    """Executor that manages a Ray actor for diffusion execution."""
    
    uses_ray: bool = True
    supports_pp: bool = False
```

#### Methods

**`__init__(od_config: OmniDiffusionConfig)`**
- Initialize executor and create Ray actor
- Ray is initialized if not already running

**`collective_rpc(method, timeout=None, args=(), kwargs=None, unique_reply_rank=None)`**
- Execute RPC call on the worker actor
- Returns result from the method call
- Raises TimeoutError if timeout exceeded

**`add_requests(requests: list[OmniDiffusionRequest]) -> DiffusionOutput`**
- Execute diffusion model for requests
- Returns DiffusionOutput with results or error

**`check_health() -> None`**
- Verify the actor is healthy
- Raises RuntimeError if unhealthy

**`shutdown() -> None`**
- Gracefully shutdown actor and cleanup
- Kills the Ray actor

### DiffusionWorkerActor

```python
@ray.remote
class DiffusionWorkerActor:
    """Ray actor that manages diffusion workers."""
```

#### Methods

**`initialize() -> bool`**
- Initialize WorkerWrapperBase
- Returns True on success, False on failure

**`collective_rpc(method, args, kwargs, unique_reply_rank) -> Any`**
- Execute method on worker
- Returns result or raises exception

**`execute_model(request_dicts: list[dict]) -> DiffusionOutput`**
- Execute diffusion model
- Returns DiffusionOutput

**`check_health() -> bool`**
- Check if worker is healthy
- Returns True if healthy

**`shutdown() -> None`**
- Cleanup resources

## Differences from Other Executors

| Feature | MultiProcExecutor | RayActorExecutor | ExternalExecutor |
|---------|-------------------|------------------|------------------|
| Worker Management | Spawns processes | Creates Ray actor | Connects to existing |
| Lifecycle | Engine-owned | Engine-owned | External |
| Distribution | Multi-process | Ray (single actor) | Custom |
| Complexity | Medium | Low | Medium |
| Ray Required | No | Yes | Optional |
| Multi-Node | No | Yes (via Ray) | Yes (custom) |

## Configuration Options

### Required

```python
distributed_executor_backend = "ray_actor"  # or RayActorExecutor class
```

### Optional Ray Actor Resources

```python
ray_actor_cpus: int = 1           # CPUs for actor
ray_actor_gpus: int = num_gpus    # GPUs for actor (defaults to num_gpus)
ray_actor_memory: int = None      # Memory in bytes
```

### Optional Ray Initialization

```python
ray_init_kwargs: dict = {}        # Passed to ray.init()
```

Example:
```python
config.ray_init_kwargs = {
    "address": "ray://head-node:10001",  # Connect to cluster
    "num_gpus": 2,
    "num_cpus": 4,
    "runtime_env": {
        "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}
    }
}
```

## Multi-Node Setup

### Option 1: Auto-connect to Cluster

```python
# On worker node, Ray is already running
# ray start --address=head-node:6379

# In your code
config = OmniDiffusionConfig(
    model="model-path",
    distributed_executor_backend="ray_actor",
    ray_init_kwargs={"address": "auto"},  # Auto-connect
)

engine = DiffusionEngine(config)
```

### Option 2: Specify Head Node

```python
config = OmniDiffusionConfig(
    model="model-path",
    distributed_executor_backend="ray_actor",
    ray_init_kwargs={
        "address": "ray://head-node:10001",
    },
)
```

### Option 3: Resource Placement

```python
# Create actor on specific node type
config.ray_actor_resources = {"node:gpu-worker": 1}
```

## Error Handling

### Timeout Errors

```python
try:
    result = engine.step(requests)
except TimeoutError:
    print("Request timed out after 300 seconds")
```

### Health Check Failures

```python
try:
    engine.check_health()
except RuntimeError as e:
    print(f"Health check failed: {e}")
    # Recreate engine or restart actor
```

### Initialization Failures

```python
try:
    engine = DiffusionEngine(config)
except RuntimeError as e:
    print(f"Failed to initialize: {e}")
    # Check Ray cluster status
    # Check GPU availability
```

## Monitoring and Debugging

### Ray Dashboard

Access Ray dashboard at `http://localhost:8265`:
- View actor status
- Check resource usage
- See logs and errors
- Monitor task execution

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Will show detailed logs from:
# - RayActorExecutor
# - DiffusionWorkerActor
# - WorkerWrapperBase
```

### Health Monitoring

```python
import time

engine = DiffusionEngine(config)

while True:
    try:
        engine.check_health()
        print("✓ Healthy")
    except RuntimeError as e:
        print(f"✗ Unhealthy: {e}")
        break
    
    time.sleep(60)  # Check every minute
```

## Performance Considerations

### Actor Placement

- Actor runs on GPU node by default
- Ensure GPU resources available
- Monitor GPU utilization

### Memory Management

- Ray actor holds model in memory
- Specify memory limits to prevent OOM
- Consider model size when allocating

### Network Overhead

- Minimal for single-actor setup
- Data serialized between engine and actor
- Use object store for large tensors (future enhancement)

## Troubleshooting

### Problem: "Ray is not installed"

**Solution:**
```bash
pip install ray
```

### Problem: "Failed to initialize Ray DiffusionWorkerActor"

**Solutions:**
1. Check Ray dashboard for actor errors
2. Verify GPU availability
3. Check model path is accessible
4. Review actor logs

### Problem: "RPC call timed out"

**Solutions:**
1. Increase timeout: `engine.collective_rpc(method, timeout=600)`
2. Check model is loaded correctly
3. Verify GPU is not hung
4. Check Ray actor is responsive

### Problem: Actor keeps dying

**Solutions:**
1. Check GPU memory - may be OOM
2. Review actor logs in Ray dashboard
3. Ensure model fits in allocated GPU memory
4. Check for CUDA errors

## Best Practices

1. **Always cleanup**: Call `engine.close()` in finally block
2. **Health checks**: Monitor health for long-running applications
3. **Resource limits**: Set appropriate memory limits
4. **Error handling**: Catch and handle timeout errors
5. **Logging**: Enable DEBUG logging for troubleshooting

## Example: Production Usage

```python
import logging
import time
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDiffusionService:
    def __init__(self, model_path: str):
        self.config = OmniDiffusionConfig(
            model=model_path,
            distributed_executor_backend="ray_actor",
            ray_actor_gpus=1,
            ray_actor_memory=32 * 1024**3,  # 32GB
        )
        self.engine: Optional[DiffusionEngine] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize engine with retry logic."""
        for attempt in range(3):
            try:
                self.engine = DiffusionEngine(self.config)
                self.engine.check_health()
                logger.info("Service initialized successfully")
                return
            except Exception as e:
                logger.error(f"Init attempt {attempt + 1} failed: {e}")
                time.sleep(5)
        
        raise RuntimeError("Failed to initialize after 3 attempts")
    
    def generate(self, prompt: str, **kwargs) -> Optional[DiffusionOutput]:
        """Generate with health monitoring."""
        try:
            # Health check before generation
            self.engine.check_health()
            
            # Create request
            request = OmniDiffusionRequest(prompt=prompt, **kwargs)
            
            # Generate
            result = self.engine.step([request])
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Attempt recovery
            self._recover()
            return None
    
    def _recover(self):
        """Attempt to recover from errors."""
        logger.info("Attempting recovery...")
        try:
            if self.engine:
                self.engine.close()
        except:
            pass
        
        self._initialize()
    
    def shutdown(self):
        """Graceful shutdown."""
        if self.engine:
            self.engine.close()

# Usage
service = ProductionDiffusionService("/path/to/model")
try:
    result = service.generate("A beautiful landscape")
finally:
    service.shutdown()
```

## See Also

- [Diffusion Executor Architecture](diffusion_executor_architecture.md)
- [Ray Documentation](https://docs.ray.io/)
- [Example Script](../../examples/ray_actor_executor_example.py)
