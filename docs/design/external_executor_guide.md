# External Executor Guide

This guide explains how to use external executors for diffusion models, focusing on the Ray-based implementation.

## Overview

The External Executor pattern allows you to:
- Use externally managed workers (e.g., Ray actors, Kubernetes pods)
- Decouple worker lifecycle from the DiffusionEngine
- Enable flexible distributed execution scenarios
- Support custom orchestration systems

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DiffusionEngine                         │
│  - Request processing                                        │
│  - Pre/post processing                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ Uses
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            RayExternalDiffusionExecutor                      │
│  - Discovers Ray actors                                      │
│  - Forwards RPC calls                                        │
│  - Manages communication                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Forwards to
           ┌───────────┴───────────┐
           ▼                       ▼
┌──────────────────────┐  ┌────────────────────┐
│RayDiffusionWorker    │  │RayDiffusionWorker  │
│Actor (Rank 0)        │  │Actor (Rank 1)      │
│  - WorkerWrapperBase │  │  - WorkerWrapperBase│
│  - GPU 0             │  │  - GPU 1           │
└──────────────────────┘  └────────────────────┘
```

## Components

### 1. ExternalDiffusionExecutor (Base Class)

Abstract base class for external executors. Provides template methods:

```python
class ExternalDiffusionExecutor(DiffusionExecutor):
    def _connect_to_workers(self) -> None:
        """Connect to external workers - implement in subclass"""
        
    def _forward_rpc_to_workers(self, ...) -> Any:
        """Forward RPC to workers - implement in subclass"""
        
    def _check_worker_health(self) -> None:
        """Check worker health - implement in subclass"""
        
    def _disconnect_from_workers(self) -> None:
        """Disconnect from workers - implement in subclass"""
```

### 2. RayDiffusionWorkerActor

Ray actor that wraps a WorkerWrapperBase instance:

```python
@ray.remote(num_gpus=1)
class RayDiffusionWorkerActor:
    def initialize(self) -> dict:
        """Initialize worker"""
        
    def execute_rpc(self, method: str, args, kwargs) -> Any:
        """Execute RPC on worker"""
        
    def generate(self, requests: list) -> DiffusionOutput:
        """Generate outputs"""
```

### 3. RayExternalDiffusionExecutor

Concrete implementation for Ray:

```python
class RayExternalDiffusionExecutor(ExternalDiffusionExecutor):
    def _connect_to_workers(self):
        # Discover Ray actors by name
        
    def _forward_rpc_to_workers(self, ...):
        # Use ray.get() to execute remote calls
```

## Usage Examples

### Basic Usage

```python
import os
import ray
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.ray_external_executor import (
    RayDiffusionWorkerActor,
    RayExternalDiffusionExecutor,
)

# 1. Initialize Ray
ray.init(num_gpus=2)

# 2. Create configuration
config = OmniDiffusionConfig(
    model="your-model",
    num_gpus=2,
)

# 3. Create Ray worker actors
worker_actors = []
for rank in range(2):
    actor = RayDiffusionWorkerActor.options(
        num_gpus=1,
        name=f"diffusion_worker_{rank}",
    ).remote(rank=rank, od_config=config)
    worker_actors.append(actor)

# Initialize actors
ray.get([actor.initialize.remote() for actor in worker_actors])

# 4. Configure executor to find actors
os.environ["DIFFUSION_WORKER_ACTOR_NAMES"] = "diffusion_worker_0,diffusion_worker_1"

# 5. Create engine with Ray executor
config.distributed_executor_backend = RayExternalDiffusionExecutor
engine = DiffusionEngine(config)

# 6. Use engine
from vllm_omni.diffusion.request import OmniDiffusionRequest

requests = [
    OmniDiffusionRequest(
        prompt="a landscape",
        height=512,
        width=512,
    )
]
output = engine.step(requests)

# 7. Cleanup
engine.close()  # Disconnects from workers
ray.get([actor.shutdown.remote() for actor in worker_actors])
```

### Using "external_launcher" Backend

```python
# Alternative: Use string backend name
config = OmniDiffusionConfig(
    model="your-model",
    num_gpus=2,
    distributed_executor_backend="external_launcher",
)

# This will use the default ExternalDiffusionExecutor
# You need to implement _connect_to_workers in a subclass
```

### Custom External Executor

```python
from vllm_omni.diffusion.executor import ExternalDiffusionExecutor

class KubernetesExternalExecutor(ExternalDiffusionExecutor):
    """Custom executor for Kubernetes pods."""
    
    def _connect_to_workers(self):
        # Discover workers via Kubernetes service discovery
        from kubernetes import client, config
        
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        
        # Find worker pods
        pods = v1.list_namespaced_pod(
            namespace="default",
            label_selector="app=diffusion-worker"
        )
        
        # Connect to each pod
        for pod in pods.items:
            # Create connection (e.g., gRPC client)
            worker = self._create_grpc_client(pod.status.pod_ip)
            self.worker_handles.append(worker)
    
    def _forward_rpc_to_workers(self, method, timeout, args, kwargs, unique_reply_rank):
        # Forward via gRPC
        responses = []
        workers = [self.worker_handles[unique_reply_rank]] if unique_reply_rank else self.worker_handles
        
        for worker in workers:
            response = worker.ExecuteRPC(
                method=method,
                args=args,
                kwargs=kwargs,
                timeout=timeout,
            )
            responses.append(response)
        
        return responses[0] if unique_reply_rank else responses

# Use it
config = OmniDiffusionConfig(
    model="your-model",
    distributed_executor_backend=KubernetesExternalExecutor,
)
```

## Configuration

### Environment Variables

- `DIFFUSION_WORKER_ACTOR_NAMES`: Comma-separated list of Ray actor names
  - Example: `"diffusion_worker_0,diffusion_worker_1,diffusion_worker_2"`

### Supported Backends

1. **"mp"** - MultiProcDiffusionExecutor (default)
   - Launches workers as processes
   
2. **"external_launcher"** - ExternalDiffusionExecutor
   - Base class for external workers
   - Must subclass and implement connection logic
   
3. **Custom Class** - Your ExternalDiffusionExecutor subclass
   ```python
   config.distributed_executor_backend = RayExternalDiffusionExecutor
   ```

## Worker Actor Implementation

### Required Methods

Your worker actor must implement:

```python
class MyWorkerActor:
    def initialize(self) -> dict:
        """Initialize worker.
        
        Returns:
            {"status": "ready"} on success
            {"status": "error", "error": "..."} on failure
        """
        
    def execute_rpc(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Execute RPC call on worker.
        
        Returns:
            Result from method call
            {"status": "error", "error": "..."} on failure
        """
        
    def generate(self, requests: list) -> DiffusionOutput:
        """Generate outputs for requests."""
        
    def get_rank(self) -> int:
        """Get worker rank."""
        
    def shutdown(self) -> None:
        """Cleanup resources."""
```

### Example Worker Wrapper

```python
from vllm_omni.diffusion.worker.gpu_worker import WorkerWrapperBase

class MyWorkerWrapper:
    def __init__(self, rank: int, od_config: OmniDiffusionConfig):
        self.rank = rank
        self.od_config = od_config
        
        # Create actual worker
        self.worker = WorkerWrapperBase(rank, od_config)
    
    def generate(self, requests):
        return self.worker.generate(requests)
    
    def get_rank(self):
        return self.rank
```

## Lifecycle Management

### Worker Lifecycle

1. **Creation**: Workers created externally (Ray, K8s, etc.)
2. **Discovery**: Executor discovers workers (by name, service, etc.)
3. **Connection**: Executor connects to workers
4. **Operation**: Executor forwards RPC calls
5. **Disconnection**: Executor disconnects (workers keep running)
6. **Termination**: Workers terminated externally

### Important Notes

- **Engine closes executor**: Only disconnects, doesn't kill workers
- **Worker termination**: Responsibility of external system
- **Health checks**: Executor can verify workers are reachable
- **Failure handling**: External system should restart failed workers

## Best Practices

### 1. Worker Discovery

```python
# Use named Ray actors for easy discovery
actor = RayDiffusionWorkerActor.options(
    name=f"diffusion_worker_{rank}",  # Named for discovery
).remote(...)

# Or use service discovery
# - Kubernetes: Label selectors
# - Consul: Service registry
# - etcd: Key-value store
```

### 2. Error Handling

```python
def execute_rpc(self, method, args, kwargs):
    try:
        result = getattr(self.worker, method)(*args, **kwargs)
        return result
    except Exception as e:
        # Return error dict for executor to handle
        return {
            "status": "error",
            "error": str(e),
            "rank": self.rank,
        }
```

### 3. Resource Management

```python
# Ray: Specify resource requirements
actor = RayDiffusionWorkerActor.options(
    num_gpus=1,
    memory=32 * 1024 * 1024 * 1024,  # 32GB
    resources={"special_gpu": 1},
).remote(...)

# Kubernetes: Resource requests/limits
# containers:
#   - name: worker
#     resources:
#       requests:
#         nvidia.com/gpu: 1
#       limits:
#         nvidia.com/gpu: 1
```

### 4. Health Monitoring

```python
class RayExternalDiffusionExecutor(ExternalDiffusionExecutor):
    def _check_worker_health(self):
        for i, worker in enumerate(self.worker_handles):
            try:
                # Simple health check
                rank = ray.get(worker.get_rank.remote(), timeout=5.0)
                assert rank == i
            except Exception as e:
                raise RuntimeError(f"Worker {i} unhealthy: {e}")
```

## Troubleshooting

### Workers Not Found

```python
# Error: Failed to find Ray actor 'diffusion_worker_0'

# Solution 1: Verify actors are created
actors = ray.util.list_named_actors()
print(actors)  # Should show your actors

# Solution 2: Check environment variable
print(os.environ.get("DIFFUSION_WORKER_ACTOR_NAMES"))

# Solution 3: Use correct names
os.environ["DIFFUSION_WORKER_ACTOR_NAMES"] = "worker_0,worker_1"  # Match actual names
```

### RPC Timeout

```python
# Error: RPC call to generate timed out

# Solution: Increase timeout
engine.collective_rpc("generate", timeout=300.0, args=(requests,))

# Or configure Ray with higher timeout
actor = RayDiffusionWorkerActor.options(
    max_task_retries=3,
).remote(...)
```

### Worker Initialization Failed

```python
# Error: Worker rank 0 failed to initialize

# Check actor logs
# Ray dashboard: http://localhost:8265
# Or programmatically:
logs = ray.get(actor.execute_rpc.remote("get_logs"))
```

## Performance Considerations

### Ray Overhead

- Ray adds serialization/deserialization overhead
- Use Ray's object store for large data transfers
- Consider batching small RPC calls

### Network Considerations

- Co-locate workers and executors when possible
- Use high-bandwidth networks for multi-node setups
- Monitor network latency and throughput

### Resource Allocation

- Ensure each worker has exclusive GPU access
- Allocate sufficient CPU and memory
- Monitor GPU utilization

## Example: Multi-Node Setup

```python
# Node 1: Head node + Worker 0
ray.init(address="auto", num_gpus=1)

# Node 2: Worker 1
ray.init(address="ray://head-node:10001", num_gpus=1)

# Create actors on different nodes
actor_0 = RayDiffusionWorkerActor.options(
    num_gpus=1,
    name="diffusion_worker_0",
    resources={"node:head": 1},  # Pin to head node
).remote(rank=0, od_config=config)

actor_1 = RayDiffusionWorkerActor.options(
    num_gpus=1,
    name="diffusion_worker_1",
    resources={"node:worker": 1},  # Pin to worker node
).remote(rank=1, od_config=config)

# Use as normal
engine = DiffusionEngine(config)
```

## See Also

- [Diffusion Executor Architecture](diffusion_executor_architecture.md)
- [Ray Documentation](https://docs.ray.io/)
- [Example Script](../examples/ray_external_executor_example.py)
