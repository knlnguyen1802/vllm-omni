# Diffusion Executor Architecture

This document describes the executor abstraction layer for the vLLM-Omni diffusion engine.

## Architecture Overview

The executor pattern provides an abstraction layer between the `DiffusionEngine` and the worker processes, following the same design pattern as vLLM's v1 Executor.

```
┌─────────────────────────────────────────────────────────────┐
│                      DiffusionEngine                         │
│  - Manages pre/post processing                              │
│  - Delegates all worker operations to Executor              │
└──────────────────────┬──────────────────────────────────────┘
                       │ Uses
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   DiffusionExecutor                          │
│  (Abstract Base Class)                                       │
│  - collective_rpc()                                          │
│  - add_requests()                                            │
│  - shutdown()                                                │
│  - check_health()                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │ Implemented by
           ┌───────────┴───────────┐
           ▼                       ▼
┌──────────────────────┐  ┌────────────────────┐
│MultiProcDiffusion    │  │ External/Custom    │
│Executor              │  │ Executor           │
│- Manages Scheduler   │  │ (Future support)   │
│- Launches Workers    │  │                    │
│- Handles RPC         │  │                    │
└──────────┬───────────┘  └────────────────────┘
           │
           ├─────► Scheduler (MessageQueue based)
           │
           └─────► Workers (GPU/NPU processes)
```

## Components

### 1. DiffusionExecutor (Abstract Base Class)

Location: `vllm_omni/diffusion/executor/executor_base.py`

**Responsibilities:**
- Define the interface that all executors must implement
- Provide factory method to select appropriate executor based on configuration
- Support both built-in and custom executor implementations

**Key Methods:**
- `get_class(od_config)`: Factory method to get executor class
- `_init_executor()`: Initialize executor-specific components
- `collective_rpc()`: Execute RPC calls on workers
- `add_requests()`: Submit diffusion requests for processing
- `shutdown()`: Clean up all resources
- `check_health()`: Verify executor health

### 2. MultiProcDiffusionExecutor

Location: `vllm_omni/diffusion/executor/multiproc_executor.py`

**Responsibilities:**
- Initialize and manage the Scheduler singleton
- Launch worker processes using multiprocessing
- Handle all communication with workers via message queues
- Manage resource cleanup

**Implementation Details:**
- Uses `BackgroundResources` with `weakref.finalize` for proper cleanup
- Initializes scheduler before launching workers
- Waits for all workers to report "ready" status before proceeding
- Sets up result queue for receiving responses from workers

### 3. DiffusionEngine (Refactored)

Location: `vllm_omni/diffusion/diffusion_engine.py`

**Changes Made:**
- Removed direct worker and scheduler management
- Removed `_make_client()`, `_launch_workers()` methods
- Removed `BackgroundResources` class (moved to executor)
- Added executor initialization via `DiffusionExecutor.get_class()`
- Delegates all worker operations to executor

**Benefits:**
- Clean separation of concerns
- Engine focuses on request processing and pre/post-processing
- Executor handles all distributed execution details
- Easy to swap executor implementations

## Configuration

The executor backend is controlled by the `distributed_executor_backend` field in `OmniDiffusionConfig`:

```python
@dataclass
class OmniDiffusionConfig:
    # ...
    distributed_executor_backend: str = "mp"  # Default: multiprocessing
    # ...
```

### Supported Backends

1. **"mp"** - MultiProcDiffusionExecutor (default)
   - Uses Python multiprocessing
   - Suitable for single-node multi-GPU setups

2. **Custom Class** - Provide a subclass of `DiffusionExecutor`
   ```python
   from vllm_omni.diffusion.executor import DiffusionExecutor
   
   class MyCustomExecutor(DiffusionExecutor):
       def _init_executor(self): ...
       def collective_rpc(self, ...): ...
       def add_requests(self, ...): ...
       def shutdown(self): ...
       def check_health(self): ...
   
   # Use it
   config = OmniDiffusionConfig(
       distributed_executor_backend=MyCustomExecutor,
       ...
   )
   ```

3. **Qualified Name String** - Provide a fully qualified class name
   ```python
   config = OmniDiffusionConfig(
       distributed_executor_backend="mypackage.executors.MyExecutor",
       ...
   )
   ```

## Usage Examples

### Basic Usage (Default Executor)

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest

# Create config with default executor (mp)
config = OmniDiffusionConfig(
    model="your-model-path",
    num_gpus=2,
    # distributed_executor_backend="mp" is default
)

# Engine automatically creates the executor
engine = DiffusionEngine(config)

# Submit requests
requests = [
    OmniDiffusionRequest(
        prompt="a beautiful landscape",
        height=1024,
        width=1024,
    )
]

# Process through executor
output = engine.step(requests)

# Cleanup
engine.close()
```

### Using Custom Executor

```python
from vllm_omni.diffusion.executor import DiffusionExecutor

class ExternalLauncherExecutor(DiffusionExecutor):
    """Executor for externally launched workers."""
    
    def _init_executor(self):
        # Connect to existing workers instead of launching
        self._connect_to_external_workers()
    
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, unique_reply_rank=None):
        # Send RPC to external workers
        return self._send_to_external_workers(method, args, kwargs)
    
    def add_requests(self, requests):
        # Forward to external workers
        return self._forward_requests(requests)
    
    def shutdown(self):
        # Don't terminate workers, just disconnect
        self._disconnect_from_workers()
    
    def check_health(self):
        # Check external workers
        self._verify_external_workers()

# Use it
config = OmniDiffusionConfig(
    model="your-model-path",
    num_gpus=2,
    distributed_executor_backend=ExternalLauncherExecutor,
)

engine = DiffusionEngine(config)
```

## Migration Guide

### Old Code (Before Executor)

```python
class DiffusionEngine:
    def __init__(self, od_config):
        self._processes = []
        self._make_client()  # Direct worker management
    
    def _make_client(self):
        scheduler.initialize(self.od_config)
        processes, result_handle = self._launch_workers(...)
        # ... complex initialization
    
    def collective_rpc(self, method, ...):
        scheduler.mq.enqueue(rpc_request)
        response = scheduler.result_mq.dequeue()
        # ... direct queue access
```

### New Code (With Executor)

```python
class DiffusionEngine:
    def __init__(self, od_config):
        # Executor handles all initialization
        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
    
    def collective_rpc(self, method, ...):
        # Delegate to executor
        return self.executor.collective_rpc(method, ...)
```

## Benefits of Executor Pattern

1. **Separation of Concerns**
   - Engine: Request processing, pre/post-processing
   - Executor: Worker management, communication

2. **Flexibility**
   - Easy to swap executor implementations
   - Support for different deployment scenarios (local, distributed, external)

3. **Testability**
   - Can mock executor for testing
   - Engine logic can be tested independently

4. **Consistency with vLLM**
   - Follows vLLM's proven architecture
   - Easier for developers familiar with vLLM

5. **Future Extensions**
   - Ray-based distributed execution
   - External launcher support
   - Custom deployment scenarios

## Implementation Checklist

- [x] Create `DiffusionExecutor` abstract base class
- [x] Implement `MultiProcDiffusionExecutor`
- [x] Refactor `DiffusionEngine` to use executor
- [x] Remove direct worker/scheduler management from engine
- [x] Add factory method for executor selection
- [x] Support custom executor classes
- [x] Maintain backward compatibility with existing code
- [x] Document the architecture and usage

## Testing

To verify the implementation works correctly:

```python
# 1. Check executor initialization
config = OmniDiffusionConfig(model="test-model", num_gpus=1)
engine = DiffusionEngine(config)
assert hasattr(engine, 'executor')
assert isinstance(engine.executor, MultiProcDiffusionExecutor)

# 2. Check health
engine.check_health()  # Should not raise

# 3. Test RPC
result = engine.collective_rpc("some_method", args=())

# 4. Cleanup
engine.close()
```
