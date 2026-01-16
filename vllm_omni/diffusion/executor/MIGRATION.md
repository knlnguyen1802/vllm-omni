# Migration Guide: Executor Pattern

## What Changed?

The `DiffusionEngine` has been refactored to use an **Executor pattern** for better separation of concerns and extensibility. This allows you to easily swap out the worker management implementation without changing the engine code.

## For End Users

### ✅ No Changes Needed!

If you're just using the `DiffusionEngine` API, **nothing changes for you**:

```python
# This still works exactly the same
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig

config = OmniDiffusionConfig(...)
engine = DiffusionEngine(config)
output = engine.step([request])
engine.close()
```

The executor is used internally and doesn't affect the public API.

## For Advanced Users

### If You Were Directly Accessing Internal Methods

Some internal methods have moved from `DiffusionEngine` to the executor:

| Old Way (Direct Access)                    | New Way (Via Executor)                      |
|--------------------------------------------|---------------------------------------------|
| `engine._make_client()`                    | Happens automatically in executor           |
| `engine._launch_workers()`                 | `executor._launch_workers()`                |
| `scheduler.add_req(requests)`              | `executor.execute_model(requests)`          |
| `scheduler.mq.enqueue(message)`            | `executor.collective_rpc(...)`              |

### Migration Examples

#### Before: Direct Scheduler Access
```python
# Old code
from vllm_omni.diffusion.scheduler import scheduler

scheduler.add_req(requests)
output = scheduler.get_response()
```

#### After: Via Executor
```python
# New code
engine = DiffusionEngine(config)
output = engine.executor.execute_model(requests)
# OR even better, use the high-level API:
output = engine.step(requests)
```

#### Before: Direct RPC
```python
# Old code
scheduler.mq.enqueue({
    "type": "rpc",
    "method": "sleep",
    "kwargs": {"level": 1}
})
response = scheduler.result_mq.dequeue()
```

#### After: Via Executor
```python
# New code
engine.collective_rpc("sleep", kwargs={"level": 1})
# OR use the convenience method:
engine.executor.sleep(level=1)
```

## For Framework Developers

### Creating Custom Executors

Now you can easily create custom executors for different deployment scenarios:

```python
from vllm_omni.diffusion.executor import DiffusionExecutor

class MyCustomExecutor(DiffusionExecutor):
    def _init_executor(self):
        # Your initialization
        pass
    
    def collective_rpc(self, method, **kwargs):
        # Your RPC implementation
        pass
    
    def execute_model(self, requests):
        # Your model execution
        pass
    
    def check_health(self):
        # Your health check
        pass
    
    def shutdown(self):
        # Your cleanup
        pass
```

Then modify `DiffusionEngine._get_executor_class()` to return your executor.

## What Was Removed

### Removed from DiffusionEngine

1. **`BackgroundResources` class**
   - Moved to `MultiProcDiffusionExecutor`
   - Now managed by the executor

2. **`_make_client()` method**
   - Functionality moved to executor initialization
   - Called automatically during `__init__`

3. **`_launch_workers()` method**
   - Moved to `MultiProcDiffusionExecutor._launch_workers()`
   - Part of executor-specific implementation

4. **`add_req_and_wait_for_response()` method**
   - Replaced by `executor.execute_model()`
   - Higher-level abstraction

5. **Direct scheduler/process management**
   - All moved to executors
   - Engine no longer manages workers directly

## What Was Added

### New in DiffusionEngine

1. **`executor` attribute**
   ```python
   engine.executor  # Access to the executor instance
   ```

2. **`_get_executor_class()` static method**
   ```python
   DiffusionEngine._get_executor_class()  # Returns executor class to use
   ```

3. **`check_health()` method**
   ```python
   engine.check_health()  # Delegates to executor
   ```

### New Modules

1. **`vllm_omni/diffusion/executor/` package**
   - `diffusion_executor.py` - Abstract base class
   - `multiproc_executor.py` - Default implementation
   - `external_executor.py` - Template for custom executors
   - `__init__.py` - Package exports

2. **Documentation**
   - `README.md` - Comprehensive guide
   - `QUICKSTART.md` - Quick reference
   - `ARCHITECTURE.md` - Architecture diagrams
   - `IMPLEMENTATION_SUMMARY.md` - Implementation details

3. **Examples**
   - `example_custom_executor.py` - Working example

## Compatibility Matrix

| Component                  | v1 (Old)      | v2 (New)      | Compatible? |
|----------------------------|---------------|---------------|-------------|
| `DiffusionEngine.step()`   | ✅ Supported  | ✅ Supported  | ✅ Yes      |
| `DiffusionEngine.close()`  | ✅ Supported  | ✅ Supported  | ✅ Yes      |
| `DiffusionEngine.abort()`  | ⚠️ Stub       | ⚠️ Stub       | ✅ Yes      |
| `scheduler.add_req()`      | ✅ Direct     | ⚠️ Via exec   | ⚠️ Changed  |
| `_launch_workers()`        | ✅ In engine  | ⚠️ In exec    | ⚠️ Moved    |
| `collective_rpc()`         | ✅ In engine  | ✅ In engine  | ✅ Yes*     |

\* `collective_rpc()` now delegates to executor but maintains the same interface

## Benefits of Migration

### 1. Cleaner Code
- DiffusionEngine: ~400 lines → ~290 lines
- Better separation of concerns
- Easier to understand and maintain

### 2. Extensibility
- Easy to add new execution strategies
- No need to modify engine code
- Plugin-style architecture

### 3. Testability
- Can mock executor for tests
- Test engine logic independently
- Test executor implementations separately

### 4. Flexibility
- Support for external systems
- Custom distributed backends
- Easy integration with cloud services

## Timeline

### Phase 1: ✅ Completed
- Abstract executor interface
- MultiProcDiffusionExecutor (default)
- Template for custom executors
- Documentation and examples
- Backward compatible API

### Phase 2: 🔄 Future Work
- Add `executor_class` to `OmniDiffusionConfig`
- Ray-based executor
- Async/non-blocking execution
- Metrics and monitoring hooks

## Getting Help

If you encounter issues during migration:

1. **Check the documentation**
   - [README.md](README.md) - Full documentation
   - [QUICKSTART.md](QUICKSTART.md) - Quick reference
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture details

2. **Review examples**
   - [example_custom_executor.py](example_custom_executor.py)
   - [external_executor.py](external_executor.py)

3. **Common Issues**

   **Issue:** Can't find `add_req_and_wait_for_response`
   ```python
   # Before
   engine.add_req_and_wait_for_response(requests)
   
   # After
   engine.executor.execute_model(requests)
   # Or better:
   engine.step(requests)
   ```

   **Issue:** Need direct scheduler access
   ```python
   # Before
   from vllm_omni.diffusion.scheduler import scheduler
   
   # After
   # Use executor methods instead, or for advanced use:
   from vllm_omni.diffusion.scheduler import scheduler
   # (Still available, but discouraged)
   ```

   **Issue:** Custom executor not being used
   ```python
   # Modify DiffusionEngine._get_executor_class():
   @staticmethod
   def _get_executor_class():
       from my_package import MyExecutor
       return MyExecutor
   ```

## Summary

The executor pattern provides a clean abstraction for worker management while maintaining full backward compatibility with existing code. Users benefit from:

- ✅ **No breaking changes** to public API
- ✅ **Better code organization** and maintainability
- ✅ **Easy extensibility** for custom deployments
- ✅ **Template and examples** for custom executors
- ✅ **Comprehensive documentation**

If you're using the standard `DiffusionEngine` API, you don't need to change anything. If you need custom execution strategies, you now have a clean way to implement them!
