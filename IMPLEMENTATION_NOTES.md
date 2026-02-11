# FutureWrapper Implementation for Multiproc Executor

## Overview
This implementation adds asynchronous request/response handling to the `MultiprocDiffusionExecutor` using FutureWrapper pattern with request ID tracking.

## Key Components

### 1. FutureWrapper Class
- **Purpose**: Wraps `concurrent.futures.Future` to support request ID tracking and result aggregation
- **Features**:
  - Tracks unique request ID for each request
  - Maintains reference to pending_futures dict for cleanup
  - Supports custom aggregation function for responses
  - Automatically removes itself from pending futures on completion

### 2. Request ID System
- **Thread-safe ID generation**: Uses `_request_id_lock` to ensure unique IDs across threads
- **ID tracking**: `_pending_futures` dict maps request_id → FutureWrapper
- **Purpose**: Allows matching async responses back to their originating requests

### 3. Output Handler Thread
- **Name**: `_output_handler_loop`
- **Lifecycle**: Started in `_init_executor()`, runs as daemon thread
- **Responsibilities**:
  1. Continuously dequeues results from `scheduler.result_mq`
  2. Extracts `request_id` from each result
  3. Finds corresponding FutureWrapper in `_pending_futures`
  4. Sets result/exception on the future
  5. Handles errors and cleanup gracefully

### 4. Updated Methods

#### `add_req(request, non_block=False)`
**Workflow**:
1. Generate unique request ID
2. Create FutureWrapper and add to pending_futures
3. Wrap request with ID: `{"type": "add_req", "request_id": ..., "request": ...}`
4. Enqueue to scheduler
5. Return Future (non_block=True) or block for result (non_block=False)

#### `collective_rpc(method, timeout, args, kwargs, non_block, unique_reply_rank)`
**Workflow**:
1. Generate unique request ID
2. Create FutureWrapper with appropriate aggregation function
3. Add to pending_futures
4. Build RPC request: `{"type": "rpc", "request_id": ..., "method": ..., ...}`
5. Enqueue to scheduler
6. Return Future (non_block=True) or block for result (non_block=False)

**Aggregation**:
- Single reply (`unique_reply_rank` set): Returns single value
- Multiple replies: Returns list of values

### 5. Shutdown Process
Updated `shutdown()` to:
1. Set `_shutdown_event` to stop output handler loop
2. Cancel all pending futures with RuntimeError("Executor shutdown")
3. Clear pending_futures dict
4. Wait for output handler thread to finish (5 second timeout)
5. Call finalizer for resource cleanup

## Response Format Expected from Workers

Workers should return results in this format:
```python
{
    "request_id": <int>,          # Required: matches request
    "status": "success"|"error",  # Optional: default is success
    "response": <actual_data>,    # The actual response data
    "error": <error_message>      # Only if status="error"
}
```

## Usage Examples

### Blocking Request
```python
# Blocks until result is ready
result = executor.add_req(my_request, non_block=False)
```

### Non-blocking Request
```python
# Returns immediately with Future
future = executor.add_req(my_request, non_block=True)
# ... do other work ...
result = future.result()  # Block when ready for result
```

### Blocking RPC
```python
# Blocks until RPC completes
response = executor.collective_rpc("method_name", args=(arg1, arg2))
```

### Non-blocking RPC with timeout
```python
# Returns Future immediately
future = executor.collective_rpc(
    "method_name",
    timeout=10.0,
    args=(arg1,),
    non_block=True
)
# ... do other work ...
result = future.result(timeout=5.0)  # Can override timeout
```

## Thread Safety
- Request ID generation: Protected by `_request_id_lock`
- Pending futures dict: Accessed by:
  - Main thread (add/remove during request creation)
  - Output handler thread (lookup/remove during response handling)
  - Shutdown (cleanup)
- Future operations: Thread-safe via `concurrent.futures.Future` internals

## Error Handling
1. **Worker errors**: Workers return `{"status": "error", "error": "msg", "request_id": id}`
2. **Timeout**: Future.result() raises TimeoutError if timeout expires
3. **Shutdown**: All pending futures get RuntimeError exception
4. **Unknown request_id**: Logged as warning, no exception raised
5. **Output handler errors**: Logged but loop continues (resilient to transient errors)

## Monitoring
- Output handler start/stop logged at INFO level
- Unknown request_ids logged as WARNING
- Output handler errors logged as ERROR with full traceback
- Scheduler queue errors logged appropriately

## Benefits
1. **Asynchronous operations**: Non-blocking requests/RPC calls
2. **Timeout support**: Built into Future.result()
3. **Concurrent requests**: Multiple requests in flight simultaneously
4. **Clean shutdown**: Graceful cleanup of pending operations
5. **Error propagation**: Exceptions properly propagated to callers
6. **Request tracking**: Easy to correlate requests with responses
