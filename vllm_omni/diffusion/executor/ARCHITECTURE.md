# Architecture Diagram

## Overall Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        User Application                         │
│                                                                 │
│  from vllm_omni.diffusion import DiffusionEngine                │
│  engine = DiffusionEngine(config)                               │
│  output = engine.step([request])                                │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         │ calls
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                      DiffusionEngine                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Responsibilities:                                        │  │
│  │ • Request pre-processing                                 │  │
│  │ • Response post-processing                               │  │
│  │ • Format conversion (DiffusionOutput → OmniRequestOutput)│  │
│  │ • High-level API (step, close, etc.)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         │ delegates to
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                   DiffusionExecutor (ABC)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Abstract Interface:                                      │  │
│  │ • collective_rpc(method, args, kwargs)                   │  │
│  │ • execute_model(requests) → DiffusionOutput              │  │
│  │ • check_health()                                         │  │
│  │ • shutdown()                                             │  │
│  │ • sleep() / wake_up()                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         │ implemented by
                         │
         ┌───────────────┼───────────────┬────────────────────┐
         │               │               │                    │
         ▼               ▼               ▼                    ▼
┌─────────────┐  ┌─────────────┐ ┌─────────────┐  ┌──────────────┐
│  MultiProc  │  │  External   │ │    HTTP     │  │   Your       │
│  Executor   │  │  Executor   │ │  Executor   │  │   Custom     │
│  (default)  │  │ (template)  │ │  (example)  │  │   Executor   │
└──────┬──────┘  └─────────────┘ └─────────────┘  └──────────────┘
       │
       │ manages
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│              Worker Processes (GPU Workers)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Worker-0     │  │ Worker-1     │  │ Worker-N     │         │
│  │ GPU 0        │  │ GPU 1        │  │ GPU N        │         │
│  │              │  │              │  │              │         │
│  │ • Load model │  │ • Load model │  │ • Load model │         │
│  │ • Execute    │  │ • Execute    │  │ • Execute    │         │
│  │ • Return     │  │ • Return     │  │ • Return     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────────────────────────────────────────┘
```

## Request Flow (MultiProcDiffusionExecutor)

```
User Request
    │
    ├─> DiffusionEngine.step(requests)
    │       │
    │       ├─> Pre-processing (if configured)
    │       │
    │       ├─> executor.execute_model(requests)
    │       │       │
    │       │       ├─> collective_rpc("generate", args=(requests,), unique_reply_rank=0)
    │       │       │       │
    │       │       │       ├─> Prepare RPC message
    │       │       │       │       {"type": "rpc", "method": "generate", ...}
    │       │       │       │
    │       │       │       ├─> Enqueue to scheduler message queue
    │       │       │       │       │
    │       │       │       │       ├─> [Broadcast via SharedMemory]
    │       │       │       │       │
    │       │       │       │       ▼
    │       │       │       │   ┌───────────────────────────────┐
    │       │       │       │   │  Worker Processes             │
    │       │       │       │   │  - Receive via message queue  │
    │       │       │       │   │  - Execute generate()         │
    │       │       │       │   │  - Send result to result_mq   │
    │       │       │       │   └───────────────────────────────┘
    │       │       │       │
    │       │       │       ├─> Dequeue from result message queue
    │       │       │       │
    │       │       │       └─> Return DiffusionOutput
    │       │       │
    │       │       └─> Return output to engine
    │       │
    │       ├─> Post-processing (if configured)
    │       │
    │       └─> Convert to OmniRequestOutput
    │
    └─> Return to user
```

## External Executor Flow (Custom Implementation)

```
User Request
    │
    ├─> DiffusionEngine.step(requests)
    │       │
    │       ├─> Pre-processing
    │       │
    │       ├─> executor.execute_model(requests)
    │       │       │
    │       │       ├─> YOUR CUSTOM LOGIC
    │       │       │   Examples:
    │       │       │   • HTTP POST to remote server
    │       │       │   • gRPC call to inference service
    │       │       │   • Cloud API invocation
    │       │       │   • Message queue publishing
    │       │       │
    │       │       └─> Return DiffusionOutput (must be compatible!)
    │       │
    │       ├─> Post-processing
    │       │
    │       └─> Convert to OmniRequestOutput
    │
    └─> Return to user
```

## Component Interaction Matrix

| Component          | Talks To              | Via                    | Purpose                  |
|--------------------|-----------------------|------------------------|--------------------------|
| User Code          | DiffusionEngine       | Method calls           | High-level API           |
| DiffusionEngine    | Executor              | Method calls           | Delegate to executor     |
| MultiProcExecutor  | Scheduler             | Message queue setup    | Initialize workers       |
| MultiProcExecutor  | Workers               | SharedMemory queue     | Send requests/RPC        |
| Workers            | MultiProcExecutor     | Result queue           | Return outputs           |
| ExternalExecutor   | External System       | HTTP/gRPC/Cloud API    | Forward requests         |
| ExternalExecutor   | DiffusionEngine       | Return values          | Return outputs           |

## Data Flow

```
OmniDiffusionRequest  (User's request)
    │
    ├─> Pre-processing (optional)
    │
    ▼
OmniDiffusionRequest  (Processed)
    │
    ├─> Executor.execute_model()
    │
    ▼
[Worker executes model OR External system processes]
    │
    ▼
DiffusionOutput  (Raw model output)
    │
    ├─> Post-processing (optional)
    │
    ▼
Images / Tensors
    │
    ├─> DiffusionEngine.step() wraps
    │
    ▼
OmniRequestOutput  (Final user-facing output)
```

## Key Abstraction Benefits

```
┌─────────────────────────────────────────────────────────┐
│  Without Executor Pattern                               │
├─────────────────────────────────────────────────────────┤
│  DiffusionEngine                                        │
│    ├─ Worker launching                                  │
│    ├─ Message queue setup                               │
│    ├─ IPC communication                                 │
│    ├─ Process management                                │
│    ├─ Request processing                                │
│    └─ Response handling                                 │
│                                                          │
│  Problem: Mixed responsibilities, hard to customize     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  With Executor Pattern                                  │
├─────────────────────────────────────────────────────────┤
│  DiffusionEngine          │  Executor                   │
│    ├─ Request processing  │    ├─ Worker launching      │
│    └─ Response handling   │    ├─ Message queues        │
│                           │    ├─ IPC communication     │
│                           │    └─ Process management    │
│                                                          │
│  Benefit: Clean separation, easy to customize executor  │
└─────────────────────────────────────────────────────────┘
```

## Extension Points

```
┌─────────────────────────────────────────────────────────┐
│  Where You Can Customize                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Create Custom Executor                              │
│     └─> Implement DiffusionExecutor interface           │
│                                                          │
│  2. Override execute_model()                            │
│     └─> Custom inference logic                          │
│                                                          │
│  3. Implement collective_rpc()                          │
│     └─> Custom RPC mechanism                            │
│                                                          │
│  4. Add custom initialization                           │
│     └─> _init_executor() with your setup                │
│                                                          │
│  5. Custom health checking                              │
│     └─> check_health() with your logic                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
