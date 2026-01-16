# Quick Start Guide: Using Custom Executors

## For Users Who Want to Use External Systems

### Step 1: Create Your Executor Class

Create a new Python file (e.g., `my_executor.py`):

```python
from vllm_omni.diffusion.executor import DiffusionExecutor
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

class MyExecutor(DiffusionExecutor):
    """Send requests to my external system."""
    
    def _init_executor(self):
        """Set up connection to your system."""
        # Initialize your client/connection here
        self.client = YourClient(url="http://your-server")
    
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, unique_reply_rank=None):
        """Forward RPC to your system."""
        # Send the method call to your system
        result = self.client.call_method(method, *args, **(kwargs or {}))
        return result
    
    def execute_model(self, requests):
        """Execute model on your system."""
        # Send requests to your inference system
        response = self.client.generate(requests)
        
        # Convert response to DiffusionOutput format
        return DiffusionOutput(
            output=response.images,
            error=None,
            trajectory_latents=None,
            trajectory_timesteps=None,
        )
    
    def check_health(self):
        """Check if your system is working."""
        if not self.client.ping():
            raise RuntimeError("System is down!")
    
    def shutdown(self):
        """Clean up."""
        self.client.close()
```

### Step 2: Configure the Engine to Use Your Executor

**Option 1: Using String Path (Recommended)**

```python
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig

config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    num_gpus=1,
    executor_class="my_package.MyExecutor",  # String path to your executor
)

engine = DiffusionEngine(config)
```

**Option 2: Using Class Type**

```python
from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig
from my_package import MyExecutor

config = OmniDiffusionConfig(
    model_class_name="Qwen3Omni",
    num_gpus=1,
    executor_class=MyExecutor,  # Direct class reference
)

engine = DiffusionEngine(config)
```

### Step 3: Use the Engine Normally

```python
from vllm_omni.diffusion.request import OmniDiffusionRequest

# Create request
request = OmniDiffusionRequest(prompt="A sunset")
output = engine.step([request])

# Clean up
engine.close()
```

## Common Patterns

### Pattern 1: HTTP REST API

```python
class HTTPExecutor(DiffusionExecutor):
    def _init_executor(self):
        import requests
        self.session = requests.Session()
        self.url = "http://my-server:8000"
    
    def execute_model(self, requests):
        response = self.session.post(
            f"{self.url}/generate",
            json={"requests": [r.to_dict() for r in requests]}
        )
        return DiffusionOutput.from_dict(response.json())
```

### Pattern 2: gRPC Service

```python
class GRPCExecutor(DiffusionExecutor):
    def _init_executor(self):
        import grpc
        from your_proto import inference_pb2_grpc
        
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = inference_pb2_grpc.InferenceStub(channel)
    
    def execute_model(self, requests):
        # Convert to proto
        proto_requests = [to_proto(r) for r in requests]
        response = self.stub.Generate(proto_requests)
        return from_proto(response)
```

### Pattern 3: Cloud Service

```python
class AWSExecutor(DiffusionExecutor):
    def _init_executor(self):
        import boto3
        self.client = boto3.client('sagemaker-runtime')
        self.endpoint = 'my-diffusion-endpoint'
    
    def execute_model(self, requests):
        import json
        payload = json.dumps([r.to_dict() for r in requests])
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint,
            Body=payload
        )
        return DiffusionOutput.from_dict(json.loads(response['Body'].read()))
```

## Key Points

✅ **Must Implement:**
- `_init_executor()` - Setup
- `collective_rpc()` - General RPC forwarding
- `check_health()` - Health checks
- `shutdown()` - Cleanup

✅ **Optional to Override:**
- `execute_model()` - If you have a specialized API

✅ **Return Compatibility:**
- `execute_model()` must return `DiffusionOutput` or `None`
- `collective_rpc()` return type depends on the method being called

✅ **Error Handling:**
- Raise `RuntimeError` on failures
- Set `output.error` for generation errors

## Testing Checklist

- [ ] Executor initializes successfully
- [ ] `check_health()` works
- [ ] `execute_model()` returns valid `DiffusionOutput`
- [ ] `collective_rpc()` forwards calls correctly
- [ ] `shutdown()` cleans up resources
- [ ] Results are compatible with `OmniRequestOutput`

## Need Help?

See:
- [README.md](README.md) - Full documentation
- [example_custom_executor.py](example_custom_executor.py) - Working example
- [external_executor.py](external_executor.py) - Templates
