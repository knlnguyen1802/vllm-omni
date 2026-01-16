"""
Example: ZMQ External Executor

This example demonstrates how to create a custom executor that forwards
diffusion requests to an external worker via ZeroMQ (ZMQ) sockets.

Use case: Remote inference server, distributed workers, or external GPU cluster.
"""

from typing import Any

import zmq
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class ZMQExecutor(DiffusionExecutor):
    """
    Executor that forwards requests to an external worker via ZMQ sockets.
    
    This executor acts as a client that sends requests to a ZMQ server
    (the actual worker) and receives results back.
    
    Architecture:
        Engine -> ZMQExecutor (client) -> ZMQ Socket -> External Worker (server)
    """

    def _init_executor(self) -> None:
        """Initialize ZMQ context and connect to external worker."""
        logger.info("Initializing ZMQ Executor...")
        
        # Get ZMQ configuration from od_config
        self.zmq_host = getattr(self.od_config, "zmq_host", "localhost")
        self.zmq_port = getattr(self.od_config, "zmq_port", 5555)
        self.zmq_timeout = getattr(self.od_config, "zmq_timeout", 300000)  # 5 minutes in ms
        
        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)  # Request socket
        
        # Connect to external worker
        zmq_address = f"tcp://{self.zmq_host}:{self.zmq_port}"
        self.socket.connect(zmq_address)
        self.socket.setsockopt(zmq.RCVTIMEO, self.zmq_timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, self.zmq_timeout)
        
        logger.info(f"ZMQ Executor connected to {zmq_address}")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Forward RPC call to external worker via ZMQ."""
        if self._closed:
            raise RuntimeError("Executor is closed.")
        
        kwargs = kwargs or {}
        
        logger.info(f"Forwarding RPC call: {method}")
        
        # Prepare RPC message
        message = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "unique_reply_rank": unique_reply_rank,
        }
        
        try:
            # Send request to external worker
            self.socket.send_json(message)
            
            # Receive response
            response = self.socket.recv_json()
            
            # Check for errors
            if isinstance(response, dict) and response.get("status") == "error":
                raise RuntimeError(
                    f"Worker failed with error: {response.get('error')}"
                )
            
            return response
            
        except zmq.error.Again:
            raise TimeoutError(f"ZMQ RPC call to {method} timed out")
        except Exception as e:
            logger.error(f"ZMQ RPC call failed: {e}")
            raise

    def execute_model(
        self, requests: list[OmniDiffusionRequest]
    ) -> DiffusionOutput | None:
        """Execute model via ZMQ external worker."""
        if self._closed:
            raise RuntimeError("Executor is closed.")
        
        logger.info(f"Executing {len(requests)} requests via ZMQ")
        
        # Serialize requests to dict
        serialized_requests = [
            {
                "prompt": r.prompt,
                "height": r.height,
                "width": r.width,
                "num_inference_steps": r.num_inference_steps,
                "num_outputs_per_prompt": r.num_outputs_per_prompt,
                "request_id": r.request_id,
                # Add other fields as needed
            }
            for r in requests
        ]
        
        # Send execution request
        message = {
            "type": "execute",
            "requests": serialized_requests,
        }
        
        try:
            self.socket.send_json(message)
            response = self.socket.recv_json()
            
            # Check for errors
            if response.get("status") == "error":
                return DiffusionOutput(
                    output=None,
                    error=response.get("error"),
                    trajectory_latents=None,
                    trajectory_timesteps=None,
                )
            
            # Convert response to DiffusionOutput
            return DiffusionOutput(
                output=response.get("output"),
                error=None,
                trajectory_latents=response.get("trajectory_latents"),
                trajectory_timesteps=response.get("trajectory_timesteps"),
            )
            
        except zmq.error.Again:
            return DiffusionOutput(
                output=None,
                error="ZMQ request timed out",
                trajectory_latents=None,
                trajectory_timesteps=None,
            )
        except Exception as e:
            logger.error(f"ZMQ execute_model failed: {e}")
            return DiffusionOutput(
                output=None,
                error=str(e),
                trajectory_latents=None,
                trajectory_timesteps=None,
            )

    def check_health(self) -> None:
        """Check if ZMQ connection to external worker is healthy."""
        if self._closed:
            raise RuntimeError("Executor is closed.")
        
        try:
            # Send ping message
            self.socket.send_json({"type": "ping"})
            response = self.socket.recv_json()
            
            if response.get("status") != "ok":
                raise RuntimeError("Worker is unhealthy")
                
            logger.info("Health check: ZMQ worker is healthy")
            
        except zmq.error.Again:
            raise RuntimeError("Health check timed out")
        except Exception as e:
            raise RuntimeError(f"Health check failed: {e}")

    def shutdown(self) -> None:
        """Close ZMQ connection and clean up resources."""
        if self._closed:
            return
        
        logger.info("Shutting down ZMQ Executor...")
        
        try:
            # Send shutdown message to worker (optional)
            self.socket.send_json({"type": "shutdown"})
            self.socket.recv_json()  # Wait for acknowledgment
        except Exception as e:
            logger.warning(f"Error during shutdown handshake: {e}")
        finally:
            # Close socket and terminate context
            if hasattr(self, "socket"):
                self.socket.close()
            if hasattr(self, "context"):
                self.context.term()
            
            logger.info("ZMQ Executor shutdown complete")


def example_usage():
    """Example of using ZMQ executor with DiffusionEngine."""
    from vllm_omni.diffusion import DiffusionEngine
    
    # Configure with ZMQ executor
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        model_name_or_path="Qwen/Qwen3-Omni",
        num_gpus=0,  # No local GPUs needed
        executor_class=ZMQExecutor,  # Or: executor_class="__main__.ZMQExecutor"
        # ZMQ configuration
        zmq_host="localhost",  # External worker host
        zmq_port=5555,         # External worker port
        zmq_timeout=300000,    # 5 minutes timeout
    )
    
    engine = DiffusionEngine(config)
    
    try:
        # Check health
        engine.check_health()
        print("✓ ZMQ worker is healthy")
        
        # Create request
        request = OmniDiffusionRequest(
            prompt="A beautiful sunset over mountains",
            height=1024,
            width=1024,
            num_inference_steps=20,
            num_outputs_per_prompt=1,
        )
        
        # Execute
        print("Executing model via ZMQ...")
        output = engine.step([request])
        
        if output and not (hasattr(output, 'error') and output.error):
            print("✓ Model execution successful")
        else:
            error_msg = output.error if (output and hasattr(output, 'error')) else 'None'
            print(f"✗ Model execution failed: {error_msg}")
            
    finally:
        engine.close()
        print("✓ Engine shutdown complete")


if __name__ == "__main__":
    print("=" * 60)
    print("ZMQ External Executor Example")
    print("=" * 60)
    print()
    print("This example shows how to forward diffusion requests to")
    print("an external worker via ZeroMQ sockets.")
    print()
    print("Setup:")
    print("1. Start your ZMQ worker server on port 5555")
    print("2. Worker should handle messages:")
    print("   - {'type': 'ping'} -> {'status': 'ok'}")
    print("   - {'type': 'execute', 'requests': [...]} -> results")
    print("   - {'type': 'shutdown'} -> {'status': 'ok'}")
    print("3. Run this example")
    print()
    print("=" * 60)
    print()
    
    # Uncomment to run (requires ZMQ worker running)
    # example_usage()
    
    print("Example complete!")
    print()
    print("To use in your code:")
    print("  config = OmniDiffusionConfig(")
    print("      executor_class=ZMQExecutor,")
    print("      zmq_host='localhost',")
    print("      zmq_port=5555,")
    print("  )")
    print("  engine = DiffusionEngine(config)")

