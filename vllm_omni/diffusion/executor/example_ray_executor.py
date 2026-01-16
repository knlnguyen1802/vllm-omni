"""
Example: Ray Actor Executor

This example demonstrates how to create a custom executor that uses a Ray actor
to manage scheduler and worker processes internally.

Use case: Ray cluster deployment, distributed inference with Ray.
"""

from typing import Any

import ray
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


@ray.remote
class DiffusionWorkerActor:
    """
    Ray actor that encapsulates the entire worker infrastructure.
    
    This actor internally manages:
    - Scheduler initialization
    - Worker process creation
    - All worker communication
    - Resource management
    
    The executor just holds a reference to this actor and calls methods on it.
    """

    def __init__(self, od_config: OmniDiffusionConfig):
        """Initialize the actor with scheduler and workers."""
        self.od_config = od_config
        self.scheduler = None
        self.worker_processes = []
        
    def initialize(self):
        """
        Initialize scheduler and spawn worker processes.
        
        This runs inside the Ray actor, so all workers are children of this actor.
        """
        import multiprocessing as mp
        
        from vllm_omni.diffusion.scheduler import scheduler
        from vllm_omni.utils.platform_utils import get_diffusion_worker_class
        
        logger.info("Initializing DiffusionWorkerActor...")
        
        # Initialize scheduler
        scheduler.initialize(self.od_config)
        self.scheduler = scheduler
        
        # Get worker class
        worker_class = get_diffusion_worker_class(self.od_config)
        
        # Spawn worker processes
        self.worker_processes = []
        for rank in range(self.od_config.num_gpus):
            proc = mp.Process(
                target=worker_class.run_worker,
                args=(
                    rank,
                    self.scheduler.broadcast_handle,
                    self.od_config,
                ),
                name=f"DiffusionWorker-{rank}",
                daemon=True,
            )
            proc.start()
            self.worker_processes.append(proc)
            logger.info(f"Started worker process {rank}: PID {proc.pid}")
        
        # Wait for workers to be ready
        self._wait_for_workers_ready()
        
        logger.info("DiffusionWorkerActor initialization complete")
        return True
    
    def _wait_for_workers_ready(self):
        """Wait for all workers to signal ready."""
        import time
        
        logger.info("Waiting for workers to be ready...")
        ready_count = 0
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        while ready_count < self.scheduler.num_workers:
            if time.time() - start_time > timeout:
                raise TimeoutError("Workers failed to initialize within timeout")
            
            msg = self.scheduler.result_mq.dequeue(timeout=1.0)
            if msg is not None and msg.get("status") == "ready":
                ready_count += 1
                logger.info(f"Worker {ready_count}/{self.scheduler.num_workers} ready")
        
        logger.info("All workers ready")
    
    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ):
        """Forward RPC call to workers."""
        kwargs = kwargs or {}
        
        # Broadcast RPC to all workers
        message = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
        }
        
        for _ in range(self.scheduler.num_workers):
            self.scheduler.mq.enqueue(message)
        
        # Collect responses
        if unique_reply_rank is not None:
            # Only wait for specific rank
            response = self.scheduler.result_mq.dequeue(timeout=30.0)
            return response
        else:
            # Wait for all workers
            responses = []
            for _ in range(self.scheduler.num_workers):
                response = self.scheduler.result_mq.dequeue(timeout=30.0)
                responses.append(response)
            return responses
    
    def execute_model(self, requests: list[dict]):
        """Execute model inference on workers."""
        # Convert dict back to OmniDiffusionRequest if needed
        # (Ray serializes objects as dicts by default)
        
        # Broadcast execution request
        message = {
            "type": "execute",
            "requests": requests,
        }
        
        for _ in range(self.scheduler.num_workers):
            self.scheduler.mq.enqueue(message)
        
        # Collect results from workers
        outputs = []
        for _ in range(self.scheduler.num_workers):
            result = self.scheduler.result_mq.dequeue(timeout=300.0)
            if result:
                outputs.append(result)
        
        # Aggregate results (simplified)
        if outputs:
            return outputs[0]  # Return first worker's output
        return None
    
    def check_health(self):
        """Check if all workers are healthy."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        
        # Send health check to all workers
        message = {"type": "health_check"}
        for _ in range(self.scheduler.num_workers):
            self.scheduler.mq.enqueue(message)
        
        # Wait for responses
        healthy_count = 0
        for _ in range(self.scheduler.num_workers):
            response = self.scheduler.result_mq.dequeue(timeout=5.0)
            if response and response.get("status") == "ok":
                healthy_count += 1
        
        if healthy_count != self.scheduler.num_workers:
            raise RuntimeError(
                f"Only {healthy_count}/{self.scheduler.num_workers} workers are healthy"
            )
        
        return True
    
    def shutdown(self):
        """Shutdown all workers and clean up resources."""
        from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE
        
        logger.info("Shutting down DiffusionWorkerActor...")
        
        # Send shutdown signal to all workers
        if self.scheduler:
            for _ in range(self.scheduler.num_workers):
                self.scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
            
            # Wait for workers to terminate
            for proc in self.worker_processes:
                proc.join(timeout=30)
                if proc.is_alive():
                    logger.warning(f"Terminating worker {proc.name}")
                    proc.terminate()
            
            self.scheduler.close()
        
        logger.info("DiffusionWorkerActor shutdown complete")
        return True


class RayActorExecutor(DiffusionExecutor):
    """
    Executor that delegates to a Ray actor.
    
    This executor is lightweight - it just creates and manages a reference
    to a Ray actor that handles all the heavy lifting internally.
    
    Architecture:
        Engine -> RayActorExecutor -> Ray Actor -> Scheduler + Workers
    """

    uses_ray: bool = True
    supports_pp: bool = False

    def _init_executor(self) -> None:
        """Initialize Ray and create the worker actor."""
        logger.info("Initializing Ray Actor Executor...")
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray_init_kwargs = getattr(self.od_config, "ray_init_kwargs", {})
            ray.init(**ray_init_kwargs)
            logger.info("Ray initialized")
        
        # Get Ray actor configuration
        num_cpus = getattr(self.od_config, "ray_actor_cpus", 1)
        num_gpus = getattr(self.od_config, "ray_actor_gpus", self.od_config.num_gpus)
        memory = getattr(self.od_config, "ray_actor_memory", None)
        
        # Create the Ray actor
        actor_options = {
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        }
        if memory:
            actor_options["memory"] = memory
        
        logger.info(f"Creating Ray actor with options: {actor_options}")
        
        # Create actor with specified resources
        self.worker_actor = DiffusionWorkerActor.options(**actor_options).remote(
            self.od_config
        )
        
        # Initialize the actor (this spawns workers inside the actor)
        # Use ray.get() to wait for initialization to complete
        init_result = ray.get(self.worker_actor.initialize.remote())
        
        if not init_result:
            raise RuntimeError("Failed to initialize Ray worker actor")
        
        logger.info("Ray Actor Executor initialized successfully")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Forward RPC call to Ray actor."""
        if self._closed:
            raise RuntimeError("Executor is closed.")
        
        logger.info(f"Forwarding RPC to Ray actor: {method}")
        
        # Call method on Ray actor and wait for result
        future = self.worker_actor.collective_rpc.remote(
            method=method,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )
        
        # Use timeout if specified
        if timeout:
            result = ray.get(future, timeout=timeout)
        else:
            result = ray.get(future)
        
        return result

    def execute_model(
        self, requests: list[OmniDiffusionRequest]
    ) -> DiffusionOutput | None:
        """Execute model via Ray actor."""
        if self._closed:
            raise RuntimeError("Executor is closed.")
        
        logger.info(f"Executing {len(requests)} requests via Ray actor")
        
        # Serialize requests to dict (Ray handles serialization)
        serialized_requests = [
            {
                "prompt": r.prompt,
                "height": r.height,
                "width": r.width,
                "num_inference_steps": r.num_inference_steps,
                "num_outputs_per_prompt": r.num_outputs_per_prompt,
                "request_id": r.request_id,
            }
            for r in requests
        ]
        
        # Call execute_model on Ray actor
        future = self.worker_actor.execute_model.remote(serialized_requests)
        
        try:
            result = ray.get(future, timeout=300)  # 5 minute timeout
            
            # Convert to DiffusionOutput
            if isinstance(result, dict):
                return DiffusionOutput(
                    output=result.get("output"),
                    error=result.get("error"),
                    trajectory_latents=result.get("trajectory_latents"),
                    trajectory_timesteps=result.get("trajectory_timesteps"),
                )
            
            return result
            
        except ray.exceptions.RayTaskError as e:
            logger.error(f"Ray task failed: {e}")
            return DiffusionOutput(
                output=None,
                error=str(e),
                trajectory_latents=None,
                trajectory_timesteps=None,
            )
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return DiffusionOutput(
                output=None,
                error=str(e),
                trajectory_latents=None,
                trajectory_timesteps=None,
            )

    def check_health(self) -> None:
        """Check if Ray actor and workers are healthy."""
        if self._closed:
            raise RuntimeError("Executor is closed.")
        
        try:
            result = ray.get(self.worker_actor.check_health.remote(), timeout=10)
            if not result:
                raise RuntimeError("Health check failed")
            logger.info("Health check: Ray actor and workers are healthy")
        except Exception as e:
            raise RuntimeError(f"Health check failed: {e}")

    def shutdown(self) -> None:
        """Shutdown Ray actor and clean up resources."""
        if self._closed:
            return
        
        logger.info("Shutting down Ray Actor Executor...")
        
        try:
            # Tell actor to shutdown its workers
            ray.get(self.worker_actor.shutdown.remote(), timeout=60)
            
            # Kill the actor
            ray.kill(self.worker_actor)
            
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        
        logger.info("Ray Actor Executor shutdown complete")


async def example_usage():
    """Example of using Ray actor executor with AsyncOmniDiffusion."""
    import asyncio
    
    from vllm_omni.entrypoints.omni_diffusion import AsyncOmniDiffusion
    
    # Configure with Ray executor
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        model_name_or_path="Qwen/Qwen3-Omni",
        num_gpus=2,  # Number of GPUs for workers
        executor_class=RayActorExecutor,  # Pass RayActorExecutor as external executor
        # Ray configuration
        ray_init_kwargs={
            "ignore_reinit_error": True,
            "num_cpus": 4,
            "num_gpus": 2,
        },
        ray_actor_cpus=1,  # CPUs for the actor itself
        ray_actor_gpus=2,  # GPUs allocated to the actor
        ray_actor_memory=8 * 1024**3,  # 8GB memory
    )
    
    print("Creating AsyncOmniDiffusion with Ray executor...")
    async_diffusion = AsyncOmniDiffusion(config)
    
    try:
        # Create generation request
        print("\nGenerating image: 'A beautiful sunset over mountains'")
        print("Parameters: 1024x1024, 20 steps")
        print("-" * 60)
        
        # Generate using the async API
        results = []
        async for output in async_diffusion.generate(
            prompt="A beautiful sunset over mountains",
            height=1024,
            width=1024,
            num_inference_steps=20,
            num_outputs_per_prompt=1,
        ):
            results.append(output)
            print(f"✓ Received output: {type(output)}")
        
        print("-" * 60)
        print(f"✓ Generation completed! Received {len(results)} output(s)")
        
        # Show results
        for idx, result in enumerate(results):
            print(f"\nResult {idx + 1}:")
            if hasattr(result, 'outputs') and result.outputs:
                for output_idx, output in enumerate(result.outputs):
                    print(f"  Output {output_idx + 1}:")
                    if hasattr(output, 'image') and output.image:
                        print(f"    - Image data: {type(output.image)}")
                        if hasattr(output.image, 'size'):
                            print(f"    - Image size: {output.image.size}")
                    if hasattr(output, 'error') and output.error:
                        print(f"    - Error: {output.error}")
            
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing AsyncOmniDiffusion...")
        await async_diffusion.close()
        print("✓ AsyncOmniDiffusion shutdown complete")


if __name__ == "__main__":
    import asyncio
    import sys
    
    print("=" * 60)
    print("Ray Actor Executor Example")
    print("=" * 60)
    print()
    print("This example shows how to use a Ray actor to manage")
    print("scheduler and worker processes internally.")
    print()
    print("Benefits:")
    print("- All worker management happens inside the Ray actor")
    print("- Can deploy to Ray clusters easily")
    print("- Better resource isolation")
    print("- Scales to distributed environments")
    print()
    print("Setup:")
    print("1. Install Ray: pip install ray")
    print("2. (Optional) Start Ray cluster: ray start --head")
    print("3. Run this example")
    print()
    print("=" * 60)
    print()
    
    # Check if Ray is available
    try:
        import ray
        print("✓ Ray is installed")
    except ImportError:
        print("✗ Ray not found. Install with: pip install ray")
        print()
        print("To use in your code:")
        print("  config = OmniDiffusionConfig(")
        print("      executor_class=RayActorExecutor,")
        print("      num_gpus=2,")
        print("      ray_init_kwargs={'address': 'auto'},")
        print("  )")
        print("  async_diffusion = AsyncOmniDiffusion(config)")
        sys.exit(1)
    
    # Run the example
    try:
        print("Running example...")
        print()
        asyncio.run(example_usage())
        print()
        print("=" * 60)
        print("✓ Example completed successfully!")
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Note: This example requires:")
        print("- Ray installed: pip install ray")
        print("- GPU resources available (or modify num_gpus=0 for CPU)")
        print("- Model weights available")
    
    print()
    print("To use in your code:")
    print("  config = OmniDiffusionConfig(")
    print("      executor_class=RayActorExecutor,")
    print("      num_gpus=2,")
    print("      ray_init_kwargs={'address': 'auto'},")
    print("  )")
    print("  async_diffusion = AsyncOmniDiffusion(config)")
    print("  async for output in async_diffusion.generate(prompt='...', ...):")
    print("      # Process output")
