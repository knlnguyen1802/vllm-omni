# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Ray-based external executor for diffusion models.

This module demonstrates how to create an external executor that works with
Ray actors. Each actor wraps a WorkerWrapperBase instance and handles RPC calls.

Example Usage:
    # 1. Start Ray and create worker actors
    import ray
    from vllm_omni.diffusion.executor.ray_external_executor import (
        RayDiffusionWorkerActor,
        RayExternalDiffusionExecutor,
    )
    
    ray.init()
    
    # Create worker actors
    num_gpus = 2
    worker_actors = []
    for rank in range(num_gpus):
        actor = RayDiffusionWorkerActor.options(
            num_gpus=1,
            name=f"diffusion_worker_{rank}",
        ).remote(rank=rank, od_config=od_config)
        worker_actors.append(actor)
    
    # Wait for actors to be ready
    ray.get([actor.initialize.remote() for actor in worker_actors])
    
    # 2. Create engine with Ray external executor
    config = OmniDiffusionConfig(
        model="your-model",
        num_gpus=num_gpus,
        distributed_executor_backend=RayExternalDiffusionExecutor,
    )
    
    # Pass actor names to executor via environment or config
    os.environ["DIFFUSION_WORKER_ACTOR_NAMES"] = ",".join(
        [f"diffusion_worker_{i}" for i in range(num_gpus)]
    )
    
    engine = DiffusionEngine(config)
    
    # 3. Use engine normally
    requests = [OmniDiffusionRequest(prompt="test", ...)]
    output = engine.step(requests)
    
    # 4. Cleanup
    engine.close()
"""

import os
import time
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.external_executor import ExternalDiffusionExecutor

logger = init_logger(__name__)

# Import Ray only when needed
try:
    import ray
    from ray.actor import ActorHandle

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ActorHandle = Any  # Type hint fallback


class RayDiffusionWorkerActor:
    """Ray actor that wraps a WorkerWrapperBase instance.

    This actor runs on a GPU and handles diffusion model inference.
    It receives RPC calls from the executor and executes them on the worker.
    """

    def __init__(self, rank: int, od_config: OmniDiffusionConfig):
        """Initialize the Ray worker actor.

        Args:
            rank: The rank of this worker (0-indexed)
            od_config: Diffusion configuration
        """
        self.rank = rank
        self.od_config = od_config
        self.worker = None

    def initialize(self) -> dict[str, Any]:
        """Initialize the worker.

        Returns:
            Initialization status dict
        """
        logger.info(f"Initializing RayDiffusionWorkerActor rank {self.rank}")

        try:
            # Import here to avoid loading on driver
            from vllm_omni.utils.platform_utils import get_diffusion_worker_class

            # Get worker class
            worker_class = get_diffusion_worker_class()

            # Create worker wrapper
            # Note: This assumes WorkerWrapperBase can be instantiated directly
            # You may need to adapt this based on your actual worker implementation
            self.worker = worker_class(
                rank=self.rank,
                od_config=self.od_config,
            )

            logger.info(f"RayDiffusionWorkerActor rank {self.rank} initialized successfully")

            return {
                "status": "ready",
                "rank": self.rank,
            }

        except Exception as e:
            logger.error(f"Failed to initialize worker rank {self.rank}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "rank": self.rank,
            }

    def execute_rpc(self, method: str, args: tuple = (), kwargs: dict | None = None) -> Any:
        """Execute an RPC call on the worker.

        Args:
            method: Method name to call on the worker
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Result from the method call
        """
        if self.worker is None:
            raise RuntimeError(f"Worker rank {self.rank} not initialized")

        kwargs = kwargs or {}

        try:
            # Get the method from worker
            method_func = getattr(self.worker, method)

            # Execute the method
            result = method_func(*args, **kwargs)

            return result

        except Exception as e:
            logger.error(f"RPC call failed on worker rank {self.rank}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "rank": self.rank,
            }

    def generate(self, requests: list) -> Any:
        """Generate outputs for diffusion requests.

        Args:
            requests: List of OmniDiffusionRequest objects

        Returns:
            DiffusionOutput
        """
        # This is a special method that workers expose for generation
        return self.execute_rpc("generate", args=(requests,))

    def get_rank(self) -> int:
        """Get the rank of this worker.

        Returns:
            Worker rank
        """
        return self.rank

    def shutdown(self) -> None:
        """Shutdown the worker."""
        logger.info(f"Shutting down RayDiffusionWorkerActor rank {self.rank}")
        if self.worker:
            # Clean up worker resources
            # The worker may have cleanup methods
            if hasattr(self.worker, "shutdown"):
                self.worker.shutdown()
        self.worker = None


class RayExternalDiffusionExecutor(ExternalDiffusionExecutor):
    """Ray-based external executor for diffusion models.

    This executor connects to Ray actors that wrap WorkerWrapperBase instances.
    It forwards all RPC calls to these Ray actors.

    The executor expects worker actors to be already created and registered
    with specific names following the pattern: "diffusion_worker_{rank}"

    Environment Variables:
        DIFFUSION_WORKER_ACTOR_NAMES: Comma-separated list of Ray actor names
            (e.g., "diffusion_worker_0,diffusion_worker_1")
    """

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        """Initialize the Ray external executor.

        Args:
            od_config: The diffusion configuration.
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is not installed. Please install it with: pip install ray"
            )

        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. Please call ray.init() before creating "
                "RayExternalDiffusionExecutor."
            )

        super().__init__(od_config)

    def _connect_to_workers(self) -> None:
        """Connect to Ray worker actors."""
        logger.info("Connecting to Ray worker actors")

        # Get actor names from environment or use default pattern
        actor_names_env = os.environ.get("DIFFUSION_WORKER_ACTOR_NAMES")

        if actor_names_env:
            # Parse comma-separated actor names
            actor_names = [name.strip() for name in actor_names_env.split(",")]
        else:
            # Use default naming pattern
            actor_names = [f"diffusion_worker_{i}" for i in range(self.od_config.num_gpus)]

        logger.info(f"Looking for Ray actors: {actor_names}")

        # Get handles to Ray actors
        self.worker_handles = []
        for actor_name in actor_names:
            try:
                actor = ray.get_actor(actor_name)
                self.worker_handles.append(actor)
                logger.info(f"Connected to Ray actor: {actor_name}")
            except ValueError as e:
                raise RuntimeError(
                    f"Failed to find Ray actor '{actor_name}'. "
                    "Make sure worker actors are created before initializing the executor. "
                    f"Error: {e}"
                )

        logger.info(f"Connected to {len(self.worker_handles)} Ray worker actors")

    def _forward_rpc_to_workers(
        self,
        method: str,
        timeout: float | None,
        args: tuple,
        kwargs: dict,
        unique_reply_rank: int | None,
    ) -> Any:
        """Forward RPC call to Ray worker actors.

        Args:
            method: Method name to call
            timeout: Optional timeout
            args: Positional arguments
            kwargs: Keyword arguments
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Results from workers
        """
        # Determine which workers to query
        if unique_reply_rank is not None:
            if unique_reply_rank >= len(self.worker_handles):
                raise ValueError(
                    f"unique_reply_rank {unique_reply_rank} out of range "
                    f"(have {len(self.worker_handles)} workers)"
                )
            workers_to_query = [self.worker_handles[unique_reply_rank]]
        else:
            workers_to_query = self.worker_handles

        # Submit RPC calls to Ray actors
        futures = []
        for worker in workers_to_query:
            future = worker.execute_rpc.remote(method, args, kwargs)
            futures.append(future)

        # Wait for results
        deadline = None if timeout is None else time.monotonic() + timeout

        try:
            if deadline is not None:
                remaining_timeout = deadline - time.monotonic()
                if remaining_timeout <= 0:
                    raise TimeoutError(f"RPC call to {method} timed out before submission.")

                results = ray.get(futures, timeout=remaining_timeout)
            else:
                results = ray.get(futures)

            # Check for errors in results
            for result in results:
                if isinstance(result, dict) and result.get("status") == "error":
                    raise RuntimeError(
                        f"Worker failed with error '{result.get('error')}', "
                        f"rank {result.get('rank')}"
                    )

            return results[0] if unique_reply_rank is not None else results

        except ray.exceptions.GetTimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e

    def _check_worker_health(self) -> None:
        """Check health of Ray worker actors."""
        for i, worker in enumerate(self.worker_handles):
            try:
                # Try to get rank as a simple health check
                rank = ray.get(worker.get_rank.remote(), timeout=5.0)
                if rank != i:
                    raise RuntimeError(
                        f"Worker {i} returned unexpected rank {rank}"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Worker {i} health check failed: {e}"
                )

    def _disconnect_from_workers(self) -> None:
        """Disconnect from Ray workers.

        Note: This does NOT terminate the Ray actors. They should be
        managed by the caller.
        """
        logger.info("Disconnecting from Ray worker actors")
        # Just clear the handles; actors remain running
        self.worker_handles = []
