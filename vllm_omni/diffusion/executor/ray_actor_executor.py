# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
RayActorExecutor

Executor that creates and manages a Ray actor for diffusion execution.
The Ray actor internally manages WorkerWrapperBase instances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor.executor_base import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

# Import Ray only when needed
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class RayActorExecutor(DiffusionExecutor):
    """Executor that delegates diffusion execution to a Ray actor.

    This executor creates and manages a single Ray actor that handles
    all diffusion model execution. The actor internally manages workers
    and provides a unified interface for RPC calls.
    """

    uses_ray: bool = True
    supports_pp: bool = False

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        """Initialize the Ray actor executor.

        Args:
            od_config: Diffusion configuration
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed. Install it with: pip install ray")

        self.worker_actor = None
        self._closed = False
        super().__init__(od_config)

    def _init_executor(self) -> None:
        """Initialize the executor by creating and initializing the Ray actor."""
        logger.info("Initializing RayActorExecutor...")

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray_init_kwargs = getattr(self.od_config, "ray_init_kwargs", {})
            ray.init(**ray_init_kwargs)
            logger.info("Ray initialized")

        # Get Ray actor resource configuration
        num_cpus = getattr(self.od_config, "ray_actor_cpus", 1)
        num_gpus = getattr(self.od_config, "ray_actor_gpus", self.od_config.num_gpus)
        memory = getattr(self.od_config, "ray_actor_memory", None)

        actor_opts = {"num_cpus": num_cpus, "num_gpus": num_gpus}
        if memory:
            actor_opts["memory"] = memory

        logger.info(f"Creating Ray actor with options: {actor_opts}")

        # Import and create the worker actor
        from vllm_omni.diffusion.executor.ray_diffusion_worker_actor import DiffusionWorkerActor

        self.worker_actor = DiffusionWorkerActor.options(**actor_opts).remote(self.od_config)

        # Initialize the actor
        result = ray.get(self.worker_actor.initialize.remote())
        if not result:
            raise RuntimeError("Failed to initialize Ray DiffusionWorkerActor")

        logger.info("RayActorExecutor initialized successfully")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Execute an RPC call on the worker actor.

        Args:
            method: Method name to call
            timeout: Optional timeout in seconds
            args: Positional arguments
            kwargs: Keyword arguments
            unique_reply_rank: Ignored in single-actor mode

        Returns:
            Result from the RPC call
        """
        if self._closed:
            raise RuntimeError("RayActorExecutor is closed")

        if not isinstance(method, str):
            raise ValueError(f"RayActorExecutor only supports string method names, got {type(method)}")

        kwargs = kwargs or {}

        # Call the actor's collective_rpc method
        future = self.worker_actor.collective_rpc.remote(
            method=method,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )

        # Wait for result with optional timeout
        try:
            if timeout is not None:
                result = ray.get(future, timeout=timeout)
            else:
                result = ray.get(future)

            return result

        except ray.exceptions.GetTimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out after {timeout}s") from e

    def add_requests(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """Execute diffusion model for the given requests.

        Args:
            requests: List of OmniDiffusionRequest objects

        Returns:
            DiffusionOutput from processing the requests
        """
        if self._closed:
            raise RuntimeError("RayActorExecutor is closed")

        logger.info(f"Executing {len(requests)} request(s) via Ray actor")

        # Convert requests to dicts for serialization
        request_dicts = []
        for req in requests:
            req_dict = {
                "prompt": req.prompt,
                "height": req.height,
                "width": req.width,
                "num_inference_steps": req.num_inference_steps,
                "num_outputs_per_prompt": req.num_outputs_per_prompt,
                "request_id": req.request_id,
                "pil_image": req.pil_image,
            }
            request_dicts.append(req_dict)

        # Call actor's execute_model method
        future = self.worker_actor.execute_model.remote(request_dicts)

        try:
            result = ray.get(future, timeout=300.0)  # 5 minute timeout

            # Convert result to DiffusionOutput if needed
            if isinstance(result, dict):
                return DiffusionOutput(
                    output=result.get("output"),
                    error=result.get("error"),
                    trajectory_latents=result.get("trajectory_latents"),
                    trajectory_timesteps=result.get("trajectory_timesteps"),
                )
            elif isinstance(result, DiffusionOutput):
                return result
            else:
                # Unexpected result type
                return DiffusionOutput(
                    output=result,
                    error=None,
                    trajectory_latents=None,
                    trajectory_timesteps=None,
                )

        except ray.exceptions.GetTimeoutError:
            logger.error("Model execution timed out after 300 seconds")
            return DiffusionOutput(
                output=None,
                error="Execution timed out",
                trajectory_latents=None,
                trajectory_timesteps=None,
            )
        except Exception as e:
            logger.error(f"Model execution failed: {e}", exc_info=True)
            return DiffusionOutput(
                output=None,
                error=str(e),
                trajectory_latents=None,
                trajectory_timesteps=None,
            )

    def check_health(self) -> None:
        """Check if the executor is healthy.

        Raises:
            RuntimeError: If the executor is not healthy
        """
        if self._closed:
            raise RuntimeError("RayActorExecutor is closed")

        try:
            healthy = ray.get(self.worker_actor.check_health.remote(), timeout=10.0)
            if not healthy:
                raise RuntimeError("Worker actor health check failed")

            logger.debug("RayActorExecutor health check passed")

        except ray.exceptions.GetTimeoutError:
            raise RuntimeError("Health check timed out")
        except Exception as e:
            raise RuntimeError(f"Health check failed: {e}")

    def shutdown(self) -> None:
        """Shutdown the executor and the Ray actor."""
        if self._closed:
            return

        logger.info("Shutting down RayActorExecutor...")
        self._closed = True

        try:
            # Tell actor to shutdown gracefully
            ray.get(self.worker_actor.shutdown.remote(), timeout=60.0)

            # Kill the actor
            ray.kill(self.worker_actor)

            logger.info("RayActorExecutor shutdown complete")

        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
