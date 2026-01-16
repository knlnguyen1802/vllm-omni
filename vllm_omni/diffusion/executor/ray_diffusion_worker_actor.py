# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Ray Actor for Diffusion Workers

This Ray actor manages diffusion workers internally and provides a unified
interface for executing diffusion models.
"""

from typing import Any

import ray
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class DiffusionWorkerActor:
    """Ray actor that manages diffusion workers internally.

    This actor initializes and manages WorkerWrapperBase instances for
    diffusion model execution. It provides methods for RPC calls and
    request processing.
    """

    def __init__(self, od_config: OmniDiffusionConfig):
        """Initialize the diffusion worker actor.

        Args:
            od_config: Diffusion configuration
        """
        self.od_config = od_config
        self.worker = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the worker wrapper.

        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            logger.info("Initializing DiffusionWorkerActor...")

            # Import worker wrapper
            from vllm_omni.utils.platform_utils import get_diffusion_worker_class

            # Get the worker wrapper class
            worker_wrapper_class = get_diffusion_worker_class()

            # For Ray actor, we use rank 0 and manage single or multiple workers
            # The worker wrapper will handle the actual initialization
            self.worker = worker_wrapper_class(
                rank=0,
                od_config=self.od_config,
            )

            # Initialize the worker
            if hasattr(self.worker, "initialize"):
                self.worker.initialize()

            self._initialized = True
            logger.info("DiffusionWorkerActor initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DiffusionWorkerActor: {e}", exc_info=True)
            return False

    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Execute an RPC call on the worker.

        Args:
            method: Method name to call
            args: Positional arguments
            kwargs: Keyword arguments
            unique_reply_rank: Ignored in single-actor mode

        Returns:
            Result from the method call
        """
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call initialize() first.")

        kwargs = kwargs or {}

        try:
            # Get method from worker
            if not hasattr(self.worker, method):
                raise AttributeError(f"Worker has no method '{method}'")

            method_func = getattr(self.worker, method)
            result = method_func(*args, **kwargs)

            logger.debug(f"RPC call '{method}' completed successfully")
            return result

        except Exception as e:
            logger.error(f"RPC call '{method}' failed: {e}", exc_info=True)
            raise

    def execute_model(self, request_dicts: list[dict]) -> DiffusionOutput | dict:
        """Execute diffusion model for the given requests.

        Args:
            request_dicts: List of request dictionaries

        Returns:
            DiffusionOutput or dict with output and error
        """
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call initialize() first.")

        try:
            logger.info(f"Executing model with {len(request_dicts)} request(s)")

            # Convert dicts back to OmniDiffusionRequest objects
            requests = []
            for req_dict in request_dicts:
                req = OmniDiffusionRequest(
                    prompt=req_dict.get("prompt"),
                    height=req_dict.get("height", 1024),
                    width=req_dict.get("width", 1024),
                    num_inference_steps=req_dict.get("num_inference_steps", 50),
                    num_outputs_per_prompt=req_dict.get("num_outputs_per_prompt", 1),
                    request_id=req_dict.get("request_id"),
                    pil_image=req_dict.get("pil_image"),
                )
                requests.append(req)

            # Call the worker's generate method
            if hasattr(self.worker, "generate"):
                result = self.worker.generate(requests)
            else:
                raise RuntimeError("Worker does not have 'generate' method")

            # Return as DiffusionOutput or dict
            if isinstance(result, DiffusionOutput):
                return result
            else:
                # Wrap in DiffusionOutput if needed
                return DiffusionOutput(
                    output=result,
                    error=None,
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

    def check_health(self) -> bool:
        """Check if the worker is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized:
                return False

            # Simple health check - verify worker exists and is responsive
            if self.worker is None:
                return False

            # If worker has a health check method, use it
            if hasattr(self.worker, "check_health"):
                return self.worker.check_health()

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the worker and clean up resources."""
        try:
            logger.info("Shutting down DiffusionWorkerActor...")

            if self.worker and hasattr(self.worker, "shutdown"):
                self.worker.shutdown()

            self.worker = None
            self._initialized = False

            logger.info("DiffusionWorkerActor shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
