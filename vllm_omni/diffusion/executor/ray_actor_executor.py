"""
RayActorExecutor

Custom executor that launches a Ray actor to manage diffusion workers internally.
"""

from __future__ import annotations
from typing import Any
import ray

from vllm.logger import init_logger
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest

from vllm_omni.diffusion.executor.ray_diffusion_worker_actor import DiffusionWorkerActor

logger = init_logger(__name__)


class RayActorExecutor(DiffusionExecutor):
    """Executor that delegates diffusion execution to a Ray actor."""

    uses_ray = True
    supports_pp = False

    # ------------------------------------------------------------------ #
    def _init_executor(self):
        logger.info("Initializing Ray Actor Executor...")

        if not ray.is_initialized():
            ray_init_kwargs = getattr(self.od_config, "ray_init_kwargs", {})
            ray.init(**ray_init_kwargs)
            logger.info("Ray initialized")

        num_cpus = getattr(self.od_config, "ray_actor_cpus", 1)
        num_gpus = getattr(self.od_config, "ray_actor_gpus", self.od_config.num_gpus)
        memory = getattr(self.od_config, "ray_actor_memory", None)

        actor_opts = {"num_cpus": num_cpus, "num_gpus": num_gpus}
        if memory:
            actor_opts["memory"] = memory

        logger.info(f"Creating Ray actor with: {actor_opts}")
        self.worker_actor = DiffusionWorkerActor.options(**actor_opts).remote(self.od_config)

        result = ray.get(self.worker_actor.initialize.remote())
        if not result:
            raise RuntimeError("Failed to initialize Ray DiffusionWorkerActor")
        logger.info("RayActorExecutor initialized successfully")

    # ------------------------------------------------------------------ #
    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("Executor closed")

        future = self.worker_actor.collective_rpc.remote(
            method=method, args=args, kwargs=kwargs, unique_reply_rank=unique_reply_rank
        )
        return ray.get(future, timeout=timeout) if timeout else ray.get(future)

    # ------------------------------------------------------------------ #
    def execute_model(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput | None:
        if self._closed:
            raise RuntimeError("Executor closed")

        logger.info(f"Executing {len(requests)} request(s) via Ray actor")

        request_dicts = [
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

        future = self.worker_actor.execute_model.remote(request_dicts)
        try:
            result = ray.get(future, timeout=300)
            if isinstance(result, dict):
                return DiffusionOutput(
                    output=result.get("output"),
                    error=result.get("error"),
                    trajectory_latents=result.get("trajectory_latents"),
                    trajectory_timesteps=result.get("trajectory_timesteps"),
                )
            return result
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return DiffusionOutput(output=None, error=str(e))

    # ------------------------------------------------------------------ #
    def check_health(self):
        if self._closed:
            raise RuntimeError("Executor closed")
        try:
            ok = ray.get(self.worker_actor.check_health.remote(), timeout=10)
            if not ok:
                raise RuntimeError("Health check failed")
            logger.info("RayActorExecutor health OK")
        except Exception as e:
            raise RuntimeError(f"Health check failed: {e}")

    # ------------------------------------------------------------------ #
    def shutdown(self):
        if self._closed:
            return
        logger.info("Shutting down RayActorExecutor...")
        try:
            ray.get(self.worker_actor.shutdown.remote(), timeout=60)
            ray.kill(self.worker_actor)
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        logger.info("RayActorExecutor shutdown complete")