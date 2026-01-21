# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Iterable
from typing import Any

import PIL.Image
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    def __init__(self, od_config: OmniDiffusionConfig):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, requests: list[OmniDiffusionRequest]):
        try:
            # Apply pre-processing if available
            if self.pre_process_func is not None:
                preprocess_start_time = time.time()
                requests = self.pre_process_func(requests)
                preprocess_time = time.time() - preprocess_start_time
                logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

            output = self.add_req_and_wait_for_response(requests)
            if output.error:
                raise Exception(f"{output.error}")
            logger.info("Generation completed successfully.")

            if output.output is None:
                logger.warning("Output is None, returning empty OmniRequestOutput")
                # Return empty output for the first request
                if len(requests) > 0:
                    request = requests[0]
                    request_id = request.request_id or ""
                    prompt = request.prompt
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else None
                    return OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        diffusion_output=output,
                        images=[],
                        prompt=prompt,
                        metrics={},
                        latents=None,
                    )
                return None

            postprocess_start_time = time.time()
            images = self.post_process_func(output.output) if self.post_process_func is not None else output.output
            postprocess_time = time.time() - postprocess_start_time
            logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

            # Convert to OmniRequestOutput format
            # Ensure images is a list
            if not isinstance(images, list):
                images = [images] if images is not None else []

            # Handle single request or multiple requests
            if len(requests) == 1:
                # Single request: return single OmniRequestOutput
                request = requests[0]
                request_id = request.request_id or ""
                prompt = request.prompt
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else None

                metrics = {}
                if output.trajectory_timesteps is not None:
                    metrics["trajectory_timesteps"] = output.trajectory_timesteps

                return OmniRequestOutput.from_diffusion(
                    request_id=request_id,
                    diffusion_output=output,
                    images=images,
                    prompt=prompt,
                    metrics=metrics,
                    latents=output.trajectory_latents,
                )
            else:
                # Multiple requests: return list of OmniRequestOutput
                # Split images based on num_outputs_per_prompt for each request
                results = []
                image_idx = 0

                for request in requests:
                    request_id = request.request_id or ""
                    prompt = request.prompt
                    if isinstance(prompt, list):
                        prompt = prompt[0] if prompt else None

                    # Get images for this request
                    num_outputs = request.num_outputs_per_prompt
                    request_images = images[image_idx : image_idx + num_outputs] if image_idx < len(images) else []
                    image_idx += num_outputs

                    metrics = {}
                    if output.trajectory_timesteps is not None:
                        metrics["trajectory_timesteps"] = output.trajectory_timesteps

                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            diffusion_output=output,
                            images=request_images,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                        )
                    )

                return results
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    @staticmethod
    def make_engine(config: OmniDiffusionConfig) -> "DiffusionEngine":
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config)

    def add_req_and_wait_for_response(self, requests: list[OmniDiffusionRequest]):
        return self.executor.add_req(requests)

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        # DEBUGGING PURPOSES ONLY
        return None
        prompt = "dummy run"
        num_inference_steps = 1
        height = 1024
        width = 1024
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it

            dummy_image = PIL.Image.new("RGB", (width, height), color=(0, 0, 0))
        else:
            dummy_image = None
        req = OmniDiffusionRequest(
            prompt=prompt,
            height=height,
            width=width,
            pil_image=dummy_image,
            num_inference_steps=num_inference_steps,
            num_outputs_per_prompt=1,
        )
        logger.info("dummy run to warm up the model")
        requests = self.pre_process_func([req]) if self.pre_process_func is not None else [req]
        self.add_req_and_wait_for_response(requests)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        Args:
            method: The method name (str) to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        assert isinstance(method, str), "Only string method names are supported for now"
        return self.executor.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )

    def close(self) -> None:
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        # TODO implement it
        logger.warning("DiffusionEngine abort is not implemented yet")
        pass
