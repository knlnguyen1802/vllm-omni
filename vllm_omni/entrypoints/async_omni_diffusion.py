# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Async entrypoint for vLLM-Omni diffusion model inference.

Provides an asynchronous interface for running diffusion models,
enabling concurrent request handling and streaming generation.
"""

import asyncio
import uuid
import weakref
from collections.abc import AsyncGenerator, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def _weak_close_async_omni_diffusion(engine: DiffusionEngine, executor: ThreadPoolExecutor) -> None:
    """Best-effort diffusion cleanup for GC finalization."""
    try:
        engine.close()
    except Exception:
        pass
    try:
        executor.shutdown(wait=False)
    except Exception:
        pass


class AsyncOmniDiffusion:
    """Async entry point for vLLM-Omni diffusion model inference.

    This class provides an asynchronous interface for running diffusion models,
    enabling concurrent request handling. It wraps the DiffusionEngine and
    provides async methods for image generation.

    Args:
        model: Model name or path to load
        od_config: Optional OmniDiffusionConfig. If not provided, it will be
            created from kwargs
        **kwargs: Additional keyword arguments passed to OmniDiffusionConfig

    Example:
        >>> async_diffusion = AsyncOmniDiffusion(model="Qwen/Qwen-Image")
        >>> result = await async_diffusion.generate(
        ...     prompt="A beautiful sunset over the ocean",
        ...     request_id="req-1",
        ... )
        >>> print(result.images)
    """

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig | None = None,
        batch_size: int = 1,
        **kwargs: Any,
    ):
        self.model = model

        # Set batch size (default 1 for backward compatibility)
        self._batch_size = max(1, batch_size)

        # Capture stage info from kwargs before they might be filtered out
        stage_id = kwargs.get("stage_id")
        engine_input_source = kwargs.get("engine_input_source")
        cfg_kv_collect_func = kwargs.pop("cfg_kv_collect_func", None)

        # Build config
        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(model=model, **kwargs)
        elif isinstance(od_config, dict):
            # If config is dict, check it too (priority to kwargs if both exist)
            if stage_id is None:
                stage_id = od_config.get("stage_id")
            if engine_input_source is None:
                engine_input_source = od_config.get("engine_input_source")
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        # Inject stage info into omni_kv_config if present
        if stage_id is not None:
            self.od_config.omni_kv_config.setdefault("stage_id", stage_id)
        if engine_input_source is not None:
            self.od_config.omni_kv_config.setdefault("engine_input_source", engine_input_source)

        # Diffusers-style models expose `model_index.json` with `_class_name`.
        # Non-diffusers models (e.g. Bagel, NextStep) only have `config.json`,
        # so we fall back to reading that and mapping model_type manually.
        try:
            config_dict = get_hf_file_to_dict("model_index.json", od_config.model)
            if config_dict is not None:
                if od_config.model_class_name is None:
                    od_config.model_class_name = config_dict.get("_class_name", None)
                od_config.update_multimodal_support()

                tf_config_dict = get_hf_file_to_dict("transformer/config.json", od_config.model)
                od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
            else:
                raise FileNotFoundError("model_index.json not found")
        except (AttributeError, OSError, ValueError, FileNotFoundError):
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            if cfg is None:
                raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

            od_config.tf_model_config = TransformerConfig.from_dict(cfg)
            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []
            # Bagel/NextStep models don't have a model_index.json, so we set the pipeline class name manually
            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                od_config.model_class_name = "BagelPipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            elif model_type == "nextstep":
                if od_config.model_class_name is None:
                    od_config.model_class_name = "NextStep11Pipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            elif architectures and len(architectures) == 1:
                od_config.model_class_name = architectures[0]
            else:
                raise

        if cfg_kv_collect_func is not None:
            od_config.cfg_kv_collect_func = cfg_kv_collect_func

        # Initialize engine
        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)

        # Thread pool for running sync engine in async context
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False
        self._weak_finalizer = weakref.finalize(
            self,
            _weak_close_async_omni_diffusion,
            self.engine,
            self._executor,
        )

        # Batching infrastructure
        self._batch_timeout: float = 0.01  # seconds to wait for more requests
        self._batch_queue: (
            asyncio.Queue[
                tuple[
                    OmniPromptType,
                    OmniDiffusionSamplingParams,
                    str,
                    LoRARequest | None,
                    asyncio.Future[OmniRequestOutput],
                ]
            ]
            | None
        ) = None
        self._batch_worker_task: asyncio.Task | None = None

        logger.info("AsyncOmniDiffusion initialized with model: %s, batch_size: %d", model, self._batch_size)

    # ------------------------------------------------------------------
    # batch_size / batch_timeout properties
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """Return the configured batch size for request batching."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if not isinstance(value, int) or value < 1:
            raise ValueError("batch_size must be a positive integer")
        self._batch_size = value

    @property
    def batch_timeout(self) -> float:
        """Seconds the batch worker waits for additional requests before
        dispatching an incomplete batch.  Default 0.01 s."""
        return self._batch_timeout

    @batch_timeout.setter
    def batch_timeout(self, value: float) -> None:
        if value < 0:
            raise ValueError("batch_timeout must be non-negative")
        self._batch_timeout = value

    # ------------------------------------------------------------------
    # Batch worker – collects up to batch_size requests then dispatches
    # ------------------------------------------------------------------

    def _ensure_batch_worker(self) -> None:
        """Lazily start the background batch-worker task.

        Called on the first ``generate()`` invocation when *batch_size > 1*.
        The worker is created lazily so that the event loop is guaranteed to
        exist (``__init__`` may run outside an async context).
        """
        if self._batch_worker_task is None or self._batch_worker_task.done():
            self._batch_queue = asyncio.Queue()
            self._batch_worker_task = asyncio.create_task(self._batch_worker_loop())

    async def _batch_worker_loop(self) -> None:
        """Background coroutine that drains the batch queue.

        1. Blocks until at least one request arrives.
        2. Tries to collect up to ``batch_size`` requests within
           ``batch_timeout`` seconds.
        3. Sends the collected batch to ``_generate_batch``.
        4. Resolves each caller's Future with its result (or exception).
        """
        assert self._batch_queue is not None
        while not self._closed:
            batch: list[
                tuple[
                    OmniPromptType,
                    OmniDiffusionSamplingParams,
                    str,
                    LoRARequest | None,
                    asyncio.Future[OmniRequestOutput],
                ]
            ] = []

            # --- wait for the first item (blocking) ---
            try:
                first = await self._batch_queue.get()
            except asyncio.CancelledError:
                return
            batch.append(first)

            # --- greedily collect more items up to batch_size ---
            for _ in range(self._batch_size - 1):
                try:
                    item = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=self._batch_timeout,
                    )
                    batch.append(item)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    break

            # Unpack batch
            prompts = [b[0] for b in batch]
            # Use the first request's sampling_params (shared across batch)
            sampling_params = batch[0][1]
            request_ids = [b[2] for b in batch]
            lora_requests = [b[3] for b in batch]
            futures = [b[4] for b in batch]

            # Pick the first non-None LoRA request (if any)
            lora_request = next((lr for lr in lora_requests if lr is not None), None)

            logger.debug(
                "Batch worker dispatching %d/%d requests: %s",
                len(batch),
                self._batch_size,
                request_ids,
            )

            try:
                results = await self._generate_batch(
                    prompts,
                    sampling_params,
                    request_ids[0],
                    lora_request,
                )
                # The combined result contains all images; each caller gets the same output
                for fut in futures:
                    if not fut.done():
                        fut.set_result(results)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

    # ------------------------------------------------------------------
    # Public batch generation API
    # ------------------------------------------------------------------

    async def generate_batch(
        self,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        request_id: str | None = None,
        lora_request: LoRARequest | None = None,
    ) -> OmniRequestOutput:
        """Generate images from multiple prompts in a single engine call.

        Unlike the queue-based batching used by ``generate()`` with
        ``batch_size > 1``, this method explicitly batches the given
        prompts into **one** ``DiffusionEngine.step()`` call and returns
        a single ``OmniRequestOutput`` containing all generated images.

        Args:
            prompts: List of text prompts describing the desired images.
            sampling_params: Shared sampling parameters for all prompts.
            request_id: Optional unique identifier. Auto-generated when *None*.
            lora_request: Optional LoRA adapter to apply.

        Returns:
            A single ``OmniRequestOutput`` with all images combined.
        """
        if request_id is None:
            request_id = f"diff-batch-{uuid.uuid4().hex[:8]}"
        return await self._generate_batch(prompts, sampling_params, request_id, lora_request)

    # ------------------------------------------------------------------
    # Internal batch generation
    # ------------------------------------------------------------------

    async def _generate_batch(
        self,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
    ) -> OmniRequestOutput:
        """Generate images from multiple prompts in a single engine call."""
        if not prompts:
            return OmniRequestOutput(request_id=request_id, images=[], final_output_type="image")

        if sampling_params.guidance_scale:
            sampling_params.guidance_scale_provided = True

        if lora_request is not None:
            sampling_params.lora_request = lora_request

        request = OmniDiffusionRequest(
            prompts=prompts,
            sampling_params=sampling_params,
            request_ids=[f"{request_id}-{i}" for i in range(len(prompts))],
        )

        logger.debug("Starting batch generation for %d prompts, request_id=%s", len(prompts), request_id)

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                self._executor,
                self.engine.step,
                request,
            )
        except Exception as e:
            logger.error("Batch generation failed for request %s: %s", request_id, e)
            raise RuntimeError(f"Diffusion batch generation failed: {e}") from e

        # Combine all per-prompt results into a single OmniRequestOutput
        all_images = []
        for result in results:
            all_images.extend(result.images)

        return OmniRequestOutput(
            request_id=request_id,
            images=all_images,
            final_output_type="image",
            finished=True,
        )

    # ------------------------------------------------------------------
    # Public generate API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        request_id: str | None = None,
        lora_request: LoRARequest | None = None,
    ) -> OmniRequestOutput:
        """Generate images asynchronously from a text prompt.

        When ``batch_size > 1`` the request is placed into an internal queue
        and the background batch-worker will group up to ``batch_size``
        concurrent requests into a single ``engine.step`` call.  The caller
        transparently awaits its individual result.

        When ``batch_size == 1`` (default) the request is executed directly
        with zero queuing overhead – fully backward-compatible.

        Args:
            prompt: Text prompt describing the desired image
            sampling_params: Sampling parameters
            request_id: Optional unique identifier for tracking the request
            lora_request: Optional LoRA adapter to apply

        Returns:
            OmniRequestOutput containing generated images

        Raises:
            RuntimeError: If generation fails
        """
        if request_id is None:
            request_id = f"diff-{uuid.uuid4().hex[:16]}"

        # ----- batched path (batch_size > 1) -----
        if self._batch_size > 1:
            self._ensure_batch_worker()
            loop = asyncio.get_event_loop()
            fut: asyncio.Future[OmniRequestOutput] = loop.create_future()
            await self._batch_queue.put(  # type: ignore[union-attr]
                (prompt, sampling_params, request_id, lora_request, fut),
            )
            return await fut

        # ----- direct path (batch_size == 1, no queuing overhead) -----
        if sampling_params.guidance_scale:
            sampling_params.guidance_scale_provided = True

        if lora_request is not None:
            sampling_params.lora_request = lora_request

        request = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_ids=[request_id],
        )

        logger.debug("Starting generation for request %s", request_id)

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                self.engine.step,
                request,
            )
            result = result[0]
        except Exception as e:
            logger.error("Generation failed for request %s: %s", request_id, e)
            raise RuntimeError(f"Diffusion generation failed: {e}") from e

        if not result.request_id:
            result.request_id = request_id
        return result

    async def generate_stream(
        self,
        prompt: str,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate images with streaming progress updates.

        Currently, diffusion models don't support true streaming, so this
        yields a single result after generation completes. Future implementations
        may support step-by-step progress updates.

        Args:
            prompt: Text prompt describing the desired image
            request_id: Optional unique identifier for tracking the request
            **kwargs: Additional generation parameters

        Yields:
            OmniRequestOutput with generation progress/results
        """
        result = await self.generate(prompt=prompt, request_id=request_id, **kwargs)
        yield result

    def close(self) -> None:
        """Close the engine and release resources.

        Should be called when done using the AsyncOmniDiffusion instance.
        """
        if self._closed:
            return
        self._closed = True

        # Cancel the batch worker if running
        if self._batch_worker_task is not None and not self._batch_worker_task.done():
            self._batch_worker_task.cancel()
            self._batch_worker_task = None

        # Drain any pending futures so callers don't hang
        if self._batch_queue is not None:
            while not self._batch_queue.empty():
                try:
                    _, _, _, _, fut = self._batch_queue.get_nowait()
                    if not fut.done():
                        fut.set_exception(RuntimeError("AsyncOmniDiffusion closed"))
                except Exception:
                    break

        finalizer = getattr(self, "_weak_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer.detach()

        try:
            self.engine.close()
        except Exception as e:
            logger.warning("Error closing diffusion engine: %s", e)

        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            logger.warning("Error shutting down executor: %s", e)

        logger.info("AsyncOmniDiffusion closed")

    def shutdown(self) -> None:
        """Alias for close() method."""
        self.close()

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request."""
        self.engine.abort(request_id)

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return not self._closed

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self._closed

    async def remove_lora(self, adapter_id: int) -> bool:
        """Remove a LoRA"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.engine.collective_rpc,
            "remove_lora",
            None,
            (adapter_id,),
            {},
            None,
        )
        return all(results) if isinstance(results, list) else results

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add a LoRA adapter"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.engine.collective_rpc,
            "add_lora",
            None,
            (),
            {"lora_request": lora_request},
            None,
        )
        return all(results) if isinstance(results, list) else results

    async def list_loras(self) -> list[int]:
        """List all registered LoRA adapter IDs."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.engine.collective_rpc,
            "list_loras",
            None,
            (),
            {},
            None,
        )
        # collective_rpc returns list from workers; flatten unique ids
        if not isinstance(results, list):
            return results or []
        merged: set[int] = set()
        for part in results:
            merged.update(part or [])
        return sorted(merged)

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self.engine.collective_rpc,
            "pin_lora",
            None,
            (),
            {"adapter_id": lora_id},
            None,
        )
        return all(results) if isinstance(results, list) else results

    async def start_profile(self, trace_filename: str | None = None) -> None:
        """Start profiling for the diffusion model.

        Args:
            trace_filename: Optional base filename for trace files.
                           If None, a timestamp-based name will be generated.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self.engine.start_profile,
            trace_filename,
        )

    async def stop_profile(self) -> dict:
        """Stop profiling and return profiling results.

        Returns:
            Dictionary containing paths to trace and table files.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.engine.stop_profile,
        )
