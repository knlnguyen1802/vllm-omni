"""
Stage manager for orchestrating multiple engines in vLLM-Omni.

Enhanced to encapsulate per-stage process lifecycle and worker logic
(device setup, LLM init, batching, shared-memory IPC), while preserving
the original input processing utilities for cross-stage data wiring.
"""

import asyncio
import fcntl
import importlib
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
from dataclasses import fields
from typing import Any

from setproctitle import setproctitle
from vllm.inputs import TextPrompt
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.distributed.omni_connectors import build_stage_connectors
from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector
from vllm_omni.distributed.ray_utils.utils import kill_ray_actor, start_ray_actor
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM
from vllm_omni.entrypoints.log_utils import count_tokens_from_outputs
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.entrypoints.omni_llm import OmniLLM
from vllm_omni.entrypoints.stage_utils import (
    SHUTDOWN_TASK,
    OmniStageTaskType,
    _to_dict,
    is_profiler_task,
    maybe_dump_to_shm,
    set_stage_devices,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.utils import detect_device_type

logger = init_logger(__name__)


class WorkerIOHandler:
    """Abstract interface for worker input/output communication.

    This abstraction allows easy swapping of communication backends
    (e.g., multiprocessing queues, ZMQ, RPC, etc.).
    """

    def recv_task(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Receive a task from the input channel.

        Args:
            timeout: Optional timeout in seconds. None means blocking.

        Returns:
            Task dictionary or None if no task available (non-blocking)
            or timeout occurred.
        """
        raise NotImplementedError

    def send_result(self, result: dict[str, Any]) -> None:
        """Send a result to the output channel.

        Args:
            result: Result dictionary to send.
        """
        raise NotImplementedError

    def send_task(self, task: dict[str, Any]) -> None:
        """Send a task to the output channel.

        Args:
            task: Task dictionary to send.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the communication channels."""
        pass


class MPQueueIOHandler(WorkerIOHandler):
    """Multiprocessing queue-based IO handler."""

    def __init__(self, in_q: mp.Queue, out_q: mp.Queue):
        self.in_q = in_q
        self.out_q = out_q

    def recv_task(self, timeout: float | None = None) -> dict[str, Any] | None:
        if timeout == 0:
            # Non-blocking
            try:
                return self.in_q.get_nowait()
            except queue.Empty:
                return None
        elif timeout is None:
            # Blocking
            return self.in_q.get()
        else:
            # Blocking with timeout
            try:
                return self.in_q.get(timeout=timeout)
            except queue.Empty:
                return None

    def send_result(self, result: dict[str, Any]) -> None:
        self.out_q.put(result)

    def send_task(self, task: dict[str, Any]) -> None:
        self.out_q.put(task)


class AsyncMPQueueIOHandler:
    """Async multiprocessing queue-based IO handler."""

    def __init__(self, in_q: mp.Queue, out_q: mp.Queue):
        self.in_q = in_q
        self.out_q = out_q

    async def recv_task(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Async receive task from queue.

        Args:
            timeout: Optional timeout. 0 for non-blocking, None for blocking.

        Returns:
            Task or None if not available.
        """
        if timeout == 0:
            # Non-blocking
            try:
                return self.in_q.get_nowait()
            except queue.Empty:
                return None
        else:
            # Blocking - use asyncio sleep to yield control
            while True:
                try:
                    return self.in_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.001)

    def send_result(self, result: dict[str, Any]) -> None:
        """Send result to output queue (sync operation)."""
        self.out_q.put(result)

    def send_task(self, task: dict[str, Any]) -> None:
        """Send task to output queue (sync operation)."""
        self.out_q.put(task)


def _build_od_config(engine_args: dict[str, Any], model: str) -> dict[str, Any]:
    """Build OmniDiffusionConfig kwargs from engine args."""
    od_config = engine_args.get("od_config", {})
    if not od_config:
        od_config = {"model": model}
        od_field_names = {f.name for f in fields(OmniDiffusionConfig)}
        for key, value in engine_args.items():
            if key in od_field_names:
                od_config[key] = value
    return od_config


def prepare_sampling_params(sampling_params: Any, stage_type: str) -> Any:
    """Prepare sampling parameters for the given stage type.

    Args:
        sampling_params: Raw sampling parameters (dict or SamplingParams)
        stage_type: Either "llm" or "diffusion"

    Returns:
        Processed sampling parameters ready for engine consumption
    """
    if stage_type == "diffusion":
        # For diffusion stages: extract kwargs, handling different input types
        if isinstance(sampling_params, dict):
            diffusion_kwargs = dict(sampling_params)
        else:
            diffusion_kwargs = getattr(sampling_params, "__dict__", {}) or {}

        # Remove 'prompt' and 'request_id' to avoid conflict with explicit arguments
        diffusion_kwargs.pop("prompt", None)
        diffusion_kwargs.pop("request_id", None)
        return diffusion_kwargs

    else:  # stage_type == "llm"
        # For LLM stages: ensure we have a SamplingParams object
        if isinstance(sampling_params, dict):
            return SamplingParams(**sampling_params)
        return sampling_params


class OmniStage:
    """Stage manager for orchestrating a single stage in the omni pipeline.

    Encapsulates per-stage process lifecycle and worker logic, including
    device setup, LLM initialization, batching, and shared-memory IPC.
    Preserves input processing utilities for cross-stage data wiring.

    Args:
        stage_config: Stage configuration object containing engine arguments,
            runtime settings, and stage-specific parameters
    """

    def __init__(self, stage_config: Any, stage_init_timeout: int = 300):
        logger.info(f"[OmniStage] stage_config: {stage_config}")
        self.stage_config = stage_config
        self.engine = None
        self.async_engine = None
        self.vllm_config = None
        self.tokenizer = None
        self.input_preprocessor = None
        self.is_tracing_enabled = False
        self.stage_id = stage_config.stage_id
        self.engine_args = stage_config.engine_args
        self.model_stage = stage_config.engine_args.model_stage
        self.requires_multimodal_data = getattr(stage_config.runtime, "requires_multimodal_data", False)
        self.engine_input_source = getattr(stage_config, "engine_input_source", [])
        self.engine_output_type = getattr(stage_config.engine_args, "engine_output_type", None)
        self.engine_outputs = None
        self.is_comprehension = getattr(stage_config, "is_comprehension", False)
        # Support for different stage types: "llm" (default) or "diffusion"
        self.stage_type = getattr(stage_config, "stage_type", "llm")
        if hasattr(stage_config, "custom_process_input_func"):
            # Import the module specified in the config (already a full module path)
            module_path, func_name = stage_config.custom_process_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_input_func = getattr(module, func_name)
        else:
            self.custom_process_input_func = None

        self.final_output = getattr(stage_config, "final_output", False)
        self.final_output_type = getattr(stage_config, "final_output_type", None)
        default_sampling_params = getattr(stage_config, "default_sampling_params", {})
        # For LLM stage, this can directly be a SamplingParams-compatible dict;
        # For diffusion stage, this only serves as default values for diffusion kwargs.
        self.default_sampling_params = _to_dict(default_sampling_params)
        # Runtime orchestration state (added)
        self._in_q: mp.Queue | None = None
        self._out_q: mp.Queue | None = None
        self._proc: mp.Process | None = None
        self._shm_threshold_bytes: int = 65536
        self._stage_init_timeout: int = stage_init_timeout

    def set_engine(self, engine: LLMEngine) -> None:
        """Set the LLM engine for this stage.

        Args:
            engine: LLMEngine instance to use for this stage
        """
        self.engine = engine

    def set_async_engine(self, async_engine: AsyncLLM) -> None:
        """Set the async LLM engine for this stage.

        Args:
            async_engine: AsyncLLM instance to use for this stage
        """
        self.async_engine = async_engine

    def set_vllm_config(self, vllm_config: Any) -> None:
        """Set the vLLM configuration for this stage.

        Args:
            vllm_config: VllmConfig instance received from worker process
        """
        self.vllm_config = vllm_config

    def set_tokenizer(self, tokenizer: TokenizerLike) -> None:
        """Set the tokenizer for this stage.

        Args:
            tokenizer: Tokenizer instance received from worker process
        """
        self.tokenizer = tokenizer

    def set_input_preprocessor(self, input_preprocessor: InputPreprocessor) -> None:
        """Set the input preprocessor for this stage.

        Args:
            input_preprocessor: InputPreprocessor instance received from worker process
        """
        self.input_preprocessor = input_preprocessor

    def set_is_tracing_enabled(self, is_tracing_enabled: bool) -> None:
        """Set whether tracing is enabled for this stage.

        Args:
            is_tracing_enabled: Boolean indicating if tracing is enabled
        """
        self.is_tracing_enabled = is_tracing_enabled

    def set_engine_outputs(self, engine_outputs: EngineCoreOutput) -> None:
        """Set the engine outputs for this stage.

        Args:
            engine_outputs: EngineCoreOutput from this stage's processing
        """
        self.engine_outputs = engine_outputs

    # ----------------- New Orchestration APIs -----------------
    def attach_queues(self, in_q: mp.Queue, out_q: mp.Queue) -> None:
        """Attach input and output queues for IPC communication.

        Args:
            in_q: Input queue for receiving tasks from orchestrator
            out_q: Output queue for sending results to orchestrator
        """
        self._in_q = in_q
        self._out_q = out_q
        # For orchestrator: recv_task reads from out_q (results), send_task writes to in_q
        self._io_handler = MPQueueIOHandler(out_q, in_q)

    def init_stage_worker(
        self,
        model: str,
        *,
        is_async: bool = False,
        shm_threshold_bytes: int = 65536,
        ctx: mp.context.BaseContext | None = None,
        batch_timeout: int = 10,
        connectors_config: dict | None = None,
        worker_backend: str = "multi_process",
        **kwargs: Any,
    ) -> None:
        """Initialize and start the stage worker process.

        Creates a worker process that runs the LLM engine for this stage.
        The worker handles batching, generation, and IPC communication.

        Args:
            model: Model name or path to load
            is_async: Whether to use async engine (default: False)
            shm_threshold_bytes: Threshold for using shared memory for IPC
            ctx: Optional multiprocessing context (default: spawn)
            batch_timeout: Timeout in seconds for batching requests
            connectors_config: Configuration for stage connectors
            worker_backend: Backend type ("multi_process" or "ray")
            **kwargs: Additional arguments (e.g. ray_placement_group)

        Raises:
            AssertionError: If queues are not attached before calling this method
        """
        assert self._in_q is not None and self._out_q is not None, "Queues must be attached before start_process"

        if worker_backend == "ray":
            ray_placement_group = kwargs.get("ray_placement_group", None)
            assert ray_placement_group is not None, "Ray placement group must be provided"
            self._shm_threshold_bytes = sys.maxsize
        else:
            self._shm_threshold_bytes = shm_threshold_bytes

        ctx = ctx or mp.get_context("spawn")
        # Prepare lightweight dict config for worker
        engine_args = _to_dict(self.engine_args)
        runtime_cfg = _to_dict(getattr(self.stage_config, "runtime", {}))
        stage_payload: dict[str, Any] = {
            "stage_id": self.stage_id,
            "engine_args": engine_args,
            "runtime": runtime_cfg,
            "shm_threshold_bytes": self._shm_threshold_bytes,
            "connectors_config": connectors_config or {},
            "stage_type": self.stage_type,
        }
        try:
            old_env = os.environ.get("VLLM_LOGGING_PREFIX")
            new_env = f"[Stage-{self.stage_id}] {'' if old_env is None else old_env}"
            os.environ["VLLM_LOGGING_PREFIX"] = new_env
            if worker_backend == "ray":
                if is_async:
                    self._ray_actor = start_ray_actor(
                        _stage_worker_async_entry,
                        ray_placement_group,
                        self.stage_id,
                        self,
                        model=model,
                        stage_payload=stage_payload,
                        batch_timeout=batch_timeout,
                        stage_init_timeout=self._stage_init_timeout,
                    )
                else:
                    self._ray_actor = start_ray_actor(
                        _stage_worker,
                        ray_placement_group,
                        self.stage_id,
                        model=model,
                        stage_payload=stage_payload,
                        in_q=self._in_q,
                        out_q=self._out_q,
                        batch_timeout=batch_timeout,
                        stage_init_timeout=self._stage_init_timeout,
                    )
            else:
                if is_async:
                    self._proc = ctx.Process(
                        target=_stage_worker_async_entry,
                        args=(
                            self,
                            model,
                            stage_payload,
                            batch_timeout,
                            self._stage_init_timeout,
                        ),
                    )
                else:
                    self._proc = ctx.Process(
                        target=_stage_worker,
                        args=(
                            model,
                            stage_payload,
                            self._in_q,
                            self._out_q,
                            batch_timeout,
                            self._stage_init_timeout,
                        ),
                    )
                self._proc.start()
        finally:
            if old_env is None:
                os.environ.pop("VLLM_LOGGING_PREFIX", None)
            else:
                os.environ["VLLM_LOGGING_PREFIX"] = old_env

    def stop_stage_worker(self) -> None:
        """Stop the stage worker process gracefully.

        Sends shutdown signal to the worker and waits for it to terminate.
        If graceful shutdown fails, forcefully terminates the process.
        Handles both multiprocessing Process and Ray Actor.
        """
        if self._in_q is not None:
            try:
                self._in_q.put_nowait(SHUTDOWN_TASK)
            except Exception as e:
                logger.warning("Failed to send shutdown to in_q: %s", e)

        if hasattr(self, "_ray_actor") and self._ray_actor:
            kill_ray_actor(self._ray_actor)
            self._ray_actor = None
        elif self._proc is not None:
            try:
                self._proc.join(timeout=5)
            except Exception as e:
                logger.debug("join() failed: %s", e)
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception as e:
                    logger.warning("terminate() failed: %s", e)

    def submit(self, payload: dict[str, Any]) -> None:
        """Submit a task to the stage worker.

        Args:
            payload: Dictionary containing task data (request_id, engine_inputs,
                sampling_params, etc.)
        """
        assert self._io_handler is not None
        self._io_handler.send_task(payload)

    def try_collect(self) -> dict[str, Any] | None:
        """Try to collect a result from the stage worker without blocking.

        Returns:
            Result dictionary if available, None otherwise. Result contains
            request_id, engine_outputs (or engine_outputs_shm), and metrics.
        """
        assert self._io_handler is not None
        return self._io_handler.recv_task(timeout=0)

    def process_engine_inputs(
        self, stage_list: list[Any], prompt: OmniTokensPrompt | TextPrompt = None
    ) -> list[OmniTokensPrompt | TextPrompt]:
        """Process engine inputs for this stage from upstream stage outputs.

        Derives inputs for this stage from outputs of upstream stages.
        Uses engine_input_source configuration to determine which upstream
        stage outputs to use. Supports custom processing functions.

        Args:
            stage_list: List of all stages in the pipeline
            prompt: Optional original prompt (for multimodal data preservation)

        Returns:
            List of processed engine inputs ready for this stage

        Raises:
            ValueError: If engine_input_source is empty or invalid
        """
        if self.custom_process_input_func is None:
            engine_inputs = []
            if len(self.engine_input_source) == 0:
                raise ValueError("engine_input_source is empty")
            source_stage_id = self.engine_input_source[0]
            source_outputs = stage_list[source_stage_id].engine_outputs
            if not isinstance(prompt, list):
                prompt = [prompt]
            multi_modal_data = {
                source_output.request_id: p.get("multi_modal_data", None)
                for source_output, p in zip(source_outputs, prompt)
            }

            for source_output in source_outputs:
                engine_input = OmniTokensPrompt(
                    prompt_token_ids=source_output.outputs[0].token_ids,
                    multi_modal_data=(
                        multi_modal_data[source_output.request_id]
                        if self.requires_multimodal_data and multi_modal_data
                        else None
                    ),
                )
                engine_inputs.append(engine_input)
            return engine_inputs

        else:
            engine_input_source = self.engine_input_source
            return self.custom_process_input_func(
                stage_list, engine_input_source, prompt, self.requires_multimodal_data
            )


def _get_method_name_from_task_type(task_type: OmniStageTaskType) -> str:
    """Map task type to method name."""
    mapping = {
        OmniStageTaskType.PROFILER_START: "start_profile",
        OmniStageTaskType.PROFILER_STOP: "stop_profile",
        OmniStageTaskType.SHUTDOWN: "shutdown",
        OmniStageTaskType.GENERATE: "generate",
    }
    return mapping.get(task_type, "generate")


class OmniStageWorkerProc:
    """Worker process for omni stage orchestration."""

    def __init__(
        self,
        model: str,
        stage_payload: dict[str, Any],
        in_q: mp.Queue,
        out_q: mp.Queue,
        batch_timeout: int = 10,
        stage_init_timeout: int = 300,
    ):
        setproctitle("OmniStageWorkerProc")
        self.model = model
        self.stage_id = stage_payload["stage_id"]
        self.engine_args = stage_payload.get("engine_args", {})
        self.runtime_cfg = stage_payload.get("runtime", {})
        self.shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))
        self.connectors_config = stage_payload.get("connectors_config", {})
        self.stage_type = stage_payload.get("stage_type", "llm")
        self.io_handler = MPQueueIOHandler(in_q, out_q)
        self.batch_timeout = batch_timeout
        self.stage_init_timeout = stage_init_timeout

        # Aggregates for running average
        self.agg_total_tokens = 0
        self.agg_total_gen_time_ms = 0.0
        self.batch_seq = 0

        # Device setup
        self._setup_devices()

        # Sequential initialization with device locking
        self._acquire_device_locks()

        try:
            # Initialize engine_core based on stage type
            self.engine_core = self._create_engine_core()
        finally:
            self._release_device_locks()

        # Initialize connectors
        self.connectors = self._init_connectors()

        # Signal readiness
        self._signal_ready()

        # Get max batch size
        self.max_batch_size = int(self.runtime_cfg.get("max_batch_size", 1) or 1)
        logger.info(f"Max batch size: {self.max_batch_size}")

    def _setup_devices(self) -> None:
        """Setup device configuration."""
        self.device_type = None
        try:
            self.device_type = detect_device_type()
            set_stage_devices(
                self.stage_id,
                self.runtime_cfg.get("devices"),
                device_type=self.device_type,
            )
        except Exception as e:
            logger.warning("Device setup failed: %s", e)

    def _acquire_device_locks(self) -> None:
        """Acquire exclusive locks for devices during initialization."""
        self.lock_files = []
        if self.device_type != "cuda":
            return

        try:
            import torch

            if not torch.cuda.is_available():
                return

            # Get parallel sizes from engine_args
            if "parallel_config" in self.engine_args:
                parallel_config = self.engine_args["parallel_config"]
                tensor_parallel_size = parallel_config.get("tensor_parallel_size", 1)
                pipeline_parallel_size = parallel_config.get("pipeline_parallel_size", 1)
                data_parallel_size = parallel_config.get("data_parallel_size", 1)
                prefill_context_parallel_size = 1
                sequence_parallel_size = parallel_config.get("sequence_parallel_size", 1)
            else:
                tensor_parallel_size = self.engine_args.get("tensor_parallel_size", 1)
                pipeline_parallel_size = self.engine_args.get("pipeline_parallel_size", 1)
                data_parallel_size = self.engine_args.get("data_parallel_size", 1)
                prefill_context_parallel_size = self.engine_args.get("prefill_context_parallel_size", 1)
                sequence_parallel_size = 1

            num_devices_per_stage = (
                tensor_parallel_size
                * pipeline_parallel_size
                * data_parallel_size
                * prefill_context_parallel_size
                * sequence_parallel_size
            )

            # Get physical device IDs
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            physical_devices = []

            if cuda_visible_devices:
                try:
                    physical_devices = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
                except (ValueError, IndexError):
                    pass

            if not physical_devices:
                num_devices = torch.cuda.device_count()
                physical_devices = list(range(num_devices))

            num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
            devices_to_lock = sorted(physical_devices[:num_devices_to_lock])

            logger.debug(
                "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d; will lock %d devices: %s",
                tensor_parallel_size,
                pipeline_parallel_size,
                data_parallel_size,
                prefill_context_parallel_size,
                sequence_parallel_size,
                num_devices_to_lock,
                devices_to_lock,
            )

            wait_start = time.time()
            acquired_lock_fds = []

            for device_id in devices_to_lock:
                lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
                lock_acquired = False

                while not lock_acquired:
                    try:
                        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            os.ftruncate(lock_fd, 0)
                            os.write(lock_fd, f"{os.getpid()}\n".encode())
                            os.fsync(lock_fd)
                            lock_acquired = True
                            acquired_lock_fds.append(lock_fd)
                            logger.debug("Acquired exclusive lock for device %s", device_id)
                        except BlockingIOError:
                            os.close(lock_fd)
                            if time.time() - wait_start > self.stage_init_timeout:
                                logger.warning(
                                    "Timeout waiting for device %s initialization lock, proceeding anyway",
                                    device_id,
                                )
                                break
                            time.sleep(0.1)
                    except OSError as e:
                        logger.debug(
                            "Failed to acquire lock for device %s: %s, continuing anyway",
                            device_id,
                            e,
                        )
                        try:
                            os.close(lock_fd)
                        except (OSError, NameError):
                            pass
                        break

            self.lock_files = acquired_lock_fds
        except Exception as e:
            logger.debug("[Stage-%s] Failed to set up sequential initialization lock: %s", self.stage_id, e)

    def _release_device_locks(self) -> None:
        """Release all acquired device locks."""
        for lock_fd in self.lock_files:
            try:
                os.close(lock_fd)
                logger.debug("Released initialization lock (fd=%s)", lock_fd)
            except (OSError, ValueError):
                pass
        self.lock_files = []

    def _create_engine_core(self) -> Any:
        """Create appropriate engine based on stage type."""
        logger.debug(
            "[Stage-%s] Initializing %s engine with args keys=%s",
            self.stage_id,
            self.stage_type,
            list(self.engine_args.keys()),
        )

        if self.stage_type == "diffusion":
            engine_args = {k: v for k, v in self.engine_args.items() if k != "model_stage"}
            return OmniDiffusion(**engine_args)
        else:
            return OmniLLM(model=self.model, **self.engine_args)

    def _init_connectors(self) -> dict:
        """Initialize OmniConnectors if configured."""
        connectors = {}
        if self.connectors_config:
            built_connectors = build_stage_connectors(
                stage_id=self.stage_id,
                connectors_config=self.connectors_config,
            )
            if built_connectors is not None:
                connectors = built_connectors
        return connectors

    def _signal_ready(self) -> None:
        """Signal readiness to orchestrator."""
        try:
            self.io_handler.send_result({"type": "stage_ready", "stage_id": self.stage_id})
        except Exception:
            pass

    def start_profile(self) -> None:
        """Start profiling."""
        if hasattr(self.engine_core, "start_profile"):
            try:
                self.engine_core.start_profile()
                logger.info("[Stage-%s] Profiler started via engine", self.stage_id)
            except Exception as e:
                logger.warning("[Stage-%s] Failed to start profiler: %s", self.stage_id, e)

    def stop_profile(self) -> None:
        """Stop profiling."""
        if hasattr(self.engine_core, "stop_profile"):
            try:
                self.engine_core.stop_profile()
                logger.info("[Stage-%s] Profiler stopped via engine", self.stage_id)
            except Exception as e:
                logger.warning("[Stage-%s] Failed to stop profiler: %s", self.stage_id, e)

    def shutdown(self) -> None:
        """Shutdown worker."""
        logger.info("[Stage-%s] Worker shutting down", self.stage_id)
        # Any cleanup logic can go here
        return "SHUTDOWN"

    def generate(self, task: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a batch of generation requests.

        This method collects tasks from the queue, prepares inputs,
        executes through the engine, and returns results.

        Args:
            task: Optional initial task. If not provided, will dequeue from queue.

        Returns:
            List of result dictionaries to be sent via IO handler.
        """
        # Collect batch of tasks
        if task is None:
            task = self.io_handler.recv_task()
            if task is None:
                return []
        batch_tasks: list[dict[str, Any]] = [task]

        if self.max_batch_size > 1:
            start_time = time.time()
            while len(batch_tasks) < self.max_batch_size:
                extra = self.io_handler.recv_task(timeout=0)
                if extra is not None:
                    if extra == SHUTDOWN_TASK:
                        self.io_handler.send_result(SHUTDOWN_TASK)
                        break
                    # Skip profiler tasks during batching
                    extra_type = extra.get("type") if isinstance(extra, dict) else None
                    if is_profiler_task(extra_type):
                        continue
                    batch_tasks.append(extra)
                    if time.time() - start_time > self.batch_timeout:
                        break
                else:
                    time.sleep(0.05)
                    if time.time() - start_time > self.batch_timeout:
                        break

        # Prepare batch inputs and collect metrics
        batch_request_ids: list[Any] = []
        batch_engine_inputs: list[Any] = []
        rx_bytes_by_rid: dict[Any, int] = {}
        rx_decode_ms_by_rid: dict[Any, float] = {}
        in_flight_ms_by_rid: dict[Any, float] = {}
        recv_dequeue_ts = time.time()

        for t in batch_tasks:
            rid = t["request_id"]
            try:
                sent_ts = float(t.get("sent_ts", None)) if isinstance(t, dict) else None
                if sent_ts is not None:
                    in_flight_ms_by_rid[rid] = (recv_dequeue_ts - sent_ts) * 1000.0
                else:
                    in_flight_ms_by_rid[rid] = 0.0
            except Exception:
                in_flight_ms_by_rid[rid] = 0.0

            ein, rx_metrics = try_recv_via_connector(
                task=t,
                connectors=self.connectors,
                stage_id=self.stage_id,
            )

            if ein is None or rx_metrics is None:
                raise RuntimeError(
                    f"[Stage-{self.stage_id}] Missing connector payload for request {rid}. "
                    "Ensure connectors are configured for all incoming edges."
                )

            rx_decode_ms_by_rid[rid] = float(rx_metrics.get("rx_decode_time_ms", 0.0))
            rx_bytes_by_rid[rid] = int(rx_metrics.get("rx_transfer_bytes", 0))

            batch_request_ids.append(rid)
            if isinstance(ein, list):
                batch_engine_inputs.extend(ein)
            elif isinstance(ein, dict):
                batch_engine_inputs.append(ein)
            elif isinstance(ein, str):
                batch_engine_inputs.append(ein)
            else:
                batch_engine_inputs.append(ein)

        sampling_params = batch_tasks[0]["sampling_params"]

        logger.debug(
            "Received batch size=%d, request_ids=%s",
            len(batch_request_ids),
            batch_request_ids,
        )

        # Execute batch through engine_core
        self.batch_seq += 1
        gen_t0 = time.time()

        # Execute based on stage type
        if self.stage_type == "diffusion":
            # Convert inputs to prompts for diffusion
            prompts = []
            for ein in batch_engine_inputs:
                if isinstance(ein, str):
                    prompts.append(ein)
                elif isinstance(ein, dict) and "prompt" in ein:
                    prompts.append(ein["prompt"])
                elif hasattr(ein, "prompt"):
                    prompts.append(ein.prompt)
                else:
                    prompts.append(str(ein))

            # Prepare diffusion kwargs
            diffusion_kwargs = prepare_sampling_params(sampling_params, "diffusion")
            diffusion_results = self.engine_core.generate(prompts, **diffusion_kwargs)

            # Convert to list format
            if isinstance(diffusion_results, list):
                gen_outputs = diffusion_results
            else:
                gen_outputs = [diffusion_results]
        else:
            # LLM engine
            llm_sampling_params = prepare_sampling_params(sampling_params, "llm")
            gen_outputs: list[Any] = []
            for ro in self.engine_core.generate(batch_engine_inputs, llm_sampling_params, use_tqdm=False):
                gen_outputs.append(ro)

        gen_t1 = time.time()
        gen_ms = (gen_t1 - gen_t0) * 1000.0

        logger.debug(f"Generate done: batch={len(batch_request_ids)}, req_ids={batch_request_ids}, gen_ms={gen_ms:.1f}")

        # Group outputs per request id
        req_to_outputs: dict[Any, list[Any]] = {rid: [] for rid in batch_request_ids}
        unmapped: list[Any] = []

        for ro in gen_outputs:
            rid = getattr(ro, "request_id", None)
            if rid in req_to_outputs:
                req_to_outputs[rid].append(ro)
            else:
                unmapped.append(ro)

        # Map unmapped outputs
        if unmapped:
            for idx, ro in enumerate(unmapped):
                target_rid = batch_request_ids[idx % len(batch_request_ids)]
                ro.request_id = target_rid
                req_to_outputs[target_rid].append(ro)

        self.agg_total_gen_time_ms += gen_ms

        # Prepare per-request results
        results = []
        for i, rid in enumerate(batch_request_ids):
            r_outputs = req_to_outputs.get(rid, [])
            metrics = make_request_stats(
                r_outputs,
                gen_ms,
                int(self.batch_seq),
                int(len(batch_request_ids)),
                float(rx_decode_ms_by_rid.get(rid, 0.0)),
                int(rx_bytes_by_rid.get(rid, 0)),
                float(in_flight_ms_by_rid.get(rid, 0.0)),
            )
            self.agg_total_tokens += metrics.num_tokens_out

            if i == len(batch_request_ids) - 1:
                metrics.stage_stats = make_stage_stats(self.agg_total_tokens, self.agg_total_gen_time_ms)
            else:
                metrics.stage_stats = None

            try:
                use_shm, payload = maybe_dump_to_shm(r_outputs, self.shm_threshold_bytes)
                if use_shm:
                    result = {
                        "request_id": rid,
                        "stage_id": self.stage_id,
                        "engine_outputs_shm": payload,
                        "metrics": metrics,
                    }
                else:
                    result = {
                        "request_id": rid,
                        "stage_id": self.stage_id,
                        "engine_outputs": payload,
                        "metrics": metrics,
                    }
            except Exception:
                result = {
                    "request_id": rid,
                    "stage_id": self.stage_id,
                    "engine_outputs": r_outputs,
                    "metrics": metrics,
                }

            results.append(result)
            logger.debug("Prepared result for request %s", rid)

        return results

    def handle_input(self, task: dict[str, Any]) -> tuple[Any, tuple, dict]:
        """Handle input task: prepare method call for execution.

        Args:
            task: Task dictionary containing type, method, args, kwargs, etc.

        Returns:
            Tuple of (func, args, kwargs) ready for execution.
        """
        # Get task type and method name
        task_type = task.get("type", OmniStageTaskType.GENERATE)
        method_name = task.get("method", None)

        # If method name not provided, infer from task type
        if method_name is None:
            method_name = _get_method_name_from_task_type(task_type)

        assert isinstance(method_name, str)
        func = getattr(self, method_name)

        # Get args and kwargs from task
        args = task.get("args", ())
        kwargs = task.get("kwargs", {})

        # Always provide task in kwargs for any method that needs it
        kwargs["task"] = task

        return func, args, kwargs

    def handle_output(self, output: Any) -> None:
        """Handle output from method execution and send results.

        Args:
            output: Output from method execution (None, dict, or list[dict])
        """
        if output is None:
            return

        # Handle result collection and emission
        # Methods can return:
        # - None: no result to send
        # - dict: single result to send
        # - list[dict]: multiple results to send
        if isinstance(output, list):
            for result in output:
                self.io_handler.send_result(result)
        elif isinstance(output, dict):
            self.io_handler.send_result(output)

    def worker_busy_loop(self, cancel: threading.Event | None = None) -> None:
        """Main busy loop for Multiprocessing Workers.

        Handles task dispatch, method execution, and result collection.
        This abstraction makes it easy to swap communication backends
        (e.g., ZMQ, RPC) by replacing the IO handler.
        """
        while True:
            # Check for shutdown
            if cancel is not None and cancel.is_set():
                logger.info("Worker shutdown requested via cancel event")
                break

            try:
                # Dequeue task (blocking)
                task = self.io_handler.recv_task()

                # Check for shutdown task
                if task == SHUTDOWN_TASK:
                    logger.info("Received shutdown signal")
                    break

                # Handle input: prepare method call
                func, args, kwargs = self.handle_input(task)

                # Execute the method
                output = func(*args, **kwargs)

                # If shutdown was called, break the loop
                if output == "SHUTDOWN":
                    break

                # Handle result collection and emission
                self.handle_output(output)

            except Exception as e:
                # Add traceback note if supported (Python 3.11+)
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                continue


class AsyncOmniStageWorkerProc:
    """Async worker process for omni stage orchestration."""

    def __init__(
        self,
        model: str,
        stage_payload: dict[str, Any],
        in_q: mp.Queue,
        out_q: mp.Queue,
        batch_timeout: int = 10,
        stage_init_timeout: int = 300,
    ):
        setproctitle("AsyncOmniStageWorkerProc")
        self.model = model
        self.stage_id = stage_payload["stage_id"]
        self.engine_args = stage_payload.get("engine_args", {})
        self.runtime_cfg = stage_payload.get("runtime", {})
        self.shm_threshold_bytes = int(stage_payload.get("shm_threshold_bytes", 65536))
        self.connectors_config = stage_payload.get("connectors_config", {})
        self.stage_type = stage_payload.get("stage_type", "llm")
        self.io_handler = AsyncMPQueueIOHandler(in_q, out_q)
        self.batch_timeout = batch_timeout
        self.stage_init_timeout = stage_init_timeout

        # Aggregates for running average
        self.agg_total_tokens = 0
        self.agg_total_gen_time_ms = 0.0
        self.batch_seq = 0

        # Device setup
        self._setup_devices()

        # Get max batch size before executor creation
        self.max_batch_size = int(self.runtime_cfg.get("max_batch_size", 1) or 1)
        self.engine_args["max_num_seqs"] = self.max_batch_size

        # Initialize connectors
        self.connectors = self._init_connectors()

    def _setup_devices(self) -> None:
        """Setup device configuration."""
        self.device_type = None
        try:
            self.device_type = detect_device_type()
            set_stage_devices(
                self.stage_id,
                self.runtime_cfg.get("devices"),
                device_type=self.device_type,
            )
        except Exception as e:
            logger.warning("Device setup failed: %s", e)

    def _acquire_device_locks(self) -> None:
        """Acquire exclusive locks for devices during initialization."""
        self.lock_files = []
        if self.device_type != "cuda":
            return

        try:
            import torch

            if not torch.cuda.is_available():
                return

            # Get parallel sizes from engine_args
            tensor_parallel_size = self.engine_args.get("tensor_parallel_size", 1)
            pipeline_parallel_size = self.engine_args.get("pipeline_parallel_size", 1)
            data_parallel_size = self.engine_args.get("data_parallel_size", 1)
            prefill_context_parallel_size = self.engine_args.get("prefill_context_parallel_size", 1)

            num_devices_per_stage = (
                tensor_parallel_size * pipeline_parallel_size * data_parallel_size * prefill_context_parallel_size
            )

            # Get physical device IDs
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            physical_devices = []

            if cuda_visible_devices:
                try:
                    physical_devices = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
                except (ValueError, IndexError):
                    pass

            if not physical_devices:
                num_devices = torch.cuda.device_count()
                physical_devices = list(range(num_devices))

            num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
            devices_to_lock = sorted(physical_devices[:num_devices_to_lock])

            logger.debug(
                "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d; will lock %d devices: %s",
                tensor_parallel_size,
                pipeline_parallel_size,
                data_parallel_size,
                prefill_context_parallel_size,
                num_devices_to_lock,
                devices_to_lock,
            )

            wait_start = time.time()
            acquired_lock_fds = []

            for device_id in devices_to_lock:
                lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
                lock_acquired = False

                while not lock_acquired:
                    try:
                        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            os.ftruncate(lock_fd, 0)
                            os.write(lock_fd, f"{os.getpid()}\n".encode())
                            os.fsync(lock_fd)
                            lock_acquired = True
                            acquired_lock_fds.append(lock_fd)
                            logger.debug("Acquired exclusive lock for device %s", device_id)
                        except BlockingIOError:
                            os.close(lock_fd)
                            if time.time() - wait_start > self.stage_init_timeout:
                                logger.warning(
                                    "Timeout waiting for device %s initialization lock, proceeding anyway",
                                    device_id,
                                )
                                break
                            time.sleep(0.1)
                    except OSError as e:
                        logger.debug(
                            "Failed to acquire lock for device %s: %s, continuing anyway",
                            device_id,
                            e,
                        )
                        try:
                            os.close(lock_fd)
                        except (OSError, NameError):
                            pass
                        break

            self.lock_files = acquired_lock_fds
        except Exception as e:
            logger.debug("[Stage-%s] Failed to set up sequential initialization lock: %s", self.stage_id, e)

    def _release_device_locks(self) -> None:
        """Release all acquired device locks."""
        for lock_fd in self.lock_files:
            try:
                os.close(lock_fd)
                logger.debug("Released initialization lock (fd=%s)", lock_fd)
            except (OSError, ValueError):
                pass
        self.lock_files = []

    async def _create_engine_core(self) -> Any:
        """Create appropriate async engine based on stage type."""
        logger.debug(
            "[Stage-%s] Initializing %s engine with args keys=%s",
            self.stage_id,
            self.stage_type,
            list(self.engine_args.keys()),
        )

        # Acquire device locks
        self._acquire_device_locks()

        try:
            if self.stage_type == "diffusion":
                # For diffusion, extract diffusion-specific config
                od_config = _build_od_config(self.engine_args, self.model)
                logger.debug(f"[Stage-%s] Initializing diffusion engine with config: {od_config}", self.stage_id)
                engine_core = AsyncOmniDiffusion(
                    model=self.model,
                    od_config=od_config,
                    **{k: v for k, v in self.engine_args.items() if k not in {"od_config", "model"}},
                )
                self.vllm_config = None  # Diffusion doesn't use vllm_config
            else:
                # LLM engine
                omni_engine_args = AsyncOmniEngineArgs(model=self.model, **self.engine_args)
                usage_context = UsageContext.OPENAI_API_SERVER
                self.vllm_config = omni_engine_args.create_engine_config(usage_context=usage_context)
                engine_core = AsyncOmniLLM.from_vllm_config(
                    vllm_config=self.vllm_config,
                    usage_context=usage_context,
                    engine_args=omni_engine_args,
                )

            # Reset mm cache for LLM engines
            if self.stage_type != "diffusion":
                await engine_core.reset_mm_cache()

            return engine_core
        finally:
            self._release_device_locks()

    def _init_connectors(self) -> dict:
        """Initialize OmniConnectors if configured."""
        connectors = {}
        if self.connectors_config:
            built_connectors = build_stage_connectors(
                stage_id=self.stage_id,
                connectors_config=self.connectors_config,
            )
            if built_connectors is not None:
                connectors = built_connectors
        return connectors

    async def _signal_ready(self) -> None:
        """Signal readiness to orchestrator."""
        try:
            stage_ready_payload = {
                "type": "stage_ready",
                "stage_id": self.stage_id,
                "vllm_config": self.vllm_config,
                "tokenizer": getattr(self.engine_core, "tokenizer", None),
            }
            # Only add is_tracing_enabled for LLM engines
            if self.stage_type != "diffusion":
                stage_ready_payload["is_tracing_enabled"] = await self.engine_core.is_tracing_enabled()
            self.io_handler.send_result(stage_ready_payload)
        except Exception as e:
            logger.warning("Failed to send stage ready signal: %s", e)

    async def start_profile(self) -> None:
        """Start profiling."""
        if hasattr(self.engine_core, "start_profile"):
            try:
                await self.engine_core.start_profile()
                logger.info("[Stage-%s] Profiler started via engine", self.stage_id)
            except Exception as e:
                logger.warning("[Stage-%s] Failed to start profiler: %s", self.stage_id, e)

    async def stop_profile(self) -> None:
        """Stop profiling."""
        if hasattr(self.engine_core, "stop_profile"):
            try:
                await self.engine_core.stop_profile()
                logger.info("[Stage-%s] Profiler stopped via engine", self.stage_id)
            except Exception as e:
                logger.warning("[Stage-%s] Failed to stop profiler: %s", self.stage_id, e)

    def shutdown(self) -> str:
        """Shutdown worker."""
        logger.info("[Stage-%s] Worker shutting down", self.stage_id)
        if hasattr(self.engine_core, "shutdown"):
            self.engine_core.shutdown()
        return "SHUTDOWN"

    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        if hasattr(self.engine_core, "abort"):
            await self.engine_core.abort(request_id)

    async def generate(self, task: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Execute a single generation request (async version).

        Args:
            task: Task to execute. Contains request_id, sampling_params, etc.

        Returns:
            Result dictionary to be sent via IO handler.
        """
        if task is None:
            return None

        result = await self._generation_single_request(task)
        return result

    async def _generation_single_request(self, task: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single generation request.

        Returns:
            Result dictionary to be sent via IO handler, or None on error.
        """
        recv_dequeue_ts = time.time()
        rid = task["request_id"]

        try:
            sent_ts = float(task.get("sent_ts", None)) if isinstance(task, dict) else None
            if sent_ts is not None:
                in_flight_time_ms = (recv_dequeue_ts - sent_ts) * 1000.0
            else:
                in_flight_time_ms = 0.0
        except Exception:
            in_flight_time_ms = 0.0

        try:
            ein, rx_metrics = try_recv_via_connector(
                task=task,
                connectors=self.connectors,
                stage_id=self.stage_id,
            )
            if ein is None or rx_metrics is None:
                raise RuntimeError(
                    f"[Stage-{self.stage_id}] Missing connector payload for request {rid}. "
                    "Ensure connectors are configured for all incoming edges."
                )
            rx_decode_time_ms = float(rx_metrics.get("rx_decode_time_ms", 0.0))
            rx_transfer_bytes = int(rx_metrics.get("rx_transfer_bytes", 0))

            sampling_params = task["sampling_params"]
            logger.debug("Received batch size=1, request_ids=%s", rid)
            gen_t0 = time.time()

            if isinstance(ein, list):
                ein = ein[0]

            # Execute through engine_core based on stage type
            gen_t0 = time.time()

            if self.stage_type == "diffusion":
                # Convert input to prompt
                if isinstance(ein, str):
                    prompt = ein
                elif isinstance(ein, dict) and "prompt" in ein:
                    prompt = ein["prompt"]
                elif hasattr(ein, "prompt"):
                    prompt = ein.prompt
                else:
                    prompt = str(ein)

                # Prepare diffusion kwargs
                diffusion_kwargs = prepare_sampling_params(sampling_params, "diffusion")
                gen_output = await self.engine_core.generate(prompt=prompt, request_id=rid, **diffusion_kwargs)
            else:
                # LLM engine
                llm_sampling_params = prepare_sampling_params(sampling_params, "llm")
                gen_output = None
                async for res in self.engine_core.generate(ein, llm_sampling_params, rid):
                    gen_output = res

            gen_t1 = time.time()
            gen_ms = (gen_t1 - gen_t0) * 1000.0

            # Prepare metrics and result
            metrics = make_request_stats(
                [gen_output],
                gen_ms,
                int(self.batch_seq),
                1,
                rx_decode_time_ms,
                rx_transfer_bytes,
                in_flight_time_ms,
            )
            self.agg_total_tokens += metrics.num_tokens_out

            # Prepare result
            r_outputs = [gen_output]
            try:
                use_shm, payload = maybe_dump_to_shm(r_outputs, self.shm_threshold_bytes)
                if use_shm:
                    return {
                        "request_id": rid,
                        "stage_id": self.stage_id,
                        "engine_outputs_shm": payload,
                        "metrics": metrics,
                    }
                else:
                    return {
                        "request_id": rid,
                        "stage_id": self.stage_id,
                        "engine_outputs": payload,
                        "metrics": metrics,
                    }
            except Exception:
                return {
                    "request_id": rid,
                    "stage_id": self.stage_id,
                    "engine_outputs": r_outputs,
                    "metrics": metrics,
                }
        except Exception as e:
            logger.exception("Failed on request %s: %s", rid, e)
            return {
                "request_id": rid,
                "stage_id": self.stage_id,
                "error": str(e),
            }

    def handle_input(self, task: dict[str, Any]) -> tuple[Any, tuple, dict]:
        """Handle input task: prepare method call for execution.

        Args:
            task: Task dictionary containing type, method, args, kwargs, etc.

        Returns:
            Tuple of (func, args, kwargs) ready for execution.
        """
        # Get task type and method name
        task_type = task.get("type", OmniStageTaskType.GENERATE)
        method_name = task.get("method", None)

        # If method name not provided, infer from task type
        if method_name is None:
            method_name = _get_method_name_from_task_type(task_type)

        assert isinstance(method_name, str)
        func = getattr(self, method_name)

        # Get args and kwargs from task
        args = task.get("args", ())
        kwargs = task.get("kwargs", {})

        # Always provide task in kwargs for any method that needs it
        kwargs["task"] = task

        # Special handling for abort - extract request_id if not in args
        if method_name == "abort" and not args:
            args = (task["request_id"],)

        return func, args, kwargs

    def handle_output(self, output: Any, batch_gen_t0: float) -> float:
        """Handle output from method execution and send results.

        Args:
            output: Output from method execution (None, dict, or list[dict])
            batch_gen_t0: Batch generation start time

        Returns:
            Updated batch generation start time
        """
        if output is None:
            return batch_gen_t0

        self.batch_seq += 1

        # Update stage stats for this batch
        batch_gen_t1 = time.time()
        self.agg_total_gen_time_ms += (batch_gen_t1 - batch_gen_t0) * 1000

        if isinstance(output, dict):
            # Add stage stats to the result
            if "metrics" in output:
                output["metrics"].stage_stats = make_stage_stats(self.agg_total_tokens, self.agg_total_gen_time_ms)
            self.io_handler.send_result(output)
            logger.debug("Sent result for request %s", output.get("request_id"))
        elif isinstance(output, list):
            # Multiple results
            for i, result in enumerate(output):
                # Add stage stats to last result only
                if i == len(output) - 1 and "metrics" in result:
                    result["metrics"].stage_stats = make_stage_stats(self.agg_total_tokens, self.agg_total_gen_time_ms)
                self.io_handler.send_result(result)
                logger.debug("Sent result for request %s", result.get("request_id"))

        return batch_gen_t1

    async def worker_busy_loop(self, cancel: threading.Event | None = None) -> None:
        """Main async busy loop for worker.

        Handles task dispatch, method execution, and result collection.
        This abstraction makes it easy to swap communication backends.
        """
        batch_gen_t0 = time.time()

        while True:
            # Check for shutdown
            if cancel is not None and cancel.is_set():
                logger.info("Worker shutdown requested via cancel event")
                break

            try:
                # Dequeue task (non-blocking)
                task = await self.io_handler.recv_task(timeout=0)

                if task is None:
                    await asyncio.sleep(0.001)
                    continue

                # Check for shutdown task
                if task == SHUTDOWN_TASK:
                    logger.info("Received shutdown signal")
                    break

                # Handle input: prepare method call
                func, args, kwargs = self.handle_input(task)

                # Execute the method (await if coroutine)
                if asyncio.iscoroutinefunction(func):
                    output = await func(*args, **kwargs)
                else:
                    output = func(*args, **kwargs)

                # If shutdown was called, break the loop
                if output == "SHUTDOWN":
                    break

                # Handle result collection and emission
                batch_gen_t0 = self.handle_output(output, batch_gen_t0)

            except Exception as e:
                # Add traceback note if supported (Python 3.11+)
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                continue

        logger.info("Stage worker exiting")


def _stage_worker(
    model: str,
    stage_payload: dict[str, Any],
    in_q: mp.Queue,
    out_q: mp.Queue,
    batch_timeout: int = 10,
    stage_init_timeout: int = 300,
) -> None:
    """Stage worker entry: device setup, LLM init, batching, SHM IPC.

    This is the main entry point for the worker process. It handles
    signal setup, death monitoring, worker initialization, and the
    main busy loop.
    """
    logger.info(f"Starting stage worker with model: {model}")

    # Signal handler used for graceful termination
    shutdown_requested = False
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            shutdown_event.set()
            raise SystemExit()

    # Either SIGTERM or SIGINT will terminate the worker
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker = None
    try:
        # Initialize worker
        worker = OmniStageWorkerProc(
            model=model,
            stage_payload=stage_payload,
            in_q=in_q,
            out_q=out_q,
            batch_timeout=batch_timeout,
            stage_init_timeout=stage_init_timeout,
        )

        # Run worker busy loop
        worker.worker_busy_loop(cancel=shutdown_event)

    except Exception:
        if shutdown_event.is_set():
            logger.info("WorkerProc shutting down.")
        else:
            logger.exception("WorkerProc failed.")
        # Set this value so we don't re-throw SystemExit()
        shutdown_requested = True

    finally:
        # Cleanup
        if worker is not None:
            logger.debug("Worker cleanup complete")


def _stage_worker_async_entry(
    omni_stage: OmniStage,
    model: str,
    stage_payload: dict[str, Any],
    batch_timeout: int = 10,
    stage_init_timeout: int = 300,
) -> None:
    asyncio.run(_stage_worker_async(omni_stage, model, stage_payload, batch_timeout, stage_init_timeout))


async def _stage_worker_async(
    omni_stage: OmniStage,
    model: str,
    stage_payload: dict[str, Any],
    batch_timeout: int = 10,
    stage_init_timeout: int = 300,
) -> None:
    """Async stage worker entry: device setup, LLM init, batching, SHM IPC.

    This is the main entry point for the async worker process. It handles
    worker initialization and the main async busy loop.
    """
    logger.info(f"Starting async stage worker with model: {model}")

    worker = None
    try:
        # Initialize worker
        worker = AsyncOmniStageWorkerProc(
            model=model,
            stage_payload=stage_payload,
            in_q=omni_stage._in_q,
            out_q=omni_stage._out_q,
            batch_timeout=batch_timeout,
            stage_init_timeout=stage_init_timeout,
        )

        # Initialize engine_core
        worker.engine_core = await worker._create_engine_core()

        # Set engine in omni_stage for access from main process
        omni_stage.set_async_engine(worker.engine_core)

        # Signal readiness
        await worker._signal_ready()

        logger.debug("[Stage-%s] Engine initialized", worker.stage_id)

        # Run worker busy loop
        await worker.worker_busy_loop()

    except Exception:
        logger.exception("WorkerProc failed.")

    finally:
        # Cleanup
        if worker is not None:
            logger.debug("Worker cleanup complete")


def count_prompt_tokens_from_outputs(engine_outputs: list[Any]) -> int:
    """Count prompt tokens from engine outputs."""
    total = 0
    for _ro in engine_outputs:
        try:
            prompt_token_ids = getattr(_ro, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                total += len(prompt_token_ids)
        except Exception:
            pass
    return total


def make_request_stats(
    req_output: list[Any],
    stage_gen_time_ms: float,
    batch_id: int,
    batch_size: int,
    rx_decode_time_ms: float,
    rx_transfer_bytes: int,
    rx_in_flight_time_ms: float,
):
    from vllm_omni.entrypoints.log_utils import (
        StageRequestMetrics,
    )

    num_tokens_in = count_prompt_tokens_from_outputs(req_output)
    num_tokens_out = count_tokens_from_outputs(req_output)
    return StageRequestMetrics(
        num_tokens_in=num_tokens_in,
        num_tokens_out=num_tokens_out,
        stage_gen_time_ms=stage_gen_time_ms,
        batch_id=batch_id,
        batch_size=batch_size,
        rx_decode_time_ms=rx_decode_time_ms,
        rx_transfer_bytes=rx_transfer_bytes,
        rx_in_flight_time_ms=rx_in_flight_time_ms,
        stage_stats=None,
    )


def make_stage_stats(_agg_total_tokens: int, _agg_total_gen_time_ms: float):
    from vllm_omni.entrypoints.log_utils import StageStats

    return StageStats(total_token=_agg_total_tokens, total_gen_time=_agg_total_gen_time_ms)
