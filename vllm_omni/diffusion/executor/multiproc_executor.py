# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import time
import weakref
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, OmniDiffusionConfig
from vllm_omni.diffusion.executor.executor_base import DiffusionExecutor
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.utils.platform_utils import get_diffusion_worker_class

logger = init_logger(__name__)


class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    Create a BackgroundResources instance to encapsulate all background resources
    (e.g., the scheduler and worker processes) that need explicit cleanup.
    This object holds references to external system resources that are not managed
    by Python's garbage collector (like OS processes, message queues, etc.),
    so they must be cleaned up manually to avoid resource leaks or zombie processes.
    """

    scheduler: Scheduler | None = None
    processes: list[mp.Process] | None = None

    def __call__(self):
        """Clean up background resources."""
        if self.scheduler is not None:
            try:
                for _ in range(self.scheduler.num_workers):
                    self.scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
                self.scheduler.close()
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)
        
        if self.processes:
            for proc in self.processes:
                if not proc.is_alive():
                    continue
                proc.join(30)
                if proc.is_alive():
                    logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                    proc.terminate()
                    proc.join(30)


class MultiProcDiffusionExecutor(DiffusionExecutor):
    """Multiprocessing executor for diffusion models.

    This executor manages worker processes and a scheduler, handling all
    communication between the engine and workers via message queues.
    """

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        """Initialize the multiprocessing executor.

        Args:
            od_config: The diffusion configuration.
        """
        self.scheduler: Scheduler | None = None
        self.processes: list[mp.Process] = []
        self._closed = False
        super().__init__(od_config)
        # Set up cleanup finalizer
        self.resources = BackgroundResources()
        self.resources.scheduler = self.scheduler
        self.resources.processes = self.processes
        self._finalizer = weakref.finalize(self, self.resources)

    def _init_executor(self) -> None:
        """Initialize the executor by setting up scheduler and workers."""
        logger.info("Initializing MultiProcDiffusionExecutor")

        # Initialize scheduler
        self._init_scheduler()

        # Launch workers
        self._launch_workers()

        logger.info("MultiProcDiffusionExecutor initialized successfully")

    def _init_scheduler(self) -> None:
        """Initialize the scheduler."""
        self.scheduler = Scheduler()
        self.scheduler.initialize(self.od_config)
        logger.debug("Scheduler initialized")

    def _launch_workers(self) -> None:
        """Launch worker processes."""
        od_config = self.od_config
        num_gpus = od_config.num_gpus

        logger.info(f"Launching {num_gpus} worker processes...")

        # Set multiprocessing start method
        mp.set_start_method("spawn", force=True)

        # Get broadcast handle from scheduler
        broadcast_handle = self.scheduler.get_broadcast_handle()

        # Get the appropriate worker class for current device
        worker_proc = get_diffusion_worker_class()

        # Extract worker_extension_cls from config if provided
        worker_extension_cls = od_config.worker_extension_cls

        # Launch all worker processes
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            
            process = mp.Process(
                target=worker_proc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
                    worker_extension_cls,  # Pass worker_extension_cls
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            scheduler_pipe_readers.append(reader)
            process.start()
            self.processes.append(process)

        # Wait for all workers to be ready
        scheduler_infos = []
        result_handle = None
        
        # Close writer pipes in parent process
        for writer in scheduler_pipe_writers:
            writer.close()

        # Receive initialization status from all workers
        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(
                    f"Rank {i} scheduler is dead. Please check if there are relevant logs."
                )
                self.processes[i].join()
                logger.error(f"Exit code: {self.processes[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError(
                    "Initialization failed. Please see the error messages above."
                )

            if i == 0:
                result_handle = data.get("result_handle")

            scheduler_infos.append(data)
            reader.close()

        # Initialize result queue in scheduler
        if result_handle is not None:
            self.scheduler.initialize_result_queue(result_handle)
        else:
            logger.error("Failed to get result queue handle from workers")
            raise RuntimeError("Result queue handle not received from workers")

        logger.debug("All workers are ready")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results.

        Args:
            method: The method name (str) or callable to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        if self._closed:
            raise RuntimeError("MultiProcDiffusionExecutor is closed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Ensure method is a string for serialization
        if not isinstance(method, str):
            raise ValueError(
                "MultiProcDiffusionExecutor only supports string method names, "
                f"got {type(method)}"
            )

        # Prepare RPC request message
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self.scheduler.mq.enqueue(rpc_request)

            # Determine which workers we expect responses from
            num_responses = 1 if unique_reply_rank is not None else self.od_config.num_gpus

            responses = []
            for _ in range(num_responses):
                dequeue_timeout = None if deadline is None else (deadline - time.monotonic())
                try:
                    if self.scheduler.result_mq is None:
                        raise RuntimeError("Result queue not initialized")

                    response = self.scheduler.result_mq.dequeue(timeout=dequeue_timeout)

                    # Check if response indicates an error
                    if isinstance(response, dict) and response.get("status") == "error":
                        raise RuntimeError(
                            f"Worker failed with error '{response.get('error')}', "
                            "please check the stack trace above for the root cause"
                        )

                    responses.append(response)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e

            return responses[0] if unique_reply_rank is not None else responses

        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise

    def add_requests(self, requests: list) -> Any:
        """Add diffusion requests to be processed.

        Args:
            requests: List of OmniDiffusionRequest objects

        Returns:
            DiffusionOutput from processing the requests
        """
        if self._closed:
            raise RuntimeError("MultiProcDiffusionExecutor is closed.")

        return self.scheduler.add_req(requests)

    def check_health(self) -> None:
        """Check if the executor is healthy.

        Raises:
            RuntimeError: If any worker process has died
        """
        if self._closed:
            raise RuntimeError("MultiProcDiffusionExecutor is closed.")

        for i, proc in enumerate(self.processes):
            if not proc.is_alive():
                raise RuntimeError(
                    f"Worker process {i} (pid={proc.pid}) is not alive. "
                    f"Exit code: {proc.exitcode}"
                )

    def shutdown(self) -> None:
        """Shutdown the executor and clean up all resources."""
        if self._closed:
            return

        logger.info("Shutting down MultiProcDiffusionExecutor")
        self._closed = True

        # Trigger cleanup via finalizer
        self._finalizer()

        logger.info("MultiProcDiffusionExecutor shutdown complete")
