# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multi-process executor for diffusion models.

This module provides a concrete implementation of DiffusionExecutor that uses
Python's multiprocessing to spawn worker processes. This is the default executor
for local multi-GPU execution.
"""

import multiprocessing as mp
import time
import weakref
from collections.abc import Callable
from typing import Any

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, OmniDiffusionConfig
from vllm_omni.diffusion.executor.diffusion_executor import DiffusionExecutor
from vllm_omni.diffusion.scheduler import Scheduler, scheduler
from vllm_omni.utils.platform_utils import get_diffusion_worker_class

logger = init_logger(__name__)


class BackgroundResources:
    """Resource manager for background processes and message queues.

    Used as a finalizer for clean shutdown. This object holds references to
    external system resources that are not managed by Python's garbage collector
    (like OS processes, message queues, etc.), so they must be cleaned up manually
    to avoid resource leaks or zombie processes.
    """

    def __init__(
        self, scheduler: Scheduler | None = None, processes: list[mp.Process] | None = None
    ):
        self.scheduler = scheduler
        self.processes = processes or []

    def __call__(self):
        """Clean up background resources."""
        if self.scheduler is not None:
            try:
                for _ in range(self.scheduler.num_workers):
                    self.scheduler.mq.enqueue(SHUTDOWN_MESSAGE)
                self.scheduler.close()
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)

        for proc in self.processes:
            if not proc.is_alive():
                continue
            proc.join(30)
            if proc.is_alive():
                logger.warning(
                    "Terminating diffusion worker %s after timeout", proc.name
                )
                proc.terminate()
                proc.join(30)


class MultiProcDiffusionExecutor(DiffusionExecutor):
    """Multi-process executor for diffusion models.

    This executor spawns multiple worker processes using Python's multiprocessing
    module. Each worker runs on a separate GPU and communicates with the engine
    through shared memory message queues.

    Architecture:
        - Main process (engine) coordinates execution
        - N worker processes (one per GPU)
        - Shared memory message queue for broadcasting requests
        - Result queue for collecting outputs

    Communication Flow:
        1. Engine sends requests/RPC calls via scheduler's message queue
        2. Workers receive and process requests
        3. Workers send results back via result queue
        4. Engine collects and returns results
    """

    uses_ray: bool = False
    supports_pp: bool = False

    def _init_executor(self) -> None:
        """Initialize the multi-process executor.

        This method:
        1. Initializes the scheduler with message queues
        2. Spawns worker processes
        3. Sets up communication channels
        4. Waits for workers to be ready
        """
        # Initialize scheduler with configuration
        scheduler.initialize(self.od_config)

        # Get the broadcast handle from the initialized scheduler
        broadcast_handle = scheduler.get_broadcast_handle()

        # Launch worker processes
        processes, result_handle = self._launch_workers(
            broadcast_handle=broadcast_handle,
        )

        if result_handle is not None:
            scheduler.initialize_result_queue(result_handle)
        else:
            logger.error("Failed to get result queue handle from workers")
            raise RuntimeError("Failed to initialize result queue")

        self._processes = processes

        # Set up resource cleanup
        self.resources = BackgroundResources(
            scheduler=scheduler, processes=self._processes
        )

        # Use weakref.finalize for proper cleanup during interpreter shutdown
        # This ensures cleanup happens before Python's shutdown sequence destroys
        # global state, preventing AttributeError during cleanup
        self._finalizer = weakref.finalize(self, self.resources)

    def _launch_workers(
        self, broadcast_handle: Any
    ) -> tuple[list[mp.Process], Any | None]:
        """Launch worker processes and establish communication channels.

        Args:
            broadcast_handle: Handle for the broadcast message queue.

        Returns:
            Tuple of (worker processes list, result queue handle).

        Raises:
            RuntimeError: If worker initialization fails.
        """
        od_config = self.od_config
        logger.info("Starting diffusion workers...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Get the appropriate worker class for current device
        worker_proc = get_diffusion_worker_class()

        # Extract worker_extension_cls from config if provided
        worker_extension_cls = od_config.worker_extension_cls

        # Create pipes for initial handshake
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        # Launch all worker processes
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
            processes.append(process)

        # Close writer ends in parent process
        for writer in scheduler_pipe_writers:
            writer.close()

        # Wait for all workers to be ready
        scheduler_infos = []
        result_handle = None

        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(
                    f"Rank {i} worker failed to start. Please check logs for errors."
                )
                processes[i].join()
                logger.error(f"Worker {i} exit code: {processes[i].exitcode}")
                raise RuntimeError(f"Worker {i} initialization failed")

            if data["status"] != "ready":
                raise RuntimeError(
                    f"Worker {i} initialization failed. Status: {data['status']}"
                )

            # Get result handle from first worker
            if i == 0:
                result_handle = data.get("result_handle")

            scheduler_infos.append(data)
            reader.close()

        logger.info(f"All {num_gpus} diffusion workers are ready")

        return processes, result_handle

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Execute an RPC call on worker processes.

        Args:
            method: The method name (str) or callable to execute on workers.
            timeout: Optional timeout in seconds.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.
            unique_reply_rank: If set, only get reply from this rank.

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results.

        Raises:
            RuntimeError: If executor is closed or RPC fails.
            TimeoutError: If RPC call times out.
        """
        if self._closed:
            raise RuntimeError("Executor is closed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Currently only support string method names
        if not isinstance(method, str):
            raise NotImplementedError("Callable methods not yet supported")

        send_method = method

        # Prepare RPC request message
        rpc_request = {
            "type": "rpc",
            "method": send_method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            scheduler.mq.enqueue(rpc_request)

            # Determine which workers we expect responses from
            num_responses = (
                1 if unique_reply_rank is not None else self.od_config.num_gpus
            )

            responses = []
            for _ in range(num_responses):
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    if scheduler.result_mq is None:
                        raise RuntimeError("Result queue not initialized")

                    response = scheduler.result_mq.dequeue(timeout=dequeue_timeout)

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
            logger.error(f"RPC call to {method} failed: {e}")
            raise

    def check_health(self) -> None:
        """Check if all worker processes are healthy.

        Raises:
            RuntimeError: If any worker process has died.
        """
        if self._closed:
            raise RuntimeError("Executor is closed.")

        for i, proc in enumerate(self._processes):
            if not proc.is_alive():
                raise RuntimeError(
                    f"Worker process {i} (PID {proc.pid}) has died. "
                    f"Exit code: {proc.exitcode}"
                )

    def shutdown(self) -> None:
        """Shutdown the executor and clean up all resources.

        This method:
        1. Sends shutdown messages to all workers
        2. Waits for workers to terminate gracefully
        3. Forces termination if workers don't respond
        4. Closes message queues and other resources
        """
        if self._closed:
            return

        logger.info("Shutting down diffusion executor...")

        # Trigger cleanup through finalizer
        self._finalizer()

        logger.info("Diffusion executor shutdown complete.")
