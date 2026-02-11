import multiprocessing as mp
import threading
import time
import uuid
import weakref
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.diffusion.worker import WorkerProc

logger = init_logger(__name__)


class FutureWrapper(Future):
    """Wrapper for Future that supports request ID tracking and aggregation."""

    def __init__(
        self,
        request_id: str,
        pending_futures: dict[str, "FutureWrapper"],
        aggregate: Callable = lambda x: x,
    ):
        self.request_id = request_id
        self.pending_futures = pending_futures
        self.aggregate = aggregate
        super().__init__()

    def result(self, timeout=None):
        """Get the result of the future."""
        return super().result(timeout=timeout)

    def set_response(self, response: Any):
        """Set the response for this future."""
        try:
            aggregated_response = self.aggregate(response)
            with suppress(InvalidStateError):
                self.set_result(aggregated_response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)
        finally:
            # Remove from pending futures once completed
            self.pending_futures.pop(self.request_id, None)


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
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


class MultiprocDiffusionExecutor(DiffusionExecutor):
    uses_multiproc: bool = True

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False
        self._pending_futures: dict[str, FutureWrapper] = {}
        self._shutdown_event = threading.Event()

        # Initialize scheduler
        self.scheduler = Scheduler()
        self.scheduler.initialize(self.od_config)
        broadcast_handle = self.scheduler.get_broadcast_handle()

        # Launch workers
        processes, result_handle = self._launch_workers(broadcast_handle)

        if result_handle is not None:
            self.scheduler.initialize_result_queue(result_handle)
        else:
            logger.error("Failed to get result queue handle from workers")

        self._processes = processes

        # Start output handler thread
        self._output_handler_thread = threading.Thread(
            target=self._output_handler_loop,
            daemon=True,
            name="DiffusionOutputHandler"
        )
        self._output_handler_thread.start()

        self.resources = BackgroundResources(scheduler=self.scheduler, processes=self._processes)
        self._finalizer = weakref.finalize(self, self.resources)

    def _launch_workers(self, broadcast_handle):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Launch all worker processes
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            process = mp.Process(
                target=WorkerProc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            scheduler_pipe_readers.append(reader)
            process.start()
            processes.append(process)

        # Wait for all workers to be ready
        scheduler_infos = []
        result_handle = None
        for writer in scheduler_pipe_writers:
            writer.close()

        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(f"Rank {i} scheduler is dead. Please check if there are relevant logs.")
                processes[i].join()
                logger.error(f"Exit code: {processes[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError("Initialization failed. Please see the error messages above.")

            if i == 0:
                result_handle = data.get("result_handle")

            scheduler_infos.append(data)
            reader.close()

        logger.debug("All workers are ready")

        return processes, result_handle

    @staticmethod
    def _get_next_request_id() -> str:
        """Generate unique request ID using UUID."""
        return str(uuid.uuid4())

    def _output_handler_loop(self) -> None:
        """Background thread that collects results from scheduler and sets futures."""
        logger.info("Output handler loop started")
        
        while not self._shutdown_event.is_set():
            try:
                if self.scheduler.result_mq is None:
                    logger.warning("Result queue not initialized, waiting...")
                    time.sleep(0.1)
                    continue

                # Dequeue result with timeout to allow checking shutdown event
                try:
                    result = self.scheduler.result_mq.dequeue(timeout=0.1)
                except Exception:
                    # Timeout or other dequeue error, continue loop
                    continue

                # Extract request_id from result
                if isinstance(result, dict) and "request_id" in result:
                    request_id = result["request_id"]
                    
                    # Find the corresponding future
                    future = self._pending_futures.get(request_id)
                    if future is not None:
                        # Extract actual response (remove request_id wrapper)
                        actual_response = result.get("response")
                        
                        # Check for errors
                        if result.get("status") == "error":
                            error_msg = result.get("error", "Unknown error")
                            future.set_exception(RuntimeError(error_msg))
                        else:
                            future.set_response(actual_response)
                    # Ignore results for unknown request_id (already completed/cleaned up)
                # Ignore results without request_id (legacy format)

            except Exception as e:
                if not self._shutdown_event.is_set():
                    logger.error(f"Error in output handler loop: {e}", exc_info=True)
                    time.sleep(0.1)  # Avoid tight loop on persistent errors
        
        logger.info("Output handler loop stopped")

    def add_req(self, request: OmniDiffusionRequest, non_block: bool = False) -> DiffusionOutput | Future[DiffusionOutput]:
        """Add a diffusion request.
        
        Args:
            request: The diffusion request to process
            non_block: If True, returns a Future immediately; if False, blocks until result is ready
            
        Returns:
            DiffusionOutput if non_block=False, Future[DiffusionOutput] if non_block=True
        """
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")
        
        # Generate unique request ID
        request_id = self._get_next_request_id()
        
        # Create future for this request
        future = FutureWrapper(
            request_id=request_id,
            pending_futures=self._pending_futures,
            aggregate=lambda x: x
        )
        self._pending_futures[request_id] = future
        
        # Wrap request with ID for tracking
        request_with_id = {
            "type": "add_req",
            "request_id": request_id,
            "request": request,
        }
        
        # Send to scheduler
        self.scheduler.mq.enqueue(request_with_id)
        
        if non_block:
            return future
        else:
            # Block and wait for result
            return future.result()

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
    ) -> Any | Future[Any]:
        """Execute RPC call across workers.
        
        Args:
            method: Method name to call on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            non_block: If True, returns a Future immediately; if False, blocks until result is ready
            unique_reply_rank: If specified, only this rank returns a response
            
        Returns:
            Result or Future depending on non_block parameter
        """
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")

        kwargs = kwargs or {}
        
        # Generate unique request ID
        request_id = self._get_next_request_id()
        
        # Determine aggregation function
        if unique_reply_rank is not None:
            aggregate = lambda x: x  # Single response, no aggregation needed
        else:
            # Multiple responses - return as list
            aggregate = lambda x: x if isinstance(x, list) else [x]
        
        # Create future for this request
        future = FutureWrapper(
            request_id=request_id,
            pending_futures=self._pending_futures,
            aggregate=aggregate
        )
        self._pending_futures[request_id] = future
        
        # Prepare RPC request message with request ID
        rpc_request = {
            "type": "rpc",
            "request_id": request_id,
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self.scheduler.mq.enqueue(rpc_request)
            
            if non_block:
                return future
            else:
                # Block and wait for result with optional timeout
                return future.result(timeout=timeout)

        except Exception as e:
            # Clean up future on error
            self._pending_futures.pop(request_id, None)
            logger.error(f"RPC call failed: {e}")
            raise

    def check_health(self) -> None:
        # Simple check if processes are alive
        for p in self._processes:
            if not p.is_alive():
                raise RuntimeError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        """Shutdown executor and clean up resources."""
        if self._closed:
            return
            
        self._closed = True
        
        # Signal output handler to stop
        self._shutdown_event.set()
        
        # Cancel all pending futures
        for future in list(self._pending_futures.values()):
            if not future.done():
                with suppress(InvalidStateError):
                    future.set_exception(RuntimeError("Executor shutdown"))
        
        self._pending_futures.clear()
        
        # Wait for output handler thread to finish
        if hasattr(self, '_output_handler_thread') and self._output_handler_thread.is_alive():
            self._output_handler_thread.join(timeout=5)
        
        # Clean up resources
        self._finalizer()
