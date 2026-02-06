import multiprocessing as mp
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable
import threading
import uuid
from collections import deque
from concurrent.futures import Future
from contextlib import suppress
from concurrent.futures import InvalidStateError
from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.utils.platform_utils import get_diffusion_worker_class

logger = init_logger(__name__)


class FutureWrapper(Future):
    def __init__(
        self,
        futures_queue: deque[tuple["FutureWrapper", Callable]],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.aggregate = aggregate
        super().__init__()

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")
        # Drain any futures ahead of us in the queue.
        while not self.done():
            future, get_response = self.futures_queue.pop()
            future.wait_for_response(get_response)
        return super().result()

    def wait_for_response(self, get_response: Callable):
        try:
            response = self.aggregate(get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    """

    scheduler: Scheduler | None = None
    processes: list[mp.Process] | None = None
    handler_thread: threading.Thread | None = None
    shutdown_event: threading.Event | None = None

    def __call__(self):
        """Clean up background resources."""
        if self.shutdown_event is not None:
            self.shutdown_event.set()
        if self.handler_thread is not None:
            self.handler_thread.join(timeout=5)
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
    """
    Multiprocess diffusion executor with future-based async request handling.
    
    This executor uses a future queue mechanism to handle asynchronous requests:
    - Each request (add_req or collective_rpc) is assigned a unique call_id (UUID)
    - Requests are queued as FutureWrapper objects with their associated get_response callable
    - A background result handler thread processes results from workers
    - Results include the call_id and are matched to their corresponding futures
    - This design avoids race conditions and provides clean async semantics
    """
    uses_multiproc: bool = True

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False
        self._queue_lock = threading.Lock()
        
        # Initialize future queue and call tracking
        self.futures_queue: deque[tuple[FutureWrapper, Callable]] = deque()
        self.pending_futures: dict[str, FutureWrapper] = {}  # call_id -> future
        self.futures_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
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

        # Start result handler thread
        self.handler_thread = threading.Thread(
            target=self._result_handler,
            name="ResultHandler",
            daemon=True,
        )
        self.handler_thread.start()

        self.resources = BackgroundResources(
            scheduler=self.scheduler,
            processes=self._processes,
            handler_thread=self.handler_thread,
            shutdown_event=self.shutdown_event,
        )
        self._finalizer = weakref.finalize(self, self.resources)

    def _launch_workers(self, broadcast_handle):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Get the appropriate worker class for current device
        worker_proc = get_diffusion_worker_class()

        # Extract worker_extension_cls and custom_pipeline_args from config if provided
        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)

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
                    worker_extension_cls,
                    custom_pipeline_args,
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

    def _result_handler(self):
        """Background thread that processes results from the result queue."""
        logger.info("Result handler thread started")
        while not self.shutdown_event.is_set():
            try:
                if self.scheduler.result_mq is None:
                    logger.warning("Result queue not initialized, waiting...")
                    time.sleep(0.1)
                    continue

                # Try to dequeue with a timeout to check shutdown_event periodically
                try:
                    result = self.scheduler.result_mq.dequeue(timeout=0.1)
                except TimeoutError:
                    continue

                # Extract call_id from result
                if isinstance(result, dict):
                    call_id = result.get("call_id")
                    actual_result = result.get("result")
                    
                    if call_id is None:
                        logger.warning(f"Result without call_id: {result}")
                        continue

                    # Find and set the corresponding future
                    with self.futures_lock:
                        future = self.pending_futures.pop(call_id, None)
                    
                    if future is not None:
                        if result.get("status") == "error":
                            error_msg = result.get("error", "Unknown error")
                            future.set_exception(RuntimeError(error_msg))
                        else:
                            future.set_result(actual_result)
                    else:
                        logger.warning(f"Received result for unknown call_id: {call_id}")
                else:
                    logger.warning(f"Unexpected result format: {type(result)}")

            except Exception as e:
                if not self.shutdown_event.is_set():
                    logger.error(f"Error in result handler: {e}", exc_info=True)
                    time.sleep(0.1)
        
        logger.info("Result handler thread stopped")

    def add_req(self, requests: list[OmniDiffusionRequest]):
        # Generate unique call_id
        call_id = str(uuid.uuid4())
        
        # Create future for this request
        future = FutureWrapper(self.futures_queue)
        
        with self.futures_lock:
            self.pending_futures[call_id] = future
        
        # Create get_response callable
        def get_response():
            with self._queue_lock:
                # Prepare RPC request for generation
                rpc_request = {
                    "type": "rpc",
                    "method": "generate",
                    "args": (requests,),
                    "kwargs": {},
                    "output_rank": 0,
                    "exec_all_ranks": True,
                    "call_id": call_id,
                }
                # Broadcast RPC request to all workers
                self.scheduler.mq.enqueue(rpc_request)
            return None  # Result will be set by handler thread
        
        # Add to futures queue
        self.futures_queue.appendleft((future, get_response))
        
        return future

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")

        # Generate unique call_id
        call_id = str(uuid.uuid4())
        kwargs = kwargs or {}
        
        # Determine aggregation function based on unique_reply_rank
        if unique_reply_rank is not None:
            aggregate = lambda x: x  # Single response
        else:
            # Multiple responses - collect them into a list
            aggregate = lambda x: x if isinstance(x, list) else [x]
        
        # Create future for this request
        future = FutureWrapper(self.futures_queue, aggregate=aggregate)
        
        with self.futures_lock:
            self.pending_futures[call_id] = future
        
        # Create get_response callable
        def get_response():
            with self._queue_lock:
                # Prepare RPC request message
                rpc_request = {
                    "type": "rpc",
                    "method": method,
                    "args": args,
                    "kwargs": kwargs,
                    "output_rank": unique_reply_rank,
                    "call_id": call_id,
                }

                try:
                    # Broadcast RPC request to all workers via unified message queue
                    self.scheduler.mq.enqueue(rpc_request)
                except Exception as e:
                    logger.error(f"RPC call failed: {e}")
                    raise
            
            return None  # Result will be set by handler thread
        
        # Add to futures queue
        self.futures_queue.appendleft((future, get_response))
        
        # Wait for result with timeout
        try:
            result = future.result(timeout=timeout)
            return result
        except Exception as e:
            # Clean up pending future if it fails
            with self.futures_lock:
                self.pending_futures.pop(call_id, None)
            raise

    def check_health(self) -> None:
        # Simple check if processes are alive
        for p in self._processes:
            if not p.is_alive():
                raise RuntimeError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        self._finalizer()
