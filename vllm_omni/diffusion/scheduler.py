# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import zmq
import threading
import time
import uuid
from typing import Any
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        existing_context = getattr(self, "context", None)
        if existing_context is not None and not existing_context.closed:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config
        self.context = zmq.Context()  # Standard synchronous context

        # Initialize single MessageQueue for all message types (generation & RPC)
        # Assuming all readers are local for now as per current launch_engine implementation
        self.mq = MessageQueue(
            n_reader=self.num_workers,
            n_local_reader=self.num_workers,
            local_reader_ranks=list(range(self.num_workers)),
        )

        self.result_mq = None
        self._closed = False
        
        # Result collection dict: request_id -> {event, result}
        self._pending_results: dict[str, dict] = {}
        self._result_lock = threading.Lock()
        
        # Background handler thread
        self._handler_thread = None

    def initialize_result_queue(self, handle):
        # Initialize MessageQueue for receiving results
        # We act as rank 0 reader for this queue
        self.result_mq = MessageQueue.create_from_handle(handle, rank=0)
        logger.info("SyncScheduler initialized result MessageQueue")
        
        # Start the result collection handler thread
        self._handler_thread = threading.Thread(target=self._result_handler, daemon=True)
        self._handler_thread.start()
        logger.info("Started result handler thread")

    def get_broadcast_handle(self):
        return self.mq.export_handle()

    def _result_handler(self):
        """Background thread that collects results from workers into dict."""
        logger.info("Result handler thread started")
        while not self._closed:
            try:
                if self.result_mq is None:
                    time.sleep(0.01)
                    continue
                
                # Dequeue result with timeout
                result = self.result_mq.dequeue(timeout=0.1)
                
                # Extract request_id from result
                request_id = None
                if isinstance(result, dict):
                    request_id = result.get("request_id")
                
                if request_id:
                    with self._result_lock:
                        if request_id in self._pending_results:
                            # Store result and signal waiting thread
                            self._pending_results[request_id]["result"] = result
                            self._pending_results[request_id]["event"].set()
                        else:
                            logger.warning(f"Received result for unknown request_id: {request_id}")
                else:
                    logger.warning(f"Received result without request_id: {result}")
                    
            except TimeoutError:
                # Normal timeout, continue polling
                continue
            except Exception as e:
                if not self._closed:
                    logger.error(f"Error in result handler: {e}", exc_info=True)
                break
        
        logger.info("Result handler thread stopped")

    def add_req(self, requests: list[OmniDiffusionRequest], request_id: str = None, timeout: float = None) -> DiffusionOutput:
        """Sends a request to the scheduler and waits for the response."""
        if request_id is None:
            raise ValueError("request_id is required")
        
        # Create event for this request
        event = threading.Event()
        with self._result_lock:
            self._pending_results[request_id] = {
                "event": event,
                "result": None
            }
        
        try:
            # Prepare RPC request for generation
            rpc_request = {
                "type": "rpc",
                "method": "generate",
                "args": (requests,),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": True,
                "request_id": request_id,
            }
            
            # Enqueue request (non-blocking)
            self.mq.enqueue(rpc_request)
            
            # Wait for result to be collected by handler
            if not event.wait(timeout=timeout):
                raise TimeoutError("Scheduler did not respond in time.")
            
            # Retrieve result
            with self._result_lock:
                result = self._pending_results[request_id]["result"]
            
            return result
            
        except zmq.error.Again:
            logger.error("Timeout waiting for response from scheduler.")
            raise TimeoutError("Scheduler did not respond in time.")
        finally:
            # Clean up
            with self._result_lock:
                self._pending_results.pop(request_id, None)

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        num_workers: int = None,
    ) -> Any:
        """Execute RPC across workers and collect results.
        
        Note: Currently only rank 0 has result_mq, so we always expect 1 response
        regardless of how many workers execute the RPC.
        """
        request_id = str(uuid.uuid4())
        kwargs = kwargs or {}
        
        # Always expect 1 response since only rank 0 has result_mq
        # Even if all workers execute (exec_all_ranks=True), only rank 0 replies
        num_expected = 1
        
        # Use single response tracking (same as add_req)
        event = threading.Event()
        with self._result_lock:
            self._pending_results[request_id] = {
                "event": event,
                "result": None
            }
        
        try:
            # Prepare RPC request message
            rpc_request = {
                "type": "rpc",
                "method": method,
                "args": args,
                "kwargs": kwargs,
                "output_rank": unique_reply_rank,
                "request_id": request_id,
            }
            
            # Enqueue request (non-blocking)
            self.mq.enqueue(rpc_request)
            
            # Wait for response (always 1 response from rank 0)
            if not event.wait(timeout=timeout):
                raise TimeoutError(f"RPC call to {method} timed out.")
            
            # Retrieve result (single response like add_req)
            with self._result_lock:
                result = self._pending_results[request_id]["result"]
            
            # Check for errors
            if isinstance(result, dict) and result.get("status") == "error":
                raise RuntimeError(
                    f"Worker failed with error '{result.get('error')}', "
                    "please check the stack trace above for the root cause"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            raise
        finally:
            # Clean up
            with self._result_lock:
                self._pending_results.pop(request_id, None)

    def close(self):
        """Closes the socket and terminates the context."""
        self._closed = True
        
        # Wait for handler thread to finish
        if self._handler_thread and self._handler_thread.is_alive():
            self._handler_thread.join(timeout=5)
        
        if hasattr(self, "context"):
            self.context.term()
        self.context = None
        self.mq = None
        self.result_mq = None
        
        # Clear pending results
        with self._result_lock:
            self._pending_results.clear()
