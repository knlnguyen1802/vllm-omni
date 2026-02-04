import multiprocessing as mp
import time
import weakref
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.utils.platform_utils import get_diffusion_worker_class

logger = init_logger(__name__)


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
                # Send shutdown message to all workers via request pipes
                for pipe in self.scheduler.request_pipes:
                    try:
                        pipe.send(SHUTDOWN_MESSAGE)
                    except Exception as exc:
                        logger.warning("Failed to send shutdown signal to worker: %s", exc)
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

        # Initialize scheduler
        self.scheduler = Scheduler()
        self.scheduler.initialize(self.od_config)

        # Launch workers and get pipes
        processes = self._launch_workers()

        self._processes = processes

        self.resources = BackgroundResources(scheduler=self.scheduler, processes=self._processes)
        self._finalizer = weakref.finalize(self, self.resources)

    def _launch_workers(self):
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

        # Create pipes for worker communication
        # For each worker:
        # - request_pipe: scheduler sends requests to worker
        # - result_pipe: worker sends results to scheduler
        scheduler_pipe_readers = []  # For initial status messages
        scheduler_pipe_writers = []
        
        scheduler_request_pipes = []  # Scheduler side of request pipes (for sending)
        worker_request_pipes = []     # Worker side of request pipes (for receiving)
        
        scheduler_result_pipes = []   # Scheduler side of result pipes (for receiving)
        worker_result_pipes = []      # Worker side of result pipes (for sending)

        for i in range(num_gpus):
            # Create status pipe for initialization
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_readers.append(reader)
            scheduler_pipe_writers.append(writer)
            
            # Create request pipe (scheduler -> worker)
            sched_req_conn, worker_req_conn = mp.Pipe(duplex=False)
            scheduler_request_pipes.append(sched_req_conn)
            worker_request_pipes.append(worker_req_conn)
            
            # Create result pipe (worker -> scheduler)
            sched_res_conn, worker_res_conn = mp.Pipe(duplex=False)
            scheduler_result_pipes.append(sched_res_conn)
            worker_result_pipes.append(worker_res_conn)
            
            process = mp.Process(
                target=worker_proc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,  # status pipe
                    worker_req_conn,  # request pipe (receive)
                    worker_res_conn,  # result pipe (send)
                    worker_extension_cls,
                    custom_pipeline_args,
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            process.start()
            processes.append(process)

        # Close writer ends on scheduler side
        for writer in scheduler_pipe_writers:
            writer.close()

        # Wait for all workers to be ready
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

            reader.close()

        logger.debug("All workers are ready")
        
        # Initialize scheduler with pipe connections
        self.scheduler.initialize_pipes(scheduler_request_pipes, scheduler_result_pipes)

        return processes

    def add_req(self, requests: list[OmniDiffusionRequest]):
        return self.scheduler.add_req(requests)

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

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # Prepare RPC request message
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": unique_reply_rank,
        }

        try:
            # Send RPC request to all workers via request pipes
            for pipe in self.scheduler.request_pipes:
                pipe.send(rpc_request)

            # Determine which workers we expect responses from
            num_responses = 1 if unique_reply_rank is not None else self.od_config.num_gpus

            responses = []
            for i in range(num_responses):
                try:
                    # Determine which pipe to read from
                    if unique_reply_rank is not None:
                        result_pipe = self.scheduler.result_pipes[unique_reply_rank]
                    else:
                        result_pipe = self.scheduler.result_pipes[i]
                    
                    # Set timeout if deadline is specified
                    if deadline is not None:
                        remaining_time = deadline - time.monotonic()
                        if remaining_time <= 0:
                            raise TimeoutError(f"RPC call to {method} timed out.")
                        if result_pipe.poll(remaining_time):
                            response = result_pipe.recv()
                        else:
                            raise TimeoutError(f"RPC call to {method} timed out.")
                    else:
                        response = result_pipe.recv()

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

    def check_health(self) -> None:
        # Simple check if processes are alive
        for p in self._processes:
            if not p.is_alive():
                raise RuntimeError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        self._finalizer()
