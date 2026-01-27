"""
DiffusionWorkerActor
Ray actor that initializes and manages scheduler and diffusion worker processes.
Part of the RayActorExecutor system.
"""

from __future__ import annotations

import time

import ray
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


@ray.remote
class DiffusionWorkerActor:
    """
    Ray actor that encapsulates the entire worker infrastructure.
    It initializes the scheduler and spawns worker processes inside the actor.
    """

    def __init__(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config
        self.scheduler = None
        self.worker_processes = []

    # --------------------------------------------------------------------- #
    # Initialization
    # --------------------------------------------------------------------- #
    def initialize(self) -> bool:
        """Initialize scheduler and worker processes."""
        from vllm_omni.diffusion.worker.gpu_worker import WorkerWrapperBase

        self.worker = WorkerWrapperBase(
            gpu_id=0,
            od_config=self.od_config,
        )
        return True
        # from vllm_omni.diffusion.scheduler import Scheduler
        # from vllm_omni.utils.platform_utils import get_diffusion_worker_class

        # logger.info("Initializing DiffusionWorkerActor...")
        # scheduler = Scheduler()
        # # Initialize scheduler
        # scheduler.initialize(self.od_config)
        # self.scheduler = scheduler

        # broadcast_handle = scheduler.get_broadcast_handle()

        # num_gpus = self.od_config.num_gpus
        # try:
        #     mp.set_start_method("spawn", force=True)
        # except RuntimeError:
        #     pass

        # worker_proc = get_diffusion_worker_class()
        # # Launch all worker processes
        # scheduler_pipe_readers = []
        # scheduler_pipe_writers = []

        # self.worker_processes = []

        # for i in range(num_gpus):
        #     reader, writer = mp.Pipe(duplex=False)
        #     scheduler_pipe_writers.append(writer)
        #     proc = mp.Process(
        #         target=worker_proc.worker_main,
        #         args=(i, self.od_config, writer, broadcast_handle),
        #         name=f"DiffusionWorker-{i}",
        #         daemon=True,
        #     )
        #     scheduler_pipe_readers.append(reader)
        #     proc.start()
        #     self.worker_processes.append(proc)

        # # Wait for all workers to be ready
        # scheduler_infos = []
        # result_handle = None
        # for writer in scheduler_pipe_writers:
        #     writer.close()

        # for i, reader in enumerate(scheduler_pipe_readers):
        #     try:
        #         data = reader.recv()
        #     except EOFError:
        #         logger.error(f"Rank {i} scheduler is dead. Please check if there are relevant logs.")
        #         processes[i].join()
        #         logger.error(f"Exit code: {processes[i].exitcode}")
        #         raise

        #     if data["status"] != "ready":
        #         raise RuntimeError("Initialization failed. Please see the error messages above.")

        #     if i == 0:
        #         result_handle = data.get("result_handle")

        #     scheduler_infos.append(data)
        #     reader.close()
        # scheduler.initialize_result_queue(result_handle)
        # logger.debug("All workers are ready")
        # #self._wait_for_workers_ready()
        # logger.info("DiffusionWorkerActor initialization complete")
        # return True

    def add_req(self, requests: list[OmniDiffusionRequest]):
        return self.worker.generate(requests)

    def _wait_for_workers_ready(self):
        """Wait until all workers report ready."""
        logger.info("Waiting for workers to be ready...")
        timeout = 300
        start = time.time()
        ready_count = 0
        while ready_count < self.scheduler.num_workers:
            if time.time() - start > timeout:
                raise TimeoutError("Workers did not initialize within timeout")
            msg = self.scheduler.result_mq.dequeue(timeout=1.0)
            if msg and msg.get("status") == "ready":
                ready_count += 1
                logger.info(f"Worker {ready_count}/{self.scheduler.num_workers} ready")
        logger.info("All workers ready")

    # --------------------------------------------------------------------- #
    # RPC and Model Execution
    # --------------------------------------------------------------------- #
    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ):
        kwargs = kwargs or {}
        message = {"type": "rpc", "method": method, "args": args, "kwargs": kwargs}

        for _ in range(self.scheduler.num_workers):
            self.scheduler.mq.enqueue(message)

        if unique_reply_rank is not None:
            return self.scheduler.result_mq.dequeue(timeout=30.0)

        responses = []
        for _ in range(self.scheduler.num_workers):
            responses.append(self.scheduler.result_mq.dequeue(timeout=30.0))
        return responses

    def execute_model(self, requests: list[dict]):
        """Execute model inference on workers."""
        from vllm_omni.diffusion.request import OmniDiffusionRequest

        logger.info(f"Request is ======================================================== {requests}")
        # Convert dict representations back to OmniDiffusionRequest objects
        # because gpu_worker.execute_model expects OmniDiffusionRequest objects
        omni_requests = []
        for req_dict in requests:
            # Reconstruct OmniDiffusionRequest from dict
            omni_req = OmniDiffusionRequest(
                prompt=req_dict.get("prompt"),
                height=req_dict.get("height"),
                width=req_dict.get("width"),
                num_inference_steps=req_dict.get("num_inference_steps"),
                num_outputs_per_prompt=req_dict.get("num_outputs_per_prompt"),
                request_id=req_dict.get("request_id"),
                seed=req_dict.get("seed"),
                # Add other fields as needed
            )
            omni_requests.append(omni_req)

        # Send OmniDiffusionRequest objects directly to workers
        for _ in range(self.scheduler.num_workers):
            self.scheduler.mq.enqueue(omni_requests)

        results = []
        for _ in range(self.scheduler.num_workers):
            res = self.scheduler.result_mq.dequeue(timeout=300.0)
            if res:
                results.append(res)

        return results[0] if results else None

    # --------------------------------------------------------------------- #
    # Health Check / Shutdown
    # --------------------------------------------------------------------- #
    def check_health(self):
        """Check all worker heartbeats."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")

        message = {"type": "health_check"}
        for _ in range(self.scheduler.num_workers):
            self.scheduler.mq.enqueue(message)

        healthy = 0
        for _ in range(self.scheduler.num_workers):
            res = self.scheduler.result_mq.dequeue(timeout=5.0)
            if res and res.get("status") == "ok":
                healthy += 1

        if healthy != self.scheduler.num_workers:
            raise RuntimeError(f"Only {healthy}/{self.scheduler.num_workers} workers healthy")

        return True

    def shutdown(self):
        """Terminate worker processes and release scheduler."""
        from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE

        logger.info("Shutting down DiffusionWorkerActor...")
        if self.scheduler:
            for _ in range(self.scheduler.num_workers):
                self.scheduler.mq.enqueue(SHUTDOWN_MESSAGE)

            for p in self.worker_processes:
                p.join(timeout=30)
                if p.is_alive():
                    logger.warning(f"Terminating {p.name}")
                    p.terminate()

            self.scheduler.close()
        logger.info("DiffusionWorkerActor shutdown complete")
        return True
