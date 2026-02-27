# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import queue

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        existing_queues = getattr(self, "broadcast_queues", None)
        if existing_queues is not None:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config

        # One mp.Queue per worker to simulate broadcast semantics.
        # The scheduler enqueues the same message into every queue so that
        # each worker receives a copy.
        self.broadcast_queues: list[mp.Queue] = [mp.Queue() for _ in range(self.num_workers)]

        # Single result queue: only rank-0 worker writes results back.
        self.result_queue: mp.Queue = mp.Queue()

    def get_broadcast_queues(self) -> list[mp.Queue]:
        """Return the list of per-worker broadcast queues."""
        return self.broadcast_queues

    def get_result_queue(self) -> mp.Queue:
        """Return the shared result queue."""
        return self.result_queue

    def broadcast(self, message) -> None:
        """Put *message* into every worker's broadcast queue."""
        for q in self.broadcast_queues:
            q.put(message)

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Sends a request to the scheduler and waits for the response."""
        # Prepare RPC request for generation
        rpc_request = {
            "type": "rpc",
            "method": "generate",
            "args": (request,),
            "kwargs": {},
            "output_rank": 0,
            "exec_all_ranks": True,
        }

        # Broadcast RPC request to all workers
        self.broadcast(rpc_request)

        # Wait for result from Rank 0
        output = self.result_queue.get()  # blocks until available
        if isinstance(output, dict) and output.get("status") == "error":
            raise RuntimeError("worker error")
        return output

    def close(self):
        """Closes all queues."""
        for q in getattr(self, "broadcast_queues", []):
            q.close()
            q.join_thread()
        self.broadcast_queues = []
        if hasattr(self, "result_queue") and self.result_queue is not None:
            self.result_queue.close()
            self.result_queue.join_thread()
            self.result_queue = None
