# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        existing_context = getattr(self, "context", None)
        if existing_context is not None:
            logger.warning("SyncSchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.num_workers = od_config.num_gpus
        self.od_config = od_config

        # Initialize pipes for worker communication
        # request_pipes: scheduler sends requests to workers
        # result_pipes: scheduler receives results from workers
        self.request_pipes = []  # List of Connection objects to send to workers
        self.result_pipes = []   # List of Connection objects to receive from workers

    def initialize_pipes(self, request_pipes, result_pipes):
        """Initialize Pipe connections for communicating with workers.
        
        Args:
            request_pipes: List of Connection objects for sending requests to workers
            result_pipes: List of Connection objects for receiving results from workers
        """
        self.request_pipes = request_pipes
        self.result_pipes = result_pipes
        logger.info("Scheduler initialized Pipe connections")

    def get_pipe_endpoints(self):
        """Return pipe endpoints for workers.
        
        Returns:
            Tuple of (request_pipes, result_pipes) for workers to use
        """
        return self.request_pipes, self.result_pipes

    def add_req(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """Sends a request to the scheduler and waits for the response."""
        try:
            # Prepare RPC request for generation
            rpc_request = {
                "type": "rpc",
                "method": "generate",
                "args": (requests,),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": True,
            }
            # Send RPC request to all workers via their request pipes
            for pipe in self.request_pipes:
                pipe.send(rpc_request)
            
            # Wait for result from Rank 0 via result pipe
            if not self.result_pipes:
                raise RuntimeError("Result pipes not initialized")

            output = self.result_pipes[0].recv()
            return output
        except Exception as e:
            logger.error(f"Error communicating with workers: {e}")
            raise

    def close(self):
        """Closes the pipe connections."""
        if hasattr(self, "request_pipes"):
            for pipe in self.request_pipes:
                try:
                    pipe.close()
                except Exception:
                    pass
        if hasattr(self, "result_pipes"):
            for pipe in self.result_pipes:
                try:
                    pipe.close()
                except Exception:
                    pass
        self.request_pipes = []
        self.result_pipes = []
