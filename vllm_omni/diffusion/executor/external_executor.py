# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.executor_base import DiffusionExecutor

logger = init_logger(__name__)


class ExternalDiffusionExecutor(DiffusionExecutor):
    """External launcher executor for diffusion models.

    This executor connects to externally managed workers (e.g., Ray actors)
    instead of launching its own worker processes. It's designed for scenarios
    where workers are managed by an external orchestration system.

    The executor assumes workers are already initialized and ready to accept
    requests. It forwards all RPC calls to these external workers.
    """

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        """Initialize the external executor.

        Args:
            od_config: The diffusion configuration.
        """
        self.worker_handles: list[Any] = []
        self._closed = False
        super().__init__(od_config)

    def _init_executor(self) -> None:
        """Initialize the executor by connecting to external workers.

        This method discovers and connects to externally managed workers.
        Subclasses should override this to implement specific discovery mechanisms.
        """
        logger.info("Initializing ExternalDiffusionExecutor")

        # Discover and connect to external workers
        self._connect_to_workers()

        logger.info(
            f"ExternalDiffusionExecutor initialized with {len(self.worker_handles)} workers"
        )

    def _connect_to_workers(self) -> None:
        """Connect to externally managed workers.

        This is a hook for subclasses to implement worker discovery and connection.
        The default implementation raises NotImplementedError.

        Subclasses should:
        1. Discover available workers (e.g., via service discovery, config, etc.)
        2. Establish connections to workers
        3. Store worker handles in self.worker_handles
        """
        raise NotImplementedError(
            "ExternalDiffusionExecutor._connect_to_workers must be implemented by subclass. "
            "Override this method to connect to your external workers."
        )

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on external worker processes and get results.

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
            raise RuntimeError("ExternalDiffusionExecutor is closed.")

        if not self.worker_handles:
            raise RuntimeError("No external workers connected.")

        kwargs = kwargs or {}

        # Ensure method is a string for serialization
        if not isinstance(method, str):
            raise ValueError(
                "ExternalDiffusionExecutor only supports string method names, "
                f"got {type(method)}"
            )

        # Forward to external workers
        return self._forward_rpc_to_workers(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )

    def _forward_rpc_to_workers(
        self,
        method: str,
        timeout: float | None,
        args: tuple,
        kwargs: dict,
        unique_reply_rank: int | None,
    ) -> Any:
        """Forward RPC call to external workers.

        This is a hook for subclasses to implement the actual RPC forwarding logic.

        Args:
            method: Method name to call
            timeout: Optional timeout
            args: Positional arguments
            kwargs: Keyword arguments
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Results from workers
        """
        raise NotImplementedError(
            "ExternalDiffusionExecutor._forward_rpc_to_workers must be implemented by subclass. "
            "Override this method to forward RPC calls to your external workers."
        )

    def add_requests(self, requests: list) -> Any:
        """Add diffusion requests to be processed.

        Args:
            requests: List of OmniDiffusionRequest objects

        Returns:
            DiffusionOutput from processing the requests
        """
        if self._closed:
            raise RuntimeError("ExternalDiffusionExecutor is closed.")

        # Forward generate request to workers
        # The first worker (rank 0) typically returns the result
        return self.collective_rpc(
            method="generate",
            args=(requests,),
            unique_reply_rank=0,
        )

    def check_health(self) -> None:
        """Check if the executor is healthy.

        Raises:
            RuntimeError: If any worker is not healthy
        """
        if self._closed:
            raise RuntimeError("ExternalDiffusionExecutor is closed.")

        # Verify all workers are reachable
        self._check_worker_health()

    def _check_worker_health(self) -> None:
        """Check health of external workers.

        This is a hook for subclasses to implement health checking logic.
        """
        # Default: assume workers are healthy if they're connected
        if not self.worker_handles:
            raise RuntimeError("No external workers connected.")

    def shutdown(self) -> None:
        """Shutdown the executor.

        Note: This does NOT terminate external workers, it only disconnects from them.
        External workers should be managed by their own lifecycle management system.
        """
        if self._closed:
            return

        logger.info("Shutting down ExternalDiffusionExecutor")
        self._closed = True

        # Disconnect from workers but don't terminate them
        self._disconnect_from_workers()

        logger.info("ExternalDiffusionExecutor shutdown complete")

    def _disconnect_from_workers(self) -> None:
        """Disconnect from external workers.

        This is a hook for subclasses to implement cleanup logic.
        Workers should continue running after disconnect.
        """
        # Clear handles
        self.worker_handles = []
