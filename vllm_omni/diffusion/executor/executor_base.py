# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


class DiffusionExecutor(ABC):
    """Abstract base class for diffusion executors.

    An executor is responsible for executing the diffusion model on one or more devices.
    It abstracts worker and scheduler initialization and manages all communication
    between the engine and workers.
    """

    @staticmethod
    def get_class(od_config: OmniDiffusionConfig) -> type["DiffusionExecutor"]:
        """Get the appropriate executor class based on configuration.

        Args:
            od_config: The diffusion configuration.

        Returns:
            The executor class to use.
        """
        executor_class: type[DiffusionExecutor]
        distributed_executor_backend = od_config.distributed_executor_backend

        # Handle custom executor class
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, DiffusionExecutor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"DiffusionExecutor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "mp":
            from vllm_omni.diffusion.executor.multiproc_executor import MultiProcDiffusionExecutor

            executor_class = MultiProcDiffusionExecutor
        elif distributed_executor_backend == "external_launcher":
            from vllm_omni.diffusion.executor.external_executor import ExternalDiffusionExecutor

            executor_class = ExternalDiffusionExecutor
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, DiffusionExecutor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"DiffusionExecutor. Got {executor_class}."
                )
        else:
            raise ValueError(f"Unknown distributed executor backend: {distributed_executor_backend}")

        return executor_class

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        """Initialize the executor.

        Args:
            od_config: The diffusion configuration.
        """
        self.od_config = od_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        """Initialize the executor-specific components.

        This should handle:
        - Scheduler initialization
        - Worker process creation and initialization
        - Communication channels setup
        """
        raise NotImplementedError

    @abstractmethod
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
            method: The method name (str) or callable to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results

        Note:
            This method should handle all communication with workers, including
            broadcasting the request and collecting responses.
        """
        raise NotImplementedError

    @abstractmethod
    def add_requests(self, requests: list) -> Any:
        """Add diffusion requests to be processed.

        Args:
            requests: List of OmniDiffusionRequest objects

        Returns:
            DiffusionOutput from processing the requests
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources.

        This should:
        - Send shutdown signals to all workers
        - Wait for workers to terminate gracefully
        - Clean up communication channels
        - Release any other resources
        """
        raise NotImplementedError

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """Register a function to be called if the executor enters a permanent failed state.

        Args:
            callback: Function to call on failure
        """
        pass

    @abstractmethod
    def check_health(self) -> None:
        """Check if the executor is healthy.

        Raises:
            RuntimeError: If the executor is not healthy
        """
        raise NotImplementedError
