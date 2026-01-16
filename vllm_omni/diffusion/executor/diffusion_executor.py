# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Abstract base class for diffusion model executors.

This module defines the abstract interface for executing diffusion models.
The executor serves as an abstraction layer between the DiffusionEngine and
the underlying worker processes, enabling different execution strategies
(local multi-process, Ray, external executors, etc.).
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

FailureCallback = Callable[[], None]


class DiffusionExecutor(ABC):
    """Abstract base class for diffusion model executors.

    An executor is responsible for executing the diffusion model on one or more devices.
    It provides an abstraction layer between the DiffusionEngine and the actual workers,
    allowing for different distributed execution strategies while maintaining a consistent
    interface.

    The executor handles:
    - Worker process management
    - Inter-process communication
    - RPC calls to workers
    - Resource cleanup
    - Health checking
    """

    uses_ray: bool = False  # Whether the executor uses Ray for orchestration
    supports_pp: bool = False  # Whether the executor supports pipeline parallelism

    @staticmethod
    def get_class(od_config: "OmniDiffusionConfig") -> type["DiffusionExecutor"]:
        """Get the executor class based on configuration.

        This method provides a centralized way to resolve executor classes with
        support for shorthand names, string paths, and direct class references.

        Args:
            od_config: Configuration object containing executor_class specification.

        Returns:
            The executor class to instantiate.

        Raises:
            TypeError: If executor_class is not a valid type or subclass.
            ValueError: If executor_class string cannot be resolved.

        Example:
            ```python
            config = OmniDiffusionConfig(executor_class="multiproc")
            executor_class = DiffusionExecutor.get_class(config)
            executor = executor_class(config)
            ```
        """
        from vllm.utils.import_utils import resolve_obj_by_qualname

        executor_class_spec = getattr(od_config, 'executor_class', None)

        # Default to MultiProcDiffusionExecutor if not specified
        if executor_class_spec is None:
            from vllm_omni.diffusion.executor.multiproc_executor import (
                MultiProcDiffusionExecutor,
            )
            return MultiProcDiffusionExecutor

        # If it's already a type, validate and return it
        if isinstance(executor_class_spec, type):
            if not issubclass(executor_class_spec, DiffusionExecutor):
                raise TypeError(
                    f"executor_class must be a subclass of DiffusionExecutor, "
                    f"got {executor_class_spec}"
                )
            return executor_class_spec

        # If it's a string, resolve it
        if isinstance(executor_class_spec, str):
            # Check for shorthand names first
            executor_class: type[DiffusionExecutor]
            
            if executor_class_spec == "multiproc" or executor_class_spec == "mp":
                from vllm_omni.diffusion.executor.multiproc_executor import (
                    MultiProcDiffusionExecutor,
                )
                executor_class = MultiProcDiffusionExecutor
            elif executor_class_spec == "external":
                from vllm_omni.diffusion.executor.external_executor import (
                    ExternalDiffusionExecutor,
                )
                executor_class = ExternalDiffusionExecutor
            elif executor_class_spec == "http":
                from vllm_omni.diffusion.executor.external_executor import (
                    HTTPDiffusionExecutor,
                )
                executor_class = HTTPDiffusionExecutor
            else:
                # Try to resolve as a fully qualified name
                try:
                    resolved_class = resolve_obj_by_qualname(executor_class_spec)
                    if not isinstance(resolved_class, type):
                        raise TypeError(
                            f"executor_class '{executor_class_spec}' resolved to "
                            f"{resolved_class}, which is not a class"
                        )
                    if not issubclass(resolved_class, DiffusionExecutor):
                        raise TypeError(
                            f"executor_class '{executor_class_spec}' must be a "
                            f"subclass of DiffusionExecutor, got {resolved_class}"
                        )
                    executor_class = resolved_class
                except Exception as e:
                    raise ValueError(
                        f"Failed to resolve executor_class '{executor_class_spec}': {e}"
                    ) from e
            
            return executor_class

        raise TypeError(
            f"executor_class must be a string or class type, "
            f"got {type(executor_class_spec)}"
        )

    def __init__(self, od_config: "OmniDiffusionConfig") -> None:
        """Initialize the executor with the given configuration.

        Args:
            od_config: Configuration for the diffusion model and execution environment.
        """
        self.od_config = od_config
        self._closed = False
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        """Initialize executor-specific resources.

        This method should set up any executor-specific resources such as:
        - Worker processes
        - Communication channels
        - Message queues
        - Distributed backends

        Raises:
            RuntimeError: If initialization fails.
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

        This is the core method that enables communication with workers. All calls
        from the engine to workers should go through this method.

        Args:
            method: The method name (str) or callable to execute on workers.
            timeout: Optional timeout in seconds. None means wait indefinitely.
            args: Positional arguments to pass to the method.
            kwargs: Keyword arguments to pass to the method.
            unique_reply_rank: If set, only get reply from this specific rank.
                If None, gather replies from all workers.

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
            from all workers.

        Raises:
            RuntimeError: If the executor is closed or RPC fails.
            TimeoutError: If the RPC call times out.

        Example:
            ```python
            # Call a method on all workers
            results = executor.collective_rpc("execute_model", args=(requests,))

            # Call a method on a specific worker
            result = executor.collective_rpc("get_cache_stats", unique_reply_rank=0)
            ```
        """
        raise NotImplementedError

    def execute_model(
        self, requests: list[OmniDiffusionRequest]
    ) -> DiffusionOutput | None:
        """Execute the diffusion model on the given requests.

        Args:
            requests: List of diffusion requests to process.

        Returns:
            DiffusionOutput containing the generated results, or None on error.

        Raises:
            RuntimeError: If execution fails.
        """
        if self._closed:
            raise RuntimeError("Executor is closed.")

        try:
            # Execute model on workers (typically rank 0 returns the result)
            results = self.collective_rpc(
                "generate", args=(requests,), unique_reply_rank=0
            )
            return results
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise

    def sleep(self, level: int = 1) -> None:
        """Put the executor into sleep mode to save resources.

        Args:
            level: Sleep level (1 = light sleep, 2 = deep sleep, etc.).
        """
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return

        time_before_sleep = time.perf_counter()
        results = self.collective_rpc("sleep", kwargs={"level": level})
        time_after_sleep = time.perf_counter()

        # Check if all workers successfully entered sleep mode
        if all(results):
            self.sleeping_tags = {"weights", "kv_cache"}
            self.is_sleeping = True
            logger.info(
                "It took %.6f seconds to fall asleep.",
                time_after_sleep - time_before_sleep,
            )
        else:
            logger.error("Failed to put all workers to sleep.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the executor from sleep mode.

        Args:
            tags: Specific tags to wake up. If None, wake up everything.
        """
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return

        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return

        time_before_wakeup = time.perf_counter()
        results = self.collective_rpc("wake_up", kwargs={"tags": tags})
        time_after_wakeup = time.perf_counter()

        # Check if all workers successfully woke up
        if all(results):
            logger.info(
                "It took %.6f seconds to wake up tags %s.",
                time_after_wakeup - time_before_wakeup,
                tags if tags is not None else self.sleeping_tags,
            )
            if tags:
                for tag in tags:
                    self.sleeping_tags.remove(tag)
            else:
                self.sleeping_tags.clear()

            if not self.sleeping_tags:
                self.is_sleeping = False
        else:
            logger.error("Failed to wake up all workers.")

    @abstractmethod
    def check_health(self) -> None:
        """Check if the executor and workers are healthy.

        Raises:
            RuntimeError: If the executor or any worker is unhealthy.
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the executor and clean up all resources.

        This method should:
        - Send shutdown signals to all workers
        - Wait for workers to terminate gracefully
        - Clean up communication channels
        - Release any other resources
        """
        raise NotImplementedError

    def register_failure_callback(self, callback: FailureCallback) -> None:  # noqa: B027
        """Register a callback to be called if the executor enters a failed state.

        Args:
            callback: Function to call on failure.
        """
        pass

    def close(self) -> None:
        """Close the executor (alias for shutdown)."""
        if not self._closed:
            self._closed = True
            self.shutdown()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False
