# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""External executor for diffusion models.

This module provides an example implementation of a custom executor that can
forward calls to external systems (e.g., remote servers, cloud services, or
custom orchestration systems). Users can extend this class to implement their
own execution strategies.
"""

import time
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor.diffusion_executor import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


class ExternalDiffusionExecutor(DiffusionExecutor):
    """External executor that forwards calls to an external system.

    This is an example/template executor that demonstrates how to implement
    a custom executor that forwards execution to external systems. Users can
    extend this class to implement their own custom execution strategies.

    Example use cases:
        - Forward to a remote inference server
        - Integrate with a custom orchestration system
        - Use a different distributed backend (e.g., MPI, Horovod)
        - Implement custom load balancing or routing logic
        - Forward to cloud-based inference services

    To use a custom executor, specify it in the OmniDiffusionConfig:
        ```python
        config = OmniDiffusionConfig(
            executor_class="path.to.MyCustomExecutor",
            # ... other config
        )
        ```

    Abstract methods to implement:
        - _init_executor(): Set up your external system connection
        - collective_rpc(): Forward RPC calls to your external system
        - check_health(): Check if your external system is healthy
        - shutdown(): Clean up external system resources
    """

    uses_ray: bool = False
    supports_pp: bool = False

    def _init_executor(self) -> None:
        """Initialize connection to external system.

        Example implementation - replace with your actual initialization logic.

        You might:
        - Connect to a remote server
        - Initialize a client library
        - Set up authentication
        - Establish communication channels
        """
        logger.info("Initializing external diffusion executor...")

        # Example: Initialize your external system here
        # self.client = YourExternalClient(
        #     host=self.od_config.external_host,
        #     port=self.od_config.external_port,
        # )

        # For this example, we'll just log
        logger.warning(
            "ExternalDiffusionExecutor is a template. "
            "Please implement your own initialization logic."
        )

        # Store any necessary state
        self._external_system_ready = False

        # Example: You might want to warm up your external system
        # self._warmup_external_system()

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Forward RPC call to external system.

        Args:
            method: The method name to execute.
            timeout: Optional timeout in seconds.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.
            unique_reply_rank: If set, only get reply from this rank.

        Returns:
            Result(s) from the external system.

        Raises:
            RuntimeError: If executor is closed or RPC fails.
            TimeoutError: If RPC call times out.
            NotImplementedError: This is a template implementation.
        """
        if self._closed:
            raise RuntimeError("Executor is closed.")

        kwargs = kwargs or {}

        # Example implementation - replace with your actual RPC logic
        logger.info(
            f"Forwarding RPC call: method={method}, "
            f"args={args}, kwargs={kwargs}, "
            f"unique_reply_rank={unique_reply_rank}"
        )

        # Example: Forward to your external system
        # result = self.client.call_remote_method(
        #     method=method,
        #     args=args,
        #     kwargs=kwargs,
        #     timeout=timeout,
        #     rank=unique_reply_rank,
        # )
        # return result

        raise NotImplementedError(
            "ExternalDiffusionExecutor is a template. "
            "Please implement your own RPC forwarding logic."
        )

    def execute_model(
        self, requests: list[OmniDiffusionRequest]
    ) -> DiffusionOutput | None:
        """Execute the diffusion model via external system.

        This is a convenience method that you can override if your external
        system has a specialized inference API that differs from the standard
        RPC interface.

        Args:
            requests: List of diffusion requests to process.

        Returns:
            DiffusionOutput containing the generated results, or None on error.
        """
        if self._closed:
            raise RuntimeError("Executor is closed.")

        try:
            # Option 1: Use the default RPC-based implementation
            # return super().execute_model(requests)

            # Option 2: Implement custom logic for your external system
            logger.info(f"Executing {len(requests)} requests on external system")

            # Example: Call your external system's inference API
            # result = self.client.generate(requests)
            # return result

            raise NotImplementedError(
                "ExternalDiffusionExecutor is a template. "
                "Please implement your own model execution logic."
            )

        except Exception as e:
            logger.error(f"External model execution failed: {e}")
            raise

    def check_health(self) -> None:
        """Check if the external system is healthy.

        Raises:
            RuntimeError: If the external system is unhealthy.
        """
        if self._closed:
            raise RuntimeError("Executor is closed.")

        # Example: Check your external system's health
        # if not self.client.is_healthy():
        #     raise RuntimeError("External system is unhealthy")

        logger.info("Health check: External system status unknown (template)")

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources.

        This method should:
        - Close connections to external systems
        - Release any allocated resources
        - Clean up temporary state
        """
        if self._closed:
            return

        logger.info("Shutting down external diffusion executor...")

        # Example: Clean up your external system connection
        # try:
        #     self.client.disconnect()
        # except Exception as e:
        #     logger.warning(f"Error during shutdown: {e}")

        logger.info("External diffusion executor shutdown complete.")


class HTTPDiffusionExecutor(ExternalDiffusionExecutor):
    """Example HTTP-based executor for remote inference.

    This is a more concrete example showing how to implement an executor
    that forwards requests to a remote HTTP inference server.
    """

    def _init_executor(self) -> None:
        """Initialize HTTP client for remote inference."""
        logger.info("Initializing HTTP diffusion executor...")

        # Example: You would initialize your HTTP client here
        # import requests
        # self.session = requests.Session()
        # self.base_url = self.od_config.inference_url or "http://localhost:8000"

        # For demonstration purposes only
        self._external_system_ready = True
        logger.info("HTTP executor initialized (template)")

    def execute_model(
        self, requests: list[OmniDiffusionRequest]
    ) -> DiffusionOutput | None:
        """Execute model via HTTP API.

        Example implementation showing how to forward requests to an HTTP endpoint.
        """
        if self._closed:
            raise RuntimeError("Executor is closed.")

        # Example HTTP request flow:
        # 1. Serialize requests to JSON
        # payload = {
        #     "requests": [req.to_dict() for req in requests],
        #     "config": self.od_config.to_dict(),
        # }
        #
        # 2. Send HTTP POST request
        # response = self.session.post(
        #     f"{self.base_url}/v1/diffusion/generate",
        #     json=payload,
        #     timeout=300,  # 5 minutes
        # )
        #
        # 3. Parse response
        # if response.status_code == 200:
        #     result_data = response.json()
        #     return DiffusionOutput.from_dict(result_data)
        # else:
        #     raise RuntimeError(f"HTTP error: {response.status_code}")

        raise NotImplementedError(
            "HTTPDiffusionExecutor is a template. "
            "Implement actual HTTP client logic."
        )

    def shutdown(self) -> None:
        """Close HTTP session."""
        if self._closed:
            return

        logger.info("Shutting down HTTP diffusion executor...")

        # Example: Close HTTP session
        # if hasattr(self, "session"):
        #     self.session.close()

        logger.info("HTTP executor shutdown complete.")
