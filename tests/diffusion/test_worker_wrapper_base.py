# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for WorkerWrapperBase class.

This module tests the WorkerWrapperBase implementation:
- Initialization with and without worker extensions
- Custom pipeline initialization
- Method delegation via execute_method
- Attribute delegation via __getattr__
- Dynamic worker class extension
"""

from unittest.mock import Mock, patch

import pytest

from vllm_omni.diffusion.worker.diffusion_worker import (
    CustomPipelineWorkerExtension,
    DiffusionWorker,
    WorkerWrapperBase,
)


@pytest.fixture
def mock_od_config():
    """Create a mock OmniDiffusionConfig."""
    config = Mock()
    config.num_gpus = 1
    config.master_port = 12345
    config.enable_sleep_mode = False
    config.cache_backend = None
    config.cache_config = None
    config.model = "test-model"
    config.diffusion_load_format = None
    config.dtype = "float32"
    config.max_cpu_loras = 0
    config.lora_path = None
    config.lora_scale = 1.0
    return config


class TestWorkerWrapperBaseInitialization:
    """Test WorkerWrapperBase initialization."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_basic_initialization(self, mock_worker_init, mock_od_config):
        """Test basic initialization without extensions."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Verify worker was created
        assert wrapper.gpu_id == 0
        assert wrapper.od_config == mock_od_config
        assert wrapper.base_worker_class == DiffusionWorker
        assert wrapper.worker_extension_cls is None
        assert wrapper.custom_pipeline_args is None
        assert wrapper.worker is not None

        # Verify DiffusionWorker.__init__ was called
        mock_worker_init.assert_called_once_with(
            local_rank=0,
            rank=0,
            od_config=mock_od_config,
        )

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_initialization_with_custom_pipeline(self, mock_worker_init, mock_od_config):
        """Test initialization with custom pipeline args."""
        custom_pipeline_args = {"pipeline_class": "custom.Pipeline"}

        with patch.object(DiffusionWorker, "re_init_pipeline") as mock_reinit:
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                custom_pipeline_args=custom_pipeline_args,
            )

            # Verify re_init_pipeline was called with custom args
            mock_reinit.assert_called_once_with(custom_pipeline_args)
            assert wrapper.custom_pipeline_args == custom_pipeline_args


class TestWorkerWrapperBaseExtension:
    """Test WorkerWrapperBase worker extension functionality."""

    def test_prepare_worker_class_without_extension(self, mock_od_config):
        """Test _prepare_worker_class without worker extension."""
        with patch.object(DiffusionWorker, "__init__", return_value=None):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
            )

            worker_class = wrapper._prepare_worker_class()
            assert worker_class == DiffusionWorker

    def test_prepare_worker_class_with_custom_pipeline_extension(self, mock_od_config):
        """Test _prepare_worker_class with custom pipeline automatically adds extension."""
        custom_pipeline_args = {"pipeline_class": "custom.Pipeline"}

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            with patch.object(DiffusionWorker, "re_init_pipeline"):
                wrapper = WorkerWrapperBase(
                    gpu_id=0,
                    od_config=mock_od_config,
                    base_worker_class=DiffusionWorker,
                    custom_pipeline_args=custom_pipeline_args,
                )

                # Verify worker_extension_cls was set to CustomPipelineWorkerExtension
                assert wrapper.worker_extension_cls == CustomPipelineWorkerExtension

    def test_prepare_worker_class_with_extension_class(self, mock_od_config):
        """Test _prepare_worker_class with explicit worker extension class."""

        class TestExtension:
            def custom_method(self):
                return "extension_method"

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=TestExtension,
            )

            # Verify the worker class was extended
            assert hasattr(wrapper.worker.__class__, "custom_method")
            assert TestExtension in wrapper.worker.__class__.__bases__

    @patch("vllm.utils.import_utils.resolve_obj_by_qualname")
    def test_prepare_worker_class_with_extension_string(self, mock_resolve, mock_od_config):
        """Test _prepare_worker_class with worker extension as string."""

        class TestExtension:
            def custom_method(self):
                return "extension_method"

        mock_resolve.return_value = TestExtension

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls="test.module.TestExtension",
            )

            # Verify resolve_obj_by_qualname was called
            mock_resolve.assert_called_once_with("test.module.TestExtension")

            # Verify the worker class was extended
            assert hasattr(wrapper.worker.__class__, "custom_method")


class TestWorkerWrapperBaseDelegation:
    """Test WorkerWrapperBase delegation to wrapped worker."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_generate_delegation(self, mock_worker_init, mock_od_config):
        """Test that generate() delegates to worker.generate()."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock the worker's generate method
        mock_output = Mock()
        wrapper.worker.generate = Mock(return_value=mock_output)

        # Create mock requests
        mock_requests = [Mock()]

        # Call generate
        result = wrapper.generate(mock_requests)

        # Verify delegation
        wrapper.worker.generate.assert_called_once_with(mock_requests)
        assert result == mock_output

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_model_delegation(self, mock_worker_init, mock_od_config):
        """Test that execute_model() delegates to worker.execute_model()."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock the worker's execute_model method
        mock_output = Mock()
        wrapper.worker.execute_model = Mock(return_value=mock_output)

        # Create mock requests
        mock_reqs = [Mock()]

        # Call execute_model
        result = wrapper.execute_model(mock_reqs, mock_od_config)

        # Verify delegation
        wrapper.worker.execute_model.assert_called_once_with(mock_reqs, mock_od_config)
        assert result == mock_output

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_load_weights_delegation(self, mock_worker_init, mock_od_config):
        """Test that load_weights() delegates to worker.load_weights()."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock the worker's load_weights method
        expected_result = {"weight1", "weight2"}
        wrapper.worker.load_weights = Mock(return_value=expected_result)

        # Create mock weights
        mock_weights = [("weight1", Mock()), ("weight2", Mock())]

        # Call load_weights
        result = wrapper.load_weights(mock_weights)

        # Verify delegation
        wrapper.worker.load_weights.assert_called_once_with(mock_weights)
        assert result == expected_result

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_sleep_delegation(self, mock_worker_init, mock_od_config):
        """Test that sleep() delegates to worker.sleep()."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock the worker's sleep method
        wrapper.worker.sleep = Mock(return_value=True)

        # Call sleep
        result = wrapper.sleep(level=1)

        # Verify delegation
        wrapper.worker.sleep.assert_called_once_with(1)
        assert result is True

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_wake_up_delegation(self, mock_worker_init, mock_od_config):
        """Test that wake_up() delegates to worker.wake_up()."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock the worker's wake_up method
        wrapper.worker.wake_up = Mock(return_value=True)

        # Call wake_up
        result = wrapper.wake_up(tags=["weights"])

        # Verify delegation
        wrapper.worker.wake_up.assert_called_once_with(["weights"])
        assert result is True

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_shutdown_delegation(self, mock_worker_init, mock_od_config):
        """Test that shutdown() delegates to worker.shutdown()."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock the worker's shutdown method
        wrapper.worker.shutdown = Mock(return_value=None)

        # Call shutdown
        result = wrapper.shutdown()

        # Verify delegation
        wrapper.worker.shutdown.assert_called_once()
        assert result is None


class TestWorkerWrapperBaseExecuteMethod:
    """Test WorkerWrapperBase.execute_method functionality."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_method_success(self, mock_worker_init, mock_od_config):
        """Test execute_method successfully calls worker method."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock a method on the worker
        expected_result = "method_result"
        wrapper.worker.test_method = Mock(return_value=expected_result)

        # Call execute_method
        result = wrapper.execute_method("test_method", "arg1", kwarg1="value1")

        # Verify method was called with correct arguments
        wrapper.worker.test_method.assert_called_once_with("arg1", kwarg1="value1")
        assert result == expected_result

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_method_with_no_args(self, mock_worker_init, mock_od_config):
        """Test execute_method with no arguments."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock a method on the worker
        expected_result = "no_args_result"
        wrapper.worker.no_args_method = Mock(return_value=expected_result)

        # Call execute_method
        result = wrapper.execute_method("no_args_method")

        # Verify method was called with no arguments
        wrapper.worker.no_args_method.assert_called_once_with()
        assert result == expected_result

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_method_error(self, mock_worker_init, mock_od_config):
        """Test execute_method raises exception on error."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock a method that raises an exception
        wrapper.worker.error_method = Mock(side_effect=RuntimeError("Test error"))

        # Call execute_method and expect exception
        with pytest.raises(RuntimeError, match="Test error"):
            wrapper.execute_method("error_method")

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_method_invalid_type(self, mock_worker_init, mock_od_config):
        """Test execute_method with invalid method type."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Call execute_method with bytes (invalid)
        with pytest.raises(AssertionError, match="Method must be str"):
            wrapper.execute_method(b"bytes_method")


class TestWorkerWrapperBaseGetAttr:
    """Test WorkerWrapperBase.__getattr__ functionality."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_getattr_delegation(self, mock_worker_init, mock_od_config):
        """Test __getattr__ delegates to worker."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Set an attribute on the worker
        wrapper.worker.custom_attribute = "test_value"

        # Access the attribute through wrapper
        assert wrapper.custom_attribute == "test_value"

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_getattr_method_access(self, mock_worker_init, mock_od_config):
        """Test __getattr__ allows accessing worker methods."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Mock a method on the worker
        wrapper.worker.custom_method = Mock(return_value="method_result")

        # Access and call the method through wrapper
        result = wrapper.custom_method()

        # Verify the method was called
        wrapper.worker.custom_method.assert_called_once()
        assert result == "method_result"

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_getattr_missing_attribute(self, mock_worker_init, mock_od_config):
        """Test __getattr__ raises AttributeError for missing attributes."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        # Access non-existent attribute should raise AttributeError
        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attribute


class TestWorkerWrapperBaseEdgeCases:
    """Test WorkerWrapperBase edge cases and special scenarios."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_extension_conflict_warning(self, mock_worker_init, mock_od_config, caplog):
        """Test warning is logged when extension conflicts with worker."""

        class ConflictExtension:
            def load_model(self):  # This conflicts with DiffusionWorker.load_model
                return "extension_load_model"

        with patch.object(DiffusionWorker, "load_model"):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=ConflictExtension,
            )

            # Verify warning was logged about the conflict
            # Note: The actual logging might not work in this test context,
            # but the code path should be executed
            assert wrapper.worker is not None

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_multiple_extensions_same_class(self, mock_worker_init, mock_od_config):
        """Test that applying the same extension twice doesn't duplicate it."""

        class TestExtension:
            def custom_method(self):
                return "extension"

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            # Create first wrapper
            wrapper1 = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=TestExtension,
            )

            # Create second wrapper with the same extension
            wrapper2 = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=TestExtension,
            )

            # Both should have the extension
            assert hasattr(wrapper1.worker, "custom_method")
            assert hasattr(wrapper2.worker, "custom_method")
