# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for GPUWorker class.

This module tests the GPUWorker implementation:
- _maybe_get_memory_pool_context: context manager for memory pool allocation
- sleep: save buffers and offload memory
- wake_up: restore buffers and reload memory
- _sleep_saved_buffers: buffer management during sleep mode
"""

from contextlib import nullcontext
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.worker.gpu_worker import GPUWorker


class TestGPUWorkerMemoryManagement:
    """Test GPUWorker memory management functionality."""

    @pytest.fixture
    def mock_od_config(self):
        """Create a mock OmniDiffusionConfig."""
        config = Mock(spec=OmniDiffusionConfig)
        config.num_gpus = 1
        config.master_port = 12345
        config.enable_sleep_mode = False
        config.cache_backend = None
        config.cache_config = None
        return config

    @pytest.fixture
    def mock_worker_no_init(self, mock_od_config):
        """Create a GPUWorker without calling init_device_and_model."""
        with patch.object(GPUWorker, 'init_device_and_model'):
            worker = GPUWorker(
                local_rank=0,
                rank=0,
                od_config=mock_od_config,
            )
            worker.pipeline = Mock()
            return worker

    def test_sleep_saved_buffers_initialization(self, mock_od_config):
        """Test that _sleep_saved_buffers is initialized as empty dict."""
        with patch.object(GPUWorker, 'init_device_and_model'):
            worker = GPUWorker(
                local_rank=0,
                rank=0,
                od_config=mock_od_config,
            )
            assert hasattr(worker, '_sleep_saved_buffers')
            assert isinstance(worker._sleep_saved_buffers, dict)
            assert len(worker._sleep_saved_buffers) == 0

    def test_maybe_get_memory_pool_context_sleep_mode_disabled(self, mock_worker_no_init):
        """Test _maybe_get_memory_pool_context returns nullcontext when sleep mode is disabled."""
        mock_worker_no_init.od_config.enable_sleep_mode = False
        
        context = mock_worker_no_init._maybe_get_memory_pool_context(tag="weights")
        
        # nullcontext doesn't have a specific type, but we can check behavior
        assert context is not None
        # Verify it's a context manager
        assert hasattr(context, '__enter__')
        assert hasattr(context, '__exit__')

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    def test_maybe_get_memory_pool_context_sleep_mode_enabled(self, mock_allocator_class, mock_worker_no_init):
        """Test _maybe_get_memory_pool_context returns memory pool context when sleep mode is enabled."""
        mock_worker_no_init.od_config.enable_sleep_mode = True
        
        # Mock the allocator instance and its methods
        mock_allocator = Mock()
        mock_allocator.get_current_usage.return_value = 0
        mock_allocator.use_memory_pool.return_value = Mock()
        mock_allocator_class.get_instance.return_value = mock_allocator
        
        context = mock_worker_no_init._maybe_get_memory_pool_context(tag="weights")
        
        # Verify allocator methods were called
        mock_allocator_class.get_instance.assert_called_once()
        mock_allocator.get_current_usage.assert_called_once()
        mock_allocator.use_memory_pool.assert_called_once_with(tag="weights")

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    def test_maybe_get_memory_pool_context_weights_tag_assertion(self, mock_allocator_class, mock_worker_no_init):
        """Test _maybe_get_memory_pool_context asserts current usage is 0 for weights tag."""
        mock_worker_no_init.od_config.enable_sleep_mode = True
        
        # Mock the allocator with non-zero current usage
        mock_allocator = Mock()
        mock_allocator.get_current_usage.return_value = 100
        mock_allocator_class.get_instance.return_value = mock_allocator
        
        with pytest.raises(AssertionError, match="Sleep mode can only be used for one instance per process"):
            mock_worker_no_init._maybe_get_memory_pool_context(tag="weights")

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    @patch('vllm_omni.diffusion.worker.gpu_worker.torch')
    def test_sleep_level_1(self, mock_torch, mock_allocator_class, mock_worker_no_init):
        """Test sleep method with level 1 (offload weights only)."""
        # Mock CUDA memory info
        mock_torch.cuda.mem_get_info.return_value = (1000, 2000)  # (free, total)
        
        # Mock allocator
        mock_allocator = Mock()
        mock_allocator_class.get_instance.return_value = mock_allocator
        
        result = mock_worker_no_init.sleep(level=1)
        
        assert result is True
        # Verify allocator.sleep was called with correct tags
        mock_allocator.sleep.assert_called_once_with(offload_tags=("weights",))
        # Verify buffers were not saved for level 1
        assert len(mock_worker_no_init._sleep_saved_buffers) == 0

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    @patch('vllm_omni.diffusion.worker.gpu_worker.torch')
    def test_sleep_level_2_saves_buffers(self, mock_torch, mock_allocator_class, mock_worker_no_init):
        """Test sleep method with level 2 saves model buffers."""
        # Mock CUDA memory info
        mock_torch.cuda.mem_get_info.return_value = (1000, 2000)
        
        # Mock allocator
        mock_allocator = Mock()
        mock_allocator_class.get_instance.return_value = mock_allocator
        
        # Mock pipeline with buffers
        mock_buffer1 = Mock(spec=torch.Tensor)
        mock_buffer1_cpu = Mock(spec=torch.Tensor)
        mock_buffer1.cpu.return_value.clone.return_value = mock_buffer1_cpu
        
        mock_buffer2 = Mock(spec=torch.Tensor)
        mock_buffer2_cpu = Mock(spec=torch.Tensor)
        mock_buffer2.cpu.return_value.clone.return_value = mock_buffer2_cpu
        
        mock_worker_no_init.pipeline.named_buffers.return_value = [
            ("buffer1", mock_buffer1),
            ("buffer2", mock_buffer2),
        ]
        
        result = mock_worker_no_init.sleep(level=2)
        
        assert result is True
        # Verify allocator.sleep was called with empty tags for level 2
        mock_allocator.sleep.assert_called_once_with(offload_tags=tuple())
        # Verify buffers were saved
        assert len(mock_worker_no_init._sleep_saved_buffers) == 2
        assert "buffer1" in mock_worker_no_init._sleep_saved_buffers
        assert "buffer2" in mock_worker_no_init._sleep_saved_buffers
        assert mock_worker_no_init._sleep_saved_buffers["buffer1"] == mock_buffer1_cpu
        assert mock_worker_no_init._sleep_saved_buffers["buffer2"] == mock_buffer2_cpu

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    @patch('vllm_omni.diffusion.worker.gpu_worker.torch')
    def test_sleep_memory_assertion(self, mock_torch, mock_allocator_class, mock_worker_no_init):
        """Test sleep method asserts that memory was freed."""
        # Mock CUDA memory info showing memory usage increased (invalid state)
        mock_torch.cuda.mem_get_info.side_effect = [
            (1000, 2000),  # Before sleep
            (500, 2000),   # After sleep (less free memory)
        ]
        
        # Mock allocator
        mock_allocator = Mock()
        mock_allocator_class.get_instance.return_value = mock_allocator
        
        with pytest.raises(AssertionError, match="Memory usage increased after sleeping"):
            mock_worker_no_init.sleep(level=1)

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    def test_wake_up_without_saved_buffers(self, mock_allocator_class, mock_worker_no_init):
        """Test wake_up method when no buffers were saved."""
        # Mock allocator
        mock_allocator = Mock()
        mock_allocator_class.get_instance.return_value = mock_allocator
        
        # Ensure no saved buffers
        mock_worker_no_init._sleep_saved_buffers = {}
        
        result = mock_worker_no_init.wake_up(tags=["weights"])
        
        assert result is True
        # Verify allocator.wake_up was called
        mock_allocator.wake_up.assert_called_once_with(["weights"])
        # Verify saved buffers remain empty
        assert len(mock_worker_no_init._sleep_saved_buffers) == 0

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    def test_wake_up_restores_buffers(self, mock_allocator_class, mock_worker_no_init):
        """Test wake_up method restores saved buffers."""
        # Mock allocator
        mock_allocator = Mock()
        mock_allocator_class.get_instance.return_value = mock_allocator
        
        # Mock saved buffers
        mock_saved_buffer1 = Mock(spec=torch.Tensor)
        mock_saved_buffer1.data = Mock()
        mock_saved_buffer2 = Mock(spec=torch.Tensor)
        mock_saved_buffer2.data = Mock()
        
        mock_worker_no_init._sleep_saved_buffers = {
            "buffer1": mock_saved_buffer1,
            "buffer2": mock_saved_buffer2,
        }
        
        # Mock pipeline buffers to restore to
        mock_buffer1 = Mock(spec=torch.Tensor)
        mock_buffer1.data = Mock()
        mock_buffer2 = Mock(spec=torch.Tensor)
        mock_buffer2.data = Mock()
        
        mock_worker_no_init.pipeline.named_buffers.return_value = [
            ("buffer1", mock_buffer1),
            ("buffer2", mock_buffer2),
        ]
        
        result = mock_worker_no_init.wake_up(tags=None)
        
        assert result is True
        # Verify allocator.wake_up was called
        mock_allocator.wake_up.assert_called_once_with(None)
        # Verify buffers were restored
        mock_buffer1.data.copy_.assert_called_once_with(mock_saved_buffer1.data)
        mock_buffer2.data.copy_.assert_called_once_with(mock_saved_buffer2.data)
        # Verify saved buffers were cleared
        assert len(mock_worker_no_init._sleep_saved_buffers) == 0

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    def test_wake_up_partial_buffer_restore(self, mock_allocator_class, mock_worker_no_init):
        """Test wake_up method only restores buffers that exist in saved state."""
        # Mock allocator
        mock_allocator = Mock()
        mock_allocator_class.get_instance.return_value = mock_allocator
        
        # Mock saved buffers (only buffer1)
        mock_saved_buffer1 = Mock(spec=torch.Tensor)
        mock_saved_buffer1.data = Mock()
        
        mock_worker_no_init._sleep_saved_buffers = {
            "buffer1": mock_saved_buffer1,
        }
        
        # Mock pipeline buffers (buffer1 and buffer2)
        mock_buffer1 = Mock(spec=torch.Tensor)
        mock_buffer1.data = Mock()
        mock_buffer2 = Mock(spec=torch.Tensor)
        mock_buffer2.data = Mock()
        
        mock_worker_no_init.pipeline.named_buffers.return_value = [
            ("buffer1", mock_buffer1),
            ("buffer2", mock_buffer2),
        ]
        
        result = mock_worker_no_init.wake_up(tags=None)
        
        assert result is True
        # Verify only buffer1 was restored
        mock_buffer1.data.copy_.assert_called_once_with(mock_saved_buffer1.data)
        mock_buffer2.data.copy_.assert_not_called()
        # Verify saved buffers were cleared
        assert len(mock_worker_no_init._sleep_saved_buffers) == 0


class TestGPUWorkerIntegration:
    """Test GPUWorker integration scenarios."""

    @pytest.fixture
    def mock_od_config_sleep_enabled(self):
        """Create a mock OmniDiffusionConfig with sleep mode enabled."""
        config = Mock(spec=OmniDiffusionConfig)
        config.num_gpus = 1
        config.master_port = 12345
        config.enable_sleep_mode = True
        config.cache_backend = None
        config.cache_config = None
        return config

    @patch('vllm_omni.diffusion.worker.gpu_worker.CuMemAllocator')
    @patch('vllm_omni.diffusion.worker.gpu_worker.torch')
    def test_sleep_wake_cycle(self, mock_torch, mock_allocator_class, mock_od_config_sleep_enabled):
        """Test complete sleep-wake cycle with buffer save and restore."""
        with patch.object(GPUWorker, 'init_device_and_model'):
            worker = GPUWorker(
                local_rank=0,
                rank=0,
                od_config=mock_od_config_sleep_enabled,
            )
            
            # Mock CUDA memory info
            mock_torch.cuda.mem_get_info.return_value = (1000, 2000)
            
            # Mock allocator
            mock_allocator = Mock()
            mock_allocator_class.get_instance.return_value = mock_allocator
            
            # Mock pipeline with buffers
            mock_buffer = Mock(spec=torch.Tensor)
            mock_buffer_cpu = Mock(spec=torch.Tensor)
            mock_buffer.cpu.return_value.clone.return_value = mock_buffer_cpu
            mock_buffer.data = Mock()
            mock_buffer_cpu.data = Mock()
            
            worker.pipeline = Mock()
            worker.pipeline.named_buffers.return_value = [("test_buffer", mock_buffer)]
            
            # Execute sleep
            sleep_result = worker.sleep(level=2)
            assert sleep_result is True
            assert len(worker._sleep_saved_buffers) == 1
            
            # Execute wake_up
            wake_result = worker.wake_up(tags=None)
            assert wake_result is True
            assert len(worker._sleep_saved_buffers) == 0
            
            # Verify buffer was restored
            mock_buffer.data.copy_.assert_called_once()
