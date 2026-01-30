# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncOmni LoRA API methods."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.lora.request import LoRARequest


class MockOmniStage:
    """Mock OmniStage for testing."""

    def __init__(self, stage_id: int, stage_type: str = "llm"):
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.final_output = False
        self.final_output_type = None
        self.vllm_config = None
        self.tokenizer = None
        self.is_tracing_enabled = False
        self._collected_results = []
        self._rpc_results = {}

    def collective_rpc(self, method: str, args: tuple = (), timeout: float | None = None, kwargs: dict[str, Any] | None = None):
        """Mock collective_rpc method."""
        # Return mocked results based on method
        if method in self._rpc_results:
            return self._rpc_results[method]
        # Default returns
        if method == "add_lora":
            return True
        elif method == "remove_lora":
            return True
        elif method == "list_loras":
            return []
        elif method == "pin_lora":
            return True
        return None

    def set_rpc_result(self, method: str, result: Any):
        """Set mock result for a specific RPC method."""
        self._rpc_results[method] = result

    def submit(self, task: dict):
        """Mock submit method."""
        pass

    def try_collect(self):
        """Mock try_collect method."""
        if self._collected_results:
            return self._collected_results.pop(0)
        return None

    def stop_stage_worker(self):
        """Mock stop_stage_worker method."""
        pass


@pytest.fixture
def mock_async_omni():
    """Create a mock AsyncOmni instance with mock stages."""
    with patch.object(AsyncOmni, '__init__', lambda self, *args, **kwargs: None):
        engine = AsyncOmni()
        
        # Initialize required attributes
        engine.stage_list = [
            MockOmniStage(0, "llm"),
            MockOmniStage(1, "diffusion"),
        ]
        engine._stage_in_queues = []
        engine.request_states = {}
        engine.output_handler = None
        engine._weak_finalizer = MagicMock()
        engine._pause_cond = asyncio.Condition()
        engine._paused = False
        engine._is_sleeping = False
        
        yield engine


@pytest.mark.asyncio
class TestAsyncOmniLoRA:
    """Test suite for AsyncOmni LoRA API methods."""

    async def test_add_lora_success(self, mock_async_omni):
        """Test adding a LoRA adapter successfully."""
        lora_request = LoRARequest(
            lora_name="test_lora",
            lora_int_id=1,
            lora_path="/path/to/lora",
        )
        lora_scale = 0.8
        
        # Set mock return value
        mock_async_omni.stage_list[0].set_rpc_result("add_lora", True)
        
        result = await mock_async_omni.add_lora(lora_request, lora_scale)
        
        assert result is True

    async def test_add_lora_default_scale(self, mock_async_omni):
        """Test adding a LoRA adapter with default scale."""
        lora_request = LoRARequest(
            lora_name="test_lora",
            lora_int_id=2,
            lora_path="/path/to/lora2",
        )
        
        # Set mock return value
        mock_async_omni.stage_list[0].set_rpc_result("add_lora", True)
        
        result = await mock_async_omni.add_lora(lora_request)
        
        assert result is True

    async def test_add_lora_failure(self, mock_async_omni):
        """Test handling failure when adding a LoRA adapter."""
        lora_request = LoRARequest(
            lora_name="test_lora",
            lora_int_id=3,
            lora_path="/invalid/path",
        )
        
        # Set mock return value to False
        mock_async_omni.stage_list[0].set_rpc_result("add_lora", False)
        
        result = await mock_async_omni.add_lora(lora_request)
        
        assert result is False

    async def test_remove_lora_success(self, mock_async_omni):
        """Test removing a LoRA adapter successfully."""
        adapter_id = 1
        
        # Set mock return value
        mock_async_omni.stage_list[0].set_rpc_result("remove_lora", True)
        
        result = await mock_async_omni.remove_lora(adapter_id)
        
        assert result is True

    async def test_remove_lora_nonexistent(self, mock_async_omni):
        """Test removing a non-existent LoRA adapter."""
        adapter_id = 999
        
        # Set mock return value to False (adapter not found)
        mock_async_omni.stage_list[0].set_rpc_result("remove_lora", False)
        
        result = await mock_async_omni.remove_lora(adapter_id)
        
        assert result is False

    async def test_list_loras_empty(self, mock_async_omni):
        """Test listing LoRA adapters when none are loaded."""
        # Set mock return value
        mock_async_omni.stage_list[0].set_rpc_result("list_loras", [])
        
        result = await mock_async_omni.list_loras()
        
        assert result == []

    async def test_list_loras_multiple(self, mock_async_omni):
        """Test listing multiple LoRA adapters."""
        expected_adapters = [1, 2, 3]
        
        # Set mock return value
        mock_async_omni.stage_list[0].set_rpc_result("list_loras", expected_adapters)
        
        result = await mock_async_omni.list_loras()
        
        assert result == expected_adapters
        assert len(result) == 3

    async def test_pin_lora_success(self, mock_async_omni):
        """Test pinning a LoRA adapter successfully."""
        adapter_id = 1
        
        # Set mock return value
        mock_async_omni.stage_list[0].set_rpc_result("pin_lora", True)
        
        result = await mock_async_omni.pin_lora(adapter_id)
        
        assert result is True

    async def test_pin_lora_failure(self, mock_async_omni):
        """Test handling failure when pinning a LoRA adapter."""
        adapter_id = 999
        
        # Set mock return value to False (adapter not found)
        mock_async_omni.stage_list[0].set_rpc_result("pin_lora", False)
        
        result = await mock_async_omni.pin_lora(adapter_id)
        
        assert result is False

    async def test_lora_workflow(self, mock_async_omni):
        """Test complete LoRA workflow: add, list, pin, remove."""
        # Add a LoRA adapter
        lora_request = LoRARequest(
            lora_name="workflow_lora",
            lora_int_id=10,
            lora_path="/path/to/workflow_lora",
        )
        mock_async_omni.stage_list[0].set_rpc_result("add_lora", True)
        add_result = await mock_async_omni.add_lora(lora_request)
        assert add_result is True
        
        # List LoRAs (should contain our adapter)
        mock_async_omni.stage_list[0].set_rpc_result("list_loras", [10])
        list_result = await mock_async_omni.list_loras()
        assert 10 in list_result
        
        # Pin the LoRA
        mock_async_omni.stage_list[0].set_rpc_result("pin_lora", True)
        pin_result = await mock_async_omni.pin_lora(10)
        assert pin_result is True
        
        # Remove the LoRA
        mock_async_omni.stage_list[0].set_rpc_result("remove_lora", True)
        remove_result = await mock_async_omni.remove_lora(10)
        assert remove_result is True
        
        # List LoRAs (should be empty now)
        mock_async_omni.stage_list[0].set_rpc_result("list_loras", [])
        final_list = await mock_async_omni.list_loras()
        assert final_list == []

    async def test_multiple_stages_lora_operations(self, mock_async_omni):
        """Test that LoRA operations work with multiple stages."""
        # Add LoRA to all stages
        lora_request = LoRARequest(
            lora_name="multi_stage_lora",
            lora_int_id=20,
            lora_path="/path/to/multi_stage_lora",
        )
        
        # Mock all stages to return True
        for stage in mock_async_omni.stage_list:
            stage.set_rpc_result("add_lora", True)
        
        result = await mock_async_omni.add_lora(lora_request)
        assert result is True
        
        # Verify collective_rpc is called on first stage
        # (implementation detail: uses first stage's result)
        assert mock_async_omni.stage_list[0].collective_rpc("add_lora", (lora_request, 1.0)) is True


@pytest.mark.asyncio
async def test_lora_concurrent_operations(mock_async_omni):
    """Test concurrent LoRA operations."""
    lora_requests = [
        LoRARequest(lora_name=f"lora_{i}", lora_int_id=i, lora_path=f"/path/to/lora_{i}")
        for i in range(5)
    ]
    
    # Set mock results
    mock_async_omni.stage_list[0].set_rpc_result("add_lora", True)
    
    # Add multiple LoRAs concurrently
    tasks = [mock_async_omni.add_lora(lora_req) for lora_req in lora_requests]
    results = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(results)


@pytest.mark.asyncio
async def test_lora_with_sleeping_engine(mock_async_omni):
    """Test LoRA operations when engine is sleeping."""
    # Put engine to sleep
    await mock_async_omni.sleep()
    assert await mock_async_omni.is_sleeping() is True
    
    # Try to add LoRA (should still work as RPC is independent)
    lora_request = LoRARequest(
        lora_name="sleep_lora",
        lora_int_id=30,
        lora_path="/path/to/sleep_lora",
    )
    mock_async_omni.stage_list[0].set_rpc_result("add_lora", True)
    result = await mock_async_omni.add_lora(lora_request)
    assert result is True
    
    # Wake up engine
    await mock_async_omni.wake_up()
    assert await mock_async_omni.is_sleeping() is False


@pytest.mark.asyncio
async def test_lora_with_paused_engine(mock_async_omni):
    """Test LoRA operations when engine is paused."""
    # Pause the engine
    await mock_async_omni.pause_generation()
    assert await mock_async_omni.is_paused() is True
    
    # LoRA operations should still work during pause
    lora_request = LoRARequest(
        lora_name="pause_lora",
        lora_int_id=40,
        lora_path="/path/to/pause_lora",
    )
    mock_async_omni.stage_list[0].set_rpc_result("add_lora", True)
    result = await mock_async_omni.add_lora(lora_request)
    assert result is True
    
    # Resume the engine
    await mock_async_omni.resume_generation()
    assert await mock_async_omni.is_paused() is False
