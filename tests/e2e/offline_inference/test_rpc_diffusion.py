# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
E2E test for RPC functionality in diffusion models.

This test verifies that the RPC mechanism works correctly for diffusion models,
allowing the engine to call methods on worker processes.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# Use random weights model for testing
models = ["riverclouds/qwen_image_random"]


@pytest.mark.parametrize("model_name", models)
def test_rpc_diffusion_generation(model_name: str):
    """Test RPC-based generation with diffusion model.
    
    This test verifies that the unified RPC mechanism works correctly
    for generation requests.
    """
    # Create diffusion engine
    m = OmniDiffusion(model=model_name)

    # Use minimal settings for fast testing
    height = 256
    width = 256
    num_inference_steps = 2

    # Test generation via RPC (now all generation goes through RPC)
    images = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=1,
    )

    # Verify generation succeeded
    assert images is not None
    assert len(images) == 1
    assert images[0].width == width
    assert images[0].height == height

    # Clean up
    m.close()


@pytest.mark.parametrize("model_name", models)
def test_rpc_diffusion_collective_rpc(model_name: str):
    """Test collective_rpc method for calling worker methods.
    
    This test verifies that the collective_rpc mechanism works correctly
    for calling arbitrary methods on worker processes.
    """
    # Create diffusion engine
    m = OmniDiffusion(model=model_name)

    # Test calling the generate method via collective_rpc
    # (similar to how generation works internally)
    from vllm_omni.diffusion.request import OmniDiffusionRequest

    request = OmniDiffusionRequest(
        prompt="a photo of a dog",
        height=256,
        width=256,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=1,
    )

    # Call the generate method on workers via RPC
    # unique_reply_rank=0 means only rank 0 will reply
    result = m.engine.collective_rpc(
        method="generate",
        args=([request],),
        kwargs={},
        unique_reply_rank=0,
    )

    # Verify result
    assert result is not None
    assert hasattr(result, "output"), "Result should have output attribute"

    # Clean up
    m.close()


@pytest.mark.parametrize("model_name", models)
def test_rpc_diffusion_shutdown(model_name: str):
    """Test that shutdown via unified message queue works correctly.
    
    This test verifies that the shutdown signal is properly sent through
    the unified message queue and workers shut down correctly.
    """
    # Create diffusion engine
    m = OmniDiffusion(model=model_name)

    # Generate one image to ensure everything is initialized
    images = m.generate(
        "a test image",
        height=256,
        width=256,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=1,
    )

    assert images is not None
    assert len(images) == 1

    # Close the engine (should send shutdown via unified queue)
    m.close()

    # Verify all worker processes have exited
    for proc in m.engine._processes:
        assert not proc.is_alive(), f"Worker process {proc.name} did not shut down"
