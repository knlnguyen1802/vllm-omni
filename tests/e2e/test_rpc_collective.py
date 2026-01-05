import os
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

diffusion_models = ["Tongyi-MAI/Z-Image-Turbo"]

omni_models = ["Qwen/Qwen2.5-Omni-3B"]


@pytest.mark.parametrize("model_name", omni_models)
def test_omni_model(model_name: str):
    m = Omni(model=model_name,init_timeout=3600)
    results = m.collective_rpc(
        method="sleep",
        args=(1,),
    )
    assert len(results)==3

@pytest.mark.parametrize("model_name", diffusion_models)
def test_diffusion_model(model_name: str):
    m = Omni(model=model_name)
    results = m.collective_rpc(
        method="sleep",
        args=(1,),
    )
    assert len(results)==1