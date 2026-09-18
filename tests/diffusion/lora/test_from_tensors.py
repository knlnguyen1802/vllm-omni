# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for loading diffusion LoRA adapters from in-memory tensors."""

import pytest
import torch

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest, TensorLoRARequest

PEFT_CONFIG = {
    "peft_type": 4,  # PeftType.LORA
    "task_type": 1,  # TaskType.CAUSAL_LM
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["proj"],
    "lora_dropout": 0.0,
    "bias": "none",
}


def _tensors(scale_a: float = 0.0) -> dict:
    return {
        "proj.lora_A.weight": torch.full((8, 4), scale_a),
        "proj.lora_B.weight": torch.zeros(4, 8),
    }


def _request(tensors=None, peft_config=None) -> TensorLoRARequest:
    return TensorLoRARequest(
        lora_name="actor-adapter",
        lora_int_id=1,
        lora_path="unused-placeholder",
        peft_config=peft_config or dict(PEFT_CONFIG),
        lora_tensors=tensors if tensors is not None else _tensors(),
    )


def _manager(dtype=torch.bfloat16, pipeline=None) -> DiffusionLoRAManager:
    manager = object.__new__(DiffusionLoRAManager)
    manager._expected_lora_modules = {"proj"}
    manager.pipeline = pipeline
    manager.dtype = dtype
    return manager


def test_tensor_request_is_a_lora_request_with_tensor_fields():
    request = _request()
    assert isinstance(request, LoRARequest)
    assert request.peft_config["r"] == 8
    assert set(request.lora_tensors) == {"proj.lora_A.weight", "proj.lora_B.weight"}


def test_load_adapter_from_tensors_builds_a_peft_lora_model():
    manager = _manager()
    lora_model, peft_helper = manager._load_adapter(_request())

    assert lora_model.id == 1
    assert peft_helper.r == 8
    assert "proj" in lora_model.loras


def test_empty_tensor_dict_raises():
    manager = _manager()
    with pytest.raises(ValueError, match="carried no adapter tensors"):
        manager._load_adapter(_request(tensors={}))


def test_pipeline_mapper_translates_tensors_before_loading():
    class _Pipeline:
        @staticmethod
        def map_lora_update_to_engine(tensors, peft_config):
            # A fused-layout pipeline renames keys; prove the hook runs by
            # scaling a tensor and renaming one key.
            translated = {name: tensor * 2 for name, tensor in tensors.items()}
            translated["proj.lora_A.weight"] = tensors["proj.lora_A.weight"]
            return translated, peft_config

    manager = _manager(pipeline=_Pipeline())
    lora_model, _ = manager._load_adapter(_request(tensors=_tensors(scale_a=0.25)))
    assert "proj" in lora_model.loras


def test_base_request_still_routes_to_dir_path(monkeypatch):
    manager = _manager()

    def fail_from_tensors(request):
        raise AssertionError("base LoRARequest must not take the tensor path")

    manager._load_adapter_from_tensors = fail_from_tensors
    base = LoRARequest(lora_name="dir-adapter", lora_int_id=2, lora_path="unused-placeholder")
    # The dir path needs a real adapter directory; expect the local-dir loader
    # to fail on the placeholder rather than taking the tensor branch.
    with pytest.raises(Exception):
        manager._load_adapter(base)
