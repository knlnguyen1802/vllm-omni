# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# for now, it suffices to use vLLM's implementation directly
# as this is a user-facing variable, defined here to so that user can directly import LoRARequest from vllm_omni
from msgspec import field

from vllm.lora.request import LoRARequest


class TensorLoRARequest(LoRARequest):
    """Adapter request carrying the LoRA weights as in-memory tensors.

    RL trainers push the actor's adapter straight into the engine every sync
    step; materializing a PEFT directory per step would be pure overhead.
    ``lora_path`` stays a placeholder identity (weights come from
    ``lora_tensors``), mirroring vLLM's ``load_inplace`` use case for
    asynchronous RL loops.

    ``peft_config`` is the trainer-side PEFT config as a plain dict
    (``PEFTHelper.from_dict`` shape); ``lora_tensors`` holds adapter tensors.
    Pipelines whose engine-side layout differs from the trainer's module
    names can define ``map_lora_update_to_engine(tensors, peft_config)`` to
    translate them before loading.
    """

    peft_config: dict = field(default_factory=dict)
    lora_tensors: dict = field(default_factory=dict)


__all__ = ["LoRARequest", "TensorLoRARequest"]
