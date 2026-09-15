# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU schema tests for OmniCustomPrompt token-id fields."""

from typing import get_type_hints

import pytest

from vllm_omni.inputs.data import OmniCustomPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_extra_prompt_ids_are_on_omni_custom_prompt():
    hints = get_type_hints(OmniCustomPrompt)
    assert "prompt_ids" in hints
    assert "extra_prompt_ids" in hints
    assert "extra_negative_prompt_ids" in hints
    assert "extra_prompt_masks" in hints
    assert "extra_negative_prompt_masks" in hints


def test_extra_prompt_ids_accept_encoder_map():
    prompt: OmniCustomPrompt = {
        "prompt_ids": [1, 2, 3],
        "extra_prompt_ids": {"t5": [4, 5], "clip": [6]},
        "extra_prompt_masks": {"t5": [1, 1]},
    }
    assert prompt["extra_prompt_ids"]["t5"] == [4, 5]
    assert isinstance(prompt["prompt_ids"], list)
