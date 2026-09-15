# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: qwen_image consumes OmniCustomPrompt.prompt_ids without re-tokenizing."""

import pytest

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_extract_custom_prompt_ids_reads_lists():
    prompts = [{"prompt_ids": [1, 2, 3], "prompt_mask": [1, 1, 1], "extra_prompt_ids": {"t5": [9]}}]
    ids, mask, neg_ids, neg_mask, extra = QwenImagePipeline.extract_custom_prompt_ids(prompts)
    assert ids == [[1, 2, 3]]
    assert mask == [[1, 1, 1]]
    assert neg_ids is None
    assert neg_mask is None
    assert extra == [{"t5": [9]}]


def test_extract_custom_prompt_ids_ignores_text_prompts():
    assert QwenImagePipeline.extract_custom_prompt_ids(["a cat"]) == (None, None, None, None, None)


def test_extract_prompts_skips_empty_string_when_ids_present():
    prompt, negative = QwenImagePipeline._extract_prompts(
        QwenImagePipeline, [{"prompt_ids": [1, 2], "prompt": "should not tokenize"}]
    )
    assert prompt is None
    assert negative is None
