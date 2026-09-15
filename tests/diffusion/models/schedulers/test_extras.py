# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.schedulers.extras import attach_rollout_extras, requested_rollout_extras

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_missing_key_raises():
    output = DiffusionOutput(output="img")
    with pytest.raises(ValueError, match="latents_clean"):
        attach_rollout_extras(output, frozenset({"latents_clean"}), {})


def test_unknown_request_raises():
    class _S:
        extra_args = {"rollout_extras": ["not_a_key"]}

    with pytest.raises(ValueError, match="Unknown rollout extras"):
        requested_rollout_extras(_S())


def test_attach_splits_prompt_embeddings_and_rl():
    output = DiffusionOutput(output="img")
    available = {"prompt_embeds": "E", "latents_clean": "C"}
    result = attach_rollout_extras(output, frozenset(available), available)
    payload = result.output["payload"]
    metadata = result.output["metadata"]
    assert payload["image"] == "img"
    assert payload["prompt_embeddings"]["prompt_embeds"] == "E"
    assert payload["rl"]["latents_clean"] == "C"
    assert metadata["prompt_embeddings"]["prompt_embeds"] == "E"
    assert metadata["rl"]["latents_clean"] == "C"
    assert "custom_output" not in result.__dataclass_fields__


def test_noop_when_nothing_requested():
    output = DiffusionOutput(output="img")
    assert attach_rollout_extras(output, None, {"prompt_embeds": "E"}).output == "img"
