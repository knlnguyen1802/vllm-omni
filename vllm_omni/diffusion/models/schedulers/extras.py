# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Allowlisted RL extras on the payload envelope. Do not restore custom_output."""

from __future__ import annotations

from typing import Any, Mapping

from vllm_omni.diffusion.data import DiffusionOutput

RL_EXTRA_KEYS = frozenset(
    {
        "prompt_embeds",
        "prompt_embeds_mask",
        "negative_prompt_embeds",
        "negative_prompt_embeds_mask",
        "pooled_prompt_embeds",
        "latents_clean",
        "train_timesteps",
        "condition_image_latents",
    }
)

_PROMPT_EMBED_KEYS = frozenset(
    {
        "prompt_embeds",
        "prompt_embeds_mask",
        "negative_prompt_embeds",
        "negative_prompt_embeds_mask",
        "pooled_prompt_embeds",
    }
)


def requested_rollout_extras(sampling: Any) -> frozenset[str]:
    extra_args = getattr(sampling, "extra_args", None) or {}
    raw = extra_args.get("rollout_extras") or extra_args.get("rl_extras") or ()
    requested = frozenset(raw)
    unknown = requested - RL_EXTRA_KEYS
    if unknown:
        raise ValueError(f"Unknown rollout extras {sorted(unknown)}. Allowed: {sorted(RL_EXTRA_KEYS)}")
    return requested


def attach_rollout_extras(
    output: DiffusionOutput,
    requested: Mapping[str, Any] | frozenset[str] | None,
    available: Mapping[str, Any],
) -> DiffusionOutput:
    """Put requested extras on ``output.output`` payload. Missing keys raise."""
    if not requested:
        return output
    keys = frozenset(requested) if not isinstance(requested, Mapping) else frozenset(requested.keys())
    missing = keys - set(available)
    if missing:
        raise ValueError(f"Requested rollout extras missing from pipeline: {sorted(missing)}")
    prompt_embeddings = {k: available[k] for k in keys & _PROMPT_EMBED_KEYS if available.get(k) is not None}
    rl = {k: available[k] for k in keys - _PROMPT_EMBED_KEYS if available.get(k) is not None}
    raw = output.output
    if isinstance(raw, dict) and "payload" in raw:
        payload = dict(raw.get("payload") or {})
        metadata = dict(raw.get("metadata") or {})
    else:
        payload = {"image": raw}
        metadata = {}
    # Same envelope verl-omni already reads: rl / prompt_embeddings live in metadata.
    if prompt_embeddings:
        payload["prompt_embeddings"] = prompt_embeddings
        metadata["prompt_embeddings"] = prompt_embeddings
    if rl:
        payload["rl"] = rl
        metadata["rl"] = rl
    output.output = {"payload": payload, "metadata": metadata}
    return output
