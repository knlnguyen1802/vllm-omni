import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_async_omni(*, inline_diffusion: bool = True, max_batch_size: int = 2) -> AsyncOmni:
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._inline_diffusion = inline_diffusion
    omni._pause_cond = asyncio.Condition()
    omni._paused = False
    omni.log_stats = False
    omni.request_states = {}

    runtime_cfg = {"max_batch_size": max_batch_size}
    stage = SimpleNamespace(
        stage_type="diffusion",
        stage_config=SimpleNamespace(runtime=runtime_cfg),
        final_output_type="image",
    )
    omni.stage_list = [stage]
    omni.default_sampling_params_list = [OmniDiffusionSamplingParams(num_inference_steps=1)]
    return omni


def test_generate_uses_batch_prompt_for_inline_diffusion():
    omni = _make_async_omni(inline_diffusion=True, max_batch_size=4)
    captured: dict[str, object] = {}

    async def _fake_generate_batch_diffusion(*, prompts, request_ids, sampling_params_list, output_modalities):
        captured["prompts"] = prompts
        captured["request_ids"] = request_ids
        captured["sampling_params_list"] = sampling_params_list
        captured["output_modalities"] = output_modalities
        return {rid: [rid] for rid in request_ids}

    omni._generate_batch_diffusion = _fake_generate_batch_diffusion  # type: ignore[method-assign]

    async def _run_generate() -> list[str]:
        outputs = []
        sampling_params = [OmniDiffusionSamplingParams(num_inference_steps=1)]
        async for out in omni.generate(
            prompt=["prompt-a", "prompt-b"],
            request_id="req-ignored-for-batch",
            sampling_params_list=sampling_params,
            output_modalities=["image"],
        ):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_run_generate())

    assert captured["prompts"] == ["prompt-a", "prompt-b"]
    generated_request_ids = captured["request_ids"]
    assert isinstance(generated_request_ids, list)
    assert len(generated_request_ids) == 2
    assert all(str(rid).startswith("batch-") for rid in generated_request_ids)
    assert outputs == generated_request_ids


def test_generate_keeps_list_of_token_ids_as_single_prompt():
    omni = _make_async_omni(inline_diffusion=True, max_batch_size=4)
    captured: dict[str, object] = {}

    async def _fake_generate_batch_diffusion(*, prompts, request_ids, sampling_params_list, output_modalities):
        captured["prompts"] = prompts
        captured["request_ids"] = request_ids
        return {request_ids[0]: ["ok"]}

    omni._generate_batch_diffusion = _fake_generate_batch_diffusion  # type: ignore[method-assign]

    tokenized_prompt = [1, 2, 3, 4]
    async def _run_generate() -> list[str]:
        outputs = []
        async for out in omni.generate(
            prompt=tokenized_prompt,
            request_id="req-1",
            sampling_params_list=[OmniDiffusionSamplingParams(num_inference_steps=1)],
        ):
            outputs.append(out)
        return outputs

    outputs = asyncio.run(_run_generate())

    assert captured["prompts"] == [tokenized_prompt]
    assert captured["request_ids"] == ["req-1"]
    assert outputs == ["ok"]


def test_generate_batch_diffusion_dispatches_to_inline_batch_path():
    omni = _make_async_omni(inline_diffusion=True, max_batch_size=8)
    called: dict[str, object] = {}

    async def _fake_generate_batch_inline(*, prompts, request_ids, sampling_params_list):
        called["prompts"] = prompts
        called["request_ids"] = request_ids
        called["sampling_params_list"] = sampling_params_list
        return {"r1": ["out1"], "r2": ["out2"]}

    omni._generate_batch_inline = _fake_generate_batch_inline  # type: ignore[method-assign]

    sampling_params = [OmniDiffusionSamplingParams(num_inference_steps=2)]
    result = asyncio.run(
        omni._generate_batch_diffusion(
            prompts=["p1", "p2"],
            request_ids=["r1", "r2"],
            sampling_params_list=sampling_params,
        ))

    assert called["prompts"] == ["p1", "p2"]
    assert called["request_ids"] == ["r1", "r2"]
    assert called["sampling_params_list"] == sampling_params
    assert result == {"r1": ["out1"], "r2": ["out2"]}


def test_async_omni_diffusion_generate_batch_fills_missing_request_ids():
    diffusion = AsyncOmniDiffusion.__new__(AsyncOmniDiffusion)
    captured: dict[str, object] = {}

    def _step(request):
        captured["request"] = request
        return [
            SimpleNamespace(request_id=None, images=["i1"]),
            SimpleNamespace(request_id="already-set", images=["i2"]),
        ]

    diffusion.engine = SimpleNamespace(step=_step)
    diffusion._executor = None

    sampling_params = OmniDiffusionSamplingParams(num_inference_steps=4, guidance_scale=2.0)
    outputs = asyncio.run(
        diffusion.generate_batch(
            prompts=["a", "b"],
            sampling_params=sampling_params,
            request_ids=["rid-1", "rid-2"],
        ))

    request = captured["request"]
    assert request.prompts == ["a", "b"]
    assert request.request_ids == ["rid-1", "rid-2"]
    assert sampling_params.guidance_scale_provided is True
    assert outputs[0].request_id == "rid-1"
    assert outputs[1].request_id == "already-set"


def test_async_omni_diffusion_generate_batch_validates_request_id_count():
    diffusion = AsyncOmniDiffusion.__new__(AsyncOmniDiffusion)
    diffusion.engine = SimpleNamespace(step=lambda _request: [])
    diffusion._executor = None

    with pytest.raises(ValueError, match="Expected 2 request_ids, got 1"):
        asyncio.run(
            diffusion.generate_batch(
                prompts=["a", "b"],
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
                request_ids=["only-one"],
            ))
