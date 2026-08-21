"""
E2E offline tests for Omni model with video input and audio output.
"""

import asyncio
import os
from contextlib import ExitStack

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_video
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.platforms import current_omni_platform

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]
thinker_only_models = ["Qwen/Qwen3-Omni-30B-A3B-Captioner"]

# Single CI deploy YAML; rocm/xpu deltas are picked automatically via the
# platforms: section. Only CUDA needs an extra enforce_eager tweak.
_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")


def get_cuda_graph_config():
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
            },
        },
    )


if current_omni_platform.is_xpu():
    stage_configs = [_CI_DEPLOY]
else:
    stage_configs = [get_cuda_graph_config()]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]
# we can use the same config for a model that only has thinker (i.e., does not
# enable audio output) because the resolver should figure out that it doesn't
# need the full pipeline based on the HF config.
thinker_test_params = [(model, stage_config) for model in thinker_only_models for stage_config in stage_configs]


def get_question(prompt_type="video"):
    prompts = {
        "video": "Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["video"])


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {"prompts": get_question(), "videos": video, "modalities": ["audio"]}

    # Test single completion
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", thinker_test_params, indirect=True)
def test_thinker_only_model_request(omni_runner, omni_runner_handler) -> None:
    """Test that we can load and run a request through a model that only has the thinker stage."""
    request_config = {"prompts": "what color is the sky?", "modalities": ["text"]}

    # Test single completion
    omni_runner_handler.send_omni_request(request_config)


def _output_token_ids(output) -> list[int]:
    if not getattr(output, "outputs", None):
        return []
    completion = output.outputs[0]
    token_ids = list(completion.token_ids or [])
    if token_ids:
        return token_ids
    return list(getattr(completion, "cumulative_token_ids", None) or [])


def _finish_reason(output) -> str | None:
    if not getattr(output, "outputs", None):
        return None
    return getattr(output.outputs[0], "finish_reason", None)


def _colocate_async_deploy() -> str:
    """CI Qwen3-Omni deploy with sleep mode enabled on every stage."""
    return modify_stage_config(
        get_cuda_graph_config(),
        updates={
            "stages": {
                0: {"enable_sleep_mode": True, "enforce_eager": True},
                1: {"enable_sleep_mode": True, "enforce_eager": True},
                2: {"enable_sleep_mode": True, "enforce_eager": True},
            },
        },
    )


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_colocate_async_abort_tokens_and_sleep_admission() -> None:
    """Prove the two remaining zhtmike/vllm-omni#1 bugs are fixed.

    **required** regression. Control-plane APIs (``abort`` / ``sleep`` /
    ``wake_up`` / ``resume_generation``) are not in ``send_omni_request``.

    On ``origin/main`` this test fails:
    - abort: ``generate()`` never yields ``finish_reason="abort"`` with the
      tokens produced so far, so resume has an empty prefix (or the stream
      hangs after frontend state is dropped).
    - sleep: ``generate()`` is admitted into EngineCore while weights/KV are
      offloaded, so the task errors (asleep / corrupted ADD frame) instead of
      waiting for ``resume_generation()``.

    On this PR both legs pass: abort returns a resumeable prefix, and sleep
    holds ``generate()`` until wake + resume.
    """
    prompt = "What color is the sky? Write a long, detailed explanation."
    with ExitStack() as after:
        engine = AsyncOmni(
            model=models[0],
            deploy_config=_colocate_async_deploy(),
            enable_sleep_mode=True,
        )
        after.callback(engine.shutdown)

        request_id = "qwen3-abort-partial"
        outputs: list = []

        async def _generate(max_tokens: int, req_id: str, gen_prompt) -> None:
            async for output in engine.generate(
                prompt=gen_prompt,
                request_id=req_id,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
                output_modalities=["text"],
            ):
                outputs.append(output)

        abort_task = asyncio.create_task(_generate(256, request_id, prompt))
        prefix: list[int] = []
        prompt_token_ids: list[int] = []
        for _ in range(600):
            if abort_task.done():
                break
            if outputs:
                latest = outputs[-1]
                prefix = _output_token_ids(latest)
                prompt_token_ids = list(getattr(latest, "prompt_token_ids", None) or [])
                if prefix:
                    break
            await asyncio.sleep(0.1)

        assert prefix, "generate produced no tokens before abort"
        assert not abort_task.done(), "generate finished before abort; raise max_tokens"
        await engine.abort(request_id)
        await asyncio.wait_for(abort_task, timeout=90)

        final = outputs[-1]
        assert final.finished
        assert _finish_reason(final) == "abort"
        abort_prefix = _output_token_ids(final)
        assert abort_prefix, "abort dropped the generated prefix (main-branch behavior)"

        resume_prompt: str | TokensPrompt = (
            TokensPrompt(prompt_token_ids=prompt_token_ids + abort_prefix) if prompt_token_ids else prompt
        )
        outputs.clear()
        async for output in engine.generate(
            prompt=resume_prompt,
            request_id="qwen3-abort-resume",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
            output_modalities=["text"],
        ):
            outputs.append(output)
        assert outputs
        assert any(_output_token_ids(out) for out in outputs)

        # Trainer order is pause → abort → sleep. Pause here after the
        # resume generate so EngineCore is idle before CuMem offload.
        await engine.pause_generation(mode="abort", clear_cache=True)
        await engine.sleep(level=1)
        outputs.clear()
        sleep_task = asyncio.create_task(_generate(8, "qwen3-sleep-admission", prompt))
        await asyncio.sleep(1.0)
        assert not sleep_task.done(), "generate() ran while EngineCore was sleeping (main-branch admission race)"
        await engine.wake_up()
        await asyncio.sleep(0.5)
        assert not sleep_task.done(), "generate() resumed before resume_generation()"
        await engine.resume_generation()
        await asyncio.wait_for(sleep_task, timeout=180)
        assert outputs
        assert _finish_reason(outputs[-1]) != "abort"
    await asyncio.sleep(5)
