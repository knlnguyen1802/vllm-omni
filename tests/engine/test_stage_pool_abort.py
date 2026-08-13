# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abort contract for StagePool and Orchestrator before route binding exists."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind

from vllm_omni.engine.messages import OutputMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStageClient:
    def __init__(self) -> None:
        self.abort_calls: list[list[str]] = []

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))


class _RecordingOutputProcessor:
    def __init__(self) -> None:
        self.request_states: dict[str, list[int]] = {}
        self.processed_ids: list[str] = []

    def process_outputs(self, engine_core_outputs, *_args, **_kwargs):
        outputs: list[RequestOutput] = []
        for eco in engine_core_outputs:
            request_id = eco.request_id
            self.processed_ids.append(request_id)
            token_ids = self.request_states.pop(request_id, None)
            if token_ids is None:
                continue
            outputs.append(_abort_request_output(request_id, token_ids))
        return SimpleNamespace(request_outputs=outputs)


def _abort_request_output(request_id: str, token_ids: list[int]) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=token_ids[-1:] if token_ids else [],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="abort",
        stop_reason=None,
    )
    completion.cumulative_token_ids = list(token_ids)
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )


def _pool(client: _FakeStageClient, processor: Any) -> StagePool:
    return StagePool(
        0,
        [client],
        output_processor=processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )


@pytest.mark.asyncio
async def test_abort_returns_output_processor_abort_without_binding() -> None:
    client = _FakeStageClient()
    processor = _RecordingOutputProcessor()
    processor.request_states["unbound-req"] = [11, 12, 13]
    pool = _pool(client, processor)

    abort_outputs = await pool.abort_requests(["unbound-req"])

    assert not client.abort_calls
    assert [output.request_id for output in abort_outputs] == ["unbound-req"]
    assert list(abort_outputs[0].outputs[0].cumulative_token_ids) == [11, 12, 13]
    assert "unbound-req" not in processor.request_states


@pytest.mark.asyncio
async def test_abort_engine_rpc_only_for_bound_requests() -> None:
    client = _FakeStageClient()
    processor = _RecordingOutputProcessor()
    processor.request_states["bound-req"] = [1]
    processor.request_states["unbound-req"] = [2, 3]
    pool = _pool(client, processor)
    pool._request_bindings["bound-req"] = 0

    abort_outputs = await pool.abort_requests(["bound-req", "unbound-req", "unknown-req"])

    assert client.abort_calls == [["bound-req"]]
    assert [output.request_id for output in abort_outputs] == ["bound-req", "unbound-req"]
    assert processor.processed_ids == ["bound-req", "unbound-req"]
    assert not processor.request_states


@pytest.mark.asyncio
async def test_abort_without_output_processor_state_is_not_delivered() -> None:
    client = _FakeStageClient()
    processor = _RecordingOutputProcessor()
    pool = _pool(client, processor)

    abort_outputs = await pool.abort_requests(["missing-req"])

    assert abort_outputs == []
    assert not client.abort_calls
    assert processor.processed_ids == []


@pytest.mark.asyncio
async def test_orchestrator_abort_routes_unbound_output_processor_outputs() -> None:
    client = _FakeStageClient()
    processor = _RecordingOutputProcessor()
    processor.request_states["unbound-req"] = [4, 5]
    pool = _pool(client, processor)

    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [pool]
    orchestrator.output_async_queue = asyncio.Queue()

    delivered = await orchestrator._abort_request_ids(["unbound-req", "missing-req"])

    assert delivered == ["unbound-req"]
    assert not client.abort_calls
    msg = orchestrator.output_async_queue.get_nowait()
    assert isinstance(msg, OutputMessage)
    assert msg.request_id == "unbound-req"
    assert msg.finished is True
    assert list(msg.engine_outputs.outputs[0].cumulative_token_ids) == [4, 5]
    assert orchestrator.output_async_queue.empty()


@pytest.mark.asyncio
async def test_output_processor_abort_preserves_cumulative_token_ids(mocker) -> None:
    detokenizer = mocker.Mock()
    detokenizer.output_token_ids = [7, 8, 9]
    detokenizer.get_next_output_text.return_value = ""
    detokenizer.num_output_tokens.return_value = 3
    detokenizer.output_text = "abc"

    logprobs = mocker.Mock()
    logprobs.logprobs = None
    logprobs.cumulative_logprob = None
    logprobs.prompt_logprobs = None

    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    state = OmniRequestState(
        request_id="ar-req",
        external_req_id="ar-req",
        parent_req=None,
        request_index=0,
        lora_request=None,
        prompt=None,
        prompt_token_ids=[0],
        prompt_embeds=None,
        logprobs_processor=logprobs,
        detokenizer=detokenizer,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    processor.request_states[state.request_id] = state
    processor.external_req_ids[state.external_req_id].append(state.request_id)

    pool = _pool(_FakeStageClient(), processor)
    abort_outputs = await pool.abort_requests(["ar-req"])

    assert len(abort_outputs) == 1
    assert abort_outputs[0].finished is True
    assert str(abort_outputs[0].outputs[0].finish_reason).lower() == "abort"
    assert list(abort_outputs[0].outputs[0].cumulative_token_ids) == [7, 8, 9]
    assert "ar-req" not in processor.request_states
