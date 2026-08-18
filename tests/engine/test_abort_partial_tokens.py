"""Regression tests: AR abort delivers partial tokens before frontend teardown."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.engine.messages import AbortResultMessage, OutputMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeDetokenizer:
    def __init__(self, token_ids: list[int], text: str = "partial") -> None:
        self.output_token_ids = list(token_ids)
        self._text = text

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return self._text

    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)


def _make_ar_req_state(
    *,
    request_id: str = "req-internal",
    external_req_id: str = "req-internal",
    token_ids: list[int] | None = None,
    logprobs: list | None = None,
    output_kind: RequestOutputKind = RequestOutputKind.DELTA,
) -> OmniRequestState:
    if token_ids is None:
        token_ids = [11, 22, 33]
    else:
        token_ids = list(token_ids)
    if logprobs is None:
        logprobs = [{"token_id": tid, "logprob": -0.1 * i} for i, tid in enumerate(token_ids)]
    logprobs_processor = MagicMock(
        logprobs=logprobs,
        cumulative_logprob=sum(lp["logprob"] for lp in logprobs) if logprobs else None,
        prompt_logprobs=None,
        pop_prompt_logprobs=MagicMock(return_value=None),
    )
    return OmniRequestState(
        request_id=request_id,
        external_req_id=external_req_id,
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=output_kind,
        prompt="hello",
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        logprobs_processor=logprobs_processor,
        detokenizer=_FakeDetokenizer(token_ids),
        max_tokens_param=64,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
    )


def test_abort_requests_collecting_outputs_preserves_partial_tokens_and_logprobs():
    """OP abort builds terminal abort output from partial generation state."""
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    req_state = _make_ar_req_state(
        request_id="req-engine-uuid",
        external_req_id="req-orch",
        token_ids=[7, 8, 9],
        output_kind=RequestOutputKind.DELTA,
    )
    processor.request_states["req-engine-uuid"] = req_state
    processor.external_req_ids["req-orch"].append("req-engine-uuid")

    aborted_ids, outputs = processor.abort_requests_collecting_outputs(["req-orch"], internal=False)

    assert aborted_ids == ["req-engine-uuid"]
    assert "req-engine-uuid" not in processor.request_states
    assert len(outputs) == 1
    out = outputs[0]
    assert isinstance(out, RequestOutput)
    assert out.finished is True
    assert out.request_id == "req-orch"
    assert len(out.outputs) == 1
    completion = out.outputs[0]
    assert completion.finish_reason == str(FinishReason.ABORT)
    assert list(completion.token_ids) == [7, 8, 9]
    assert completion.logprobs is not None
    assert len(completion.logprobs) == len(completion.token_ids)


def test_abort_requests_collecting_outputs_empty_when_no_tokens_yet():
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    req_state = _make_ar_req_state(token_ids=[], logprobs=[])
    processor.request_states["r"] = req_state
    processor.external_req_ids["r"].append("r")

    aborted_ids, outputs = processor.abort_requests_collecting_outputs(["r"], internal=False)
    assert aborted_ids == ["r"]
    assert len(outputs) == 1
    assert list(outputs[0].outputs[0].token_ids) == []
    assert outputs[0].outputs[0].finish_reason == str(FinishReason.ABORT)


@pytest.mark.asyncio
async def test_stage_pool_abort_returns_ar_outputs_before_engine_abort():
    """StagePool collects OP abort outputs before EngineCore abort."""
    call_order: list[str] = []

    class _OP:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], bool]] = []

        def abort_requests_collecting_outputs(self, request_ids, *, internal: bool):
            call_order.append("op")
            self.calls.append((list(request_ids), internal))
            return (
                ["engine-id"],
                [
                    RequestOutput(
                        request_id="req-orch",
                        prompt=None,
                        prompt_token_ids=[1],
                        prompt_logprobs=None,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="hi",
                                token_ids=[5, 6],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="abort",
                            )
                        ],
                        finished=True,
                    )
                ],
            )

    class _Client:
        stage_type = "llm"
        final_output = True
        final_output_type = "text"

        def __init__(self) -> None:
            self.abort_calls: list[list[str]] = []

        async def abort_requests_async(self, request_ids: list[str]) -> None:
            call_order.append("engine")
            self.abort_calls.append(list(request_ids))

    client = _Client()
    op = _OP()
    pool = StagePool(0, [client], output_processor=op)
    pool._request_bindings["req-orch"] = 0

    outputs = await pool.abort_requests(["req-orch"])

    assert call_order == ["op", "engine"]
    assert op.calls == [(["req-orch"], False)]
    assert client.abort_calls == [["engine-id"]]
    assert len(outputs) == 1
    assert list(outputs[0].outputs[0].token_ids) == [5, 6]


@pytest.mark.asyncio
async def test_stage_pool_diffusion_abort_has_no_partial_prefix_outputs():
    """Diffusion abort must not fabricate AR-style partial token outputs."""

    class _Client:
        stage_type = "diffusion"
        final_output = True
        final_output_type = "image"

        def __init__(self) -> None:
            self.abort_calls: list[list[str]] = []

        async def abort_requests_async(self, request_ids: list[str]) -> None:
            self.abort_calls.append(list(request_ids))

    class _OP:
        def abort_requests_collecting_outputs(self, *_args, **_kwargs):
            raise AssertionError("diffusion abort must not collect AR OP outputs")

    client = _Client()
    pool = StagePool(0, [client], output_processor=_OP())
    pool._request_bindings["diff-req"] = 0

    outputs = await pool.abort_requests(["diff-req"])

    assert outputs == []
    assert client.abort_calls == [["diff-req"]]


@pytest.mark.asyncio
async def test_orchestrator_abort_result_includes_final_stage_outputs():
    class _Pool:
        def __init__(self, stage_id: int, stage_type: str, *, final_output: bool) -> None:
            self.stage_id = stage_id
            self.stage_type = stage_type
            self.final_output = final_output
            self.stage_client = SimpleNamespace(
                stage_type=stage_type,
                final_output=final_output,
                final_output_type="text" if stage_type == "llm" else "image",
            )
            self.released: list[list[str]] = []

        async def abort_requests(self, request_ids: list[str]):
            if self.stage_type == "diffusion":
                return []
            return [
                RequestOutput(
                    request_id=request_ids[0],
                    prompt=None,
                    prompt_token_ids=[1],
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="partial",
                            token_ids=[9, 8],
                            cumulative_logprob=None,
                            logprobs=[{"x": 1}, {"x": 2}],
                            finish_reason="abort",
                        )
                    ],
                    finished=True,
                )
            ]

        def release_bindings(self, request_ids: list[str]) -> None:
            self.released.append(list(request_ids))

        def get_bound_replica_id(self, _request_id: str) -> int:
            return 0

    orch = object.__new__(Orchestrator)
    orch.stage_pools = [
        _Pool(0, "llm", final_output=True),
        _Pool(1, "diffusion", final_output=True),
    ]
    orch.request_states = {
        "req-a": OrchestratorRequestState(
            request_id="req-a",
            final_stage_id=0,
            final_output_stage_ids={0},
            stage_submit_ts={0: 1.0},
        )
    }
    orch._cfg_tracker = SimpleNamespace(
        get_parent_id=lambda _rid: None,
        is_companion_done=lambda _rid: True,
        cleanup_parent=lambda _rid: [],
        pop_pending_parent=lambda _pid: None,
    )
    orch.duplex_control_plane = None
    orch._pd_kv_params = {}
    orch._running_counter = None
    orch.output_async_queue = asyncio.Queue()
    orch.rpc_async_queue = asyncio.Queue()

    from vllm_omni.engine.messages import AbortRequestMessage

    await orch._handle_abort(AbortRequestMessage(request_ids=["req-a"], rpc_id="rpc-1"))
    result = orch.rpc_async_queue.get_nowait()
    assert isinstance(result, AbortResultMessage)
    assert result.success is True
    assert result.abort_outputs is not None
    assert len(result.abort_outputs) == 1
    msg = result.abort_outputs[0]
    assert isinstance(msg, OutputMessage)
    assert msg.finished is True
    assert msg.request_id == "req-a"
    engine_out = msg.engine_outputs
    assert isinstance(engine_out, OmniRequestOutput)
    assert list(engine_out.outputs[0].token_ids) == [9, 8]
    assert engine_out.outputs[0].finish_reason == "abort"
    assert len(engine_out.outputs[0].logprobs) == 2


@pytest.mark.asyncio
async def test_async_omni_abort_enqueues_output_before_state_removal():
    """Frontend must enqueue abort output before popping request_states."""
    order: list[str] = []

    abort_msg = OutputMessage(
        request_id="req-1",
        stage_id=0,
        engine_outputs=OmniRequestOutput.from_pipeline(
            stage_id=0,
            final_output_type="text",
            request_output=RequestOutput(
                request_id="req-1",
                prompt=None,
                prompt_token_ids=[1],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="x",
                        token_ids=[1, 2],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="abort",
                    )
                ],
                finished=True,
            ),
        ),
        finished=True,
    )

    class _Engine:
        async def abort_async(self, request_ids: list[str]):
            order.append("engine_abort")
            assert request_ids == ["req-1"]
            return [abort_msg]

    omni = object.__new__(AsyncOmni)
    omni.engine = _Engine()
    omni.log_stats = False
    state = ClientRequestState(request_id="req-1", external_request_id="external")
    omni.request_states = {"req-1": state}

    original_put = state.queue.put

    async def _tracked_put(item):
        order.append("queue_put")
        assert "req-1" in omni.request_states
        await original_put(item)

    state.queue.put = _tracked_put  # type: ignore[method-assign]

    await AsyncOmni._abort(omni, ["req-1"])

    assert order == ["engine_abort", "queue_put"]
    assert "req-1" not in omni.request_states
    queued = state.queue.get_nowait()
    assert queued is abort_msg
    assert list(queued.engine_outputs.outputs[0].token_ids) == [1, 2]
    assert queued.engine_outputs.outputs[0].finish_reason == "abort"
