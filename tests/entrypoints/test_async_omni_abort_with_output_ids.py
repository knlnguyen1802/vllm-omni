# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AsyncOmni.abort_with_output_ids: external-ID abort with output delivery + pause."""

from __future__ import annotations

import pytest
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.engine.messages import OutputMessage
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeEngine:
    def __init__(self, abort_outputs: list[OutputMessage] | None = None) -> None:
        self.abort_calls: list[tuple[list[str], bool]] = []
        self._abort_outputs = abort_outputs or []

    async def abort_async(self, request_ids: list[str], *, pause: bool = False) -> list[OutputMessage]:
        self.abort_calls.append((list(request_ids), pause))
        return list(self._abort_outputs)


def _abort_output_message(request_id: str, token_ids: list[int]) -> OutputMessage:
    return OutputMessage(
        request_id=request_id,
        stage_id=0,
        replica_id=None,
        engine_outputs=OmniRequestOutput.from_stage_output(
            RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=[1],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="",
                        token_ids=token_ids,
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="abort",
                    )
                ],
                finished=True,
            ),
            request_id=request_id,
            stage_id=0,
            final_output_type="text",
            finished=True,
        ),
        metrics=None,
        finished=True,
    )


def _omni(engine: _FakeEngine) -> AsyncOmni:
    omni = object.__new__(AsyncOmni)
    omni.log_stats = False
    omni.request_states = {}
    omni._orphaned_abort_states = {}
    omni.engine = engine
    return omni


@pytest.mark.asyncio
async def test_abort_with_output_ids_returns_internal_ids_with_delivered_outputs() -> None:
    msg = _abort_output_message("internal-1", [5, 6, 7])
    engine = _FakeEngine([msg])
    omni = _omni(engine)
    state = ClientRequestState(request_id="internal-1", external_request_id="ext-1")
    omni.request_states["internal-1"] = state
    # A second request that produced no abort output (e.g. diffusion or no OP state).
    other = ClientRequestState(request_id="internal-2", external_request_id="ext-2")
    omni.request_states["internal-2"] = other

    delivered = await omni.abort_with_output_ids(["ext-1", "ext-2"])

    assert delivered == ["internal-1"]
    assert engine.abort_calls == [(["internal-1", "internal-2"], False)]
    received = state.queue.get_nowait()
    assert received is msg
    assert list(received.engine_outputs.outputs[0].token_ids) == [5, 6, 7]
    assert "internal-1" not in omni.request_states
    assert "internal-2" not in omni.request_states


@pytest.mark.asyncio
async def test_abort_with_output_ids_empty_list_with_pause_still_pauses_engine() -> None:
    engine = _FakeEngine()
    omni = _omni(engine)

    delivered = await omni.abort_with_output_ids([], pause=True)

    assert delivered == []
    # The pause must reach the engine even with no mapped internal IDs: an
    # empty frontend request list does not prove the backend is idle.
    assert engine.abort_calls == [([], True)]


@pytest.mark.asyncio
async def test_abort_with_output_ids_empty_list_without_pause_is_noop() -> None:
    engine = _FakeEngine()
    omni = _omni(engine)

    delivered = await omni.abort_with_output_ids([])

    assert delivered == []
    assert engine.abort_calls == []


@pytest.mark.asyncio
async def test_abort_with_output_ids_unmapped_external_ids_still_pause() -> None:
    engine = _FakeEngine()
    omni = _omni(engine)
    omni.request_states["internal-1"] = ClientRequestState(request_id="internal-1", external_request_id="ext-1")

    delivered = await omni.abort_with_output_ids(["ext-unknown"], pause=True)

    assert delivered == []
    assert engine.abort_calls == [([], True)]
    # Unrelated request state is untouched by an empty mapped abort.
    assert "internal-1" in omni.request_states
