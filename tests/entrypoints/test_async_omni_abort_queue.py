# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend abort must still deliver a terminal queue message to generate()."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.engine.messages import OutputMessage
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _omni(abort_async) -> AsyncOmni:
    omni = object.__new__(AsyncOmni)
    omni.log_stats = False
    omni.request_states = {}
    omni._orphaned_abort_states = {}
    omni.engine = SimpleNamespace(abort_async=abort_async)
    return omni


@pytest.mark.asyncio
async def test_abort_keeps_orphaned_state_so_generate_receives_terminal_output() -> None:
    aborted: list[list[str]] = []

    async def fake_abort_async(request_ids):
        aborted.append(list(request_ids))

    omni = _omni(fake_abort_async)
    state = ClientRequestState(request_id="internal-1", external_request_id="ext-1")
    omni.request_states["internal-1"] = state

    waiter = asyncio.create_task(state.queue.get())
    await omni._abort(["internal-1"])

    assert aborted == [["internal-1"]]
    assert "internal-1" not in omni.request_states
    assert omni._orphaned_abort_states["internal-1"] is state
    assert not waiter.done()

    abort_output = OmniRequestOutput(request_id="internal-1", finished=True)
    abort_output.outputs = [
        SimpleNamespace(token_ids=[9], cumulative_token_ids=[7, 8, 9], finish_reason="abort")
    ]
    msg = OutputMessage(
        request_id="internal-1",
        stage_id=0,
        replica_id=None,
        engine_outputs=abort_output,
        metrics=None,
        finished=True,
    )
    should_continue, req_id, stage_id, req_state = omni._handle_output_message(msg)
    assert should_continue is False
    assert req_id == "internal-1"
    assert stage_id == 0
    assert req_state is state

    await req_state.queue.put(msg)
    omni._release_orphaned_abort_state("internal-1")

    received = await asyncio.wait_for(waiter, timeout=1)
    assert received.finished is True
    assert list(received.engine_outputs.outputs[0].cumulative_token_ids) == [7, 8, 9]
    assert "internal-1" not in omni._orphaned_abort_states


@pytest.mark.asyncio
async def test_generate_cancel_abort_does_not_keep_orphaned_state() -> None:
    async def fake_abort_async(request_ids):
        del request_ids

    omni = _omni(fake_abort_async)
    state = ClientRequestState(request_id="internal-2", external_request_id="ext-2")
    omni.request_states["internal-2"] = state

    await omni._abort(["internal-2"], keep_queue=False)

    assert "internal-2" not in omni.request_states
    assert "internal-2" not in omni._orphaned_abort_states
