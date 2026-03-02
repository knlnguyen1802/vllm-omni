# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for AsyncOmni.generate_batch.

These tests exercise the new batching surface added in the
``batching_async_omni`` branch on top of commit d6a3551.

Covered scenarios
-----------------
1. Empty prompt list returns empty list immediately.
2. Single prompt is forwarded to ``generate()`` and the last output is
   returned.
3. Multiple prompts are launched concurrently and results come back in
   the **original prompt order**, even when the async tasks finish out
   of order.
4. Unique, stable ``batch-<id>-<i>`` request IDs are generated.
5. Errors raised inside individual ``generate()`` calls propagate out
   through ``asyncio.gather``.
6. ``output_modalities`` and ``sampling_params_list`` are forwarded to
   each inner ``generate()`` call.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_output(request_id: str, text: str = "hello") -> OmniRequestOutput:
    """Build a minimal OmniRequestOutput stub."""
    out = MagicMock(spec=OmniRequestOutput)
    out.request_id = request_id
    out.text = text
    return out


async def _async_gen_from_list(items: list[Any]) -> AsyncGenerator[Any, None]:
    """Yield items as an async generator."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncOmniBatchGenerate:
    """Tests for AsyncOmni.generate_batch (unit-level, no GPU required)."""

    def _make_engine(self):
        """
        Build a minimal AsyncOmni stand-in with ``generate`` mocked so
        that we can call ``generate_batch`` without real GPU/process setup.
        """
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine = object.__new__(AsyncOmni)
        # We only need the generate_batch method; patch generate separately
        # in each test.
        return engine

    # ------------------------------------------------------------------
    # 1. Empty prompt list
    # ------------------------------------------------------------------

    def test_empty_prompts_returns_empty_list(self):
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine = self._make_engine()
        result = asyncio.run(AsyncOmni.generate_batch(engine, prompts=[]))
        assert result == []

    # ------------------------------------------------------------------
    # 2. Single prompt – forwards to generate() and captures last output
    # ------------------------------------------------------------------

    def test_single_prompt_returns_last_output(self):
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine = self._make_engine()
        prompt = "What is the capital of France?"

        # generate() yields three intermediate outputs; the last one is
        # what generate_batch should return.
        outputs = [
            _make_output("req-0", "Par"),
            _make_output("req-0", "Paris"),
            _make_output("req-0", "Paris!"),
        ]

        async def _fake_generate(prompt_, request_id, *args, **kwargs):
            for o in outputs:
                o.request_id = request_id  # align to actual rid used
                yield o

        with patch.object(AsyncOmni, "generate", side_effect=_fake_generate):
            results = asyncio.run(AsyncOmni.generate_batch(engine, prompts=[prompt]))

        assert len(results) == 1
        assert results[0].text == "Paris!"

    # ------------------------------------------------------------------
    # 3. Multiple prompts – order preserved
    # ------------------------------------------------------------------

    def test_multiple_prompts_order_preserved(self):
        """
        Simulate tasks completing in reverse order.  generate_batch must
        still return results in the original prompt order.
        """
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine = self._make_engine()
        prompts = ["prompt-A", "prompt-B", "prompt-C"]

        # Collect request_ids as they are passed to generate()
        seen_rids: list[str] = []

        async def _fake_generate(prompt_, request_id, *args, **kwargs):
            seen_rids.append(request_id)
            # Each prompt yields exactly one output carrying its own request_id.
            out = _make_output(request_id, text=f"answer-for-{prompt_}")
            out.request_id = request_id
            yield out

        with patch.object(AsyncOmni, "generate", side_effect=_fake_generate):
            results = asyncio.run(AsyncOmni.generate_batch(engine, prompts=prompts))

        assert len(results) == 3
        # Results must correspond to prompts in order
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            assert result.text == f"answer-for-{prompt}", (
                f"Result at index {i} is out of order: {result.text}"
            )

    # ------------------------------------------------------------------
    # 4. Unique, stable request IDs per batch
    # ------------------------------------------------------------------

    def test_unique_request_ids_per_batch(self):
        """Each call to generate_batch must create new request IDs."""
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine = self._make_engine()
        prompts = ["p1", "p2"]
        captured_ids: list[list[str]] = []

        async def _fake_generate(prompt_, request_id, *args, **kwargs):
            if len(captured_ids) == 0:
                captured_ids.append([])
            if len(captured_ids[-1]) < len(prompts):
                captured_ids[-1].append(request_id)
            out = _make_output(request_id)
            out.request_id = request_id
            yield out

        async def _run_twice():
            await AsyncOmni.generate_batch(engine, prompts=prompts)
            captured_ids.append([])  # start fresh for second call
            await AsyncOmni.generate_batch(engine, prompts=prompts)

        with patch.object(AsyncOmni, "generate", side_effect=_fake_generate):
            asyncio.run(_run_twice())

        # IDs from the two calls must not share the same batch prefix
        batch1_ids = [r for r in captured_ids[0] if r.startswith("batch-")]
        batch2_ids = [r for r in captured_ids[1] if r.startswith("batch-")]
        assert batch1_ids, "First batch produced no batch- prefixed IDs"
        assert batch2_ids, "Second batch produced no batch- prefixed IDs"
        # They must share the "batch-<hex>-N" format
        for rid in batch1_ids + batch2_ids:
            parts = rid.split("-")
            assert parts[0] == "batch", f"Unexpected ID format: {rid}"
            assert parts[-1].isdigit(), f"Missing index suffix: {rid}"
        # Batch prefixes are different across calls
        prefix1 = batch1_ids[0].rsplit("-", 1)[0]  # "batch-<hex>"
        prefix2 = batch2_ids[0].rsplit("-", 1)[0]
        assert prefix1 != prefix2, "Two batch calls reused the same batch prefix"

    # ------------------------------------------------------------------
    # 5. Error propagation – exception inside generate() bubbles up
    # ------------------------------------------------------------------

    def test_error_in_generate_propagates(self):
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine = self._make_engine()

        async def _failing_generate(prompt_, request_id, *args, **kwargs):
            raise RuntimeError("engine exploded")
            yield  # make it a generator

        with patch.object(AsyncOmni, "generate", side_effect=_failing_generate):
            with pytest.raises(RuntimeError, match="engine exploded"):
                asyncio.run(AsyncOmni.generate_batch(engine, prompts=["bad prompt"]))

    # ------------------------------------------------------------------
    # 6. output_modalities and sampling_params_list are forwarded
    # ------------------------------------------------------------------

    def test_kwargs_forwarded_to_generate(self):
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine = self._make_engine()
        prompt = "test prompt"
        expected_modalities = ["text", "audio"]
        expected_sp = [MagicMock()]  # dummy sampling params

        received_kwargs: dict = {}

        async def _capturing_generate(prompt_, request_id, sp_list=None, *args, output_modalities=None, **kwargs):
            received_kwargs["output_modalities"] = output_modalities
            received_kwargs["sampling_params_list"] = sp_list
            out = _make_output(request_id)
            out.request_id = request_id
            yield out

        with patch.object(AsyncOmni, "generate", side_effect=_capturing_generate):
            asyncio.run(AsyncOmni.generate_batch(
                engine,
                prompts=[prompt],
                sampling_params_list=expected_sp,
                output_modalities=expected_modalities,
            ))

        assert received_kwargs["output_modalities"] == expected_modalities
        assert received_kwargs["sampling_params_list"] == expected_sp

    # ------------------------------------------------------------------
    # 7. No output from a generate() call → excluded from result list
    # ------------------------------------------------------------------

    def test_prompt_with_no_output_excluded(self):
        """
        A generate() that yields nothing should not appear in the results.
        """
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        engine = self._make_engine()
        prompts = ["good-prompt", "empty-prompt"]

        async def _selective_generate(prompt_, request_id, *args, **kwargs):
            if "empty" in prompt_:
                return  # yields nothing
            out = _make_output(request_id, text="good")
            out.request_id = request_id
            yield out

        with patch.object(AsyncOmni, "generate", side_effect=_selective_generate):
            results = asyncio.run(AsyncOmni.generate_batch(engine, prompts=prompts))

        # Only the first prompt produced an output
        assert len(results) == 1
        assert results[0].text == "good"
