# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract tests linking ``supports_request_batch`` to ``forward()`` return type.

Guard against the class of bug where a pipeline *inherits*
``supports_request_batch = True`` from a base class while its ``forward()``
returns a single ``DiffusionOutput`` instead of ``list[DiffusionOutput]``.

When ``supports_request_batch`` is ``True`` the engine routes execution
through the request-batch path (``execute_model_batch``), which calls
``_normalize_pipeline_outputs(..., allow_single_output=False)`` and rejects a
bare ``DiffusionOutput``. A pipeline whose ``forward()`` returns a single
``DiffusionOutput`` must therefore declare ``supports_request_batch = False``
so the engine uses the single-request path that accepts it.
"""

from __future__ import annotations

import typing
from contextlib import contextmanager
from types import SimpleNamespace
from typing import get_type_hints

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from tests.e2e.offline_inference.custom_pipeline.qwen_image_pipeline_with_logprob import (
    QwenImagePipelineWithLogProbForTest,
)
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _forward_returns_list(pipeline_cls: type) -> bool:
    """Return True iff ``pipeline_cls.forward`` annotates a list return type."""
    hints = get_type_hints(pipeline_cls.forward)
    annotation = hints.get("return")
    origin = typing.get_origin(annotation)
    return origin in (list, typing.List)


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


def _fake_platform_for_peak_memory():
    return SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
    )


def _make_batch_runner(pipeline):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.cache_backend = None
    runner.offload_backend = None
    runner.prompt_embed_cache = None
    runner.state_cache = {}
    runner.od_config = SimpleNamespace(
        cache_backend="none",
        enable_cache_dit_summary=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
        streaming_output=False,
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None,
    )
    runner._kv_prefetch_enabled = False
    return runner


def _make_batch_scheduler_output(num_reqs: int):
    reqs = []
    for i in range(num_reqs):
        reqs.append(
            SimpleNamespace(
                request_id=f"req-{i}",
                prompt=f"prompt-{i}",
                sampling_params=SimpleNamespace(
                    generator=None,
                    seed=None,
                    generator_device=None,
                    num_inference_steps=2,
                ),
                skip_cache_refresh=True,
                kv_sender_info=None,
            )
        )
    return SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(request_id=req.request_id, req=req) for req in reqs],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
    )


def test_test_pipeline_declares_supports_request_batch_consistent_with_forward_return():
    """QwenImagePipelineWithLogProbForTest.forward returns a single DiffusionOutput,
    so it must declare ``supports_request_batch = False`` to use the single-request
    path that accepts a bare DiffusionOutput."""
    assert QwenImagePipelineWithLogProbForTest.supports_request_batch is False
    assert _forward_returns_list(QwenImagePipelineWithLogProbForTest) is False


class _BatchSingleOutputFromBasePipeline:
    """Mirrors the verl-omni bug: inherits ``supports_request_batch = True``
    from a base but returns a bare ``DiffusionOutput`` from ``forward()``."""

    supports_request_batch = True

    def forward(self, batch: DiffusionRequestBatch) -> DiffusionOutput:
        return DiffusionOutput(output=batch.prompts[0])


def test_request_batch_pipeline_returning_single_diffusion_output_is_rejected(monkeypatch):
    """A pipeline declaring ``supports_request_batch = True`` but returning a bare
    DiffusionOutput must be rejected by the batch path -- the exact mismatch that
    breaks a subclass inheriting ``supports_request_batch = True`` while keeping a
    single-output ``forward()``."""
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    runner = _make_batch_runner(_BatchSingleOutputFromBasePipeline())
    sched = _make_batch_scheduler_output(num_reqs=1)

    with pytest.raises(RuntimeError, match="request-batch forward must return list\\[DiffusionOutput\\]"):
        DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)
