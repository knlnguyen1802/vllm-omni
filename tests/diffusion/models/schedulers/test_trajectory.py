# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for TrajectoryCollector attach helper."""

import copy

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.schedulers.trajectory import (
    TrajectoryCollector,
    attach_scheduler_trajectory,
    configure_scheduler_for_request,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Collector:
    def __init__(self, traj: dict):
        self._traj = traj

    def configure(self, **kwargs):
        self.kwargs = kwargs

    def reset(self):
        self._traj = {}

    def get_trajectory(self):
        return dict(self._traj)


def test_collector_is_runtime_checkable():
    assert isinstance(_Collector({}), TrajectoryCollector)


def test_attach_noop_without_protocol():
    output = DiffusionOutput(output="img")
    result = attach_scheduler_trajectory(output, object())
    assert result is output
    assert result.trajectory_latents is None


def test_attach_copies_fields():
    output = DiffusionOutput(output="img")
    sched = _Collector({"latents": "L", "timesteps": "T", "log_probs": "P"})
    result = attach_scheduler_trajectory(output, sched)
    assert result.trajectory_latents == "L"
    assert result.trajectory_timesteps == "T"
    assert result.trajectory_log_probs == "P"


def test_reset_clears_buffers():
    sched = _Collector({"latents": "L"})
    sched.reset()
    assert sched.get_trajectory() == {}


def test_configure_stores_kwargs():
    sched = _Collector({})
    sched.configure(sde_type="sde", noise_level=0.7)
    assert sched.kwargs == {"sde_type": "sde", "noise_level": 0.7}


def test_configure_scheduler_for_request_resets_and_applies_kwargs():
    class _S:
        extra_args = {"scheduler_configure": {"sde_type": "cps", "noise_level": 0.3}}

    sched = _Collector({"latents": "stale"})
    configure_scheduler_for_request(sched, _S())
    assert sched.get_trajectory() == {}
    assert sched.kwargs == {"sde_type": "cps", "noise_level": 0.3}


def test_deepcopy_isolates_collector_buffers():
    sched = _Collector({"latents": "L"})
    clone = copy.deepcopy(sched)
    clone.reset()
    assert sched.get_trajectory() == {"latents": "L"}
    assert clone.get_trajectory() == {}


def test_stock_scheduler_is_not_a_collector():
    from vllm_omni.diffusion.models.schedulers import FlowMatchEulerDiscreteScheduler

    assert not isinstance(FlowMatchEulerDiscreteScheduler(), TrajectoryCollector)
