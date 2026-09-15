# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional scheduler protocol for RL trajectory export.

Stock denoise loops stay unchanged. A scheduler that implements
:class:`TrajectoryCollector` can record latents / timesteps / log-probs;
:func:`attach_scheduler_trajectory` copies them onto ``DiffusionOutput``
at the pipeline tail and at ``post_decode`` so both execution modes share
one formatter.

RL math (SDE formulas, windows, log-probs) lives in the scheduler class,
typically provided by an external package via ``vllm_omni.schedulers``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vllm_omni.diffusion.data import DiffusionOutput


@runtime_checkable
class TrajectoryCollector(Protocol):
    """Optional scheduler interface for per-request RL trajectories."""

    def configure(self, **kwargs: Any) -> None:
        """Per-request knobs (sde_type, noise_level, window)."""

    def reset(self) -> None:
        """Clear buffers. Required because step-mode deep-copies carry state."""

    def get_trajectory(self) -> dict[str, Any]:
        """Return ``latents``, ``timesteps``, ``log_probs`` (any may be None)."""


def _is_export_rank() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def configure_scheduler_for_request(scheduler: Any, sampling: Any) -> Any:
    """Reset collector buffers and apply per-request ``scheduler_configure``.

    No-op when ``scheduler`` is not a :class:`TrajectoryCollector`.
    """
    if not isinstance(scheduler, TrajectoryCollector):
        return scheduler
    scheduler.reset()
    extra = getattr(sampling, "extra_args", None) or {}
    cfg = extra.get("scheduler_configure")
    if isinstance(cfg, dict) and cfg:
        scheduler.configure(**cfg)
    return scheduler


def attach_scheduler_trajectory(output: DiffusionOutput, scheduler: Any) -> DiffusionOutput:
    """Copy collector buffers onto ``output.trajectory_*``. Rank-0 only.

    No-op when ``scheduler`` is not a :class:`TrajectoryCollector` or when
    this process is not rank 0.
    """
    if output is None or not isinstance(scheduler, TrajectoryCollector):
        return output
    if not _is_export_rank():
        return output
    traj = scheduler.get_trajectory() or {}
    if traj.get("latents") is not None:
        output.trajectory_latents = traj["latents"]
    if traj.get("timesteps") is not None:
        output.trajectory_timesteps = traj["timesteps"]
    if traj.get("log_probs") is not None:
        output.trajectory_log_probs = traj["log_probs"]
    if traj.get("decoded") is not None:
        output.trajectory_decoded = traj["decoded"]
    return output
