# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.models.schedulers.registry import (
    build_pipeline_scheduler,
    ensure_scheduler_consumed,
    is_injected_scheduler,
    register_scheduler,
    resolve_scheduler_cls,
)
from vllm_omni.diffusion.models.schedulers.scheduling_dmd2_euler import DMD2EulerScheduler
from vllm_omni.diffusion.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)

from vllm_omni.diffusion.models.schedulers.trajectory import (
    TrajectoryCollector,
    attach_scheduler_trajectory,
    configure_scheduler_for_request,
)

__all__ = [
    "DMD2EulerScheduler",
    "FlowMatchEulerDiscreteScheduler",
    "FlowUniPCMultistepScheduler",
    "TrajectoryCollector",
    "attach_scheduler_trajectory",
    "build_pipeline_scheduler",
    "configure_scheduler_for_request",
    "ensure_scheduler_consumed",
    "is_injected_scheduler",
    "register_scheduler",
    "resolve_scheduler_cls",
]
