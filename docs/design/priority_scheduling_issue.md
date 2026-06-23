# Implement Priority Scheduling in vLLM-Omni

## Summary

vLLM added priority scheduling in vllm-project/vllm#5958, closing the RFC in vllm-project/vllm#6077. vLLM-Omni should expose and preserve the same request priority contract across Omni entrypoints, orchestrator submission, per-stage engine requests, and scheduler queues.

Priority scheduling lets latency-sensitive requests share an engine with long-running batch requests. A request with a higher scheduling priority should be selected before lower-priority waiting requests while preserving FCFS order among equal-priority requests. Upstream vLLM uses lower numeric priority values as higher priority, with `0` as the default.

## Motivation

Today `AsyncOmni.generate()` already accepts a `priority` argument and `OmniRequest.from_engine_core_request()` forwards `request.priority` into the vLLM request object. However, Omni request submission does not consistently pass the caller-provided priority into the internal engine/orchestrator path. As a result, callers can provide `priority`, but the scheduler may still see the default priority.

This limits several important serving patterns:

- Co-locating interactive and batch traffic on the same vLLM-Omni deployment.
- Allowing user-facing requests to move ahead of queued background work.
- Preserving upstream vLLM API compatibility for OpenAI-compatible clients that pass `priority` in `extra_body`.
- Enabling future external orchestration policies without requiring request abort and resend.

## Proposed Scope

### Phase 1: Priority propagation parity

- Preserve the public `priority: int = 0` argument on `AsyncOmni.generate()`.
- Forward `priority` into `AsyncOmniEngine.add_request_async()` for normal requests.
- Forward `priority` into the first submission for streaming input requests.
- Ensure `OmniEngineCoreRequest` carries the priority into `OmniRequest`.
- Add L1 CPU regression tests proving non-default priority survives `AsyncOmni.generate()` submission.

### Phase 2: Scheduler policy parity

- Verify that vLLM-Omni schedulers use vLLM's request queue created from `self.policy`, so enabling the upstream priority policy reorders waiting requests by priority.
- Add scheduler-level coverage for lower numeric priority being scheduled before default-priority requests when the scheduler policy is `priority`.
- Preserve FCFS order among equal-priority requests.
- Keep default behavior unchanged when the policy is FCFS.

### Phase 2b: Diffusion scheduler parity

- Add a defaulted `priority` field to `OmniDiffusionRequest`.
- Order diffusion scheduler waiting requests by `(priority, arrival order)`, with lower numeric priority scheduled first.
- Preserve batching constraints: a higher-priority incompatible request at the waiting head should not be bypassed by a lower-priority compatible request.
- Do not preempt already-running diffusion requests in the first implementation. Diffusion priority only affects waiting requests.
- Forward priority through `StagePool`, inline diffusion stage clients, and subprocess diffusion stage clients.

### Phase 3: Strict preemption parity, if needed

- Evaluate whether vLLM-Omni should mirror upstream strict priority preemption from vllm-project/vllm#5958.
- If implemented, preempt only when the configured policy is `priority`.
- Avoid regressions with chunked prefill, async chunk transfer, full-payload input waiting, and multi-stage KV transfer.
- Add GPU integration coverage before enabling this path broadly.

## Non-Goals

- Dynamic priority updates during a request lifetime.
- A new scheduling metadata dataclass.
- Global fairness or tenant quota policy inside vLLM-Omni.
- Changing priority semantics from upstream vLLM. Lower numeric values should continue to mean higher priority.

## User-Facing Contract

- Default priority is `0`.
- Lower numeric values are scheduled earlier than higher numeric values.
- Equal priorities fall back to FCFS behavior.
- Priority must be accepted by direct `AsyncOmni.generate()` callers.
- OpenAI-compatible serving should pass request `priority` from `extra_body` to the engine path where the underlying protocol exposes it.

## Acceptance Criteria

- A direct `AsyncOmni.generate(..., priority=-10)` request submits `priority=-10` to the internal engine.
- Streaming-input `AsyncOmni.generate(..., priority=-10)` submits the first request with `priority=-10`.
- Priority remains default `0` when callers omit it.
- Request construction preserves priority through `OmniEngineCoreRequest -> OmniRequest`.
- Diffusion request construction preserves `priority`, defaulting to `0`.
- Diffusion schedulers schedule waiting request priority `-10` before priority `0`.
- Diffusion schedulers preserve FCFS ordering among equal-priority waiting requests.
- When scheduler policy is `priority`, a waiting request with priority `-10` is considered before a waiting request with priority `0`.
- When scheduler policy is not `priority`, existing FCFS behavior and performance are unchanged.

## Test Plan

- L1 CPU tests for `AsyncOmni.generate()` priority forwarding.
- L1 CPU tests for streaming first-request priority forwarding.
- Scheduler unit tests for priority ordering and FCFS tie-breaks if the local scheduler path does not already inherit upstream coverage.
- L1 CPU tests for diffusion request priority defaults and waiting-queue priority ordering.
- Follow-up integration test with OpenAI-compatible `extra_body={"priority": -10}` once endpoint routing is audited.

## References

- vllm-project/vllm#6077: RFC for priority scheduling.
- vllm-project/vllm#5958: upstream priority scheduling implementation.
- vllm-project/vllm#8850: async engine priority support follow-up.