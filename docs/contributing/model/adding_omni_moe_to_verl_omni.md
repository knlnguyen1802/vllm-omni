# How to Adapt an Omni MoE Model into verl-omni

> Based on [PR #113 — Qwen3-Omni Thinker GSPO + LoRA](https://github.com/verl-project/verl-omni/pull/113)

---

## Slide 1 — The 3 Files You Need to Touch

| # | File | What You Do |
|---|---|---|
| **1** | `verl_omni/models/transformers/<model>.py` | **Model patch** — make the Omni model look like a causal LM to FSDP |
| **2** | `examples/gspo_trainer/<model>_only.yaml` | **Stage config** — tell vLLM-Omni how to run the thinker stage |
| **3** | `examples/gspo_trainer/config/<model>_gspo.yaml` + `.sh` | **Recipe** — training config + launch script |

**Loading mechanism:** `VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.<model>` (driver) + `actor_rollout_ref.model.external_lib=...` (workers). Patches loaded on-demand via `importlib`, never hard-imported.

---

### The Model Patch (`verl_omni/models/transformers/<model>.py`)

**Why?** Omni models have `ForConditionalGeneration` in the name but are NOT a standard causal LM — the thinker is buried inside. FSDP needs it exposed as one.

5 things to do (apply as import side-effect, idempotent):

```python
# 1. Register so verl dispatches to AutoModelForCausalLM
AutoModelForCausalLM.register(ConfigClass, ModelClass)
_architecture_to_auto_class.setdefault("ArchName", AutoModelForCausalLM)

# 2. Delegate forward & embeddings to self.thinker (the actual LM)
ModelClass.forward = lambda self, ...: self.thinker(...)
ModelClass.get_input_embeddings = lambda self: self.thinker.get_input_embeddings()

# 3. Fix FSDP blockers
ModelClass._no_split_modules = ["RealDecoderLayer"]       # upstream may be wrong
ModelClass._verl_strip_modules = ["talker", "code2wav"]    # drop unused heads
ConfigClass.tie_word_embeddings = _FalseTieDescriptor()    # else OOM on FSDP init

# 4. Wrap verl's hf_processor to recognize your multimodal processor
#    (match block can't be extended at runtime → monkey-patch wrapper)

# 5. Apply on import so it works as an external_lib target
apply_patches()
```

**MoE-specific considerations:**
- LoRA on `all-linear` covers every expert → adapter may span multiple IPC buckets
- Accumulate all buckets, then call `add_lora` **once** (vLLM's `pack_moe` needs all per-expert tensors together)
- `exclude_modules` strips non-thinker sub-modules + freezes vision tower

---

### The Stage Config (`<model>_only.yaml`)

Tells vLLM-Omni: "run one stage (thinker), use all GPUs, AR mode, with LoRA":

```yaml
stage_args:
  - stage_id: 0
    runtime:
      devices: "0,1,2,3"
    engine_args:
      model_stage: thinker
      model_arch: YourModelThinkerForConditionalGeneration
      worker_type: ar                          # autoregressive text tokens
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      hf_config_name: thinker_config           # reads config.thinker_config, not top-level
      tensor_parallel_size: 4
      gpu_memory_utilization: 0.4              # low — shares GPU with FSDP actor
      enable_lora: true
      max_lora_rank: 64
      enable_sleep_mode: true
      logprobs_mode: processed_logprobs
```

**Key flags:** `worker_type: ar` selects text-token generation. `hf_config_name` tells vLLM-Omni where to find the text config in a nested Omni config. Memory values are kept low because the engine shares each GPU with the FSDP actor.

---

## Slide 2 — Recipe, Weight Sync & Validation

### Recipe (`config/<model>_gspo.yaml`)

Inherits verl's `ppo_trainer`, overrides only what's needed:

```yaml
defaults: [ppo_trainer, _self]

actor_rollout_ref:
  model:
    lora_rank: 64
    target_modules: all-linear
    exclude_modules: ".*talker.*|.*code2wav.*|.*code_predictor.*"
  actor:
    strategy: fsdp
    fsdp_config: {param_offload: true, optimizer_offload: true}
    policy_loss: {loss_mode: gspo, loss_agg_mode: seq-mean-token-mean}
  rollout:
    name: vllm_omni
    mode: async
    engine_kwargs: {vllm_omni: {output_mode: ar}}
algorithm: {adv_estimator: grpo}
reward: {reward_manager: {name: dapo}}
```

### Weight Sync (actor → rollout, `workers/rollout/vllm_rollout/utils.py`)

Two paths, differentiated by worker type:

| Path | LoRA | Standard (no LoRA) |
|---|---|---|
| **AR worker** | `TensorLoRARequest` (verl's) → `add_lora` once after bucket accumulation | `model.load_weights(bucket)` per bucket → `process_weights_after_loading` once |
| **Diffusion worker** | `OmniTensorLoRARequest` (vllm-omni) | `self.load_weights(bucket)` (pipeline loader) |

**Why bucket-accumulate for MoE?** `pack_moe` inside vLLM needs all per-expert LoRA tensors at once; per-bucket `add_lora` overwrites previous buckets.

### Validation Signals

After 1 step, you should see:
- `rollout_actor_probs_pearson_corr` > 0.95 (actor ↔ rollout agree after sync)
- `actor/loss` ≈ 1e-4…1e-3, no OOM
- Reward rising with steps

### Smoke Test (CI-ready)

```bash
# 1. Build tiny random-weight model (no 60 GB download)
python tests/special_e2e/build_<model>_tiny_random.py --output-dir ~/models/tiny-random/<Model>

# 2. Create dummy dataset
python tests/special_e2e/create_dummy_math_data.py

# 3. Run 2 training steps end-to-end
bash tests/special_e2e/run_gspo_<model>_lora_smoke.sh
```

Wire into `tests/gpu_smoke/run_gpu_smoke_tests.sh`.

### Verified Stack

| vLLM | vLLM-Omni | transformers | torch | flash-attn |
|---|---|---|---|---|
| 0.22.0 | 0.22.0 | 4.57.6 (pin 4.x) | 2.11.0+cu130 | 2.8.3 |

> **Reference:** PR #113 added Qwen3-Omni-30B-A3B Thinker GSPO+LoRA. Full files at `examples/gspo_trainer/` and `verl_omni/models/transformers/qwen3_omni_thinker.py`.
