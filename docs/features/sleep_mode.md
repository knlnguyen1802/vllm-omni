# Sleep Mode

vLLM-Omni’s **Sleep Mode** allows you to temporarily release most GPU memory used by a model—such as model weights and key-value (KV) caches (for autoregressive models)—**without stopping the server or unloading the Docker container**.

This feature is inherited from [vLLM’s Sleep Mode](https://blog.vllm.ai/2025/10/26/sleep-mode.html), which provides zero-reload model switching for multi-model serving.  
It is especially useful in **RLHF**, **training**, or **cost-saving scenarios**, where GPU resources must be freed between inference workloads.

---

## 🧠 Diffusion Model Extension

We added Sleep Mode support for **diffusion models**, which previously lacked this functionality.  
In diffusion pipelines, this currently only offloads **model weight memory**, as these models typically do not use KV caches.

This means:
- Diffusion models can now enter Level 1 sleep.
- Pipeline states (e.g., noise schedulers, buffers) remain intact after waking.
- Useful for releasing VRAM between image generation or training cycles.

---

## Enable sleep mode
To enable sleep mode, set the `enable_sleep_mode` in `engine_args` to True
Example:
```
omni = Omni(model=...,enable_sleep_mode=True)
```