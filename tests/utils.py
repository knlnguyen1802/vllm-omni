# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
import threading
from contextlib import contextmanager

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from amdsmi import (
        amdsmi_get_gpu_vram_usage,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
    )

    @contextmanager
    def _nvml():
        try:
            amdsmi_init()
            yield
        finally:
            amdsmi_shut_down()
elif current_platform.is_cuda():
    from vllm.third_party.pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

    @contextmanager
    def _nvml():
        try:
            nvmlInit()
            yield
        finally:
            nvmlShutdown()
else:

    @contextmanager
    def _nvml():
        yield


def get_physical_device_indices(devices):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return devices

    visible_indices = [int(x) for x in visible_devices.split(",")]
    index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
    return [index_mapping[i] for i in devices if i in index_mapping]


@_nvml()
def wait_for_gpu_memory_to_clear(
    *,
    devices: list[int],
    threshold_bytes: int | None = None,
    threshold_ratio: float | None = None,
    timeout_s: float = 120,
) -> None:
    assert threshold_bytes is not None or threshold_ratio is not None
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    devices = get_physical_device_indices(devices)
    start_time = time.time()
    while True:
        output: dict[int, str] = {}
        output_raw: dict[int, tuple[float, float]] = {}
        for device in devices:
            if current_platform.is_rocm():
                dev_handle = amdsmi_get_processor_handles()[device]
                mem_info = amdsmi_get_gpu_vram_usage(dev_handle)
                gb_used = mem_info["vram_used"] / 2**10
                gb_total = mem_info["vram_total"] / 2**10
            else:
                dev_handle = nvmlDeviceGetHandleByIndex(device)
                mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
                gb_used = mem_info.used / 2**30
                gb_total = mem_info.total / 2**30
            output_raw[device] = (gb_used, gb_total)
            output[device] = f"{gb_used:.02f}/{gb_total:.02f}"

        print("gpu memory used/total (GiB): ", end="")
        for k, v in output.items():
            print(f"{k}={v}; ", end="")
        print("")

        if threshold_bytes is not None:
            is_free = lambda used, total: used <= threshold_bytes / 2**30  # noqa E731
            threshold = f"{threshold_bytes / 2**30} GiB"
        else:
            is_free = lambda used, total: used / total <= threshold_ratio  # noqa E731
            threshold = f"{threshold_ratio:.2f}"

        dur_s = time.time() - start_time
        if all(is_free(used, total) for used, total in output_raw.values()):
            print(f"Done waiting for free GPU memory on devices {devices=} ({threshold=}) {dur_s=:.02f}")
            break

        if dur_s >= timeout_s:
            raise ValueError(f"Memory of devices {devices=} not free after {dur_s=:.02f} ({threshold=})")

        time.sleep(5)

class GPUMemoryMonitor:
    """Poll global device memory usage via CUDA APIs."""

    def __init__(self, device_index: int, interval: float = 0.05):
        self.device_index = device_index
        self.interval = interval
        self._peak_used_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def monitor_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    with torch.cuda.device(self.device_index):
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                    used_mb = (total_bytes - free_bytes) / (1024**2)
                    self._peak_used_mb = max(self._peak_used_mb, used_mb)
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=monitor_loop, daemon=False)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_used_mb(self) -> float:
        fallback_alloc = torch.cuda.max_memory_allocated(device=self.device_index) / (1024**2)
        fallback_reserved = torch.cuda.max_memory_reserved(device=self.device_index) / (1024**2)
        return max(self._peak_used_mb, fallback_alloc, fallback_reserved)

    def __del__(self):
        self.stop()