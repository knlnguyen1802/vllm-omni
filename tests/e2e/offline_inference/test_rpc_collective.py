import os
import sys
from pathlib import Path

import pytest
import asyncio

from .utils import create_new_process_for_each_test

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.async_omni import AsyncOmni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

models = ["/mnt/nvme3n1/n0090/Z-Image-Turbo"]

@create_new_process_for_each_test()
@pytest.mark.parametrize("model_name", models)
def test_rpc_collective_omni(model_name: str):
    m = Omni(model=model_name, enable_sleep_mode=True)
    sleep_results = m.collective_rpc(
        method="sleep",
        args=(1,),
    )
    assert len(sleep_results) == 1
    wake_up_results = m.collective_rpc(
        method="wake_up",
        args=(["weights"],),
    )
    assert len(wake_up_results) == 1


@create_new_process_for_each_test()
@pytest.mark.parametrize("model_name", models)
def test_rpc_collective_async_omni(model_name: str):
    async def _run():
        m = AsyncOmni(model=model_name, enable_sleep_mode=True)
        sleep_results = await m.collective_rpc(method="sleep", args=(1,))
        assert len(sleep_results) == 1
        wake_up_results = await m.collective_rpc(method="wake_up", args=(["weights"],))
        assert len(wake_up_results) == 1

    asyncio.run(_run())

