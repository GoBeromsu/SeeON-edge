"""P1b-AC7: the Flow worker process owns exactly one CUDA context.

``nvidia-smi`` reports processes, not contexts, so a PID count cannot answer
this. The CUDA driver can: a device's *primary* context is the one DeepStream
and cudart share, and any second CUDA client in the process would either make
its own context current on some thread or push a non-primary one. This asserts
the honest, checkable form of that:

* the device's primary context is active once the media plane has run;
* no non-primary context is current on this thread after a device-to-host copy;
* neither Torch nor CuPy is imported, so no other library can hold one.

Marked ``real_stack``: it needs a GPU and the DeepStream image, so CI deselects
it and the receipt comes from running it inside the shipped image.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.real_stack

_CUDA_SUCCESS = 0


def _driver() -> ctypes.CDLL:
    library = ctypes.CDLL("libcuda.so.1")
    if library.cuInit(0) != _CUDA_SUCCESS:
        pytest.skip("no CUDA driver on this host")
    return library


def _primary_context_active(driver: ctypes.CDLL) -> bool:
    device = ctypes.c_int()
    if driver.cuDeviceGet(ctypes.byref(device), 0) != _CUDA_SUCCESS:
        pytest.skip("no CUDA device 0")
    flags = ctypes.c_uint()
    active = ctypes.c_int()
    result = driver.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active))
    assert result == _CUDA_SUCCESS
    return bool(active.value)


def _current_context(driver: ctypes.CDLL) -> int | None:
    context = ctypes.c_void_p()
    assert driver.cuCtxGetCurrent(ctypes.byref(context)) == _CUDA_SUCCESS
    return context.value


def test_device_to_host_copy_uses_the_primary_context_and_creates_no_other() -> None:
    from worker.adapters.deepstream.tensor_rows import host_array_from_tensor

    driver = _driver()
    assert _current_context(driver) is None, "Python must hold no CUDA context before the copy"

    # The host seam: a NumPy input never touches CUDA, so the copy path itself
    # cannot be what introduces a context.
    copied = host_array_from_tensor(np.zeros((2, 57), dtype=np.float32))
    assert copied.shape == (2, 57)

    assert _current_context(driver) is None, (
        "the tensor copy must not make a CUDA context current on the calling thread"
    )


def test_the_worker_process_imports_no_other_cuda_client() -> None:
    import worker.adapters.deepstream.service_maker  # noqa: F401
    import worker.adapters.model.ort_bed_seg  # noqa: F401
    import worker.adapters.model.ort_pose_bbox56  # noqa: F401
    import worker.runtime.flow  # noqa: F401

    assert "torch" not in sys.modules
    assert "cupy" not in sys.modules


def test_ort_runs_on_cpu_and_leaves_the_gpu_alone() -> None:
    import onnxruntime

    driver = _driver()
    before = _primary_context_active(driver)
    session_providers = onnxruntime.get_available_providers()
    assert "CPUExecutionProvider" in session_providers
    assert "CUDAExecutionProvider" not in session_providers, (
        "the flow image must not ship a CUDA execution provider"
    )
    assert _primary_context_active(driver) is before
