"""Host copies of the fixed-shape pose output tensor."""

from __future__ import annotations

import ctypes
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_ROW_WIDTH = 57
_CPU_DEVICE_TYPE = 1


class CudaRuntime(Protocol):
    def cudaMemcpy(self, destination: int, source: int, size: int, kind: int) -> int: ...


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_cudart: CudaRuntime | None = None


def _cuda_runtime() -> CudaRuntime:
    global _cudart
    if _cudart is None:
        runtime = ctypes.CDLL("libcudart.so")
        runtime.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        runtime.cudaMemcpy.restype = ctypes.c_int
        _cudart = cast(CudaRuntime, runtime)
    return _cudart


def rows_from_tensor(
    tensor: Any,
    *,
    cudart: CudaRuntime | None = None,
) -> NDArray[np.float32]:
    """Return the tensor's ``[N,57]`` rows as an owned host float32 array.

    A NumPy input is an intentional host seam for tests.  Vendor tensors expose
    DLPack and are copied with cudart only when their device is not CPU.
    """
    if isinstance(tensor, np.ndarray):
        rows = np.asarray(tensor, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != _ROW_WIDTH:
            raise ValueError(
                f"pose tensor must have shape [N, {_ROW_WIDTH}], received {rows.shape}"
            )
        return rows.copy()

    capsule = tensor.__dlpack__(None)
    managed = ctypes.cast(
        _capsule_pointer(capsule, b"dltensor"), ctypes.POINTER(_DLManagedTensor)
    ).contents
    dl_tensor = managed.dl_tensor
    if dl_tensor.dtype.code != 2 or dl_tensor.dtype.bits != 32 or dl_tensor.dtype.lanes != 1:
        raise ValueError("pose tensor must be float32")
    count = 1
    for index in range(dl_tensor.ndim):
        count *= int(dl_tensor.shape[index])
    if count % _ROW_WIDTH != 0:
        raise ValueError(f"pose tensor element count {count} is not divisible by {_ROW_WIDTH}")
    host = np.empty(count, dtype=np.float32)
    source = int(dl_tensor.data) + int(dl_tensor.byte_offset)
    if dl_tensor.device.device_type == _CPU_DEVICE_TYPE:
        ctypes.memmove(host.ctypes.data, source, host.nbytes)
    else:
        result = (cudart if cudart is not None else _cuda_runtime()).cudaMemcpy(
            host.ctypes.data, source, host.nbytes, _CUDA_MEMCPY_DEVICE_TO_HOST
        )
        if result != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed: {result}")
    return host.reshape((-1, _ROW_WIDTH))


__all__ = ["CudaRuntime", "rows_from_tensor"]
