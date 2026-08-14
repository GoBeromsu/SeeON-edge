"""Deterministic fake device-resident adapter for hosts without NVIDIA hardware.

This repo's CI/dev machines are Apple Silicon (no CUDA device). Every
lifecycle/backpressure/transfer-accounting/parity claim this prototype makes
still has to be proven by a real, deterministic test -- just not against real
GPU memory. ``FakeDeviceStorage`` stands in for a CUDA allocation: a plain
host-side ``numpy`` buffer tagged with a non-host ``MemoryKind`` so every
capability/ownership/lifetime rule in
``worker.adapters.decode.nvdec_device.pool`` is exercised exactly as it would
be against a real CUDA tensor, while ``FakeDeviceResidentBatcher`` proves
batch formation and numeric parity against the plain-CPU reference
computation. Neither type is ever selected by a production profile: they are
constructed only by tests and by
``worker.adapters.decode.nvdec_device.diagnostic``'s explicit
``--fake``-selected dry run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np
from numpy.typing import NDArray

from worker.adapters.decode.nvdec_device.models import DeviceResidentPoolConfig
from worker.adapters.decode.nvdec_device.pool import DeviceResidentFramePool
from worker.adapters.decode.nvdec_device.telemetry import DeviceResidencyTelemetry
from worker.types import FrameDescriptor, FrameLease, MemoryKind, PixelFormat

_RGB_CHANNELS = 3


@final
class FakeDeviceAllocator:
    """Mint/track fake device-resident NV12-shaped RGB24 buffers.

    Buffers are plain ``numpy`` arrays -- never a real device allocation --
    but every handle is opaque ``object`` to pool code, so this exercises the
    exact same code path a real ``torch.cuda`` allocator would.
    """

    def __init__(self, *, width: int, height: int, telemetry: DeviceResidencyTelemetry) -> None:
        self._width = width
        self._height = height
        self._telemetry = telemetry
        self.allocations = 0

    def __call__(self) -> tuple[object, FrameDescriptor]:
        self.allocations += 1
        buffer = np.zeros((self._height, self._width, _RGB_CHANNELS), dtype=np.uint8)
        descriptor = FrameDescriptor(
            width=self._width,
            height=self._height,
            memory_kind=MemoryKind.CUDA_DEVICE,
            pixel_format=PixelFormat.RGB24,
            plane_strides=(int(buffer.strides[0]),),
            size_bytes=int(buffer.nbytes),
        )
        return buffer, descriptor

    def upload(self, lease: FrameLease, host_image: NDArray[np.uint8]) -> None:
        """Simulate one H2D copy into an acquired device-resident lease."""
        handle = lease.device_handle
        assert isinstance(handle, np.ndarray)  # noqa: S101 - fake allocator invariant
        if handle.shape != host_image.shape:
            raise ValueError("fake device upload shape mismatch")
        handle[...] = host_image
        self._telemetry.record_h2d(int(host_image.nbytes))

    def download(self, lease: FrameLease) -> NDArray[np.uint8]:
        """Simulate one D2H readback from an acquired device-resident lease."""
        handle = lease.device_handle
        assert isinstance(handle, np.ndarray)  # noqa: S101 - fake allocator invariant
        copy = handle.copy()
        self._telemetry.record_d2h(int(copy.nbytes))
        return copy


def fake_device_resident_pool(
    *, camera_id: str, capacity: int, width: int, height: int
) -> tuple[DeviceResidentFramePool, FakeDeviceAllocator]:
    telemetry = DeviceResidencyTelemetry(pool_capacity=capacity)
    allocator = FakeDeviceAllocator(width=width, height=height, telemetry=telemetry)
    config = DeviceResidentPoolConfig(
        camera_id=camera_id, capacity=capacity, width=width, height=height
    )
    pool = DeviceResidentFramePool(config, allocate=allocator, telemetry=telemetry)
    return pool, allocator


@final
class FakeDeviceResidentBatcher:
    """Deterministic batch formation plus a numeric reference "inference".

    ``mean_rgb`` is a stand-in normalized observation -- a fixed, trivially
    checkable per-batch numeric reduction, not a real detector. It exists so
    tests can assert CPU/"CUDA" (fake) parity on the exact same numeric
    contract Todo 17's acceptance criteria requires ("pinned fixtures produce
    normalized observations within declared CPU/CUDA tolerance") without
    depending on a real model artifact.
    """

    def __init__(self, *, max_batch_size: int, allocator: FakeDeviceAllocator) -> None:
        if max_batch_size <= 0:
            raise ValueError("max batch size must be positive")
        self._max_batch_size = max_batch_size
        self._allocator = allocator

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def form_batch(self, leases: Sequence[FrameLease]) -> Sequence[FrameLease]:
        if len(leases) > self._max_batch_size:
            raise ValueError(
                f"batch of {len(leases)} exceeds max_batch_size={self._max_batch_size}"
            )
        return tuple(leases)

    def infer_mean_rgb(
        self, leases: Sequence[FrameLease]
    ) -> tuple[tuple[float, float, float], ...]:
        """Device-resident numeric reduction: never leaves device memory until this result.

        Each per-frame mean is computed directly against the pool's
        device-resident buffer (no read-through, no D2H copy of the full
        frame) -- only the three-float reduced result crosses back to host
        memory, matching the plan's "stable host-readback only at declared
        seams" requirement.
        """
        results: list[tuple[float, float, float]] = []
        for lease in leases:
            handle = lease.device_handle
            assert isinstance(handle, np.ndarray)  # noqa: S101 - fake allocator invariant
            means = handle.reshape(-1, _RGB_CHANNELS).mean(axis=0)
            results.append((float(means[0]), float(means[1]), float(means[2])))
        return tuple(results)


__all__ = [
    "FakeDeviceAllocator",
    "FakeDeviceResidentBatcher",
    "fake_device_resident_pool",
]
