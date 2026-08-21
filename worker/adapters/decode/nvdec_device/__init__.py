"""NVIDIA device-resident analysis pool.

Backs the unified ``nvidia`` profile
(``worker/runtime/profile/registry.py``) only. The host-only profiles
(``cpu-host``, ``intel-vaapi-host``, ``apple-mps-host``) construct nothing
from this package -- see ``worker/runtime/profile/boot.py:verify_device_or_raise``,
which gates this profile's concrete stages behind
``probe_device_resident_capability`` and still fails closed on any host that
cannot prove real NVDEC/CUDA-stream/DLPack support.
"""

from __future__ import annotations

from worker.adapters.decode.nvdec_device.capability import (
    DeviceResidentCapability,
    probe_device_resident_capability,
)
from worker.adapters.decode.nvdec_device.errors import (
    DevicePoolExhaustedError,
    DeviceResidentUnavailableError,
)
from worker.adapters.decode.nvdec_device.models import DeviceResidentPoolConfig
from worker.adapters.decode.nvdec_device.pool import (
    DeviceResidentFramePool,
    DeviceResidentPoolStatus,
    SlotRecycler,
    StorageAllocator,
)
from worker.adapters.decode.nvdec_device.telemetry import (
    DeviceResidencyTelemetry,
    DeviceResidencyTelemetrySnapshot,
)

__all__ = [
    "DevicePoolExhaustedError",
    "DeviceResidencyTelemetry",
    "DeviceResidencyTelemetrySnapshot",
    "DeviceResidentCapability",
    "DeviceResidentFramePool",
    "DeviceResidentPoolConfig",
    "DeviceResidentPoolStatus",
    "DeviceResidentUnavailableError",
    "SlotRecycler",
    "StorageAllocator",
    "probe_device_resident_capability",
]
