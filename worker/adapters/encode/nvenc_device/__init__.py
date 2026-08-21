"""Experimental CUDA overlay + device-input NVENC prototype (Todo 18).

Backs the unified ``nvidia`` profile
(``worker/runtime/profile/registry.py``) only, same as
``worker.adapters.decode.nvdec_device`` (Todo 17). The host-only profiles
(``cpu-host``, ``intel-vaapi-host``, ``apple-mps-host``) construct nothing
from this package.

- ``renderer``: ``CudaOverlaySceneRenderer`` -- consumes the identical
  ``worker.pipeline.output.overlay_scene.OverlaySceneBuilder`` scene the CPU
  renderer draws, on a retained device-resident ``FrameLease``.
- ``models``/``capability``/``telemetry``/``errors``: declared codec/
  container/profile candidates, combined device-resident+NVENC capability
  probe, transfer/pressure/outcome counters, and fail-closed error types.
- ``fake``: deterministic doubles for this repo's non-NVIDIA CI/dev hosts.
- ``diagnostic``: standalone operator CLI, never imported by production boot.
"""

from __future__ import annotations

from worker.adapters.encode.nvenc_device.capability import (
    DeviceInputNvencCapability,
    probe_device_input_nvenc_capability,
)
from worker.adapters.encode.nvenc_device.errors import (
    DeviceEncoderPoolExhaustedError,
    DeviceEncoderRejectedInputError,
    DeviceEncoderUnavailableError,
)
from worker.adapters.encode.nvenc_device.models import (
    DeviceEncoderCodec,
    DeviceEncoderContainer,
    DeviceEncoderPoolConfig,
    DeviceEncoderProfile,
    DeviceEncoderSelection,
)
from worker.adapters.encode.nvenc_device.renderer import (
    CudaOverlaySceneRenderer,
    DeviceRenderMetrics,
    DeviceSceneDrawer,
    DeviceSceneRenderError,
)
from worker.adapters.encode.nvenc_device.telemetry import (
    DeviceEncoderTelemetry,
    DeviceEncoderTelemetrySnapshot,
)

__all__ = [
    "CudaOverlaySceneRenderer",
    "DeviceEncoderCodec",
    "DeviceEncoderContainer",
    "DeviceEncoderPoolConfig",
    "DeviceEncoderPoolExhaustedError",
    "DeviceEncoderProfile",
    "DeviceEncoderRejectedInputError",
    "DeviceEncoderSelection",
    "DeviceEncoderTelemetry",
    "DeviceEncoderTelemetrySnapshot",
    "DeviceEncoderUnavailableError",
    "DeviceInputNvencCapability",
    "DeviceRenderMetrics",
    "DeviceSceneDrawer",
    "DeviceSceneRenderError",
    "probe_device_input_nvenc_capability",
]
