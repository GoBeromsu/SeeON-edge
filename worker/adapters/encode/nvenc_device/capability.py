"""Real capability probe for the experimental device-input NVENC encoder seam.

Answers a strictly narrower question than either existing probe alone:

- ``worker.adapters.decode.nvdec_device.capability.probe_device_resident_capability``
  proves this process can hold decoded frames device-resident with
  CUDA-event-ordered lifetime and DLPack handoff (Todo 17) -- a prerequisite
  for handing an *already device-resident* overlaid surface to NVENC without
  a host round-trip, but it says nothing about whether an NVENC encoder
  session can actually be opened.
- ``worker.adapters.device.cuda.probe.probe_nvenc_capability`` proves the
  ``ffmpeg`` binary on this host was built with the ``h264_nvenc`` encoder --
  the same check the production nvenc/libx264 clip-encoder fallback (#53)
  uses -- but that check alone says nothing about device-resident input; the
  production path always feeds NVENC from a host buffer.

``probe_device_input_nvenc_capability`` requires *both*: device-resident
concrete stages available AND an ffmpeg build that reports ``h264_nvenc``.
Neither sub-probe is weakened or reimplemented here; this module only
combines their existing, already-proven results with a fail-closed AND.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from worker.adapters.decode.nvdec_device.capability import (
    DeviceResidentCapability,
    probe_device_resident_capability,
)
from worker.adapters.device.cuda.probe import NvencCapability, probe_nvenc_capability


@dataclass(frozen=True, slots=True)
class DeviceInputNvencCapability:
    """Fail-closed answer for whether the device-input NVENC seam can open."""

    available: bool
    reason: str
    device_resident: DeviceResidentCapability
    nvenc: NvencCapability


def probe_device_input_nvenc_capability(
    *,
    device_resident_probe: Callable[[], DeviceResidentCapability] = (
        probe_device_resident_capability
    ),
    nvenc_probe: Callable[[], NvencCapability] = probe_nvenc_capability,
) -> DeviceInputNvencCapability:
    """Never raises. Reports ``available=True`` only when every gate is real.

    Order of checks (each an independent, fail-closed gate, matching
    ``probe_device_resident_capability``'s convention):

    1. Device-resident concrete stages (CUDA, NVML device identity,
       ``torch.cuda.Stream``/``Event``, DLPack) -- without this there is no
       truthful device-resident surface to feed NVENC in the first place.
    2. ``ffmpeg`` built with ``h264_nvenc`` -- without this no NVENC session
       can be opened at all, device-resident or not.
    """
    device_resident = device_resident_probe()
    if not device_resident.available:
        return DeviceInputNvencCapability(
            available=False,
            reason=f"device-resident capability unavailable: {device_resident.reason}",
            device_resident=device_resident,
            nvenc=NvencCapability(False, "not probed: device-resident capability failed first"),
        )

    nvenc = nvenc_probe()
    if not nvenc.available:
        return DeviceInputNvencCapability(
            available=False,
            reason=f"h264_nvenc encoder unavailable: {nvenc.reason}",
            device_resident=device_resident,
            nvenc=nvenc,
        )

    return DeviceInputNvencCapability(
        available=True,
        reason="device-input NVENC concrete stages are available",
        device_resident=device_resident,
        nvenc=nvenc,
    )


__all__ = ["DeviceInputNvencCapability", "probe_device_input_nvenc_capability"]
