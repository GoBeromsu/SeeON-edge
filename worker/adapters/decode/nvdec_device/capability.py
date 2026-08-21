"""Real capability probe gating the unified `nvidia` profile.

This answers a strictly narrower question than
``worker.adapters.device.cuda.probe.probe_cuda_capability``: not merely "can
this process construct a ``device='cuda'`` model", but "can this process
additionally hold NVDEC-decoded frames device-resident, move them through a
bounded pool with CUDA-event-ordered lifetime, and hand them to an in-process
CUDA inference runtime without a full-frame host round-trip". Since the NVIDIA
profiles were unified, this probe is the only device gate for ``nvidia``:
plain ``torch.cuda`` usability no longer admits any public profile. Todo 7
(`worker/runtime/profile/registry.py`) still marks the profile's concrete
stages ``concrete_stages_available=False`` until those stages ship.

Every check below reads only ``torch``'s own documented public API
(``torch.cuda.is_available``, ``torch.cuda.Stream``, ``torch.cuda.Event``,
``torch.from_dlpack``, ``Tensor.__dlpack__``) plus the existing
``probe_cuda_capability``/``probe_nvml_gpu_status`` adapters already proven in
this repo. No undocumented or guessed API is touched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from worker.adapters.device.cuda.probe import CudaCapability, probe_cuda_capability
from worker.adapters.device.nvml.probe import NvmlGpuStatus, probe_nvml_gpu_status

TorchImporter: TypeAlias = Callable[[], Any]


def _import_torch() -> Any:
    import torch

    return torch


@dataclass(frozen=True, slots=True)
class DeviceResidentCapability:
    """Fail-closed answer for whether the experimental concrete stages can boot."""

    available: bool
    reason: str
    cuda: CudaCapability
    nvml: NvmlGpuStatus
    stream_event_supported: bool
    dlpack_supported: bool


def probe_device_resident_capability(
    *,
    torch_importer: TorchImporter = _import_torch,
    cuda_probe: Callable[[], CudaCapability] = probe_cuda_capability,
    nvml_probe: Callable[[], NvmlGpuStatus] = probe_nvml_gpu_status,
) -> DeviceResidentCapability:
    """Never raises. Reports ``available=True`` only when every stage is real.

    Order of checks (each one an independent, fail-closed gate -- a later
    check never runs past an earlier failure, matching
    ``probe_cuda_capability``'s "never raise, but report the first
    disqualifying reason" convention):

    1. ``probe_cuda_capability`` -- the process can construct a CUDA model at
       all. This is a strict prerequisite: an experimental device-resident
       pipeline that cannot even do the production ``cuda`` path has nothing
       to build on.
    2. ``probe_nvml_gpu_status`` -- NVML can enumerate the real device this
       pool would be pinned to (driver/device identity for provenance).
    3. ``torch.cuda.Stream``/``torch.cuda.Event`` construct successfully --
       the concrete ownership/lifetime primitive this prototype's lease
       completion events are built on
       (``worker.types.frame_memory.CompletionEvent``/``CompletionReclaimer``).
    4. ``torch.from_dlpack`` and ``Tensor.__dlpack__`` are present -- the
       documented zero-copy handoff into an in-process CUDA inference runtime
       this prototype's batcher requires; without it there is no truthful way
       to keep a decoded frame device-resident across the preprocess ->
       inference-input seam.
    """
    cuda = cuda_probe()
    nvml = nvml_probe()

    if not cuda.available:
        return DeviceResidentCapability(
            available=False,
            reason=f"cuda capability unavailable: {cuda.reason}",
            cuda=cuda,
            nvml=nvml,
            stream_event_supported=False,
            dlpack_supported=False,
        )
    if not nvml.nvml_available:
        return DeviceResidentCapability(
            available=False,
            reason=f"nvml device identity unavailable: {nvml.reason}",
            cuda=cuda,
            nvml=nvml,
            stream_event_supported=False,
            dlpack_supported=False,
        )

    try:
        torch = torch_importer()
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency boundary
        return DeviceResidentCapability(
            available=False,
            reason=f"torch import failed: {type(exc).__name__}",
            cuda=cuda,
            nvml=nvml,
            stream_event_supported=False,
            dlpack_supported=False,
        )

    stream_event_supported = _probe_stream_event(torch)
    if not stream_event_supported:
        return DeviceResidentCapability(
            available=False,
            reason="torch.cuda.Stream/Event construction failed",
            cuda=cuda,
            nvml=nvml,
            stream_event_supported=False,
            dlpack_supported=False,
        )

    dlpack_supported = hasattr(torch, "from_dlpack") and hasattr(torch.Tensor, "__dlpack__")
    if not dlpack_supported:
        return DeviceResidentCapability(
            available=False,
            reason="torch build has no DLPack (from_dlpack/__dlpack__) support",
            cuda=cuda,
            nvml=nvml,
            stream_event_supported=True,
            dlpack_supported=False,
        )

    return DeviceResidentCapability(
        available=True,
        reason="device-resident concrete stages are available",
        cuda=cuda,
        nvml=nvml,
        stream_event_supported=True,
        dlpack_supported=True,
    )


def _probe_stream_event(torch: Any) -> bool:
    try:
        stream = torch.cuda.Stream()
        event = torch.cuda.Event()
        del stream, event
    except Exception:  # noqa: BLE001 - capability probe must never break startup
        return False
    return True


__all__ = ["DeviceResidentCapability", "TorchImporter", "probe_device_resident_capability"]
