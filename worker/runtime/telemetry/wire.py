"""Closed runtime-status wire types and projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

from contracts.decode_diagnostics import DecodeSelection


@dataclass(frozen=True, slots=True)
class ClipRecorderStatus:
    """Frozen values allowed by the existing clip-recorder wire object."""

    available: bool = False
    dropped_frames: int | None = None
    dropped_events: int | None = None
    failed_writes: int | None = None
    finalized_clips: int | None = None
    video_unavailable_clips: int | None = None
    active_clips: int | None = None
    encoder: str | None = None


class RelayDecodePayload(TypedDict):
    """Existing decode diagnostics wire fields."""

    requested: str
    selected: str | None
    fallback_count: int
    last_reason: str | None
    updated_at_sec: float


class RelayCameraPayload(TypedDict):
    """Existing per-camera runtime-status wire fields."""

    camera_id: str
    decode: RelayDecodePayload
    measured_fps: NotRequired[float]


class RelayClipRecorderPayload(TypedDict):
    """Existing clip-recorder runtime-status wire fields."""

    available: bool
    dropped_frames: int | None
    dropped_events: int | None
    failed_writes: int | None
    finalized_clips: int | None
    video_unavailable_clips: int | None
    active_clips: int | None
    encoder: str | None


class RelayClipExportPayload(TypedDict):
    """Clip-export setting currently applied by this worker."""

    enabled: bool
    version: int


class RelayGpuPayload(TypedDict):
    """Existing optional GPU runtime-status wire fields."""

    nvml_available: bool
    cuda_context_ok: bool
    driver_version: str | None
    device_name: str | None
    captured_at_sec: float
    nvml_error: str | None


class RelayWorkerPayload(TypedDict):
    """Existing optional worker runtime-status wire fields."""

    alive: bool
    pid: int | None
    started_at_sec: float | None
    profile_boot_error: str | None


class RelayRuntimeStatusPayload(TypedDict):
    """Closed runtime-status request shape accepted by the backend."""

    facility_id: str
    generation: int | None
    seq: int
    cameras: list[RelayCameraPayload]
    clip_recorder: RelayClipRecorderPayload
    clip_export: RelayClipExportPayload
    gpu: NotRequired[RelayGpuPayload]
    worker: NotRequired[RelayWorkerPayload]


def camera_payload(
    camera_id: str,
    selection: DecodeSelection,
    measured_fps: float | None,
) -> RelayCameraPayload:
    payload = RelayCameraPayload(
        camera_id=camera_id,
        decode=RelayDecodePayload(
            requested=selection.requested,
            selected=selection.selected,
            fallback_count=selection.fallback_count,
            last_reason=selection.last_reason,
            updated_at_sec=selection.updated_at_sec,
        ),
    )
    if measured_fps is not None:
        payload["measured_fps"] = measured_fps
    return payload


def facility_payload(
    facility_id: str,
    generation: int | None,
    seq: int,
    cameras: list[RelayCameraPayload],
    clip_recorder: ClipRecorderStatus,
    clip_export: RelayClipExportPayload,
    gpu: RelayGpuPayload | None,
    worker: RelayWorkerPayload | None,
) -> RelayRuntimeStatusPayload:
    payload = RelayRuntimeStatusPayload(
        facility_id=facility_id,
        generation=generation,
        seq=seq,
        cameras=sorted(cameras, key=lambda camera: camera["camera_id"]),
        clip_export=clip_export,
        clip_recorder=RelayClipRecorderPayload(
            available=clip_recorder.available,
            dropped_frames=clip_recorder.dropped_frames,
            dropped_events=clip_recorder.dropped_events,
            failed_writes=clip_recorder.failed_writes,
            finalized_clips=clip_recorder.finalized_clips,
            video_unavailable_clips=clip_recorder.video_unavailable_clips,
            active_clips=clip_recorder.active_clips,
            encoder=clip_recorder.encoder,
        ),
    )
    if gpu is not None:
        payload["gpu"] = gpu
    if worker is not None:
        payload["worker"] = worker
    return payload


__all__ = [
    "ClipRecorderStatus",
    "RelayCameraPayload",
    "RelayClipExportPayload",
    "RelayClipRecorderPayload",
    "RelayDecodePayload",
    "RelayGpuPayload",
    "RelayRuntimeStatusPayload",
    "RelayWorkerPayload",
    "camera_payload",
    "facility_payload",
]
