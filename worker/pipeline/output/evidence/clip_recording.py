from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from worker.adapters.encode.adapter_errors import (
    ClipRemuxError,
    CrossEpochFrameError,
    CrossEpochSegmentError,
    CrossGenerationSegmentError,
    EncoderPolicyError,
    EncoderStartError,
    EncoderWriteError,
)
from worker.adapters.encode.csv_segment_index import Segment
from worker.adapters.encode.encoder_setup import parse_encode_policy
from worker.adapters.encode.models import ClipArtifact, EncoderGeometry
from worker.types import BusinessEvent, FrameKey, FramePacket


class ClipReasonCode(StrEnum):
    ENCODER_FAILED = "ENCODER_FAILED"
    REMUX_FAILED = "REMUX_FAILED"
    NO_SEGMENTS = "NO_SEGMENTS"
    STREAM_EPOCH_MISMATCH = "STREAM_EPOCH_MISMATCH"


@dataclass(frozen=True, slots=True)
class ClipReady:
    clip_id: str
    artifact: ClipArtifact


@dataclass(frozen=True, slots=True)
class ClipUnavailable:
    clip_id: str
    reason_code: ClipReasonCode
    detail_reason: str | None = None
    truncation_reasons: tuple[str, ...] = ()


ClipOutcome = ClipReady | ClipUnavailable


@dataclass(frozen=True, slots=True)
class ClipWindow:
    pre_event_seconds: float = 30.0
    post_event_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.pre_event_seconds < 0 or self.post_event_seconds < 0:
            raise EncoderPolicyError("clip window bounds must not be negative")

    def bounds(self, event_time_sec: float) -> tuple[float, float]:
        return (
            event_time_sec - self.pre_event_seconds,
            event_time_sec + self.post_event_seconds,
        )


def _windowed_duration_s(
    segments: Sequence[Segment],
    start_time_sec: float,
    end_time_sec: float,
) -> float:
    """Sum each selected segment's overlap with the requested clip window.

    Segments are ffmpeg's own coarse-grained encode units (their boundaries
    depend on encoder startup timing and keyframe placement, not the
    pre/post-event window), so `select_segments` routinely pulls in
    boundary segments that only partially overlap `[start_time_sec,
    end_time_sec]`. `SegmentClipFinalizer.finalize` reports the raw sum of
    those segments' full durations as `ClipArtifact.duration_s`, which
    overstates the clip by however much the boundary segments extend past
    the requested window (observed as almost a full extra segment on each
    edge). Clip each segment's contribution to its overlap with the window
    instead, so the reported duration tracks what was actually requested --
    matching how the `ClipUnavailable` path already derives its duration
    from the requested window rather than raw segment bookkeeping.
    """
    return sum(
        max(
            0.0,
            min(segment.end_time_sec, end_time_sec) - max(segment.start_time_sec, start_time_sec),
        )
        for segment in segments
    )


class SegmentedEncoderSession(Protocol):
    """An `EncoderSession` that also exposes its own closed segments.

    The `EncoderSession` port is deliberately write/close only; evidence
    finalization additionally needs the per-generation closed-segment view, so
    the wider capability is declared structurally here instead of widening the
    shared port for every consumer.
    """

    def write(self, packet: FramePacket) -> None: ...

    def close(self) -> None: ...

    def select_segments(
        self,
        *,
        start_time_sec: float,
        end_time_sec: float,
    ) -> tuple[Segment, ...]: ...

    @property
    def worker_boot_id(self) -> str: ...

    @property
    def stream_epoch(self) -> int: ...

    @property
    def origin_time_sec(self) -> float | None:
        """Event-clock value that maps to this session's segment-local 0.

        ``select_segments`` filters segments after converting the requested
        window into this same generation-local axis (#165); callers that
        also compare against the *returned* segments' (still local) times --
        such as ``_windowed_duration_s`` below -- must apply this same
        offset to their own window bounds first, or they silently compare
        two different clocks again one level up.
        """
        ...


class SegmentedClipEncoder(Protocol):
    def open(
        self,
        camera: str,
        profile: str,
        geometry: EncoderGeometry,
    ) -> SegmentedEncoderSession: ...


class SegmentClipFinalizer(Protocol):
    def finalize(
        self,
        segments: Sequence[Segment],
        event: BusinessEvent,
    ) -> ClipArtifact: ...


class SegmentClipFinalizerFactory(Protocol):
    def __call__(self, output_dir: Path) -> SegmentClipFinalizer: ...


class ClipRecordingCoordinator:
    """Route frames to one reusable encoder per camera and remux event windows.

    Encoder loss degrades only the affected camera: the event still finalizes,
    as a typed `ClipUnavailable` outcome rather than a lost event or a raised
    failure that would reach the decision stage.
    """

    def __init__(
        self,
        *,
        encoder: SegmentedClipEncoder,
        finalizer: SegmentClipFinalizer | None = None,
        finalizer_factory: SegmentClipFinalizerFactory | None = None,
        profile: str,
        fps: float,
        window: ClipWindow | None = None,
    ) -> None:
        if (finalizer is None) == (finalizer_factory is None):
            raise EncoderPolicyError("configure exactly one clip finalizer source")
        self._encoder = encoder
        self._finalizer = finalizer
        self._finalizer_factory = finalizer_factory
        # A bad profile is a global misconfiguration, not a camera fault: fail here
        # rather than degrade every camera at its first frame.
        self._profile = parse_encode_policy(profile)
        self._fps = fps
        self._window = ClipWindow() if window is None else window
        self._sessions: dict[str, SegmentedEncoderSession] = {}
        self._degraded: set[str] = set()
        self._fps_by_camera: dict[str, float] = {}

    @property
    def degraded_cameras(self) -> frozenset[str]:
        return frozenset(self._degraded)

    def set_camera_fps(self, camera_id: str, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise EncoderPolicyError("camera fps must be finite and positive")
        self._fps_by_camera[camera_id] = fps

    def write(self, packet: FramePacket) -> bool:
        geometry = EncoderGeometry(
            packet.width,
            packet.height,
            self._fps_by_camera.get(packet.camera_id, self._fps),
        )
        try:
            session = self._encoder.open(packet.camera_id, self._profile, geometry)
            if session.stream_epoch not in (
                0,
                packet.stream_epoch,
            ) or session.worker_boot_id not in ("", packet.worker_boot_id):
                session.close()
                session = self._encoder.open(packet.camera_id, self._profile, geometry)
            session.write(packet)
        except (
            CrossEpochFrameError,
            EncoderStartError,
            EncoderWriteError,
            EncoderPolicyError,
        ):
            self._degraded.add(packet.camera_id)
            self._sessions.pop(packet.camera_id, None)
            return False
        self._sessions[packet.camera_id] = session
        self._degraded.discard(packet.camera_id)
        return True

    def finalize(
        self,
        *,
        camera_id: str,
        clip_id: str,
        event_time_sec: float,
        event: BusinessEvent,
        output_dir: Path | None = None,
        trigger_frame_key: FrameKey | None = None,
        window_bounds: tuple[float, float] | None = None,
    ) -> ClipOutcome:
        session = self._sessions.get(camera_id)
        if session is None:
            return ClipUnavailable(clip_id, ClipReasonCode.ENCODER_FAILED)
        if trigger_frame_key is not None and (
            session.worker_boot_id != trigger_frame_key.worker_boot_id
            or session.stream_epoch != trigger_frame_key.stream_epoch
        ):
            return ClipUnavailable(clip_id, ClipReasonCode.STREAM_EPOCH_MISMATCH)

        start_time_sec, end_time_sec = (
            self._window.bounds(event_time_sec) if window_bounds is None else window_bounds
        )
        try:
            segments = session.select_segments(
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
            )
        except (EncoderStartError, EncoderWriteError):
            self._degraded.add(camera_id)
            return ClipUnavailable(clip_id, ClipReasonCode.ENCODER_FAILED)

        if not segments:
            return ClipUnavailable(clip_id, ClipReasonCode.NO_SEGMENTS)

        finalizer = self._finalizer
        if finalizer is None:
            if output_dir is None or self._finalizer_factory is None:
                raise EncoderPolicyError("finalizer output directory is required")
            finalizer = self._finalizer_factory(output_dir)
        try:
            artifact = finalizer.finalize(segments, event)
        except CrossEpochSegmentError:
            return ClipUnavailable(clip_id, ClipReasonCode.STREAM_EPOCH_MISMATCH)
        except (ClipRemuxError, CrossGenerationSegmentError):
            return ClipUnavailable(clip_id, ClipReasonCode.REMUX_FAILED)
        origin = session.origin_time_sec
        local_start = start_time_sec if origin is None else start_time_sec - origin
        local_end = end_time_sec if origin is None else end_time_sec - origin
        artifact = replace(
            artifact,
            duration_s=_windowed_duration_s(segments, local_start, local_end),
        )
        return ClipReady(clip_id, artifact)

    def seal(self, camera_id: str) -> bool:
        """Close one encoder generation while retaining its segment index."""
        session = self._sessions.get(camera_id)
        if session is None:
            return False
        try:
            session.close()
        except (EncoderStartError, EncoderWriteError):
            self._degraded.add(camera_id)
            return False
        return True

    def close(self, camera_id: str) -> None:
        session = self._sessions.pop(camera_id, None)
        if session is None:
            return
        try:
            session.close()
        except (EncoderStartError, EncoderWriteError):
            self._degraded.add(camera_id)

    def close_all(self) -> None:
        for camera_id in tuple(self._sessions):
            self.close(camera_id)


__all__ = [
    "ClipOutcome",
    "ClipReady",
    "ClipReasonCode",
    "ClipRecordingCoordinator",
    "ClipUnavailable",
    "ClipWindow",
    "SegmentClipFinalizer",
    "SegmentClipFinalizerFactory",
    "SegmentedClipEncoder",
    "SegmentedEncoderSession",
]
