"""Vendor-neutral media-plane control surface (P1b, G8b).

A media plane owns decode, batching, inference, tracking and recording for a
set of RTSP sources and publishes accepted perception metadata per camera. The
runtime composes one implementation per profile; the only DeepStream-aware
implementation lives in ``worker.adapters.deepstream``.

Recording semantics follow what DeepStream Smart Record actually provides, as
measured in ``docs/research/pyservicemaker-p1b-spike.md``: a recording is
started with a lookback and a forward duration, may be stopped early, and is
sealed only when the completion callback delivers ``RecordingInfo``. There is no
extension primitive; overlapping starts are absorbed by the plane, so an actor
above this surface owns any extension policy by delaying its stop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from worker.native.deepstream.metadata import SourceBinding


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    """What the plane reports when a recording is sealed on disk."""

    session_id: int
    camera_id: str
    path: str
    duration_ms: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SourceStatus:
    camera_id: str
    binding: SourceBinding
    live: bool
    reconnects: int


@dataclass(frozen=True, slots=True)
class MediaPlaneStatus:
    sources: tuple[SourceStatus, ...]
    engine_identity: str
    nvenc_sessions_active: int


@runtime_checkable
class MediaPlane(Protocol):
    """Source lifecycle plus the evidence commands the runtime issues.

    Every method is called from the runtime's own threads; implementations
    serialize onto their pipeline thread. Source mutation returns the new
    ``SourceBinding`` so the caller can subscribe to exact accepted metadata.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def status(self) -> MediaPlaneStatus: ...

    def add_source(self, camera_id: str, uri: str) -> SourceBinding: ...

    def remove_source(self, camera_id: str) -> None: ...

    def source_failure(self, camera_id: str, category: str) -> SourceBinding: ...

    def snapshot(self, camera_id: str) -> bytes:
        """One bounded JPEG of the latest OSD-composited frame, or raise."""
        ...

    def start_recording(
        self,
        camera_id: str,
        *,
        lookback_sec: int,
        duration_sec: int,
        on_sealed: Callable[[RecordingInfo], None],
    ) -> int:
        """Begin writing the cached stream to a file; returns the session id.

        Raises ``RecordingRefused`` when the source is not live or has no
        cached I-frame yet, so a caller never believes an empty session started.
        """
        ...

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        """Seal early; ``on_sealed`` still fires exactly once."""
        ...


class RecordingRefused(RuntimeError):
    """The plane cannot begin a recording for this source right now."""
