from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from worker.types import BusinessEvent, FrameLease, FramePacket

_CameraT = TypeVar("_CameraT", contravariant=True)
_ProfileT = TypeVar("_ProfileT", contravariant=True)
_GeometryT = TypeVar("_GeometryT", contravariant=True)
_SegmentT = TypeVar("_SegmentT", contravariant=True)
_ArtifactT = TypeVar("_ArtifactT", covariant=True)


@runtime_checkable
class EncoderSession(Protocol):
    """One long-lived derivative encoder session owned by a camera."""

    def write(self, packet: FramePacket) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class ClipEncoder(Protocol[_CameraT, _ProfileT, _GeometryT]):
    """Open a camera encoder without resolving profile policy internally."""

    def open(
        self,
        camera: _CameraT,
        profile: _ProfileT,
        geometry: _GeometryT,
    ) -> EncoderSession: ...


@runtime_checkable
class ClipFinalizer(Protocol[_SegmentT, _ArtifactT]):
    """Finalize selected completed segments into an immutable clip artifact."""

    def finalize(
        self,
        segments: Sequence[_SegmentT],
        event: BusinessEvent,
    ) -> _ArtifactT: ...


@runtime_checkable
class DeviceInputEncoder(Protocol):
    """Encode explicit device-resident surfaces without host readback.

    Distinct from ``ClipEncoder``/``EncoderSession`` (the production
    host-buffer segment-muxer seam, fed one ``FramePacket`` at a time by
    ``worker.adapters.encode.ffmpeg_segment_encoder.FFmpegSegmentEncoder``):
    this port only accepts an already device-resident, ownership-tracked
    ``FrameLease`` -- never a host array -- and never silently reads it back.
    A caller that has only a host frame must materialize it through a named,
    capability-validated converter before calling ``submit``; this port
    itself performs zero implicit host<->device transfer.
    """

    def submit(self, lease: FrameLease) -> None: ...

    def close(self) -> None: ...


__all__ = ["ClipEncoder", "ClipFinalizer", "DeviceInputEncoder", "EncoderSession"]
