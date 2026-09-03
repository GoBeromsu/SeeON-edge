"""Ports and dependency bundle for the serialized clip actor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.pipeline.output.evidence.clip_publication import (
    ClipPublicationMetadata,
    PublishedClip,
)
from worker.pipeline.output.evidence.clip_recording import ClipOutcome
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    EvidenceReasonCode,
)
from worker.types import BusinessEvent, FrameKey, FramePacket


class RecordingCoordinator(Protocol):
    def write(self, packet: FramePacket) -> bool: ...
    def seal(self, camera_id: str) -> bool: ...
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
    ) -> ClipOutcome: ...
    def close(self, camera_id: str) -> None: ...
    def close_all(self) -> None: ...


class ClipPublicationPort(Protocol):
    def publish_ready(
        self,
        reservation: ClipReservation,
        artifact_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> PublishedClip | None: ...
    def publish_unavailable(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
        reason_code: EvidenceReasonCode,
    ) -> PublishedClip | None: ...


class ClipCloseHook(Protocol):
    def __call__(self, camera_id: str, clip_id: ClipId, /) -> None: ...


class ClipReleaseHook(Protocol):
    def __call__(self, camera_id: str, clip_id: ClipId, /) -> None: ...


class ClipCancelHook(Protocol):
    def __call__(self, reservation: ClipReservation, /) -> None: ...


class ClipFinalizedHook(Protocol):
    def __call__(self, clip_id: ClipId, /) -> None: ...


@dataclass(frozen=True, slots=True)
class ClipActorDependencies:
    coordinator: RecordingCoordinator
    publisher: ClipPublicationPort
    close: ClipCloseHook
    release: ClipReleaseHook
    finalized: ClipFinalizedHook | None = None
    encoder_name: str = "libx264"
    cancel: ClipCancelHook | None = None


__all__ = [
    "ClipActorDependencies",
    "ClipPublicationPort",
    "RecordingCoordinator",
]
