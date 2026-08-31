"""Ports and dependency bundle for the serialized clip actor."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
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
from worker.types import BusinessEvent, FrameKey, FramePacket, SceneRecord


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


class SceneSelectionPort(Protocol):
    def mark_active(self, camera_id: str) -> None: ...
    def clear_active(self, camera_id: str) -> None: ...
    def select(
        self,
        camera_id: str,
        trigger_epoch: int,
        start_pts_sec: Fraction,
        end_pts_sec: Fraction,
    ) -> tuple[SceneRecord, ...]: ...


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
    # Like camera_pipeline._observation_recorder, this optional telemetry tap
    # leaves primary evidence behavior unchanged when composition omits it.
    scene_selector: SceneSelectionPort | None = None


__all__ = [
    "ClipActorDependencies",
    "ClipPublicationPort",
    "RecordingCoordinator",
    "SceneSelectionPort",
]
