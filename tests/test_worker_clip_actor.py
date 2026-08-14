from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contracts.frame import Frame
from worker.adapters.encode.models import ClipArtifact
from worker.pipeline.output.evidence.clip_actor import ClipActor, ClipActorDependencies
from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.pipeline.output.evidence.clip_publication import ClipPublicationMetadata
from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
    EventMessage,
    FrameMessage,
)
from worker.pipeline.output.evidence.clip_recording import (
    ClipOutcome,
    ClipReady,
    ClipReasonCode,
    ClipUnavailable,
)
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    EvidenceReasonCode,
)
from worker.types import BusinessEvent, FramePacket


def _packet(time_sec: float) -> FramePacket:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    frame = Frame(index=round(time_sec * 10), time_sec=time_sec, image=image)
    return FramePacket("cam-1", frame, time_sec, frame.index, 8, 8, 0.0)


def _reservation(tmp_path: Path, clip_id: str = "clip-1") -> ClipReservation:
    staging_dir = tmp_path / "clips" / ".staging" / clip_id
    staging_dir.mkdir(parents=True)
    return ClipReservation(
        ClipId(clip_id),
        "cam-1",
        staging_dir,
        tmp_path / "clips" / clip_id,
    )


def _event_message(
    reservation: ClipReservation,
    event_ref: str,
    *,
    allow_new_clip: bool = True,
    time_sec: float = 10.0,
) -> EventMessage:
    event = BusinessEvent(
        "fall",
        "fall.detected",
        event_ref,
        "cam-1",
        "facility-1",
        time_sec,
        0.9,
    )
    return EventMessage(reservation, event_ref, event.event_type, event, allow_new_clip)


@dataclass(frozen=True, slots=True)
class _Coordinator:
    outcome: ClipOutcome
    packets: list[FramePacket] = field(default_factory=list)
    sealed: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)

    def write(self, packet: FramePacket) -> bool:
        self.packets.append(packet)
        return True

    def seal(self, camera_id: str) -> bool:
        self.sealed.append(camera_id)
        return True

    def finalize(
        self,
        *,
        camera_id: str,
        clip_id: str,
        event_time_sec: float,
        event: BusinessEvent,
        output_dir: Path | None = None,
    ) -> ClipOutcome:
        del camera_id, clip_id, event_time_sec, event, output_dir
        return self.outcome

    def close(self, camera_id: str) -> None:
        self.closed.append(camera_id)

    def close_all(self) -> None:
        return


@dataclass(frozen=True, slots=True)
class _Publisher:
    ready: list[tuple[ClipReservation, Path, ClipPublicationMetadata]] = field(
        default_factory=list
    )
    unavailable: list[
        tuple[ClipReservation, ClipPublicationMetadata, EvidenceReasonCode]
    ] = field(default_factory=list)

    def publish_ready(
        self,
        reservation: ClipReservation,
        artifact_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> None:
        self.ready.append((reservation, artifact_path, metadata))

    def publish_unavailable(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
        reason_code: EvidenceReasonCode,
    ) -> None:
        self.unavailable.append((reservation, metadata, reason_code))


def _actor(
    tmp_path: Path,
    outcome: ClipOutcome,
) -> tuple[ClipActor, ClipRecorderStats, _Coordinator, _Publisher, list[ClipId]]:
    stats = ClipRecorderStats()
    coordinator = _Coordinator(outcome)
    publisher = _Publisher()
    finalized: list[ClipId] = []
    actor = ClipActor(
        ClipRecorderConfig(
            store_dir=tmp_path,
            pre_event_seconds=1.0,
            post_event_seconds=2.0,
            finalize_grace_seconds=1.0,
        ),
        stats,
        ClipActorDependencies(
            coordinator=coordinator,
            publisher=publisher,
            close=lambda _camera_id, _clip_id: None,
            release=lambda _camera_id, _clip_id: None,
            finalized=finalized.append,
        ),
    )
    return actor, stats, coordinator, publisher, finalized


def test_ready_clip_publishes_after_post_event_boundary(tmp_path: Path) -> None:
    reservation = _reservation(tmp_path)
    artifact = ClipArtifact(reservation.staging_dir / "clip.mp4", 1, 2, 3.0)
    actor, stats, coordinator, publisher, finalized = _actor(
        tmp_path,
        ClipReady("clip-1", artifact),
    )

    actor.handle_frame(FrameMessage(_packet(10.0)))
    actor.handle_event(_event_message(reservation, "event-1"))
    actor.handle_frame(FrameMessage(_packet(12.0)))

    assert coordinator.sealed == ["cam-1"]
    assert publisher.ready[0][0] == reservation
    assert publisher.ready[0][2].event_refs == ("event-1",)
    assert stats.finalized_clips == 1
    assert stats.active_clips == 0
    assert finalized == [ClipId("clip-1")]


def test_unavailable_outcome_maps_no_segments_to_no_frames(tmp_path: Path) -> None:
    reservation = _reservation(tmp_path)
    actor, stats, _, publisher, _ = _actor(
        tmp_path,
        ClipUnavailable("clip-1", ClipReasonCode.NO_SEGMENTS),
    )

    actor.handle_event(_event_message(reservation, "event-1"))
    actor.flush()

    assert publisher.unavailable[0][2] is EvidenceReasonCode.NO_FRAMES
    assert stats.video_unavailable_clips == 1


def test_coalesced_refs_remain_ordered_and_unique(tmp_path: Path) -> None:
    reservation = _reservation(tmp_path)
    actor, _, _, publisher, _ = _actor(
        tmp_path,
        ClipUnavailable("clip-1", ClipReasonCode.ENCODER_FAILED),
    )

    actor.handle_event(_event_message(reservation, "event-1"))
    actor.handle_event(_event_message(reservation, "event-1"))
    actor.handle_event(_event_message(reservation, "event-2"))
    actor.flush()

    assert publisher.unavailable[0][1].event_refs == ("event-1", "event-2")


def test_coalesced_event_keeps_the_first_event_cutoff(tmp_path: Path) -> None:
    reservation = _reservation(tmp_path)
    actor, _, _, publisher, _ = _actor(
        tmp_path,
        ClipUnavailable("clip-1", ClipReasonCode.ENCODER_FAILED),
    )

    actor.handle_event(_event_message(reservation, "event-1", time_sec=10.0))
    first_cutoff = actor._active_by_camera["cam-1"].cutoff_time_sec
    actor.handle_event(_event_message(reservation, "event-2", time_sec=20.0))

    assert actor._active_by_camera["cam-1"].cutoff_time_sec == first_cutoff == 12.0
    actor.flush()
    assert publisher.unavailable[0][1].event_refs == ("event-1", "event-2")


def test_attach_only_event_after_frame_finalization_cannot_mutate_clip(tmp_path: Path) -> None:
    reservation = _reservation(tmp_path)
    artifact = ClipArtifact(reservation.staging_dir / "clip.mp4", 1, 2, 3.0)
    actor, stats, _, publisher, _ = _actor(tmp_path, ClipReady("clip-1", artifact))

    actor.handle_event(_event_message(reservation, "event-1"))
    actor.handle_frame(FrameMessage(_packet(12.0)))
    actor.handle_event(_event_message(reservation, "event-2", allow_new_clip=False))

    assert stats.attach_missed_events == 1
    assert len(publisher.ready) == 1
    assert publisher.ready[0][2].event_refs == ("event-1",)
    assert stats.finalized_clips == 1


def test_ready_clip_metadata_timestamps_stay_ordered_when_finalize_outruns_stream_time(
    tmp_path: Path,
) -> None:
    """Regression for the Linux-CI ``ReadyClipManifest`` "clip timestamps are
    not ordered" failure.

    ``duration_s`` is derived from source/stream time (segment end minus
    start, see ``worker/adapters/encode/clip_finalizer.py``'s
    ``FFmpegConcatFinalizer.finalize``), not from wall-clock elapsed time.
    ``clip_actor.ClipActor._metadata`` used to compute
    ``clip_end_at = started_at + duration_s`` from that stream-time
    duration while ``finalized_at`` is real wall-clock ``datetime.now(UTC)``.
    When frames are admitted far faster than real time -- as this
    synchronous test does, and as can happen with non-realtime or
    backlog-catch-up ingestion -- the wall-clock gap between the event and
    finalize is much smaller than the nominal stream-time duration, so
    ``clip_end_at`` landed after ``finalized_at``, violating
    ``ReadyClipManifest``'s ``clip_start_at <= clip_end_at <= finalized_at``
    invariant (``worker/pipeline/output/evidence/manifest_models.py``'s
    ``_ordered_timestamps`` validator) and making
    ``ClipActor._finalize`` swallow a ``pydantic.ValidationError`` as a
    failed write. This exercises only ``ClipActor`` and fake collaborators
    -- no ffprobe, no ``/proc``, no mediamtx -- so it is deterministic on
    every OS.
    """
    reservation = _reservation(tmp_path)
    stream_time_duration_s = 1_000.0
    artifact = ClipArtifact(
        reservation.staging_dir / "clip.mp4", 1, 2, stream_time_duration_s
    )
    actor, _, _, publisher, _ = _actor(tmp_path, ClipReady("clip-1", artifact))

    actor.handle_event(_event_message(reservation, "event-1"))
    actor.flush()

    metadata = publisher.ready[0][2]
    assert metadata.clip_start_at <= metadata.clip_end_at <= metadata.finalized_at


def test_expiry_forces_finalize_when_stream_time_stalls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import worker.pipeline.output.evidence.clip_actor as actor_module

    reservation = _reservation(tmp_path)
    actor, stats, _, _, _ = _actor(
        tmp_path,
        ClipUnavailable("clip-1", ClipReasonCode.ENCODER_FAILED),
    )
    monkeypatch.setattr(actor_module.time, "monotonic", lambda: 100.0)
    actor.handle_event(_event_message(reservation, "event-1"))
    monkeypatch.setattr(actor_module.time, "monotonic", lambda: 104.0)

    actor.expire()

    assert stats.forced_finalized == 1
    assert stats.active_clips == 0
