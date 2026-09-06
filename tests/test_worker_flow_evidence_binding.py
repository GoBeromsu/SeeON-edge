from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from worker.interfaces.media_plane import RecordingInfo, RecordingRefused
from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublicationError
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
from worker.pipeline.output.evidence.smart_record_actor import SmartRecordActor
from worker.runtime.flow.evidence import FlowEvidenceBinding
from worker.types import BusinessEvent, NativeEvidenceTrigger


@dataclass
class _Plane:
    refused: int = 0
    starts: list[int] = field(default_factory=list)
    stops: list[int] = field(default_factory=list)
    callbacks: dict[int, object] = field(default_factory=dict)

    def start_recording(
        self, camera_id: str, *, lookback_sec: int, duration_sec: int, on_sealed: object
    ) -> int:
        if self.refused:
            self.refused -= 1
            raise RecordingRefused("source has not produced a frame")
        session = len(self.starts) + 1
        self.starts.append(session)
        self.callbacks[session] = on_sealed
        return session

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        self.stops.append(session_id)

    def seal(self, session: int) -> None:
        callback = self.callbacks[session]
        callback(RecordingInfo(session, "camera-a", f"/clips/{session}.mp4", 12_000, 640, 360))


@dataclass
class _Stager:
    staged: list[dict[str, object]] = field(default_factory=list)
    completed: list[tuple[str, str | None]] = field(default_factory=list)

    def stage(self, event: dict[str, object]) -> None:
        self.staged.append(event)

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        self.completed.append((edge_event_id, clip_id))


@dataclass
class _Publisher:
    fail: bool = False

    def publish(self, sealed: object, events: object) -> object:
        del events
        if self.fail:
            raise FlowClipPublicationError("publication failed")
        return type("_Published", (), {"clip_id": sealed.clip_id})()


def _event(identity: str) -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", identity, "camera-a", "facility-a", 12.0, 0.99)


def _trigger() -> NativeEvidenceTrigger:
    return NativeEvidenceTrigger("camera-a", "boot", 1, 1, 1, 12_000_000_000, 12.0)


def _binding(
    plane: _Plane,
    now: list[float],
    dates: list[datetime],
    sidecar_directory: Path,
    *,
    extension_sec: int = 45,
) -> tuple[SmartRecordActor, FlowEvidenceBinding, _Stager, _Publisher]:
    stager = _Stager()
    publisher = _Publisher()
    sealed: list[FlowEvidenceBinding] = []
    actor = SmartRecordActor(
        camera_id="camera-a",
        media_plane=plane,
        clock=lambda: now[0],
        sink=lambda clip: sealed[0].on_sealed(clip),
        lookback_sec=10,
        extension_sec=extension_sec,
        clip_id_factory=lambda: "primary-clip",
    )
    binding = FlowEvidenceBinding(
        actor=actor,
        stager=stager,
        publisher=publisher,
        sidecars=FlowSealedSidecars(sidecar_directory),
        camera_id="camera-a",
        now=lambda: dates.pop(0),
    )
    sealed.append(binding)
    return actor, binding, stager, publisher


def test_admitted_alert_stages_one_recording_and_sealed_receipt(tmp_path: Path) -> None:
    plane, now = _Plane(), [0.0]
    actor, binding, stager, _ = _binding(plane, now, [datetime(2026, 1, 1, tzinfo=UTC)], tmp_path)
    binding.emit_for_frame(_event("one"), _trigger())
    assert plane.starts == [1]
    now[0] = 30.0
    actor.tick()
    plane.seal(1)
    assert stager.staged == [
        {
            "edge_event_id": "one",
            "event_type": "fall.detected",
            "probability": 0.99,
            "detected_at": "2026-01-01T00:00:00Z",
            "camera_id": "camera-a",
            "facility_id": "facility-a",
            "evidence": {"domain": "fall", "identity": "one", "time_sec": 12.0},
        }
    ]
    assert stager.completed == [("one", "primary-clip")]


def test_two_alerts_extend_one_clip_and_complete_distinct_incidents(tmp_path: Path) -> None:
    plane, now = _Plane(), [0.0]
    actor, binding, stager, _ = _binding(
        plane,
        now,
        [datetime(2026, 1, 1, 0, 0, 20, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)],
        tmp_path,
    )
    binding.emit_for_frame(_event("late"), _trigger())
    now[0] = 20.0
    binding.emit_for_frame(_event("early"), _trigger())
    now[0] = 50.0
    actor.tick()
    plane.seal(1)
    assert plane.starts == [1]
    assert actor.smart_record_extended_total == 1
    assert [item["detected_at"] for item in stager.staged] == [
        "2026-01-01T00:00:20Z",
        "2026-01-01T00:00:00Z",
    ]
    assert stager.completed == [("early", "primary-clip"), ("late", "primary-clip")]


def test_alert_while_stopping_starts_second_clip_without_dropping_it(tmp_path: Path) -> None:
    """A deployment that cuts clips short can race the stop; nothing is dropped.

    With a shorter extension window than the recording window the actor issues a
    real early stop, so an alert arriving in that gap cannot join the sealing
    clip: it must mark the first clip raced and open a second one.
    """
    plane, now = _Plane(), [0.0]
    actor, binding, stager, _ = _binding(
        plane,
        now,
        [
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=30),
        ],
        tmp_path,
        extension_sec=20,
    )
    binding.emit_for_frame(_event("one"), _trigger())
    now[0] = 30.0
    actor.tick()
    assert plane.stops == [1]
    binding.emit_for_frame(_event("two"), _trigger())
    plane.seal(1)
    assert actor.smart_record_extension_raced_total == 1
    assert plane.starts == [1, 2]
    now[0] = 60.0
    actor.tick()
    plane.seal(2)
    assert stager.completed == [("one", "primary-clip"), ("two", "primary-clip")]


def test_refused_recording_retries_on_tick(tmp_path: Path) -> None:
    plane, now = _Plane(refused=1), [0.0]
    actor, binding, _, _ = _binding(plane, now, [datetime(2026, 1, 1, tzinfo=UTC)], tmp_path)
    binding.emit_for_frame(_event("one"), _trigger())
    assert actor.smart_record_start_refused_total == 1
    actor.tick()
    assert plane.starts == [1]


def test_publication_failure_surfaces_without_completing_the_incident(tmp_path: Path) -> None:
    plane, now = _Plane(), [0.0]
    actor, binding, stager, publisher = _binding(
        plane, now, [datetime(2026, 1, 1, tzinfo=UTC)], tmp_path
    )
    binding.emit_for_frame(_event("one"), _trigger())
    publisher.fail = True

    with pytest.raises(FlowClipPublicationError, match="publication failed"):
        plane.seal(1)

    assert stager.completed == []
    assert actor.state.name == "FINALIZING"
    assert len(binding.sidecars.pending_for_camera("camera-a")) == 1
