from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from worker.interfaces.media_plane import RecordingInfo, RecordingRefused
from worker.pipeline.output.evidence.smart_record_actor import SmartRecordActor
from worker.runtime.flow.evidence import FlowEvidenceBinding
from worker.types import BusinessEvent, NativeEvidenceTrigger


@dataclass
class _Plane:
    refused: int = 0
    starts: list[int] = field(default_factory=list)
    stops: list[int] = field(default_factory=list)
    callbacks: dict[int, object] = field(default_factory=dict)

    def start_recording(self, camera_id: str, *, lookback_sec: int, duration_sec: int, on_sealed: object) -> int:
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


def _event(identity: str) -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", identity, "camera-a", "facility-a", 12.0, 0.99)


def _trigger() -> NativeEvidenceTrigger:
    return NativeEvidenceTrigger("camera-a", "boot", 1, 1, 1, 12_000_000_000, 12.0)


def _binding(plane: _Plane, now: list[float], dates: list[datetime]) -> tuple[SmartRecordActor, FlowEvidenceBinding, _Stager]:
    stager = _Stager()
    sealed: list[FlowEvidenceBinding] = []
    actor = SmartRecordActor(
        camera_id="camera-a", media_plane=plane, clock=lambda: now[0],
        sink=lambda clip: sealed[0].on_sealed(clip), lookback_sec=10,
        clip_id_factory=lambda: "primary-clip",
    )
    binding = FlowEvidenceBinding(actor=actor, stager=stager, now=lambda: dates.pop(0))
    sealed.append(binding)
    return actor, binding, stager


def test_admitted_alert_stages_one_recording_and_sealed_receipt() -> None:
    plane, now = _Plane(), [0.0]
    actor, binding, stager = _binding(plane, now, [datetime(2026, 1, 1, tzinfo=UTC)])
    binding.emit_for_frame(_event("one"), _trigger())
    assert plane.starts == [1]
    now[0] = 30.0
    actor.tick()
    plane.seal(1)
    assert stager.staged == [{
        "edge_event_id": "one", "event_type": "fall.detected", "probability": 0.99,
        "detected_at": "2026-01-01T00:00:00Z", "camera_id": "camera-a", "facility_id": "facility-a",
        "evidence": {"domain": "fall", "identity": "one", "time_sec": 12.0},
    }]
    assert stager.completed == [("one", "primary-clip")]


def test_two_alerts_extend_one_clip_and_complete_distinct_incidents() -> None:
    plane, now = _Plane(), [0.0]
    actor, binding, stager = _binding(plane, now, [datetime(2026, 1, 1, 0, 0, 20, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)])
    binding.emit_for_frame(_event("late"), _trigger())
    now[0] = 20.0
    binding.emit_for_frame(_event("early"), _trigger())
    now[0] = 50.0
    actor.tick()
    plane.seal(1)
    assert plane.starts == [1]
    assert actor.smart_record_extended_total == 1
    assert [item["detected_at"] for item in stager.staged] == ["2026-01-01T00:00:20Z", "2026-01-01T00:00:00Z"]
    assert stager.completed == [("early", "primary-clip"), ("late", "primary-clip")]


def test_alert_while_stopping_starts_second_clip_without_dropping_it() -> None:
    plane, now = _Plane(), [0.0]
    actor, binding, stager = _binding(plane, now, [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=30)])
    binding.emit_for_frame(_event("one"), _trigger())
    now[0] = 30.0
    actor.tick()
    binding.emit_for_frame(_event("two"), _trigger())
    plane.seal(1)
    assert actor.smart_record_extension_raced_total == 1
    assert plane.starts == [1, 2]
    now[0] = 60.0
    actor.tick()
    plane.seal(2)
    assert stager.completed == [("one", "primary-clip"), ("two", "primary-clip")]


def test_refused_recording_retries_on_tick() -> None:
    plane, now = _Plane(refused=1), [0.0]
    actor, binding, _ = _binding(plane, now, [datetime(2026, 1, 1, tzinfo=UTC)])
    binding.emit_for_frame(_event("one"), _trigger())
    assert actor.smart_record_start_refused_total == 1
    actor.tick()
    assert plane.starts == [1]
