from __future__ import annotations

from dataclasses import dataclass, field

from worker.interfaces.media_plane import RecordingInfo, RecordingRefused
from worker.pipeline.output.evidence.smart_record_actor import (
    ClipSealed,
    DuplicateRecordingSealedError,
    SmartRecordActor,
)


@dataclass
class FakePlane:
    refusals: int = 0
    starts: list[tuple[int, int]] = field(default_factory=list)
    stops: list[int] = field(default_factory=list)
    callbacks: dict[int, object] = field(default_factory=dict)

    def start_recording(
        self, camera_id: str, *, lookback_sec: int, duration_sec: int, on_sealed: object
    ) -> int:
        if self.refusals:
            self.refusals -= 1
            raise RecordingRefused("not live")
        session_id = len(self.starts) + 1
        self.starts.append((lookback_sec, duration_sec))
        self.callbacks[session_id] = on_sealed
        return session_id

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        self.stops.append(session_id)

    def seal(self, session_id: int) -> None:
        callback = self.callbacks[session_id]
        callback(
            RecordingInfo(session_id, "camera-a", f"/clips/{session_id}.mp4", 12_000, 640, 480)
        )


def _actor(plane: FakePlane, now: list[float], sink: list[object]) -> SmartRecordActor:
    return SmartRecordActor(
        camera_id="camera-a",
        media_plane=plane,
        clock=lambda: now[0],
        sink=sink.append,
        lookback_sec=10,
        clip_id_factory=lambda: f"clip-{len(plane.starts)}",
    )


def test_extends_one_session_and_orders_contributors() -> None:
    plane, now, sink = FakePlane(), [0.0], []
    actor = _actor(plane, now, sink)
    actor.admit("first", "2026-01-01T00:00:20Z")
    now[0] = 20.0
    actor.admit("second", "2026-01-01T00:00:10Z")
    now[0] = 49.9
    actor.tick()
    assert plane.stops == []
    now[0] = 50.0
    actor.tick()
    plane.seal(1)
    assert plane.starts == [(10, 180)]
    assert plane.stops == [1]
    assert actor.smart_record_extended_total == 1
    [sealed] = sink
    assert isinstance(sealed, ClipSealed)
    assert (sealed.clip_id, sealed.path, sealed.duration_ms, sealed.boundary) == (
        "clip-1",
        "/clips/1.mp4",
        12_000,
        "none",
    )
    assert [contributor.event_ref for contributor in sealed.contributors] == ["second", "first"]


def test_hard_deadline_race_refusal_and_duplicate_callback() -> None:
    plane, now, sink = FakePlane(), [0.0], []
    actor = _actor(plane, now, sink)
    actor.admit("one", "2026-01-01T00:00:00Z")
    now[0] = 179.0
    actor.admit("two", "2026-01-01T00:02:59Z")
    now[0] = 180.0
    actor.tick()
    assert plane.stops == [1]
    plane.seal(1)
    assert actor.smart_record_extended_total == 1
    assert sink[0].boundary == "extension_bounded"
    actor.admit("three", "2026-01-01T00:03:00Z")
    now[0] = 210.0
    actor.tick()
    actor.admit("four", "2026-01-01T00:03:30Z")
    assert actor.smart_record_extension_raced_total == 1
    plane.seal(2)
    assert len(plane.starts) == 3
    plane.seal(2)
    assert isinstance(sink[-1], DuplicateRecordingSealedError)


def test_refused_start_retries_on_tick() -> None:
    plane, now, sink = FakePlane(refusals=1), [0.0], []
    actor = _actor(plane, now, sink)
    actor.admit("one", "2026-01-01T00:00:00Z")
    assert actor.smart_record_start_refused_total == 1
    actor.tick()
    assert len(plane.starts) == 1
