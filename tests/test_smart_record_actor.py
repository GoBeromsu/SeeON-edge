from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from worker.interfaces.media_plane import RecordingInfo, RecordingRefused
from worker.pipeline.output.evidence.smart_record_actor import (
    ClipSealed,
    DuplicateRecordingSealedError,
    SmartRecordActor,
)


@dataclass
class FakePlane:
    """The measured plane: it seals at the duration it was given at start.

    ``stop_recording`` records the request so a test can prove the actor does
    not ask for an early stop it does not need.
    """

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
        lookback_sec=15,
        clip_id_factory=lambda: f"clip-{len(plane.starts)}",
    )


def test_one_alert_starts_one_window_of_the_configured_lookback_and_duration() -> None:
    """The actor asks the plane for exactly the window it was configured with."""
    plane, now, sink = FakePlane(), [0.0], []
    actor = _actor(plane, now, sink)

    actor.admit("event-1", "2026-01-01T00:00:00Z")
    actor.tick()

    assert plane.starts == [(15, 45)]
    assert plane.stops == []


def test_a_second_alert_inside_the_window_is_absorbed_into_the_same_clip() -> None:
    """DeepStream absorbs an overlapping start, so the actor must never issue one."""
    plane, now, sink = FakePlane(), [0.0], []
    actor = _actor(plane, now, sink)
    actor.admit("early", "2026-01-01T00:00:00Z")
    actor.tick()

    now[0] = 20.0
    actor.admit("late", "2026-01-01T00:00:20Z")
    actor.tick()

    assert plane.starts == [(15, 45)]
    assert actor.smart_record_extended_total == 1


def test_the_window_seals_itself_and_carries_both_contributors_in_order() -> None:
    plane, now, sink = FakePlane(), [0.0], []
    actor = _actor(plane, now, sink)
    actor.admit("late", "2026-01-01T00:00:20Z")
    actor.tick()
    actor.admit("early", "2026-01-01T00:00:00Z")

    now[0] = 45.0
    actor.tick()
    plane.seal(1)

    sealed = sink[-1]
    assert isinstance(sealed, ClipSealed)
    assert [contributor.event_ref for contributor in sealed.contributors] == ["early", "late"]
    assert sealed.boundary == "extension_bounded"
    # Nothing was cut short: the plane's own window bounded the clip.
    assert plane.stops == []


def test_a_refused_start_is_counted_and_retried_rather_than_dropped() -> None:
    plane, now, sink = FakePlane(refusals=1), [0.0], []
    actor = _actor(plane, now, sink)

    # The actor starts on admission; a refusal must be counted and kept
    # pending rather than discarding the alert.
    actor.admit("event-1", "2026-01-01T00:00:00Z")
    assert plane.starts == []
    assert actor.smart_record_start_refused_total == 1

    now[0] = 1.0
    actor.tick()
    assert plane.starts == [(15, 45)]


def test_a_duplicate_seal_is_surfaced_to_the_sink_not_swallowed() -> None:
    plane, now, sink = FakePlane(), [0.0], []
    actor = _actor(plane, now, sink)
    actor.admit("event-1", "2026-01-01T00:00:00Z")
    actor.tick()
    plane.seal(1)
    plane.seal(1)

    assert isinstance(sink[-1], DuplicateRecordingSealedError)
    assert len([item for item in sink if isinstance(item, ClipSealed)]) == 1


def test_an_alert_after_the_clip_sealed_opens_the_next_one() -> None:
    plane, now, sink = FakePlane(), [0.0], []
    actor = _actor(plane, now, sink)
    actor.admit("first", "2026-01-01T00:00:00Z")
    actor.tick()
    now[0] = 45.0
    actor.tick()
    plane.seal(1)

    now[0] = 61.0
    actor.admit("second", "2026-01-01T00:01:01Z")
    actor.tick()

    assert plane.starts == [(15, 45), (15, 45)]
    clips = [item for item in sink if isinstance(item, ClipSealed)]
    assert [contributor.event_ref for contributor in clips[0].contributors] == ["first"]


def test_seal_is_not_latched_until_the_sink_accepts_it() -> None:
    plane, now = FakePlane(), [0.0]
    accepted: list[ClipSealed] = []
    failures = [True]

    def sink(item: object) -> None:
        if failures[0]:
            failures[0] = False
            raise RuntimeError("publication failed")
        assert isinstance(item, ClipSealed)
        accepted.append(item)

    actor = SmartRecordActor(
        camera_id="camera-a",
        media_plane=plane,
        clock=lambda: now[0],
        sink=sink,
        lookback_sec=15,
        clip_id_factory=lambda: "clip-1",
    )
    actor.admit("event-1", "2026-01-01T00:00:00Z")

    with pytest.raises(RuntimeError, match="publication failed"):
        plane.seal(1)

    assert actor.state.name == "FINALIZING"
    plane.seal(1)
    assert [clip.clip_id for clip in accepted] == ["clip-1"]
