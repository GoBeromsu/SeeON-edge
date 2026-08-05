from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from contracts.frame import Frame
from worker.adapters.encode import ClipRemuxError, EncoderGeometry, EncoderStartError
from worker.adapters.encode.adapter_errors import (
    CrossGenerationSegmentError,
    EncoderPolicyError,
    EncoderWriteError,
)
from worker.adapters.encode.csv_segment_index import Segment
from worker.adapters.encode.models import ClipArtifact
from worker.pipeline.output.evidence import (
    ClipReady,
    ClipReasonCode,
    ClipRecordingCoordinator,
    ClipUnavailable,
    ClipWindow,
)
from worker.types import BusinessEvent, FramePacket


def _packet(seq: int, *, camera_id: str = "cam-1", width: int = 4, height: int = 2) -> FramePacket:
    image = np.full((height, width, 3), seq % 255, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=float(seq), image=image)
    return FramePacket(camera_id, frame, float(seq), seq, width, height, 1.0)


def _event() -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", 7, "cam-1", "facility-1", 12.0, 0.9)


def _segment(index: int, *, generation: int = 1) -> Segment:
    return Segment(
        path=Path(f"/tmp/seg-{index:05d}.mp4"),
        generation=generation,
        start_time_sec=float(index) * 2.0,
        end_time_sec=float(index + 1) * 2.0,
    )


@dataclass(frozen=True, slots=True)
class _FakeSession:
    """Records what one camera's encoder session received."""

    camera: str
    geometry: EncoderGeometry
    segments: tuple[Segment, ...] = ()
    packets: list[FramePacket] = field(default_factory=list)
    windows: list[tuple[float, float]] = field(default_factory=list)
    closed: list[None] = field(default_factory=list)
    write_error: Exception | None = None
    select_error: Exception | None = None
    origin_time_sec: float | None = None

    def write(self, packet: FramePacket) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.packets.append(packet)

    def close(self) -> None:
        self.closed.append(None)

    def select_segments(
        self,
        *,
        start_time_sec: float,
        end_time_sec: float,
    ) -> tuple[Segment, ...]:
        self.windows.append((start_time_sec, end_time_sec))
        if self.select_error is not None:
            raise self.select_error
        return self.segments


@dataclass(frozen=True, slots=True)
class _FakeEncoder:
    """Hands out one reusable session per camera."""

    sessions: dict[str, _FakeSession] = field(default_factory=dict)
    opens: list[None] = field(default_factory=list)
    open_error: Exception | None = None
    segments: tuple[Segment, ...] = ()

    def open(self, camera: str, profile: str, geometry: EncoderGeometry) -> _FakeSession:
        assert profile in {"libx264", "h264_nvenc"}
        self.opens.append(None)
        if self.open_error is not None:
            raise self.open_error
        session = self.sessions.get(camera)
        if session is None or session.geometry != geometry:
            session = _FakeSession(camera=camera, geometry=geometry, segments=self.segments)
            self.sessions[camera] = session
        return session


@dataclass(frozen=True, slots=True)
class _FakeFinalizer:
    calls: list[tuple[Segment, ...]] = field(default_factory=list)
    error: Exception | None = None

    def finalize(
        self,
        segments: Sequence[Segment],
        event: BusinessEvent,
    ) -> ClipArtifact:
        assert event.camera_id
        self.calls.append(tuple(segments))
        if self.error is not None:
            raise self.error
        return ClipArtifact(
            path=Path("/tmp/clip.mp4"),
            generation=segments[0].generation,
            segment_count=len(segments),
            duration_s=4.0,
        )


def _coordinator(
    encoder: _FakeEncoder,
    finalizer: _FakeFinalizer,
    *,
    window: ClipWindow | None = None,
) -> ClipRecordingCoordinator:
    return ClipRecordingCoordinator(
        encoder=encoder,
        finalizer=finalizer,
        profile="libx264",
        fps=5.0,
        window=window,
    )


def test_frames_reuse_one_session_per_camera() -> None:
    encoder = _FakeEncoder()
    coordinator = _coordinator(encoder, _FakeFinalizer())

    for seq in range(1, 11):
        assert coordinator.write(_packet(seq))

    assert len(encoder.sessions) == 1
    assert len(encoder.sessions["cam-1"].packets) == 10


def test_registered_camera_fps_controls_encoder_geometry() -> None:
    encoder = _FakeEncoder()
    coordinator = _coordinator(encoder, _FakeFinalizer())
    coordinator.set_camera_fps("cam-1", 15.0)

    assert coordinator.write(_packet(1))

    assert encoder.sessions["cam-1"].geometry.fps == 15.0


def test_sealed_session_remains_selectable_until_clip_finalizes() -> None:
    encoder = _FakeEncoder(segments=(_segment(0),))
    coordinator = _coordinator(encoder, _FakeFinalizer())
    coordinator.write(_packet(1))

    assert coordinator.seal("cam-1") is True
    outcome = coordinator.finalize(
        camera_id="cam-1",
        clip_id="clip-1",
        event_time_sec=1.0,
        event=_event(),
    )

    assert isinstance(outcome, ClipReady)
    assert len(encoder.sessions["cam-1"].closed) == 1


def test_finalize_remuxes_the_configured_pre_and_post_window() -> None:
    encoder = _FakeEncoder(segments=(_segment(0), _segment(1)))
    finalizer = _FakeFinalizer()
    coordinator = _coordinator(
        encoder,
        finalizer,
        window=ClipWindow(pre_event_seconds=10.0, post_event_seconds=20.0),
    )
    coordinator.write(_packet(1))

    outcome = coordinator.finalize(
        camera_id="cam-1",
        clip_id="clip-1",
        event_time_sec=12.0,
        event=_event(),
    )

    assert isinstance(outcome, ClipReady)
    assert outcome.clip_id == "clip-1"
    assert outcome.artifact.segment_count == 2
    assert encoder.sessions["cam-1"].windows == [(2.0, 32.0)]
    assert len(finalizer.calls) == 1


def test_finalize_reports_duration_clamped_to_the_requested_window() -> None:
    """Regression for the Linux-CI ``duration_s`` overshoot.

    Segments are ffmpeg's own coarse encode units -- their boundaries depend
    on encoder startup timing and keyframe placement, not the pre/post-event
    window -- so ``select_segments`` (``worker/adapters/encode/
    csv_segment_index.py``'s ``CsvSegmentIndex.select``) routinely returns
    boundary segments that only partially overlap ``[start_time_sec,
    end_time_sec]``. ``FFmpegConcatFinalizer.finalize`` used to be the sole
    source of ``ClipArtifact.duration_s``, reporting the raw sum of the
    *full* selected segments' durations, which overstated the clip by
    however much the boundary segments extended past the requested window
    (observed on Linux CI as 33.0s for a nominal 30.0s pre+post window).
    ``ClipRecordingCoordinator.finalize`` now clamps each segment's
    contribution to its overlap with the window before reporting
    ``duration_s``, regardless of what the finalizer itself returns -- this
    exercises only fakes, no ffmpeg/ffprobe/mediamtx, so it is deterministic
    on every OS.
    """
    # Segment 0 starts two seconds before the window and ends inside it;
    # segment 1 starts inside the window and ends eight seconds past it.
    boundary_segments = (
        Segment(path=Path("/tmp/seg-a.mp4"), generation=1, start_time_sec=0.0, end_time_sec=5.0),
        Segment(path=Path("/tmp/seg-b.mp4"), generation=1, start_time_sec=5.0, end_time_sec=40.0),
    )
    encoder = _FakeEncoder(segments=boundary_segments)
    # The fake finalizer's returned duration_s is intentionally unrelated to
    # the segments, so a passing assertion proves the coordinator itself
    # recomputed the clamped value rather than trusting the finalizer's.
    finalizer = _FakeFinalizer()
    coordinator = _coordinator(
        encoder,
        finalizer,
        window=ClipWindow(pre_event_seconds=10.0, post_event_seconds=20.0),
    )
    coordinator.write(_packet(1))

    outcome = coordinator.finalize(
        camera_id="cam-1",
        clip_id="clip-1",
        event_time_sec=12.0,
        event=_event(),
    )

    assert isinstance(outcome, ClipReady)
    assert outcome.artifact.duration_s == 30.0


def test_finalize_converts_the_duration_window_into_the_sessions_local_axis() -> None:
    """Regression for #165: a shifted generation origin must also shift the
    window used for duration clamping, not just segment selection -- else
    the same event-clock-vs-segment-clock mismatch reappears one level up
    (`_windowed_duration_s` comparing raw event-clock bounds against the
    session's local-axis segments) and reports a bogus ``duration_s``
    instead of the actual clamped overlap.
    """
    # This generation's local axis started at event-clock 12470.0 (its first
    # frame arrived then); segments are expressed in that local axis, so
    # they start at 0 rather than at the (much larger) event-clock reading.
    boundary_segments = (
        Segment(path=Path("/tmp/seg-a.mp4"), generation=7, start_time_sec=0.0, end_time_sec=40.0),
        Segment(path=Path("/tmp/seg-b.mp4"), generation=7, start_time_sec=40.0, end_time_sec=80.0),
    )
    session = _FakeSession(
        camera="cam-1",
        geometry=EncoderGeometry(4, 2, 5.0),
        segments=boundary_segments,
        origin_time_sec=12470.0,
    )
    encoder = _FakeEncoder(sessions={"cam-1": session})
    finalizer = _FakeFinalizer()
    coordinator = _coordinator(
        encoder,
        finalizer,
        window=ClipWindow(pre_event_seconds=30.0, post_event_seconds=30.0),
    )
    coordinator.write(_packet(1))

    outcome = coordinator.finalize(
        camera_id="cam-1",
        clip_id="clip-1",
        event_time_sec=12500.0,
        event=_event(),
    )

    assert isinstance(outcome, ClipReady)
    # Event window in event-clock terms: [12470.0, 12530.0].
    assert session.windows == [(12470.0, 12530.0)]
    # Converted into the session's local axis (subtract origin 12470.0):
    # [0.0, 60.0]. Overlap with segment a (local 0..40) is 40.0, with
    # segment b (local 40..80) is 20.0 -- total 60.0. Comparing the *raw*,
    # unconverted event-clock bounds ([12470.0, 12530.0]) against these
    # same local-axis segments (0..80) has no overlap at all and clamps to
    # 0.0 -- the bug this test pins.
    assert outcome.artifact.duration_s == 60.0


def test_remux_failure_yields_typed_unavailable_without_losing_the_event() -> None:
    encoder = _FakeEncoder(segments=(_segment(0),))
    finalizer = _FakeFinalizer(error=ClipRemuxError("boom"))
    coordinator = _coordinator(encoder, finalizer)
    coordinator.write(_packet(1))

    outcome = coordinator.finalize(
        camera_id="cam-1",
        clip_id="clip-1",
        event_time_sec=12.0,
        event=_event(),
    )

    assert outcome == ClipUnavailable("clip-1", ClipReasonCode.REMUX_FAILED)


def test_cross_generation_selection_is_reported_as_remux_failure() -> None:
    encoder = _FakeEncoder(segments=(_segment(0), _segment(1, generation=2)))
    finalizer = _FakeFinalizer(error=CrossGenerationSegmentError((1, 2)))
    coordinator = _coordinator(encoder, finalizer)
    coordinator.write(_packet(1))

    outcome = coordinator.finalize(
        camera_id="cam-1",
        clip_id="clip-1",
        event_time_sec=12.0,
        event=_event(),
    )

    assert outcome == ClipUnavailable("clip-1", ClipReasonCode.REMUX_FAILED)


def test_empty_window_is_reported_as_no_segments() -> None:
    encoder = _FakeEncoder(segments=())
    finalizer = _FakeFinalizer()
    coordinator = _coordinator(encoder, finalizer)
    coordinator.write(_packet(1))

    outcome = coordinator.finalize(
        camera_id="cam-1",
        clip_id="clip-1",
        event_time_sec=12.0,
        event=_event(),
    )

    assert outcome == ClipUnavailable("clip-1", ClipReasonCode.NO_SEGMENTS)
    assert finalizer.calls == []


def test_encoder_write_failure_degrades_only_that_camera() -> None:
    encoder = _FakeEncoder()
    coordinator = _coordinator(encoder, _FakeFinalizer())
    coordinator.write(_packet(1, camera_id="cam-2"))
    encoder.sessions["cam-1"] = _FakeSession(
        camera="cam-1",
        geometry=EncoderGeometry(4, 2, 5.0),
        write_error=EncoderWriteError("pipe closed"),
    )

    assert coordinator.write(_packet(2, camera_id="cam-1")) is False
    assert coordinator.write(_packet(3, camera_id="cam-2")) is True
    assert coordinator.degraded_cameras == frozenset({"cam-1"})
    assert len(encoder.sessions["cam-2"].packets) == 2


def test_finalize_without_a_live_session_reports_encoder_failure() -> None:
    coordinator = _coordinator(_FakeEncoder(), _FakeFinalizer())

    outcome = coordinator.finalize(
        camera_id="cam-1",
        clip_id="clip-1",
        event_time_sec=12.0,
        event=_event(),
    )

    assert outcome == ClipUnavailable("clip-1", ClipReasonCode.ENCODER_FAILED)


def test_encoder_start_failure_never_falls_back_to_another_encoder() -> None:
    encoder = _FakeEncoder(open_error=EncoderStartError("nvenc unavailable"))
    coordinator = _coordinator(encoder, _FakeFinalizer())

    assert coordinator.write(_packet(1)) is False
    assert coordinator.degraded_cameras == frozenset({"cam-1"})
    assert encoder.sessions == {}


def test_close_all_releases_every_camera_session() -> None:
    encoder = _FakeEncoder()
    coordinator = _coordinator(encoder, _FakeFinalizer())
    coordinator.write(_packet(1, camera_id="cam-1"))
    coordinator.write(_packet(2, camera_id="cam-2"))

    coordinator.close_all()
    coordinator.close("cam-1")

    assert len(encoder.sessions["cam-1"].closed) == 1
    assert len(encoder.sessions["cam-2"].closed) == 1


def test_unknown_profile_fails_at_construction() -> None:
    with pytest.raises(EncoderPolicyError):
        ClipRecordingCoordinator(
            encoder=_FakeEncoder(),
            finalizer=_FakeFinalizer(),
            profile="mp4v",
            fps=5.0,
        )


def test_negative_window_bounds_are_rejected() -> None:
    with pytest.raises(EncoderPolicyError):
        ClipWindow(pre_event_seconds=-1.0)
