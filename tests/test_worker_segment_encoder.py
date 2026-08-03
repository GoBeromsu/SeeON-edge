from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from contracts.frame import Frame
from worker.adapters.encode import (
    EncoderGeometry,
    EncoderPolicyError,
    EncoderStartError,
    EncoderWriteError,
    FFmpegEncoderSession,
    FFmpegSegmentEncoder,
    SegmentEncoderConfig,
)
from worker.interfaces import ClipEncoder, EncoderSession
from worker.types import FramePacket


@dataclass(slots=True)
class _FakeProcess:
    """Records what one spawned encoder process received (mutation is the point)."""

    args: tuple[str, ...]
    payloads: list[bytes] = field(default_factory=list)
    reaped: int = 0
    on_reap: Callable[[], None] | None = None

    def write(self, payload: bytes) -> None:
        self.payloads.append(payload)

    def reap(self) -> int | None:
        self.reaped += 1
        if self.on_reap is not None:
            self.on_reap()
        return 0


@dataclass(slots=True)
class _SpawnSpy:
    """Test double for the ffmpeg spawn seam."""

    processes: list[_FakeProcess] = field(default_factory=list)
    fail_encoders: tuple[str, ...] = ()

    def __call__(self, args: tuple[str, ...]) -> _FakeProcess:
        process = _FakeProcess(args=args)
        self.processes.append(process)
        encoder = args[args.index("-c:v") + 1]
        if encoder in self.fail_encoders:
            raise EncoderStartError(f"cannot start {encoder}")
        return process

    @property
    def encoders(self) -> list[str]:
        return [
            process.args[process.args.index("-c:v") + 1] for process in self.processes
        ]


def _packet(seq: int, *, width: int = 4, height: int = 2) -> FramePacket:
    image = np.full((height, width, 3), seq % 255, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=float(seq) * 0.5, image=image)
    return FramePacket("cam-1", frame, float(seq) * 0.5, seq, width, height, 1.5)


def _encoder(tmp_path: Path, spawner: _SpawnSpy) -> FFmpegSegmentEncoder:
    return FFmpegSegmentEncoder(
        SegmentEncoderConfig(store_dir=tmp_path, segment_seconds=2.0),
        spawner=spawner,
    )


def _write_segment_list(
    session: FFmpegEncoderSession,
    entries: tuple[tuple[str, float, float], ...],
) -> None:
    list_path = session.segment_list_path
    list_path.parent.mkdir(parents=True, exist_ok=True)
    _ = list_path.write_text(
        "".join(f"{name},{start:.6f},{end:.6f}\n" for name, start, end in entries),
        encoding="utf-8",
    )


def test_encoder_conforms_to_the_clip_encoder_port(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path, _SpawnSpy())

    assert isinstance(encoder, ClipEncoder)
    session = encoder.open("cam-1", "libx264", EncoderGeometry(4, 2, 5.0))
    assert isinstance(session, EncoderSession)


def test_one_camera_across_five_segment_boundaries_starts_one_process(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)
    geometry = EncoderGeometry(4, 2, 5.0)

    session = encoder.open("cam-1", "libx264", geometry)
    for seq in range(1, 26):
        # Re-opening per frame batch is the reuse contract: the long-lived
        # segment muxer must not be replaced at a segment boundary.
        assert encoder.open("cam-1", "libx264", geometry) is session
        session.write(_packet(seq))

    assert encoder.metrics.process_starts == 1
    assert encoder.metrics.recreates == 0
    assert encoder.metrics.active_sessions == 1
    assert len(spawner.processes) == 1
    assert len(spawner.processes[0].payloads) == 25


def test_two_cameras_start_two_processes_with_isolated_sessions(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)
    geometry = EncoderGeometry(4, 2, 5.0)

    first = encoder.open("cam-1", "libx264", geometry)
    second = encoder.open("cam-2", "libx264", geometry)
    first.write(_packet(1))
    second.write(_packet(2))

    assert first is not second
    assert encoder.metrics.process_starts == 2
    assert encoder.metrics.active_sessions == 2
    assert encoder.metrics.recreates == 0


def test_resolution_change_increments_exactly_one_recreate_and_new_generation(
    tmp_path: Path,
) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)

    session = encoder.open("cam-1", "libx264", EncoderGeometry(4, 2, 5.0))
    session.write(_packet(1))
    first_generation = session.generation
    session.write(_packet(2, width=8, height=4))

    assert encoder.metrics.recreates == 1
    assert encoder.metrics.process_starts == 2
    assert session.generation == first_generation + 1
    assert spawner.processes[0].reaped == 1
    assert "8x4" in spawner.processes[1].args


def test_fps_change_recreates_only_that_camera(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)

    untouched = encoder.open("cam-2", "libx264", EncoderGeometry(4, 2, 5.0))
    session = encoder.open("cam-1", "libx264", EncoderGeometry(4, 2, 5.0))
    recreated = encoder.open("cam-1", "libx264", EncoderGeometry(4, 2, 15.0))

    assert recreated is not session
    assert encoder.open("cam-2", "libx264", EncoderGeometry(4, 2, 5.0)) is untouched
    assert encoder.metrics.recreates == 1
    assert encoder.metrics.active_sessions == 2


def test_closed_segments_never_concatenate_across_generations(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)

    session = encoder.open("cam-1", "libx264", EncoderGeometry(4, 2, 5.0))
    session.write(_packet(1))
    _write_segment_list(session, (("seg-00000.mp4", 0.0, 2.0), ("seg-00001.mp4", 2.0, 4.0)))
    old_generation_segments = session.closed_segments()
    session.write(_packet(2, width=8, height=4))
    _write_segment_list(session, (("seg-00000.mp4", 0.0, 2.0),))

    current = session.closed_segments()

    assert {segment.generation for segment in old_generation_segments} == {1}
    assert {segment.generation for segment in current} == {2}
    assert session.select_segments(start_time_sec=0.0, end_time_sec=4.0) == current


def test_nvenc_session_open_failure_falls_back_to_libx264_once(tmp_path: Path) -> None:
    """#53's sanctioned exception to #43's "no implicit fallback" rule.

    #43 forbids a silent LSTM -> random-forest fallback because that changes
    a decision's *semantic content*. NVENC -> libx264 only changes encode
    *cost* (GPU vs CPU): both produce the same H.264 clip content, so a
    session-open failure on nvenc is allowed -- and required -- to demote
    that camera to libx264 rather than losing the recording entirely. This
    replaces the pre-#53 test (same name minus "_falls_back_to_libx264_once")
    which asserted the opposite: that an nvenc failure must never start the
    software encoder. That was correct under #43's original blanket rule;
    #53 is the explicit, narrower exception for encode-only cost fallback.
    """
    spawner = _SpawnSpy(fail_encoders=("h264_nvenc",))
    encoder = _encoder(tmp_path, spawner)

    session = encoder.open("cam-1", "h264_nvenc", EncoderGeometry(4, 2, 5.0))

    assert spawner.encoders == ["h264_nvenc", "libx264"]
    assert session.encoder == "libx264"
    assert encoder.metrics.failures == 0
    assert encoder.metrics.active_sessions == 1
    assert encoder.metrics.encode_fallbacks == 1
    selection = encoder.encode_selections["cam-1"]
    assert selection.requested == "h264_nvenc"
    assert selection.selected == "libx264"
    assert selection.fallback_count == 1
    assert selection.last_reason == "session_open_failed"


def test_nvenc_fallback_is_persistent_across_reopens(tmp_path: Path) -> None:
    spawner = _SpawnSpy(fail_encoders=("h264_nvenc",))
    encoder = _encoder(tmp_path, spawner)
    geometry = EncoderGeometry(4, 2, 5.0)

    first = encoder.open("cam-1", "h264_nvenc", geometry)
    # Re-requesting nvenc for the same camera must not retry nvenc: the
    # demotion is persistent once recorded.
    second = encoder.open("cam-1", "h264_nvenc", geometry)

    assert first is second
    assert spawner.encoders == ["h264_nvenc", "libx264"]
    assert encoder.metrics.encode_fallbacks == 1


def test_nvenc_and_libx264_both_failing_raises_and_records_one_failure(
    tmp_path: Path,
) -> None:
    spawner = _SpawnSpy(fail_encoders=("h264_nvenc", "libx264"))
    encoder = _encoder(tmp_path, spawner)

    with pytest.raises(EncoderStartError):
        _ = encoder.open("cam-1", "h264_nvenc", EncoderGeometry(4, 2, 5.0))

    assert spawner.encoders == ["h264_nvenc", "libx264"]
    assert encoder.metrics.failures == 1
    assert encoder.metrics.active_sessions == 0


def test_nvenc_fallback_stays_demoted_through_a_geometry_triggered_recreate(
    tmp_path: Path,
) -> None:
    spawner = _SpawnSpy(fail_encoders=("h264_nvenc",))
    encoder = _encoder(tmp_path, spawner)

    session = encoder.open("cam-1", "h264_nvenc", EncoderGeometry(4, 2, 5.0))
    # A mid-stream geometry change forces a recreate; the camera must stay on
    # libx264 rather than re-attempting nvenc.
    session.write(_packet(1, width=8, height=4))

    assert spawner.encoders == ["h264_nvenc", "libx264", "libx264"]
    assert session.encoder == "libx264"
    assert encoder.metrics.encode_fallbacks == 1


def test_profile_selects_the_codec_without_probing(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)

    _ = encoder.open("cam-1", "h264_nvenc", EncoderGeometry(256, 256, 5.0))
    _ = encoder.open("cam-2", "libx264", EncoderGeometry(4, 2, 5.0))

    assert spawner.encoders == ["h264_nvenc", "libx264"]
    with pytest.raises(EncoderPolicyError):
        _ = encoder.open("cam-3", "mp4v", EncoderGeometry(4, 2, 5.0))


def test_close_is_idempotent_reaps_the_process_and_frees_the_session(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)

    session = encoder.open("cam-1", "libx264", EncoderGeometry(4, 2, 5.0))
    session.write(_packet(1))
    session.close()
    session.close()

    assert spawner.processes[0].reaped == 1
    assert encoder.metrics.active_sessions == 0
    with pytest.raises(EncoderWriteError):
        session.write(_packet(2))


def test_close_indexes_the_final_segment_flushed_during_child_reap(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)
    session = encoder.open("cam-1", "libx264", EncoderGeometry(4, 2, 5.0))
    process = spawner.processes[0]
    process.on_reap = lambda: _write_segment_list(session, (("seg-00000.mp4", 0.0, 2.0),))

    session.close()

    closed = session.closed_segments()
    assert [(segment.start_time_sec, segment.end_time_sec) for segment in closed] == [(0.0, 2.0)]
    assert encoder.metrics.finalized_segments == 1


def test_encoder_arguments_declare_a_long_lived_segment_muxer(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)

    _ = encoder.open("cam-1", "libx264", EncoderGeometry(640, 360, 5.0))
    args = spawner.processes[0].args

    assert args[args.index("-f") + 1] == "rawvideo"
    assert args[args.index("-pix_fmt") + 1] == "rgb24"
    assert args[args.index("-i") + 1] == "pipe:0"
    assert args[args.index("-s") + 1] == "640x360"
    assert "-c:v" in args and args[args.index("-c:v") + 1] == "libx264"
    assert "yuv420p" in args
    assert "segment" in args
    assert args[args.index("-segment_time") + 1] == "2"
    assert args[args.index("-segment_list_type") + 1] == "csv"
    assert args[-1].endswith("seg-%05d.mp4")


def test_finalized_segment_counter_tracks_closed_segments(tmp_path: Path) -> None:
    spawner = _SpawnSpy()
    encoder = _encoder(tmp_path, spawner)

    session = encoder.open("cam-1", "libx264", EncoderGeometry(4, 2, 5.0))
    session.write(_packet(1))
    _write_segment_list(session, (("seg-00000.mp4", 0.0, 2.0), ("seg-00001.mp4", 2.0, 4.0)))
    session.write(_packet(2))
    _write_segment_list(
        session,
        (
            ("seg-00000.mp4", 0.0, 2.0),
            ("seg-00001.mp4", 2.0, 4.0),
            ("seg-00002.mp4", 4.0, 6.0),
        ),
    )
    session.write(_packet(3))

    assert encoder.metrics.finalized_segments == 3


def test_geometry_rejects_non_positive_dimensions() -> None:
    with pytest.raises(EncoderPolicyError):
        _ = EncoderGeometry(0, 2, 5.0)
    with pytest.raises(EncoderPolicyError):
        _ = EncoderGeometry(4, 2, 0.0)
