from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from worker.adapters.encode import (
    ClipRemuxError,
    CrossEpochSegmentError,
    CrossGenerationSegmentError,
    FFmpegConcatFinalizer,
    Segment,
)
from worker.interfaces import ClipFinalizer
from worker.types import BusinessEvent


@dataclass(slots=True)
class _RunnerSpy:
    """Records every ffmpeg invocation the finalizer performs."""

    returncode: int = 0
    commands: list[tuple[str, ...]] = field(default_factory=list)

    def __call__(self, args: tuple[str, ...]) -> int:
        self.commands.append(args)
        return self.returncode


def _event() -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", 7, "cam-1", "facility-1", 12.0, 0.91)


def _segments(tmp_path: Path, *, generation: int = 1, count: int = 2) -> tuple[Segment, ...]:
    segment_dir = tmp_path / "segments" / "cam-1" / f"gen-{generation}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []
    for index in range(count):
        path = segment_dir / f"seg-{index:05d}.mp4"
        _ = path.write_bytes(b"segment-media")
        segments.append(
            Segment(
                path=path,
                generation=generation,
                start_time_sec=float(index) * 2.0,
                end_time_sec=float(index + 1) * 2.0,
            )
        )
    return tuple(segments)


def test_finalizer_conforms_to_the_port(tmp_path: Path) -> None:
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=_RunnerSpy())

    assert isinstance(finalizer, ClipFinalizer)


def test_finalize_remuxes_with_stream_copy_and_faststart(tmp_path: Path) -> None:
    runner = _RunnerSpy()
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=runner)

    artifact = finalizer.finalize(_segments(tmp_path), _event())

    assert len(runner.commands) == 1
    args = runner.commands[0]
    assert args[args.index("-f") + 1] == "concat"
    assert args[args.index("-safe") + 1] == "0"
    assert args[args.index("-c") + 1] == "copy"
    assert args[args.index("-movflags") + 1] == "+faststart"
    assert artifact.path == Path(args[-1])
    assert artifact.segment_count == 2
    assert artifact.generation == 1
    assert artifact.duration_s == 4.0


def test_finalize_derives_media_origin_from_selected_segment_metadata(
    tmp_path: Path,
) -> None:
    runner = _RunnerSpy()
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=runner)
    segments = tuple(
        Segment(
            path=segment.path,
            generation=segment.generation,
            start_time_sec=segment.start_time_sec + 2.0,
            end_time_sec=segment.end_time_sec + 2.0,
            worker_boot_id="boot-1",
            camera_id="cam-1",
            stream_epoch=4,
            source_origin_pts_sec=100.0,
        )
        for segment in _segments(tmp_path)
    )

    artifact = finalizer.finalize(segments, _event())

    assert artifact.worker_boot_id == "boot-1"
    assert artifact.camera_id == "cam-1"
    assert artifact.stream_epoch == 4
    assert artifact.media_origin_pts_sec == 102.0


def test_finalize_refuses_segments_from_two_stream_epochs(tmp_path: Path) -> None:
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=_RunnerSpy())
    segments = _segments(tmp_path)
    mixed = (
        Segment(
            path=segments[0].path,
            generation=1,
            start_time_sec=0.0,
            end_time_sec=2.0,
            worker_boot_id="boot-1",
            camera_id="cam-1",
            stream_epoch=1,
            source_origin_pts_sec=10.0,
        ),
        Segment(
            path=segments[1].path,
            generation=1,
            start_time_sec=2.0,
            end_time_sec=4.0,
            worker_boot_id="boot-1",
            camera_id="cam-1",
            stream_epoch=2,
            source_origin_pts_sec=0.0,
        ),
    )

    with pytest.raises(CrossEpochSegmentError):
        finalizer.finalize(mixed, _event())


def test_finalize_never_starts_an_encoder_or_decodes_frames(tmp_path: Path) -> None:
    runner = _RunnerSpy()
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=runner)

    _ = finalizer.finalize(_segments(tmp_path), _event())
    args = runner.commands[0]

    assert "-c:v" not in args
    assert "libx264" not in args
    assert "h264_nvenc" not in args
    assert "rawvideo" not in args
    assert "pipe:0" not in args


def test_finalize_writes_a_concat_list_of_the_selected_segments(tmp_path: Path) -> None:
    runner = _RunnerSpy()
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=runner)
    segments = _segments(tmp_path)

    _ = finalizer.finalize(segments, _event())

    args = runner.commands[0]
    list_path = Path(args[args.index("-i") + 1])
    lines = list_path.read_text(encoding="utf-8").splitlines()
    assert lines == [f"file '{segment.path}'" for segment in segments]


def test_finalize_refuses_segments_from_two_generations(tmp_path: Path) -> None:
    runner = _RunnerSpy()
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=runner)
    mixed = _segments(tmp_path, generation=1) + _segments(tmp_path, generation=2)

    with pytest.raises(CrossGenerationSegmentError):
        _ = finalizer.finalize(mixed, _event())

    assert runner.commands == []


def test_finalize_reports_typed_failure_when_remux_exits_non_zero(tmp_path: Path) -> None:
    runner = _RunnerSpy(returncode=1)
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=runner)

    with pytest.raises(ClipRemuxError):
        _ = finalizer.finalize(_segments(tmp_path), _event())


def test_finalize_refuses_an_empty_segment_selection(tmp_path: Path) -> None:
    runner = _RunnerSpy()
    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips", runner=runner)

    with pytest.raises(ClipRemuxError):
        _ = finalizer.finalize((), _event())

    assert runner.commands == []
