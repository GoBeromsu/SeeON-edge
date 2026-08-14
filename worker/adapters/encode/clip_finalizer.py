from __future__ import annotations

import subprocess
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Protocol, final

from worker.adapters.encode.adapter_errors import (
    ClipRemuxError,
    CrossEpochSegmentError,
    CrossGenerationSegmentError,
)
from worker.adapters.encode.csv_segment_index import Segment
from worker.adapters.encode.models import ClipArtifact
from worker.types import BusinessEvent


class RemuxRunner(Protocol):
    def __call__(self, args: tuple[str, ...]) -> int: ...


def run_ffmpeg_remux(args: tuple[str, ...]) -> int:
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise ClipRemuxError(
            f"failed to start ffmpeg remux ({type(exc).__name__})"
        ) from exc
    return completed.returncode


def _concat_line(path: Path) -> str:
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'"


@final
class FFmpegConcatFinalizer:
    def __init__(
        self,
        output_dir: Path,
        *,
        runner: RemuxRunner = run_ffmpeg_remux,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self._output_dir = output_dir
        self._runner = runner
        self._ffmpeg_bin = ffmpeg_bin

    def finalize(
        self,
        segments: Sequence[Segment],
        event: BusinessEvent,
    ) -> ClipArtifact:
        del event
        if not segments:
            raise ClipRemuxError("cannot finalize an empty segment selection")
        generations = tuple(sorted({segment.generation for segment in segments}))
        if len(generations) != 1:
            raise CrossGenerationSegmentError(generations)
        stream_identities = {
            (segment.worker_boot_id, segment.camera_id, segment.stream_epoch)
            for segment in segments
        }
        if len(stream_identities) != 1:
            raise CrossEpochSegmentError(
                tuple(sorted({segment.stream_epoch for segment in segments}))
            )

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            list_path = self._output_dir / "segments.ffconcat"
            _ = list_path.write_text(
                "".join(f"{_concat_line(segment.path)}\n" for segment in segments),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ClipRemuxError(
                f"failed to prepare ffmpeg concat input ({type(exc).__name__})"
            ) from exc

        output_path = self._output_dir / "clip.mp4"
        args = (
            self._ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-map",
            "0:v:0",
            "-an",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        )
        returncode = self._runner(args)
        if returncode != 0:
            with suppress(OSError):
                output_path.unlink(missing_ok=True)
            raise ClipRemuxError("ffmpeg concat remux failed", returncode=returncode)

        duration_s = sum(
            max(0.0, segment.end_time_sec - segment.start_time_sec)
            for segment in segments
        )
        first_segment = segments[0]
        return ClipArtifact(
            path=output_path,
            generation=generations[0],
            segment_count=len(segments),
            duration_s=duration_s,
            worker_boot_id=first_segment.worker_boot_id,
            camera_id=first_segment.camera_id,
            stream_epoch=first_segment.stream_epoch,
            media_origin_pts_sec=first_segment.media_origin_pts_sec,
        )


__all__ = ["FFmpegConcatFinalizer", "RemuxRunner", "run_ffmpeg_remux"]
