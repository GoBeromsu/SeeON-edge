from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import final

from worker.adapters.encode.adapter_errors import CrossEpochFrameError
from worker.types import FramePacket


@dataclass(frozen=True, slots=True)
class Segment:
    path: Path
    generation: int
    start_time_sec: float
    end_time_sec: float
    worker_boot_id: str = ""
    camera_id: str = ""
    stream_epoch: int = 0
    source_origin_pts_sec: float | None = None

    @property
    def media_origin_pts_sec(self) -> float | None:
        if self.source_origin_pts_sec is None:
            return None
        return self.source_origin_pts_sec + self.start_time_sec


@final
class CsvSegmentIndex:
    """Track completed segments reported by one FFmpeg CSV segment list.

    Segment start/end times in this index are FFmpeg's own generation-local
    clock: seconds since *this* segment muxer process started, which resets
    to 0 every time a new generation is spawned (#165). Callers windowing by
    event time -- a decode-session-relative clock that resets on its own,
    independent schedule (RTSP reconnects) -- must not compare the two
    clocks directly. ``observe_frame``/``origin_time_sec`` track the
    (continuously re-estimated) offset between them so ``select`` can
    convert a query window into this generation's local axis before
    filtering.
    """

    def __init__(self, list_path: Path, generation: int) -> None:
        self._list_path = list_path
        self._generation = generation
        self._segments_by_path: dict[Path, Segment] = {}
        self._origin_time_sec: float | None = None
        self._frame_count: int = 0
        self._worker_boot_id = ""
        self._camera_id = ""
        self._stream_epoch = 0

    @property
    def list_path(self) -> Path:
        return self._list_path

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def worker_boot_id(self) -> str:
        return self._worker_boot_id

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def stream_epoch(self) -> int:
        return self._stream_epoch

    @property
    def origin_time_sec(self) -> float | None:
        """Event-clock value that corresponds to this generation's local 0.

        ``None`` until at least one frame has been observed.
        """
        return self._origin_time_sec

    def observe_frame(self, frame: FramePacket | float, fps: float) -> None:
        """Record a frame without coercing distinct stream epochs together.

        Numeric input is retained for legacy callers. Production supplies a
        ``FramePacket`` so the generation is anchored once to its first frame
        and a reconnect is rejected rather than silently re-basing its PTS.
        """
        if isinstance(frame, FramePacket):
            if self._frame_count == 0:
                self._worker_boot_id = frame.worker_boot_id
                self._camera_id = frame.camera_id
                self._stream_epoch = frame.stream_epoch
                self._origin_time_sec = frame.frame.time_sec if frame.pts is None else frame.pts
            elif (
                frame.worker_boot_id != self._worker_boot_id
                or frame.camera_id != self._camera_id
                or frame.stream_epoch != self._stream_epoch
            ):
                raise CrossEpochFrameError(self._stream_epoch, frame.stream_epoch)
        else:
            self._origin_time_sec = frame - (self._frame_count / fps)
        self._frame_count += 1

    def refresh(self) -> tuple[Segment, ...]:
        try:
            rows = csv.reader(self._list_path.read_text(encoding="utf-8").splitlines())
        except FileNotFoundError:
            return ()

        discovered: list[Segment] = []
        for row in rows:
            if len(row) != 3:
                continue
            try:
                start_time_sec = float(row[1])
                end_time_sec = float(row[2])
            except ValueError:
                continue
            path = Path(row[0])
            if not path.is_absolute():
                path = self._list_path.parent / path
            if path in self._segments_by_path:
                continue
            segment = Segment(
                path=path,
                generation=self._generation,
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
                worker_boot_id=self._worker_boot_id,
                camera_id=self._camera_id,
                stream_epoch=self._stream_epoch,
                source_origin_pts_sec=self._origin_time_sec,
            )
            self._segments_by_path[path] = segment
            discovered.append(segment)
        return tuple(discovered)

    def completed(self) -> tuple[Segment, ...]:
        return tuple(self._segments_by_path.values())

    def select(self, *, start_time_sec: float, end_time_sec: float) -> tuple[Segment, ...]:
        local_start, local_end = self._to_local(start_time_sec, end_time_sec)
        return tuple(
            segment
            for segment in self._segments_by_path.values()
            if segment.end_time_sec >= local_start
            and segment.start_time_sec <= local_end
        )

    def _to_local(self, start_time_sec: float, end_time_sec: float) -> tuple[float, float]:
        if self._origin_time_sec is None:
            return start_time_sec, end_time_sec
        return (
            start_time_sec - self._origin_time_sec,
            end_time_sec - self._origin_time_sec,
        )


__all__ = ["CsvSegmentIndex", "Segment"]
