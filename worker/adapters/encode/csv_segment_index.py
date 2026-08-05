from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import final


@dataclass(frozen=True, slots=True)
class Segment:
    path: Path
    generation: int
    start_time_sec: float
    end_time_sec: float


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

    @property
    def list_path(self) -> Path:
        return self._list_path

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def origin_time_sec(self) -> float | None:
        """Event-clock value that corresponds to this generation's local 0.

        ``None`` until at least one frame has been observed.
        """
        return self._origin_time_sec

    def observe_frame(self, event_time_sec: float, fps: float) -> None:
        """Record one more frame written into this generation.

        FFmpeg assigns this generation-local clock to each frame as
        ``frame_index / fps`` (constant frame rate from a raw pipe input),
        so re-deriving the origin -- ``event_time_sec - frame_index / fps``
        -- on *every* frame keeps the mapping accurate even when the event
        clock itself resets independently (an RTSP reconnect restarts the
        decode session's clock without starting a new encoder generation):
        the very next frame after such a reset recomputes the origin from
        its own (small) event time and the (unaffected, still-growing)
        frame count, so the offset self-corrects with no explicit
        reconnect detection needed.
        """
        self._origin_time_sec = event_time_sec - (self._frame_count / fps)
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
