from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final

from worker.adapters.encode.csv_segment_index import CsvSegmentIndex, Segment
from worker.adapters.encode.models import EncodePolicy, EncoderGeometry
from worker.adapters.encode.segment_process import EncoderProcess
from worker.types import FramePacket


@dataclass(frozen=True, slots=True)
class SessionGeneration:
    camera: str
    encoder: EncodePolicy
    geometry: EncoderGeometry
    number: int
    process: EncoderProcess
    index: CsvSegmentIndex


class SessionOwner(Protocol):
    def write_session(self, session: FFmpegEncoderSession, packet: FramePacket) -> None: ...

    def close_session(self, session: FFmpegEncoderSession) -> None: ...

    def segments_for(self, session: FFmpegEncoderSession) -> tuple[Segment, ...]: ...

    def select_for(
        self,
        session: FFmpegEncoderSession,
        *,
        start_time_sec: float,
        end_time_sec: float,
    ) -> tuple[Segment, ...]: ...


@final
class FFmpegEncoderSession:
    def __init__(self, owner: SessionOwner, state: SessionGeneration) -> None:
        self._owner = owner
        self._state = state
        self.closed = False

    @property
    def camera(self) -> str:
        return self._state.camera

    @property
    def encoder(self) -> EncodePolicy:
        return self._state.encoder

    @property
    def geometry(self) -> EncoderGeometry:
        return self._state.geometry

    @property
    def generation(self) -> int:
        return self._state.number

    @property
    def process(self) -> EncoderProcess:
        return self._state.process

    @property
    def index(self) -> CsvSegmentIndex:
        return self._state.index

    @property
    def origin_time_sec(self) -> float | None:
        return self._state.index.origin_time_sec

    @property
    def segment_list_path(self) -> Path:
        return self._state.index.list_path

    def matches(self, encoder: EncodePolicy, geometry: EncoderGeometry) -> bool:
        return not self.closed and self.encoder == encoder and self.geometry == geometry

    def write(self, packet: FramePacket) -> None:
        self._owner.write_session(self, packet)

    def close(self) -> None:
        self._owner.close_session(self)

    def closed_segments(self) -> tuple[Segment, ...]:
        return self._owner.segments_for(self)

    def select_segments(
        self,
        *,
        start_time_sec: float,
        end_time_sec: float,
    ) -> tuple[Segment, ...]:
        return self._owner.select_for(
            self,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
        )

    def replace(self, state: SessionGeneration) -> None:
        self._state = state

    def mark_closed(self) -> None:
        self.closed = True


__all__ = ["FFmpegEncoderSession", "SessionGeneration", "SessionOwner"]
