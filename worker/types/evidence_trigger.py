"""Image-free identity used to bind native decisions to source evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from worker.types.frame_packet import FrameKey


class EvidenceTrigger(Protocol):
    camera_id: str
    worker_boot_id: str
    stream_epoch: int
    source_generation: int
    pts: float | None

    @property
    def frame_key(self) -> FrameKey: ...

    @property
    def trigger_time_sec(self) -> float: ...

    def retain(self) -> EvidenceTrigger: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NativeEvidenceTrigger:
    camera_id: str
    worker_boot_id: str
    stream_epoch: int
    source_generation: int
    seq: int
    source_pts: int
    time_sec: float

    @property
    def pts(self) -> float:
        return self.source_pts / 1_000_000_000

    @property
    def trigger_time_sec(self) -> float:
        return self.pts

    @property
    def frame_key(self) -> FrameKey:
        return FrameKey(
            self.worker_boot_id,
            self.camera_id,
            self.stream_epoch,
            self.seq,
            self.pts,
            self.source_pts,
            source_generation=self.source_generation,
        )

    def retain(self) -> NativeEvidenceTrigger:
        return self

    def release(self) -> None:
        return None


__all__ = ["EvidenceTrigger", "NativeEvidenceTrigger"]
