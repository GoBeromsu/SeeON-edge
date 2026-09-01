from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class SceneRecord:
    """An immutable, already serialized scene frame for the in-memory ring.

    Callers must finish canonical serialization, detail shedding, and byte
    accounting on their camera thread before constructing this record. Rings
    only retain these immutable bytes and never serialize while holding a lock.
    """

    worker_boot_id: str
    camera_id: str
    stream_epoch: int
    generation: int
    source_pts_sec: Fraction
    seq: int
    payload: bytes
    size_bytes: int
    detail_shed: bool

    def __post_init__(self) -> None:
        if not self.worker_boot_id or not self.camera_id:
            raise ValueError("scene record identity must not be blank")
        if self.stream_epoch < 0 or self.generation < 0 or self.seq < 0:
            raise ValueError("scene record epoch, generation, and sequence must be non-negative")
        if self.size_bytes != len(self.payload) or self.size_bytes <= 0:
            raise ValueError("scene record size must exactly match a non-empty payload")


__all__ = ["SceneRecord"]
