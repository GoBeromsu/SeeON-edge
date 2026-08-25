from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction

from contracts.frame import Frame
from worker.types.frame_memory import FrameDescriptor, FrameLease


@dataclass(frozen=True, slots=True)
class FrameKey:
    worker_boot_id: str
    camera_id: str
    stream_epoch: int
    seq: int
    pts: float | None
    source_pts: int | None = None
    source_time_base: Fraction | None = None
    source_generation: int = 0


@dataclass(frozen=True, slots=True, init=False)
class FramePacket:
    camera_id: str
    _frame: Frame = field(compare=False, hash=False, repr=False)
    pts: float | None
    seq: int
    width: int
    height: int
    decode_time_ms: float
    worker_boot_id: str = ""
    stream_epoch: int = 0
    source_pts: int | None = None
    source_dts: int | None = None
    source_time_base: Fraction | None = None
    source_generation: int = 0
    lease: FrameLease = field(compare=False, hash=False, repr=False)

    def __init__(
        self,
        camera_id: str,
        frame: Frame | None = None,
        pts: float | None = None,
        seq: int = 0,
        width: int = 0,
        height: int = 0,
        decode_time_ms: float = 0.0,
        worker_boot_id: str = "",
        stream_epoch: int = 0,
        source_pts: int | None = None,
        source_dts: int | None = None,
        source_time_base: Fraction | None = None,
        lease: FrameLease | None = None,
        source_generation: int = 0,
        *,
        _frame: Frame | None = None,
    ) -> None:
        resolved_frame = frame if frame is not None else _frame
        if resolved_frame is None:
            raise ValueError("frame packet requires a host frame")
        resolved_lease = FrameLease.from_host(resolved_frame) if lease is None else lease
        descriptor = resolved_lease.descriptor
        if descriptor.width != width or descriptor.height != height:
            raise ValueError("frame lease descriptor does not match packet geometry")
        if resolved_lease.host_frame is not resolved_frame:
            raise ValueError("frame packet frame does not match its lease storage")
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "_frame", resolved_frame)
        object.__setattr__(self, "pts", pts)
        object.__setattr__(self, "seq", seq)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "decode_time_ms", decode_time_ms)
        object.__setattr__(self, "worker_boot_id", worker_boot_id)
        object.__setattr__(self, "stream_epoch", stream_epoch)
        object.__setattr__(self, "source_pts", source_pts)
        object.__setattr__(self, "source_dts", source_dts)
        object.__setattr__(self, "source_time_base", source_time_base)
        object.__setattr__(self, "source_generation", source_generation)
        object.__setattr__(self, "lease", resolved_lease)

    @property
    def frame(self) -> Frame:
        """Compatibility host borrow guarded by this packet's lease handle."""
        return self.borrow_host_frame()

    def borrow_host_frame(self) -> Frame:
        frame = self.lease.host_frame
        if frame is not self._frame:  # pragma: no cover - constructor invariant
            raise RuntimeError("frame packet lease storage changed unexpectedly")
        return frame

    @property
    def descriptor(self) -> FrameDescriptor:
        return self.lease.descriptor

    @property
    def frame_key(self) -> FrameKey:
        return FrameKey(
            self.worker_boot_id,
            self.camera_id,
            self.stream_epoch,
            self.seq,
            self.pts,
            self.source_pts,
            self.source_time_base,
            self.source_generation,
        )

    def retain(self) -> FramePacket:
        return replace(self, lease=self.lease.retain())

    def with_lease(self, lease: FrameLease) -> FramePacket:
        return replace(self, lease=lease)

    def release(self) -> None:
        self.lease.release()

    @property
    def released(self) -> bool:
        return self.lease.released


__all__ = ["FrameKey", "FramePacket"]
