from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceResidentPoolConfig:
    """Bounded configuration for one camera's device-resident frame pool.

    ``capacity`` bounds outstanding device-resident slots (backpressure, per
    the plan's "bounded pools/backpressure" requirement) -- never an
    unbounded queue, mirroring ``worker/pipeline/bus``'s bounded-subscription
    convention.
    """

    camera_id: str
    capacity: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("device-resident pool config requires a camera id")
        if self.capacity <= 0:
            raise ValueError("device-resident pool capacity must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("device-resident pool frame dimensions must be positive")


__all__ = ["DeviceResidentPoolConfig"]
