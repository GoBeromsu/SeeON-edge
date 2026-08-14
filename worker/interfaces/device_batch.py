from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from worker.types import FrameLease


@runtime_checkable
class DeviceResidentPool(Protocol):
    """Bounded pool owning device-resident frame storage with backpressure.

    Distinct from ``worker.interfaces.serving.BatchServingClient`` (the
    deliberately unimplemented ADR-0002 batched-inference seam): this port
    only owns acquiring/recycling bounded device-resident storage slots for
    leases already produced by a decode adapter. It never runs inference and
    never decides task/schedule policy -- those stay the runtime's and the
    domain's concern respectively.
    """

    def acquire(self) -> FrameLease:
        """Acquire one bounded device-resident storage slot as a fresh lease.

        Raises a pool-specific backpressure error (never blocks silently and
        never allocates past the configured bound) when every slot is
        outstanding.
        """
        ...

    @property
    def capacity(self) -> int: ...

    @property
    def outstanding(self) -> int: ...


@runtime_checkable
class DeviceResidentBatcher(Protocol):
    """Group device-resident leases into fixed-size batches for in-process inference.

    A typed seam only: no NVDEC/NVENC overlay production default lives behind
    this port yet. Every leased element crosses batch formation without a
    host round-trip; batch formation itself commits no per-frame device
    synchronize.
    """

    def form_batch(self, leases: Sequence[FrameLease]) -> Sequence[FrameLease]: ...

    @property
    def max_batch_size(self) -> int: ...


__all__ = ["DeviceResidentBatcher", "DeviceResidentPool"]
