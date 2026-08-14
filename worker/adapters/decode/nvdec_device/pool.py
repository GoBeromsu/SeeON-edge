"""Bounded device-resident frame pool with lease/event ownership and backpressure.

Implements ``worker.interfaces.device_batch.DeviceResidentPool``. This module
never imports ``torch`` directly: slot allocation, H2D upload, and D2H
readback are all injected as ``StorageAllocator``/``TransferHooks`` callables
so the same pool logic is exercised by a real CUDA-backed allocator
(``worker.adapters.decode.nvdec_device.cuda_storage``, constructed only on a
capability-probed NVIDIA host) and by the deterministic
``worker.adapters.decode.nvdec_device.fake`` double used in every test that
runs on this repo's non-NVIDIA CI/dev hosts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, final

from worker.adapters.decode.nvdec_device.errors import DevicePoolExhaustedError
from worker.adapters.decode.nvdec_device.models import DeviceResidentPoolConfig
from worker.adapters.decode.nvdec_device.telemetry import DeviceResidencyTelemetry
from worker.types import FrameDescriptor, FrameLease, MemoryKind


class StorageAllocator(Protocol):
    """Allocate one bounded device-resident storage slot.

    Returns the opaque device handle plus its descriptor. Called at most
    ``config.capacity`` times per pool instance (issue: pool identity must
    stay 1:1 with its bounded slot count, never grow past it).
    """

    def __call__(self) -> tuple[object, FrameDescriptor]: ...


class SlotRecycler(Protocol):
    """Return a previously acquired device handle to the allocator's free list."""

    def __call__(self, handle: object) -> None: ...


@dataclass(frozen=True, slots=True)
class DeviceResidentPoolStatus:
    capacity: int
    outstanding: int
    high_watermark: int
    exhaustion_events: int


@final
class DeviceResidentFramePool:
    """Own a fixed number of pre-allocated device-resident storage slots.

    Backpressure, not queuing: once every slot is outstanding, ``acquire``
    raises ``DevicePoolExhaustedError`` immediately rather than blocking or
    growing the pool -- the same bounded-refusal contract as
    ``worker.pipeline.bus.BoundedFrameBus``'s full-subscription drop policy,
    adapted to a caller that needs to know synchronously (this pool has no
    silent-drop option: a caller that cannot get a slot must decide to skip a
    frame or apply real backpressure upstream, not lose accounting of it).
    """

    __slots__ = (
        "_allocate",
        "_config",
        "_free_slots",
        "_lock",
        "_outstanding",
        "_recycle",
        "_slot_descriptors",
        "_telemetry",
    )

    def __init__(
        self,
        config: DeviceResidentPoolConfig,
        *,
        allocate: StorageAllocator,
        recycle: SlotRecycler | None = None,
        telemetry: DeviceResidencyTelemetry | None = None,
    ) -> None:
        self._config = config
        self._allocate = allocate
        self._recycle = recycle
        self._telemetry = telemetry or DeviceResidencyTelemetry(pool_capacity=config.capacity)
        self._lock = threading.Lock()
        self._free_slots: list[object] = []
        # Every handle this pool has ever minted keeps its immutable
        # descriptor here for the lifetime of the pool -- slots are recycled
        # (returned to `_free_slots`), never re-described, so a handle's
        # descriptor never needs reconstruction after the fact.
        self._slot_descriptors: dict[int, FrameDescriptor] = {}
        self._outstanding = 0

    @property
    def capacity(self) -> int:
        return self._config.capacity

    @property
    def outstanding(self) -> int:
        with self._lock:
            return self._outstanding

    @property
    def telemetry(self) -> DeviceResidencyTelemetry:
        return self._telemetry

    def status(self) -> DeviceResidentPoolStatus:
        snapshot = self._telemetry.snapshot()
        return DeviceResidentPoolStatus(
            capacity=snapshot.pool_capacity,
            outstanding=snapshot.pool_outstanding,
            high_watermark=snapshot.pool_high_watermark,
            exhaustion_events=snapshot.pool_exhaustion_events,
        )

    def acquire(self) -> FrameLease:
        """Acquire one slot: reuse a recycled handle, mint a new one, or refuse.

        Invariant: ``len(self._slot_descriptors)`` is the total number of
        distinct handles this pool has ever minted (it never shrinks -- a
        recycled handle returns to ``_free_slots``, it is never freed early),
        so a fresh allocation is only ever attempted while that total is
        still below ``capacity``. Once ``capacity`` handles exist and none is
        free, every further request is refused -- the pool's total handle
        count can never exceed its declared bound.
        """
        with self._lock:
            if self._free_slots:
                handle = self._free_slots.pop()
                descriptor = self._slot_descriptors[id(handle)]
            elif len(self._slot_descriptors) < self._config.capacity:
                handle, descriptor = self._allocate()
                if descriptor.memory_kind is MemoryKind.HOST:
                    raise ValueError(
                        "device-resident pool allocator returned a host-memory descriptor"
                    )
                self._slot_descriptors[id(handle)] = descriptor
            else:
                self._telemetry.record_pool_exhausted()
                raise DevicePoolExhaustedError(self._config.capacity, self._outstanding)
            self._outstanding += 1
            outstanding = self._outstanding
        self._telemetry.record_acquire(outstanding)
        return FrameLease.from_device(
            handle,
            descriptor,
            on_recycle=self._on_recycle,
        )

    def _on_recycle(self, handle: object) -> None:
        with self._lock:
            self._outstanding -= 1
            outstanding = self._outstanding
            self._free_slots.append(handle)
        self._telemetry.record_release(outstanding)
        if self._recycle is not None:
            self._recycle(handle)


__all__ = [
    "DeviceResidentFramePool",
    "DeviceResidentPoolStatus",
    "SlotRecycler",
    "StorageAllocator",
]
