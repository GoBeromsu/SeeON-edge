from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, final

import numpy as np

from contracts.frame import Frame


class MemoryKind(StrEnum):
    HOST = "host"
    CUDA_DEVICE = "cuda-device"
    VAAPI_SURFACE = "vaapi-surface"
    DMABUF = "dmabuf"
    MPS_DEVICE = "mps-device"


class PixelFormat(StrEnum):
    RGB24 = "rgb24"
    BGR24 = "bgr24"
    NV12 = "nv12"


@dataclass(frozen=True, slots=True)
class FrameDescriptor:
    width: int
    height: int
    memory_kind: MemoryKind
    pixel_format: PixelFormat
    plane_strides: tuple[int, ...]
    size_bytes: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not self.plane_strides or any(stride <= 0 for stride in self.plane_strides):
            raise ValueError("frame plane strides must be positive")
        if self.size_bytes <= 0:
            raise ValueError("frame size_bytes must be positive")


class CompletionEvent(Protocol):
    """Opaque accelerator event passed only to an injected stream/reclaimer seam."""


class ConsumerCompletion(Protocol):
    def wait_for(self, event: CompletionEvent) -> None: ...

    def record_completion(self) -> CompletionEvent: ...


class CompletionReclaimer(Protocol):
    def defer(
        self,
        completions: tuple[CompletionEvent, ...],
        recycle: Callable[[], None],
    ) -> None: ...


class FrameLeaseReleasedError(RuntimeError):
    pass


@final
class _ImmediateReclaimer:
    def defer(
        self,
        completions: tuple[CompletionEvent, ...],
        recycle: Callable[[], None],
    ) -> None:
        if completions:
            raise RuntimeError("completion events require an injected asynchronous reclaimer")
        recycle()


@dataclass(frozen=True, slots=True)
class _HostStorage:
    frame: Frame
    descriptor: FrameDescriptor


@final
class _HostRecycleAdapter:
    """Narrow `_LeaseState`'s polymorphic `Frame | object` recycle payload back to `Frame`.

    `_LeaseState._recycle` only ever invokes this callback with
    `_HostStorage.frame` (never a device handle) for a lease built through
    `from_host`, so the narrowing here is safe by construction -- it exists
    purely to keep `from_host`'s public `on_recycle: Callable[[Frame], None]`
    signature unchanged for every existing caller while `_LeaseState` itself
    stays storage-kind-agnostic.
    """

    __slots__ = ("_callback",)

    def __init__(self, callback: Callable[[Frame], None]) -> None:
        self._callback = callback

    def __call__(self, payload: Frame | object) -> None:
        if not isinstance(payload, Frame):
            raise TypeError("host lease recycle received a non-host payload")
        self._callback(payload)


@dataclass(frozen=True, slots=True)
class _DeviceStorage:
    """Opaque non-host-resident storage handle plus its descriptor.

    ``handle`` is deliberately typed ``object``: ``worker.types`` may import
    only the standard library and ``contracts`` (see ``worker/types/AGENTS.md``),
    so it can never name a concrete accelerator tensor/array type. The owning
    media-plane adapter is the only code
    that ever casts ``handle`` back to its real type; every other consumer of
    a device-resident ``FrameLease`` treats it as opaque and forwards it
    through named, capability-validated converters only.
    """

    handle: object
    descriptor: FrameDescriptor


class _LeaseState:
    __slots__ = (
        "completion_reclaimer",
        "consumer_completions",
        "count",
        "lock",
        "on_recycle",
        "producer_completion",
        "recycle_scheduled",
        "recycled",
        "storage",
    )

    def __init__(
        self,
        storage: _HostStorage | _DeviceStorage,
        *,
        producer_completion: CompletionEvent | None,
        completion_reclaimer: CompletionReclaimer,
        on_recycle: Callable[[Frame | object], None] | None,
    ) -> None:
        self.storage = storage
        self.producer_completion = producer_completion
        self.completion_reclaimer = completion_reclaimer
        self.on_recycle = on_recycle
        self.consumer_completions: list[CompletionEvent] = []
        self.count = 1
        self.lock = threading.Lock()
        self.recycle_scheduled = False
        self.recycled = False

    def reserve(self, count: int) -> None:
        with self.lock:
            if self.recycled or self.recycle_scheduled or self.count <= 0:
                raise FrameLeaseReleasedError("frame lease has already been recycled")
            self.count += count

    def release(self, completion: CompletionEvent | None) -> None:
        completions: tuple[CompletionEvent, ...] | None = None
        with self.lock:
            if completion is not None:
                self.consumer_completions.append(completion)
            self.count -= 1
            if self.count < 0:
                raise FrameLeaseReleasedError("frame lease reference count became negative")
            if self.count == 0:
                self.recycle_scheduled = True
                completions = tuple(self.consumer_completions)
        if completions is not None:
            producer = self.producer_completion
            pending = completions if producer is None else (producer, *completions)
            self.completion_reclaimer.defer(pending, self._recycle)

    def _recycle(self) -> None:
        callback: Callable[[Frame | object], None] | None
        payload: Frame | object
        with self.lock:
            if self.recycled:
                raise FrameLeaseReleasedError("frame storage was recycled more than once")
            self.recycled = True
            callback = self.on_recycle
            storage = self.storage
            payload = storage.frame if isinstance(storage, _HostStorage) else storage.handle
        if callback is not None:
            callback(payload)


@final
class FrameLease:
    """One independently releasable handle over reference-counted frame storage."""

    __slots__ = ("_consumer", "_released", "_state", "_status_lock")

    def __init__(
        self,
        state: _LeaseState,
        *,
        consumer: ConsumerCompletion | None = None,
    ) -> None:
        self._state = state
        self._consumer = consumer
        self._released = False
        self._status_lock = threading.Lock()

    @classmethod
    def from_host(
        cls,
        frame: Frame,
        *,
        pixel_format: PixelFormat = PixelFormat.RGB24,
        producer_completion: CompletionEvent | None = None,
        completion_reclaimer: CompletionReclaimer | None = None,
        on_recycle: Callable[[Frame], None] | None = None,
    ) -> FrameLease:
        image = frame.image
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("host frames must be HxWx3 uint8 arrays")
        height, width, _channels = image.shape
        descriptor = FrameDescriptor(
            width=int(width),
            height=int(height),
            memory_kind=MemoryKind.HOST,
            pixel_format=pixel_format,
            plane_strides=(int(image.strides[0]),),
            size_bytes=int(image.nbytes),
        )
        wrapped_on_recycle = None if on_recycle is None else _HostRecycleAdapter(on_recycle)
        state = _LeaseState(
            _HostStorage(frame, descriptor),
            producer_completion=producer_completion,
            completion_reclaimer=completion_reclaimer or _ImmediateReclaimer(),
            on_recycle=wrapped_on_recycle,
        )
        return cls(state)

    @classmethod
    def from_device(
        cls,
        handle: object,
        descriptor: FrameDescriptor,
        *,
        producer_completion: CompletionEvent | None = None,
        completion_reclaimer: CompletionReclaimer | None = None,
        on_recycle: Callable[[object], None] | None = None,
    ) -> FrameLease:
        """Wrap an already-allocated non-host storage handle as a fresh lease.

        Unlike ``from_host``, this never allocates or validates pixel data --
        the owning adapter (e.g. a bounded device-resident pool) is
        responsible for constructing ``handle`` and its matching
        ``descriptor`` up front, and for reclaiming the handle in
        ``on_recycle`` (typically returning it to that same bounded pool
        rather than freeing device memory per frame).
        """
        if descriptor.memory_kind is MemoryKind.HOST:
            raise ValueError("from_device requires a non-host memory kind")
        state = _LeaseState(
            _DeviceStorage(handle, descriptor),
            producer_completion=producer_completion,
            completion_reclaimer=completion_reclaimer or _ImmediateReclaimer(),
            on_recycle=on_recycle,
        )
        return cls(state)

    @property
    def descriptor(self) -> FrameDescriptor:
        self._ensure_accessible()
        return self._state.storage.descriptor

    @property
    def host_frame(self) -> Frame:
        self._ensure_accessible()
        if self._state.storage.descriptor.memory_kind is not MemoryKind.HOST:
            raise RuntimeError("frame is not host-resident; use a named materializer")
        storage = self._state.storage
        assert isinstance(storage, _HostStorage)  # noqa: S101 - memory_kind gate above
        return storage.frame

    @property
    def device_handle(self) -> object:
        self._ensure_accessible()
        if self._state.storage.descriptor.memory_kind is MemoryKind.HOST:
            raise RuntimeError("frame is host-resident; use host_frame")
        storage = self._state.storage
        assert isinstance(storage, _DeviceStorage)  # noqa: S101 - memory_kind gate above
        return storage.handle

    @property
    def released(self) -> bool:
        with self._status_lock:
            return self._released

    @property
    def ref_count(self) -> int:
        with self._state.lock:
            return self._state.count

    @property
    def recycled(self) -> bool:
        with self._state.lock:
            return self._state.recycled

    def _ensure_accessible(self) -> None:
        with self._status_lock:
            if self._released:
                raise FrameLeaseReleasedError("frame lease handle was already released")
        with self._state.lock:
            if self._state.recycled:
                raise FrameLeaseReleasedError("frame lease storage was already recycled")

    def set_recycle_callback(self, callback: Callable[[Frame | object], None]) -> None:
        with self._state.lock:
            if self._state.on_recycle is not None:
                raise ValueError("frame lease recycle callback is already configured")
            if self._state.recycle_scheduled or self._state.recycled:
                raise FrameLeaseReleasedError("cannot configure a recycled frame lease")
            self._state.on_recycle = callback

    def retain(self, *, consumer: ConsumerCompletion | None = None) -> FrameLease:
        with self._status_lock:
            if self._released:
                raise FrameLeaseReleasedError("frame lease handle was already released")
            self._state.reserve(1)
        if consumer is not None and self._state.producer_completion is not None:
            consumer.wait_for(self._state.producer_completion)
        return FrameLease(self._state, consumer=consumer)

    def precharge(self, count: int) -> LeaseBatch:
        if count < 0:
            raise ValueError("precharge count must not be negative")
        with self._status_lock:
            if self._released:
                raise FrameLeaseReleasedError("frame lease handle was already released")
            self._state.reserve(count)
        return LeaseBatch(self._state, count)

    def release(self) -> None:
        with self._status_lock:
            if self._released:
                raise FrameLeaseReleasedError("frame lease handle was already released")
            completion = None if self._consumer is None else self._consumer.record_completion()
            self._released = True
        self._state.release(completion)


@final
class LeaseBatch:
    """Atomically reserved fan-out references, sealed after dispatch."""

    __slots__ = ("_lock", "_remaining", "_sealed", "_state")

    def __init__(self, state: _LeaseState, count: int) -> None:
        self._state = state
        self._remaining = count
        self._sealed = False
        self._lock = threading.Lock()

    def take(self) -> FrameLease:
        with self._lock:
            if self._sealed:
                raise FrameLeaseReleasedError("precharged lease batch is sealed")
            if self._remaining == 0:
                raise FrameLeaseReleasedError("precharged lease batch is exhausted")
            self._remaining -= 1
        return FrameLease(self._state)

    def seal(self) -> None:
        with self._lock:
            if self._sealed:
                return
            self._sealed = True
            remaining = self._remaining
            self._remaining = 0
        for _index in range(remaining):
            self._state.release(None)


__all__ = [
    "CompletionEvent",
    "CompletionReclaimer",
    "ConsumerCompletion",
    "FrameDescriptor",
    "FrameLease",
    "FrameLeaseReleasedError",
    "LeaseBatch",
    "MemoryKind",
    "PixelFormat",
]
