from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from contracts.frame import Frame
from worker.adapters.frame.host import HostFrameMaterializer
from worker.types import (
    CopyMetrics,
    FrameDescriptor,
    FrameLease,
    FrameLeaseReleasedError,
    FramePacket,
    MemoryKind,
    PixelFormat,
)


def _frame(value: int = 1) -> Frame:
    return Frame(
        index=value,
        time_sec=float(value),
        image=np.full((2, 3, 3), value, dtype=np.uint8),
    )


def test_closed_memory_and_pixel_contract_describes_host_rgb_frame() -> None:
    lease = FrameLease.from_host(_frame())

    assert lease.descriptor == FrameDescriptor(
        width=3,
        height=2,
        memory_kind=MemoryKind.HOST,
        pixel_format=PixelFormat.RGB24,
        plane_strides=(9,),
        size_bytes=18,
    )

    lease.release()


def test_retain_prevents_early_recycle_and_each_handle_releases_once() -> None:
    recycled: list[Frame] = []
    owner = FrameLease.from_host(_frame(), on_recycle=recycled.append)
    consumer = owner.retain()

    owner.release()
    assert recycled == []
    assert consumer.ref_count == 1

    consumer.release()
    assert len(recycled) == 1
    with pytest.raises(FrameLeaseReleasedError, match="already released"):
        consumer.release()
    assert len(recycled) == 1


def test_precharge_reserves_all_fanout_handles_before_dispatch_and_seals_unused() -> None:
    recycled: list[Frame] = []
    owner = FrameLease.from_host(_frame(), on_recycle=recycled.append)
    fanout = owner.precharge(3)

    first = fanout.take()
    second = fanout.take()
    owner.release()
    first.release()
    assert recycled == []

    fanout.seal()
    assert recycled == []
    second.release()
    assert len(recycled) == 1
    with pytest.raises(FrameLeaseReleasedError, match="sealed"):
        fanout.take()


class _FakeEvent:
    def __init__(self) -> None:
        self.complete = False


class _FakeConsumer:
    def __init__(self, completion: _FakeEvent) -> None:
        self.completion = completion
        self.waited_for: list[_FakeEvent] = []

    def wait_for(self, event: _FakeEvent) -> None:
        self.waited_for.append(event)

    def record_completion(self) -> _FakeEvent:
        return self.completion


class _FakeReclaimer:
    def __init__(self) -> None:
        self.pending: list[tuple[tuple[_FakeEvent, ...], Callable[[], None]]] = []

    def defer(
        self,
        completions: tuple[_FakeEvent, ...],
        recycle: Callable[[], None],
    ) -> None:
        self.pending.append((completions, recycle))

    def complete_ready(self) -> None:
        still_pending: list[tuple[tuple[_FakeEvent, ...], Callable[[], None]]] = []
        for events, recycle in self.pending:
            if all(event.complete for event in events):
                recycle()
            else:
                still_pending.append((events, recycle))
        self.pending = still_pending


def test_completion_events_order_consumers_and_defer_recycle_without_synchronize() -> None:
    produced = _FakeEvent()
    consumed = _FakeEvent()
    reclaimer = _FakeReclaimer()
    recycled: list[Frame] = []
    owner = FrameLease.from_host(
        _frame(),
        producer_completion=produced,
        completion_reclaimer=reclaimer,
        on_recycle=recycled.append,
    )
    consumer_context = _FakeConsumer(consumed)

    consumer = owner.retain(consumer=consumer_context)
    assert consumer_context.waited_for == [produced]

    owner.release()
    consumer.release()
    assert recycled == []
    assert reclaimer.pending[0][0] == (produced, consumed)

    consumed.complete = True
    reclaimer.complete_ready()
    assert recycled == []
    produced.complete = True
    reclaimer.complete_ready()
    assert len(recycled) == 1


def test_producer_completion_defers_zero_consumer_recycle() -> None:
    produced = _FakeEvent()
    reclaimer = _FakeReclaimer()
    recycled: list[Frame] = []
    owner = FrameLease.from_host(
        _frame(),
        producer_completion=produced,
        completion_reclaimer=reclaimer,
        on_recycle=recycled.append,
    )

    owner.release()
    assert recycled == []
    assert reclaimer.pending[0][0] == (produced,)

    produced.complete = True
    reclaimer.complete_ready()
    assert len(recycled) == 1


def test_released_handle_and_packet_reject_stale_host_access() -> None:
    packet = FramePacket("camera-a", _frame(), 1.0, 1, 3, 2, 0.1)
    stale_frame = packet.borrow_host_frame()

    packet.release()

    with pytest.raises(FrameLeaseReleasedError, match="released"):
        packet.borrow_host_frame()
    with pytest.raises(FrameLeaseReleasedError, match="released"):
        _ = packet.frame
    assert stale_frame.index == 1  # contracts.Frame stays host-only and unchanged.


def test_named_host_materializer_is_the_only_counted_full_frame_copy() -> None:
    source = FrameLease.from_host(_frame(7))
    metrics = CopyMetrics()
    materializer = HostFrameMaterializer(name="clip-thread-host-clone", metrics=metrics)

    view = materializer.view(source)
    clone = materializer.materialize(source)
    clone_frame = materializer.view(clone)

    assert view is source.host_frame
    assert clone_frame is not view
    assert np.array_equal(clone_frame.image, view.image)
    snapshot = metrics.snapshot()
    assert snapshot.materializations == 1
    assert snapshot.copied_frames == 1
    assert snapshot.copied_bytes == view.image.nbytes
    assert snapshot.by_adapter == (("clip-thread-host-clone", 1),)

    source.release()
    clone.release()
