from __future__ import annotations

from collections.abc import Callable

import numpy as np

from worker.types import FrameDescriptor, FrameLease, MemoryKind, PixelFormat


class _Event:
    def __init__(self, name: str) -> None:
        self.name = name
        self.complete = False


class _ConsumerStream:
    def __init__(self, completion: _Event) -> None:
        self.completion = completion
        self.waited_for: list[_Event] = []
        self.recorded = 0

    def wait_for(self, event: _Event) -> None:
        self.waited_for.append(event)

    def record_completion(self) -> _Event:
        self.recorded += 1
        return self.completion


class _EventReclaimer:
    def __init__(self) -> None:
        self.pending: list[tuple[tuple[_Event, ...], Callable[[], None]]] = []

    def defer(
        self,
        completions: tuple[_Event, ...],
        recycle: Callable[[], None],
    ) -> None:
        self.pending.append((completions, recycle))

    def complete_ready(self) -> None:
        waiting = self.pending
        self.pending = []
        for events, recycle in waiting:
            if all(event.complete for event in events):
                recycle()
            else:
                self.pending.append((events, recycle))


def test_cuda_device_lease_balances_multi_stream_event_ownership_before_recycle() -> None:
    handle = np.zeros((3, 4, 3), dtype=np.uint8)
    descriptor = FrameDescriptor(
        width=4,
        height=3,
        memory_kind=MemoryKind.CUDA_DEVICE,
        pixel_format=PixelFormat.RGB24,
        plane_strides=(int(handle.strides[0]),),
        size_bytes=int(handle.nbytes),
    )
    produced = _Event("decode-produced")
    inference_done = _Event("inference-done")
    overlay_done = _Event("overlay-done")
    inference = _ConsumerStream(inference_done)
    overlay = _ConsumerStream(overlay_done)
    reclaimer = _EventReclaimer()
    recycled: list[object] = []
    owner = FrameLease.from_device(
        handle,
        descriptor,
        producer_completion=produced,
        completion_reclaimer=reclaimer,
        on_recycle=recycled.append,
    )

    inference_lease = owner.retain(consumer=inference)
    overlay_lease = owner.retain(consumer=overlay)
    assert inference.waited_for == [produced]
    assert overlay.waited_for == [produced]
    assert owner.ref_count == 3
    assert owner.device_handle is handle

    owner.release()
    inference_lease.release()
    overlay_lease.release()
    assert inference.recorded == overlay.recorded == 1
    assert recycled == []
    assert len(reclaimer.pending) == 1
    assert reclaimer.pending[0][0] == (produced, inference_done, overlay_done)

    produced.complete = True
    inference_done.complete = True
    reclaimer.complete_ready()
    assert recycled == []

    overlay_done.complete = True
    reclaimer.complete_ready()
    assert recycled == [handle]
    assert owner.ref_count == 0
    assert owner.recycled is True
    assert reclaimer.pending == []
