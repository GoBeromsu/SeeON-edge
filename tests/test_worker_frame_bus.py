"""RED contract for the concrete per-camera bounded frame bus.

Assumed public API:

* ``BoundedFrameBus(evidence_capacity=128, clock=...)`` constructs one bus for
  one camera.
* ``inference``, ``live``, and ``evidence`` expose canonical subscriptions.
  Their capacities are 1, 1, and ``evidence_capacity`` respectively; the
  first two use latest-only replacement and evidence uses FIFO drop-newest.
* ``subscribe(name, capacity=..., latest_only=...)`` remains available for
  named subscriptions.
* ``metrics(name)`` returns an immutable snapshot with ``published``, ``taken``,
  ``dropped``, and ``queue_age_sec`` fields.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pytest

from contracts.frame import Frame
from worker.interfaces.bus import FrameBus
from worker.pipeline.bus import BoundedFrameBus
from worker.types import FrameLease, FramePacket


class _FakeEvent:
    def __init__(self) -> None:
        self.complete = False


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
        for events, recycle in tuple(self.pending):
            if all(event.complete for event in events):
                recycle()
                self.pending.remove((events, recycle))


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now: float = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_packet(seq: int, *, camera_id: str = "camera-a") -> FramePacket:
    image = np.full((2, 3, 3), seq, dtype=np.uint8)
    return FramePacket(
        camera_id=camera_id,
        frame=Frame(index=seq, time_sec=float(seq), image=image),
        pts=float(seq),
        seq=seq,
        width=3,
        height=2,
        decode_time_ms=1.5,
    )


def test_canonical_subscriptions_have_required_capacities_and_policies() -> None:
    # Given a bus with the default configured evidence capacity
    bus = BoundedFrameBus()
    assert isinstance(bus, FrameBus)

    # Then canonical subscriptions expose inference/latest-only/cap1,
    # live/latest-only/cap1, and evidence/FIFO/cap128.
    assert bus.inference.capacity == 1
    assert bus.inference.latest_only is True
    assert bus.live.capacity == 1
    assert bus.live.latest_only is True
    assert bus.evidence.capacity == 128
    assert bus.evidence.latest_only is False


def test_canonical_evidence_capacity_is_configurable() -> None:
    # Given a bus configured with an explicit evidence capacity
    bus = BoundedFrameBus(evidence_capacity=3)

    # Then only the evidence subscription adopts that configured capacity.
    assert bus.evidence.capacity == 3
    assert bus.inference.capacity == 1
    assert bus.live.capacity == 1


def test_latest_only_replaces_oldest_pending_packet_and_counts_one_drop() -> None:
    # Given an empty latest-only subscription
    bus = BoundedFrameBus()
    first = make_packet(1)
    second = make_packet(2)

    # When two packets are published before the consumer takes one
    bus.publish(first)
    bus.publish(second)

    # Then only the newest packet remains and exactly one packet was dropped.
    taken = bus.inference.take(timeout_sec=0)
    assert taken is not None and taken.seq == second.seq
    taken.release()
    assert bus.inference.take(timeout_sec=0) is None
    assert bus.metrics("inference").dropped == 1


def test_evidence_full_queue_drops_newest_and_preserves_fifo_order() -> None:
    # Given a FIFO evidence subscription with capacity two
    bus = BoundedFrameBus(evidence_capacity=2)
    first, second, newest = make_packet(1), make_packet(2), make_packet(3)

    # When a third packet arrives while the queue is full
    bus.publish(first)
    bus.publish(second)
    bus.publish(newest)

    # Then the newest packet is dropped and the existing packets remain FIFO.
    taken_first = bus.evidence.take(timeout_sec=0)
    taken_second = bus.evidence.take(timeout_sec=0)
    assert taken_first is not None and taken_first.seq == first.seq
    assert taken_second is not None and taken_second.seq == second.seq
    taken_first.release()
    taken_second.release()
    assert bus.evidence.take(timeout_sec=0) is None
    assert bus.metrics("evidence").dropped == 1


def test_metrics_report_exact_publish_take_and_drop_counts() -> None:
    # Given packets that exercise replacement, successful take, and empty take
    bus = BoundedFrameBus(evidence_capacity=1)
    bus.publish(make_packet(1))
    bus.publish(make_packet(2))
    assert bus.inference.take(timeout_sec=0) is not None
    assert bus.inference.take(timeout_sec=0) is None

    # Then metrics distinguish publishes, takes, and drops exactly.
    snapshot = bus.metrics("inference")
    assert (snapshot.published, snapshot.taken, snapshot.dropped) == (2, 1, 1)


def test_queue_age_uses_injected_clock_deterministically() -> None:
    # Given a fake clock and one queued packet
    clock = FakeClock(100.0)
    bus = BoundedFrameBus(clock=clock)
    bus.publish(make_packet(1))

    # When time advances without consuming the packet
    clock.advance(2.5)

    # Then the metric reports the deterministic age of the oldest queued item.
    assert bus.metrics("inference").queue_age_sec == 2.5

    # And consuming it removes the queue age.
    assert bus.inference.take(timeout_sec=0) is not None
    assert bus.metrics("inference").queue_age_sec == 0.0


def test_subscriber_counters_are_independent() -> None:
    # Given two named subscriptions on one bus
    bus = BoundedFrameBus()
    inference = bus.subscribe("custom-inference", capacity=1, latest_only=True)
    _ = bus.subscribe("custom-evidence", capacity=2)

    # When both receive the same publication but only one is consumed
    bus.publish(make_packet(1))
    assert inference.take(timeout_sec=0) is not None

    # Then each subscription reports only its own counters.
    assert (
        bus.metrics("custom-inference").published,
        bus.metrics("custom-inference").taken,
    ) == (1, 1)
    assert (
        bus.metrics("custom-evidence").published,
        bus.metrics("custom-evidence").taken,
    ) == (1, 0)


def test_independent_per_camera_buses_do_not_affect_each_other() -> None:
    # Given separate concrete buses representing cameras A and B
    camera_a = BoundedFrameBus(evidence_capacity=1)
    camera_b = BoundedFrameBus(evidence_capacity=1)

    # When camera A's slow evidence consumer causes a drop
    camera_a.publish(make_packet(1, camera_id="a"))
    camera_a.publish(make_packet(2, camera_id="a"))
    camera_b.publish(make_packet(10, camera_id="b"))

    # Then camera B remains able to receive its own packet unaffected.
    assert camera_b.evidence.take(timeout_sec=0) is not None
    assert camera_a.metrics("evidence").dropped == 1
    assert camera_b.metrics("evidence").dropped == 0


def test_subscribers_receive_independent_lease_handles_over_the_same_host_frame() -> None:
    # Given canonical subscribers and one host-backed packet
    bus = BoundedFrameBus()
    packet = make_packet(7)
    source_frame = packet.frame

    # When the packet is published once
    bus.publish(packet)
    inference = bus.inference.take(timeout_sec=0)
    live = bus.live.take(timeout_sec=0)
    evidence = bus.evidence.take(timeout_sec=0)

    # Then each consumer owns a distinct releasable handle without copying the frame.
    assert inference is not None and live is not None and evidence is not None
    assert inference is not live and live is not evidence
    assert inference.frame is live.frame is evidence.frame is source_frame
    assert inference.lease is not live.lease
    inference.release()
    live.release()
    evidence.release()


def test_mutating_a_consumer_image_copy_leaves_source_and_shared_packet_unchanged() -> None:
    # Given one packet shared with two consumers
    bus = BoundedFrameBus()
    packet = make_packet(4)
    source_frame = packet.frame
    original = source_frame.image.copy()
    bus.publish(packet)

    # When a consumer copies the image before mutating its working buffer
    inference_packet = bus.inference.take(timeout_sec=0)
    assert inference_packet is not None
    consumer_image = inference_packet.frame.image.copy()
    consumer_image[0, 0, 0] = 255

    # Then the source frame and the other subscriber's packet remain unchanged.
    assert np.array_equal(source_frame.image, original)
    live_packet = bus.live.take(timeout_sec=0)
    assert live_packet is not None
    assert np.array_equal(live_packet.frame.image, original)
    inference_packet.release()
    live_packet.release()


def test_empty_nonblocking_take_returns_none() -> None:
    # Given an empty subscription
    bus = BoundedFrameBus()

    # When the consumer requests a nonblocking take
    result = bus.inference.take(timeout_sec=0)

    # Then no packet is fabricated and the call returns immediately.
    assert result is None


def test_rejected_post_close_publish_waits_for_producer_completion() -> None:
    produced = _FakeEvent()
    reclaimer = _FakeReclaimer()
    recycled: list[Frame] = []
    frame = Frame(19, 19.0, np.zeros((2, 3, 3), dtype=np.uint8))
    owner = FrameLease.from_host(
        frame,
        producer_completion=produced,
        completion_reclaimer=reclaimer,
        on_recycle=recycled.append,
    )
    packet = FramePacket("camera-a", owner.host_frame, 19.0, 19, 3, 2, 1.5, lease=owner)
    bus = BoundedFrameBus()
    bus.close()

    bus.publish(packet)
    assert recycled == []
    produced.complete = True
    reclaimer.complete_ready()
    assert len(recycled) == 1


def test_publish_take_and_close_balance_every_lease_without_early_recycle() -> None:
    recycled: list[Frame] = []
    packet = make_packet(20)
    source_frame = packet.frame
    assert packet.lease is not None
    packet.lease.set_recycle_callback(recycled.append)
    bus = BoundedFrameBus(evidence_capacity=1)

    bus.publish(packet)
    inference = bus.inference.take(timeout_sec=0)
    assert inference is not None
    bus.close()
    assert recycled == []

    inference.release()
    assert recycled == [source_frame]


def test_drop_oldest_drop_newest_and_post_close_publish_release_exactly_once() -> None:
    recycled: list[int] = []

    def tracked(seq: int) -> FramePacket:
        packet = make_packet(seq)
        assert packet.lease is not None
        packet.lease.set_recycle_callback(lambda frame: recycled.append(frame.index))
        return packet

    bus = BoundedFrameBus(evidence_capacity=1)
    bus.publish(tracked(1))
    bus.publish(tracked(2))
    bus.close()
    bus.publish(tracked(3))

    assert sorted(recycled) == [1, 2, 3]
    assert len(recycled) == 3


def test_publish_exception_releases_all_precharged_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    recycled: list[Frame] = []
    packet = make_packet(30)
    source_frame = packet.frame
    assert packet.lease is not None
    packet.lease.set_recycle_callback(recycled.append)
    bus = BoundedFrameBus()

    def raising_publish(_packet: FramePacket) -> None:
        raise RuntimeError("subscriber failed")

    monkeypatch.setattr(bus.live, "publish", raising_publish)

    with pytest.raises(RuntimeError, match="subscriber failed"):
        bus.publish(packet)
    bus.close()

    assert recycled == [source_frame]


def test_concurrent_publishers_start_at_a_barrier_and_close_balances_all_handles() -> None:
    bus = BoundedFrameBus(evidence_capacity=1)
    start = threading.Barrier(3)
    recycled: list[int] = []
    recycled_lock = threading.Lock()

    def publish(seq: int) -> None:
        packet = make_packet(seq)
        assert packet.lease is not None

        def record(frame: Frame) -> None:
            with recycled_lock:
                recycled.append(frame.index)

        packet.lease.set_recycle_callback(record)
        start.wait(timeout=1.0)
        bus.publish(packet)

    threads = tuple(threading.Thread(target=publish, args=(seq,)) for seq in (40, 41))
    for thread in threads:
        thread.start()
    start.wait(timeout=1.0)
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    bus.close()

    assert sorted(recycled) == [40, 41]


def test_metrics_snapshot_is_immutable() -> None:
    # Given a metrics snapshot
    bus = BoundedFrameBus()
    snapshot = bus.metrics("inference")

    # Then the snapshot is a frozen dataclass rather than mutable live state.
    assert is_dataclass(snapshot)
    field_name = "published"
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        setattr(snapshot, field_name, 99)
