"""Live preview decoupled from inference: ``bus.live`` has a real consumer.

Before this lane, ``bus.live`` was published to at ingest and taken by nobody
(``bus.metrics("live").taken == 0``); the operator preview was published from
inside ``CameraPipelinePump`` *after* ``analytics.process``, so every preview
frame sat behind a pose forward. These tests pin the decoupled shape:

* the live pump drains ``bus.live`` on its own thread and keeps delivering
  while the inference path is blocked (2s injected pose delay);
* the overlay comes from the *cached* observation, tagged with its age, and a
  cached observation older than the staleness threshold is never drawn as if
  it were current;
* zero viewers means zero encode work (product decision #48);
* killing the live pump leaves inference/evidence untouched.
"""

from __future__ import annotations

import threading
import time
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.observation import BoundingBox, DetectionLabel, FrameObservation
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.output.live_view import LatestFrameStore, LiveViewSubscriber
from worker.pipeline.output.live_view_pump import (
    DEFAULT_STALE_AFTER_SEC,
    LatestObservationStore,
    LiveViewPump,
)
from worker.types import FramePacket


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((4, 4, 3), seq % 251, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 4, 4, 0.25)


def _observation() -> FrameObservation:
    return FrameObservation(
        detections=(
            (BoundingBox(0, 0, 2, 2, 0.9),),
            (DetectionLabel("NORMAL", 0.9, False),),
        ),
        poses=(((1, 1, 0.9),),),
    )


@final
class _FakeClock:
    """Deterministic monotonic clock; no test here waits on wall time."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@final
class _SpyPublisher:
    """Records what the pump handed the live view, without encoding anything."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, FrameObservation, float | None]] = []
        self.stale_flags: list[bool] = []

    def publish(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[object, ...] = (),
        *,
        observation_age_sec: float | None = None,
        overlay_stale: bool = False,
    ) -> bool:
        del debug_snapshots
        self.calls.append((packet.frame.index, observation, observation_age_sec))
        self.stale_flags.append(overlay_stale)
        return True


@final
class _SpyRenderer:
    """``LiveViewRenderer`` spy: counts every encode the subscriber attempts."""

    def __init__(self) -> None:
        self.encode_count = 0

    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[object, ...] = (),
    ) -> bytes:
        del packet, observation, debug_snapshots
        self.encode_count += 1
        return b"\xff\xd8jpeg\xff\xd9"


def _drain(pump: LiveViewPump, *, cycles: int = 1) -> None:
    for _ in range(cycles):
        pump.run_once()


def test_live_pump_consumes_the_live_lane_without_any_inference_cycle() -> None:
    """``bus.metrics("live").taken`` must move with zero inference cycles."""
    bus = BoundedFrameBus()
    publisher = _SpyPublisher()
    pump = LiveViewPump("camera-a", bus.live, publisher, LatestObservationStore())

    for seq in (1, 2, 3):
        bus.publish(_packet("camera-a", seq))
        pump.run_once()

    assert bus.metrics("live").taken == 3
    # The inference lane was never drained: preview did not ride on it.
    assert bus.metrics("inference").taken == 0
    assert [index for index, _observation, _age in publisher.calls] == [1, 2, 3]


def test_preview_keeps_flowing_while_the_pose_path_is_blocked_for_two_seconds() -> None:
    """A 2s pose stall must not cost the preview a single frame.

    The "inference" side here is a thread holding a lock for 2 seconds, the
    same way a slow forward holds the coordinator. The live pump runs on its
    own thread against ``bus.live`` and must deliver every frame published
    during the stall; the overlay reports the cached observation's age.
    """
    bus = BoundedFrameBus()
    clock = _FakeClock()
    observations = LatestObservationStore(clock=clock)
    observations.record("camera-a", _observation(), (), frame_index=0)
    publisher = _SpyPublisher()
    pump = LiveViewPump(
        "camera-a",
        bus.live,
        publisher,
        observations,
        poll_timeout_sec=0.02,
        clock=clock,
        stale_after_sec=10.0,
    )

    pose_started = threading.Event()
    pose_finished = threading.Event()

    def _stalled_pose() -> None:
        pose_started.set()
        # A real 2-second forward. The live pump must not be behind it.
        time.sleep(2.0)
        pose_finished.set()

    inference = threading.Thread(target=_stalled_pose, daemon=True)
    live = threading.Thread(target=pump.run, name="live-view-pump", daemon=True)
    live.start()
    inference.start()
    assert pose_started.wait(timeout=1.0)

    delivered = threading.Event()
    try:
        for seq in (1, 2, 3, 4, 5):
            bus.publish(_packet("camera-a", seq))
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if len(publisher.calls) >= seq:
                    break
                delivered.wait(0.005)
            assert len(publisher.calls) >= seq, (
                "live pump stalled behind inference at frame "
                f"{seq}; delivered={len(publisher.calls)}"
            )
    finally:
        pump.stop()
        live.join(timeout=2.0)

    assert not pose_finished.is_set(), "the injected 2s pose delay ended too early"
    assert not live.is_alive()
    assert [index for index, _observation, _age in publisher.calls][:5] == [1, 2, 3, 4, 5]


def test_overlay_reports_the_cached_observation_age() -> None:
    bus = BoundedFrameBus()
    clock = _FakeClock()
    observations = LatestObservationStore(clock=clock)
    observations.record("camera-a", _observation(), (), frame_index=7)
    publisher = _SpyPublisher()
    pump = LiveViewPump(
        "camera-a", bus.live, publisher, observations, clock=clock, stale_after_sec=2.0
    )

    clock.advance(0.5)
    bus.publish(_packet("camera-a", 1))
    _drain(pump)

    _index, observation, age = publisher.calls[0]
    assert age == pytest.approx(0.5)
    assert observation.poses == _observation().poses


def test_a_stale_cached_observation_is_never_drawn_as_current() -> None:
    """Past the threshold the skeleton is dropped, not presented as fresh."""
    bus = BoundedFrameBus()
    clock = _FakeClock()
    observations = LatestObservationStore(clock=clock)
    observations.record("camera-a", _observation(), (), frame_index=7)
    publisher = _SpyPublisher()
    pump = LiveViewPump(
        "camera-a", bus.live, publisher, observations, clock=clock, stale_after_sec=1.0
    )

    clock.advance(1.5)
    bus.publish(_packet("camera-a", 9))
    _drain(pump)

    _index, observation, age = publisher.calls[0]
    assert age == pytest.approx(1.5)
    assert observation.poses == ()
    assert observation.boxes == ()
    assert pump.stale_overlay_count == 1
    assert publisher.stale_flags == [True]


def test_no_cached_observation_yet_publishes_a_bare_frame() -> None:
    bus = BoundedFrameBus()
    publisher = _SpyPublisher()
    pump = LiveViewPump("camera-a", bus.live, publisher, LatestObservationStore())

    bus.publish(_packet("camera-a", 1))
    _drain(pump)

    _index, observation, age = publisher.calls[0]
    assert age is None
    assert observation.poses == ()


def test_zero_viewers_means_zero_encode_calls() -> None:
    """Product decision #48: no viewers means no encoding at all."""
    bus = BoundedFrameBus()
    store = LatestFrameStore()
    store.register_camera("camera-a")
    renderer = _SpyRenderer()
    pump = LiveViewPump(
        "camera-a", bus.live, LiveViewSubscriber(store, renderer=renderer), LatestObservationStore()
    )

    for seq in (1, 2, 3):
        bus.publish(_packet("camera-a", seq))
        pump.run_once()

    assert renderer.encode_count == 0
    assert store.get_latest("camera-a") is None
    # Frames were still consumed from the lane: the pump does not back up.
    assert bus.metrics("live").taken == 3


def test_viewer_connect_and_disconnect_toggles_encoding_cleanly() -> None:
    bus = BoundedFrameBus()
    store = LatestFrameStore()
    store.register_camera("camera-a")
    renderer = _SpyRenderer()
    pump = LiveViewPump(
        "camera-a", bus.live, LiveViewSubscriber(store, renderer=renderer), LatestObservationStore()
    )

    seq = 0
    counts: list[int] = []
    for connected in (False, True, False, True, False):
        if connected:
            store.mark_viewer_connected("camera-a")
        seq += 1
        bus.publish(_packet("camera-a", seq))
        pump.run_once()
        counts.append(renderer.encode_count)
        if connected:
            store.mark_viewer_disconnected("camera-a")

    assert counts == [0, 1, 1, 2, 2]
    assert not store.has_viewers("camera-a")


def test_a_failing_publisher_never_stops_the_live_pump() -> None:
    @final
    class _Exploding:
        def publish(
            self,
            packet: FramePacket,
            observation: FrameObservation,
            debug_snapshots: tuple[object, ...] = (),
            *,
            observation_age_sec: float | None = None,
            overlay_stale: bool = False,
        ) -> bool:
            del packet, observation, debug_snapshots
            del observation_age_sec, overlay_stale
            raise RuntimeError("overlay renderer exploded")

    bus = BoundedFrameBus()
    pump = LiveViewPump("camera-a", bus.live, _Exploding(), LatestObservationStore())

    for seq in (1, 2):
        bus.publish(_packet("camera-a", seq))
        pump.run_once()

    assert pump.failure_count == 2
    assert bus.metrics("live").taken == 2


def test_live_pump_releases_every_packet_it_takes() -> None:
    bus = BoundedFrameBus()
    publisher = _SpyPublisher()
    pump = LiveViewPump("camera-a", bus.live, publisher, LatestObservationStore())

    bus.publish(_packet("camera-a", 1))
    live_packet = None

    @final
    class _CapturingPublisher:
        def publish(
            self,
            packet: FramePacket,
            observation: FrameObservation,
            debug_snapshots: tuple[object, ...] = (),
            *,
            observation_age_sec: float | None = None,
            overlay_stale: bool = False,
        ) -> bool:
            del observation, debug_snapshots, observation_age_sec, overlay_stale
            nonlocal live_packet
            live_packet = packet
            return True

    pump = LiveViewPump("camera-a", bus.live, _CapturingPublisher(), LatestObservationStore())
    pump.run_once()

    assert live_packet is not None
    assert live_packet.released


def test_default_staleness_threshold_is_sub_second_scale() -> None:
    """The threshold must be tight enough that a skeleton cannot lag visibly."""
    assert 0.2 <= DEFAULT_STALE_AFTER_SEC <= 2.0


def test_killing_the_live_pump_leaves_inference_and_evidence_flowing() -> None:
    """Isolation: preview is a tap, so losing it costs the pipeline nothing."""
    bus = BoundedFrameBus()
    publisher = _SpyPublisher()
    pump = LiveViewPump(
        "camera-a", bus.live, publisher, LatestObservationStore(), poll_timeout_sec=0.02
    )
    live = threading.Thread(target=pump.run, name="live-view-pump", daemon=True)
    live.start()

    bus.publish(_packet("camera-a", 1))
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not publisher.calls:
        time.sleep(0.005)
    assert publisher.calls

    pump.stop()
    live.join(timeout=2.0)
    assert not live.is_alive()

    for seq in (2, 3):
        bus.publish(_packet("camera-a", seq))
        inference_packet = bus.inference.take(timeout_sec=0)
        assert inference_packet is not None and inference_packet.frame.index == seq
        inference_packet.release()

    # ``evidence`` is a FIFO, so frame 1 is still queued ahead of 2 and 3.
    evidence_indexes: list[int] = []
    while (evidence_packet := bus.evidence.take(timeout_sec=0)) is not None:
        evidence_indexes.append(evidence_packet.frame.index)
        evidence_packet.release()

    assert evidence_indexes == [1, 2, 3]
    assert bus.metrics("inference").taken == 2
    assert bus.metrics("evidence").taken == 3
    assert len(publisher.calls) == 1


def test_observation_store_keeps_cameras_independent() -> None:
    clock = _FakeClock()
    observations = LatestObservationStore(clock=clock)
    observations.record("camera-a", _observation(), (), frame_index=1)
    clock.advance(5.0)
    observations.record("camera-b", FrameObservation(), (), frame_index=2)

    cached_a = observations.latest("camera-a")
    cached_b = observations.latest("camera-b")
    assert cached_a is not None and cached_b is not None
    assert cached_a.frame_index == 1
    assert cached_b.frame_index == 2
    assert clock.now - cached_a.recorded_at == pytest.approx(5.0)
    assert clock.now - cached_b.recorded_at == pytest.approx(0.0)
    assert observations.latest("camera-c") is None
