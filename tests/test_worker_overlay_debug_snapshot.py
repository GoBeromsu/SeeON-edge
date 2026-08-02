from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import numpy as np

from contracts.frame import Frame
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains import DOMAIN_REGISTRY
from worker.domains.bed_exit import BedExitConfig, BedExitDebugSnapshot, BedExitMonitor
from worker.pipeline.output.live_view import LatestFrameStore, LiveViewSubscriber
from worker.types import BusinessEvent, DecisionInput, FramePacket

BED = BoundingBox(0, 0, 100, 100, 0.9)
PERSON_IN_BED = BoundingBox(10, 10, 50, 50, 0.9)
PERSON_OUTSIDE_BED = BoundingBox(105, 10, 119, 50, 0.9)


class _CountingMonitor(BedExitMonitor):
    def __init__(self) -> None:
        super().__init__(
            config=BedExitConfig(
                camera_id="camera-1",
                facility_id="facility-1",
                hold_frames=1,
                grace_frames=0,
            ),
            clock=lambda: datetime(2026, 7, 31, 12, tzinfo=UTC),
        )
        self.update_count = 0

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        self.update_count += 1
        return super().update(input_value)

    def counters(self, person_id: int) -> tuple[int | None, int | None, int, int]:
        assignment = self._assignments[person_id]
        return (
            assignment.bed_id,
            assignment.candidate_bed_id,
            assignment.candidate_frames,
            assignment.grace_frames,
        )


@dataclass(frozen=True, slots=True)
class _TargetPass:
    monitor: _CountingMonitor
    packet: FramePacket
    observation: FrameObservation
    snapshot: BedExitDebugSnapshot
    events: tuple[BusinessEvent, ...]
    counters: tuple[int | None, int | None, int, int]


class _FailingRenderer:
    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...],
    ) -> bytes:
        del packet, observation, debug_snapshots
        raise cv2.error("render failed")


class _SpyRenderer:
    """Fails the test if ``encode_jpeg`` runs at all.

    Used to prove viewer gating (#48) skips rendering entirely with zero
    viewers, rather than merely discarding the result afterwards.
    """

    def __init__(self) -> None:
        self.calls = 0

    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...],
    ) -> bytes:
        del packet, observation, debug_snapshots
        self.calls += 1
        raise AssertionError("renderer must not run with zero viewers")


def _decision_input(box: BoundingBox, *, frame_index: int) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((box,), ()),
            regions=((BED,), ()),
            track_ids=(7,),
        ),
        frame_width=120,
        frame_height=120,
        live_track_ids=(7,),
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH, age_frames=0),
    )


def _run_target_pass() -> _TargetPass:
    monitor = _CountingMonitor()
    _ = monitor.update(_decision_input(PERSON_IN_BED, frame_index=0))
    monitor.update_count = 0
    target_input = _decision_input(PERSON_OUTSIDE_BED, frame_index=1)
    events = monitor.update(target_input)
    adapter = DOMAIN_REGISTRY["bed_exit"].debug_snapshot_adapter
    assert adapter is not None
    snapshot = adapter(monitor, target_input.frame_index)
    assert snapshot is not None
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    packet = FramePacket(
        camera_id="camera-1",
        frame=Frame(index=1, time_sec=1.0, image=image),
        pts=1.0,
        seq=41,
        width=120,
        height=120,
        decode_time_ms=0.5,
    )
    return _TargetPass(
        monitor=monitor,
        packet=packet,
        observation=target_input.observation,
        snapshot=snapshot,
        events=events,
        counters=monitor.counters(7),
    )


def test_one_detector_pass_feeds_event_and_live_debug_render() -> None:
    # Given: one target-frame detector update already produced an event and debug snapshot.
    target = _run_target_pass()
    store = LatestFrameStore()
    # Viewer gating (#48): publish() is a no-op with zero viewers, so this
    # regression check for "still works with viewers" needs one connected.
    store.mark_viewer_connected("camera-1")
    subscriber = LiveViewSubscriber(store)
    original_bytes = target.packet.frame.image.tobytes()

    # When: live output renders those frozen target-frame values.
    published = subscriber.publish(
        target.packet,
        target.observation,
        (target.snapshot,),
    )

    # Then: no second domain pass or mutation occurs and the stored JPEG matches the seq.
    latest = store.get_latest("camera-1")
    assert published is True
    assert target.monitor.update_count == 1
    assert len(target.events) == 1
    assert target.events[0].event_type == "bed-exit"
    assert target.monitor.counters(7) == target.counters
    assert target.packet.frame.image.tobytes() == original_bytes
    assert latest is not None
    assert latest.seq == target.packet.seq
    assert latest.frame_index == target.packet.frame.index
    assert latest.jpeg.startswith(b"\xff\xd8")


def test_live_render_error_preserves_event_domain_state_and_previous_frame() -> None:
    # Given: an admitted target-frame event, prior live JPEG, and a failing renderer.
    target = _run_target_pass()
    store = LatestFrameStore()
    store.publish_jpeg("camera-1", b"previous", seq=40, frame_index=0)
    previous = store.get_latest("camera-1")
    # Viewer gating (#48): publish() is a no-op with zero viewers, so this
    # regression check for "still renders (and fails) with viewers" needs one
    # connected.
    store.mark_viewer_connected("camera-1")
    subscriber = LiveViewSubscriber(store, renderer=_FailingRenderer())
    original_bytes = target.packet.frame.image.tobytes()

    # When: cosmetic rendering fails after the event decision.
    published = subscriber.publish(
        target.packet,
        target.observation,
        (target.snapshot,),
    )

    # Then: inference can continue with the event/state/source frame unchanged.
    assert published is False
    assert len(target.events) == 1
    assert target.monitor.update_count == 1
    assert target.monitor.counters(7) == target.counters
    assert target.packet.frame.image.tobytes() == original_bytes
    assert store.get_latest("camera-1") is previous


def test_publish_with_zero_viewers_skips_rendering_entirely() -> None:
    """Confirmed product decision (#48): no viewers means no ``encode_jpeg`` call."""
    target = _run_target_pass()
    store = LatestFrameStore()
    spy = _SpyRenderer()
    subscriber = LiveViewSubscriber(store, renderer=spy)

    published = subscriber.publish(
        target.packet,
        target.observation,
        (target.snapshot,),
    )

    assert published is False
    assert spy.calls == 0
    assert store.get_latest("camera-1") is None


def test_publish_with_pending_snapshot_demand_encodes_once_with_zero_viewers() -> None:
    """A snapshot request is a momentary viewer for exactly one ``publish()``."""
    target = _run_target_pass()
    store = LatestFrameStore()
    store.request_snapshot_refresh("camera-1")
    subscriber = LiveViewSubscriber(store)

    published = subscriber.publish(
        target.packet,
        target.observation,
        (target.snapshot,),
    )

    assert published is True
    assert store.get_latest("camera-1") is not None

    # The demand flag is one-shot: a second publish with still-zero viewers
    # is gated again.
    published_again = subscriber.publish(
        target.packet,
        target.observation,
        (target.snapshot,),
    )
    assert published_again is False
