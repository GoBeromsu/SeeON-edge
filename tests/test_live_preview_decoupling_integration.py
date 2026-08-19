"""Two cameras, a real viewer, a stalled pose lane: preview must keep moving.

The unit tests in ``tests/test_live_view_pump.py`` pin the live pump against
spies. This file assembles the real objects the worker composes -- two
``BoundedFrameBus`` instances, the real ``CapabilityInferenceCoordinator``
over a deliberately slow batched client, the real ``CameraPipelinePump``
(which now only *caches* its observation), the real ``LiveViewPump`` +
``LiveViewSubscriber`` + ``LatestFrameStore`` behind viewer gating -- and
proves the two lanes are actually independent: JPEGs keep landing in the store
for a connected viewer while every pose forward is blocked.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import final

import numpy as np

from contracts.frame import Frame
from contracts.runner import RunnerResult, pose_result
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import (
    CapabilityInferenceCoordinator,
    InferenceResultSlot,
)
from worker.pipeline.output.live_view import LatestFrameStore, LiveViewSubscriber
from worker.pipeline.output.live_view_pump import LatestObservationStore, LiveViewPump
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.types import BusinessEvent, DecisionInput, FramePacket

CAMERAS = ("camera-a", "camera-b")


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((8, 8, 3), seq % 251, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 8, 8, 0.25)


@final
class _StalledBatchClient:
    """Every batched forward blocks until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()

    def create(self, task: str) -> object:  # pragma: no cover - protocol filler
        raise AssertionError(f"single-frame path must not be used: {task}")

    def infer_batch(
        self, task: str, frames: Sequence[FramePacket], **_kwargs: object
    ) -> tuple[RunnerResult, ...]:
        assert task == "pose"
        self.entered.set()
        assert self.release.wait(timeout=10.0)
        return tuple(pose_result((), ()) for _frame in frames)


@final
class _NullWatchdog:
    def guard(self, **_kwargs: object) -> _NullWatchdog:
        return self

    def __enter__(self) -> int:
        return 0

    def __exit__(self, *_exc: object) -> bool:
        return False


@final
class _NoDecider:
    def update(self, _input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        return ()


@final
class _NullSink:
    def emit(self, _event: BusinessEvent) -> None:
        return None


def _analytics(camera_id: str) -> CompositeExtractor:
    return CompositeExtractor(
        extractors=(),
        scheduler=Scheduler(task_intervals={}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id=camera_id),
    )


def _wait_for(predicate: object, *, timeout_sec: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.005)
    return bool(predicate())  # type: ignore[operator]


def test_two_cameras_keep_previewing_while_every_pose_forward_is_blocked() -> None:
    client = _StalledBatchClient()
    coordinator = CapabilityInferenceCoordinator(
        client,  # type: ignore[arg-type]
        _NullWatchdog(),  # type: ignore[arg-type]
    )
    observations = LatestObservationStore()
    frames = LatestFrameStore()
    subscriber = LiveViewSubscriber(frames)

    buses: dict[str, BoundedFrameBus] = {}
    live_pumps: list[LiveViewPump] = []
    pipeline_pumps: list[CameraPipelinePump] = []
    for camera_id in CAMERAS:
        bus = BoundedFrameBus()
        buses[camera_id] = bus
        frames.register_camera(camera_id)
        # A real viewer is attached to both cameras: encoding is expected here.
        frames.mark_viewer_connected(camera_id)
        results = InferenceResultSlot()
        coordinator.register(camera_id, bus.inference, results)
        pipeline_pumps.append(
            CameraPipelinePump(
                camera_id,
                results,
                _analytics(camera_id),
                EventAggregator(deciders=(_NoDecider(),), incidents=IncidentManager()),
                _NullSink(),
                poll_timeout_sec=0.02,
                observation_recorder=observations,
            )
        )
        live_pumps.append(
            LiveViewPump(
                camera_id, bus.live, subscriber, observations, poll_timeout_sec=0.02
            )
        )

    threads = [
        threading.Thread(target=runnable.run, daemon=True, name=name)
        for runnable, name in (
            [(coordinator, "coordinator")]
            + [(pump, f"pipeline-{pump.camera_id}") for pump in pipeline_pumps]
            + [(pump, f"live-{pump.camera_id}") for pump in live_pumps]
        )
    ]
    for thread in threads:
        thread.start()

    try:
        # Frame 1 enters both lanes; the coordinator's forward blocks there.
        for camera_id in CAMERAS:
            buses[camera_id].publish(_packet(camera_id, 1))
        assert client.entered.wait(timeout=5.0)

        # With pose stuck, keep feeding frames: preview must keep encoding.
        for seq in (2, 3, 4):
            for camera_id in CAMERAS:
                buses[camera_id].publish(_packet(camera_id, seq))
            assert _wait_for(
                lambda seq=seq: all(
                    (latest := frames.get_latest(camera_id)) is not None
                    and latest.frame_index >= seq
                    for camera_id in CAMERAS
                )
            ), f"preview stalled behind inference at frame {seq}"

        for camera_id in CAMERAS:
            latest = frames.get_latest(camera_id)
            assert latest is not None
            assert latest.jpeg.startswith(b"\xff\xd8")
            # No observation has been cached yet (pose never returned), so the
            # frame is published bare rather than with someone else's skeleton.
            assert latest.observation_age_sec is None
            assert observations.latest(camera_id) is None
            assert buses[camera_id].metrics("live").taken >= 4

        # Release inference: observations start landing, overlays go fresh.
        client.release.set()
        assert _wait_for(
            lambda: all(observations.latest(camera_id) is not None for camera_id in CAMERAS)
        )
        for camera_id in CAMERAS:
            buses[camera_id].publish(_packet(camera_id, 5))
        assert _wait_for(
            lambda: all(
                (latest := frames.get_latest(camera_id)) is not None
                and latest.observation_age_sec is not None
                and latest.overlay_stale is False
                for camera_id in CAMERAS
            )
        )
    finally:
        client.release.set()
        coordinator.stop()
        for pipeline_pump in pipeline_pumps:
            pipeline_pump.stop()
        for live_pump in live_pumps:
            live_pump.stop()
        for thread in threads:
            thread.join(timeout=5.0)

    assert not any(thread.is_alive() for thread in threads)
