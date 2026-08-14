from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from worker.pipeline.analytics import CompositeExtractor, NamedExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.types import FramePacket


class _Runner:
    def __init__(self, result: RunnerResult) -> None:
        self._result: RunnerResult = result
        self.calls: int = 0
        self.last_image: Image | None = None

    def run(self, image: Image) -> RunnerResult:
        self.calls += 1
        self.last_image = image
        return self._result


def _extractor(module_name: str, runner: _Runner) -> NamedExtractor:
    return NamedExtractor(
        module_name=module_name,
        runner=runner,
        _call=runner.run,
        _clock=lambda: 0.0,
    )


def _packet() -> FramePacket:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))
    return FramePacket(
        camera_id="camera-a",
        frame=Frame(index=0, time_sec=0.0, image=image),
        pts=0.0,
        seq=0,
        width=8,
        height=6,
        decode_time_ms=0.25,
    )


def _pose_result() -> RunnerResult:
    keypoints = tuple((index, index + 1, 0.9) for index in range(17))
    flattened = tuple(value for point in keypoints for value in point)
    return pose_result((flattened,), ((1, 1, 4, 5, 0.8),))


def _composite(
    extractors: Sequence[NamedExtractor],
    camera_id: str,
) -> CompositeExtractor:
    return CompositeExtractor(
        extractors=extractors,
        scheduler=Scheduler({"pose": 1, "person": 1, "bed": 30}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id),
    )


def test_raw_frame_fanout_keeps_pixels_with_model_evidence_and_live_consumers() -> None:
    pose = _Runner(_pose_result())
    composite = _composite((_extractor("pose", pose),), "camera-a")
    bus = BoundedFrameBus()
    packet = _packet()
    source_frame = packet.frame
    source_frame_key = packet.frame_key
    source_lease = packet.lease

    bus.publish(packet)
    model_packet = bus.inference.take(timeout_sec=0)
    evidence_packet = bus.evidence.take(timeout_sec=0)
    live_packet = bus.live.take(timeout_sec=0)
    assert model_packet is not None
    assert evidence_packet is not None
    assert live_packet is not None
    assert model_packet is not packet
    assert evidence_packet is not packet
    assert live_packet is not packet
    assert model_packet is not evidence_packet is not live_packet
    assert model_packet.frame_key == evidence_packet.frame_key == live_packet.frame_key
    assert model_packet.frame_key == source_frame_key
    assert model_packet.lease is not source_lease
    assert evidence_packet.lease is not source_lease
    assert live_packet.lease is not source_lease
    assert model_packet.lease is not evidence_packet.lease
    assert evidence_packet.lease is not live_packet.lease
    assert model_packet.frame is evidence_packet.frame is live_packet.frame is source_frame

    result = composite.process(model_packet)

    assert pose.last_image is source_frame.image
    assert evidence_packet.frame.image is source_frame.image
    assert live_packet.frame.image is source_frame.image
    assert not hasattr(result.decision_input, "frame")
    assert not hasattr(result.decision_input, "image")
