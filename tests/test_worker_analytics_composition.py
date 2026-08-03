from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from contracts.frame import Frame
from contracts.runner import (
    Image,
    RunnerProtocol,
    RunnerResult,
    bed_result,
    person_result,
    pose_result,
)
from worker.pipeline.analytics import CompositeExtractor, NamedExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.model_composition import compose_yolo_extractors
from worker.types import FramePacket

ServingOption = str | int | float | bool | None


class _Runner:
    def __init__(self, result: RunnerResult) -> None:
        self._result: RunnerResult = result
        self.calls: int = 0
        self.last_image: Image | None = None

    def run(self, image: Image) -> RunnerResult:
        self.calls += 1
        self.last_image = image
        return self._result


class _ServingClient:
    def __init__(self, runners: Mapping[str, RunnerProtocol]) -> None:
        self._runners: Mapping[str, RunnerProtocol] = runners
        self.create_calls: list[tuple[str, dict[str, ServingOption]]] = []

    def create(self, task: str, **kwargs: ServingOption) -> RunnerProtocol:
        self.create_calls.append((task, dict(kwargs)))
        return self._runners[task]


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


def test_production_composition_provisions_yolo_tasks_once_through_serving_seam() -> None:
    runners = {
        "pose": _Runner(_pose_result()),
        "person": _Runner(person_result(((1, 1, 4, 5, 0.9),))),
        "bed": _Runner(bed_result(())),
    }
    client = _ServingClient(runners)

    # box_source="person" (issue #44): exercises all three production tasks
    # provisioned once and shared across cameras, including person.
    shared = compose_yolo_extractors(client, device="cuda:0", box_source="person")
    camera_a = _composite(shared.extractors, "camera-a")
    camera_b = _composite(shared.extractors, "camera-b")

    assert tuple(extractor.module_name for extractor in shared.extractors) == (
        "pose",
        "person",
        "bed",
    )
    assert client.create_calls == [
        ("pose", {"device": "cuda:0"}),
        ("person", {"device": "cuda:0"}),
        ("bed", {"device": "cuda:0"}),
    ]
    assert camera_a.extractors == camera_b.extractors == shared.extractors
    assert all(
        first.runner is second.runner
        for first, second in zip(camera_a.extractors, camera_b.extractors, strict=True)
    )
    assert camera_a.tracker is not camera_b.tracker
    assert camera_a.scene_state is not camera_b.scene_state


def test_raw_frame_fanout_keeps_pixels_with_model_evidence_and_live_consumers() -> None:
    pose = _Runner(_pose_result())
    client = _ServingClient({
        "pose": pose,
        "person": _Runner(person_result(())),
        "bed": _Runner(bed_result(())),
    })
    shared = compose_yolo_extractors(client, device="cpu")
    composite = _composite(shared.extractors, "camera-a")
    bus = BoundedFrameBus()
    packet = _packet()

    bus.publish(packet)
    model_packet = bus.inference.take(timeout_sec=0)
    evidence_packet = bus.evidence.take(timeout_sec=0)
    live_packet = bus.live.take(timeout_sec=0)
    assert model_packet is not None
    assert evidence_packet is not None
    assert live_packet is not None
    assert model_packet is packet
    assert evidence_packet is packet
    assert live_packet is packet

    result = composite.process(model_packet)

    assert pose.last_image is packet.frame.image
    assert evidence_packet.frame.image is packet.frame.image
    assert live_packet.frame.image is packet.frame.image
    assert not hasattr(result.decision_input, "frame")
    assert not hasattr(result.decision_input, "image")
