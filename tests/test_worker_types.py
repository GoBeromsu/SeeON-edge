from __future__ import annotations

import ast
from dataclasses import MISSING, FrozenInstanceError, fields, replace
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest

from contracts.event import EventPayload
from contracts.frame import Frame
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    DetectionLabel,
    DetectionResult,
    FrameObservation,
)
from contracts.runner import DetectionRunnerResult, RunnerResult
from worker.types import (
    BusinessEvent,
    DecisionInput,
    FrameBedPoseFeatures,
    FrameKey,
    FrameLease,
    FramePacket,
    ModuleResult,
)

_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_FIELDS = {
    FramePacket: (
        "camera_id",
        "_frame",
        "pts",
        "seq",
        "width",
        "height",
        "decode_time_ms",
        "worker_boot_id",
        "stream_epoch",
        "source_pts",
        "source_dts",
        "source_time_base",
        "lease",
    ),
    ModuleResult: ("module_name", "result", "elapsed_ms", "output_adapter"),
    DecisionInput: (
        "observation",
        "frame_width",
        "frame_height",
        "live_track_ids",
        "time_sec",
        "frame_index",
        "bed_region",
        "bed_pose_features",
    ),
    BusinessEvent: (
        "domain",
        "event_type",
        "identity",
        "camera_id",
        "facility_id",
        "time_sec",
        "probability",
        "person_id",
        "bed_id",
        "audit",
        "snapshot_jpeg",
    ),
}


def _build_pipeline_values() -> tuple[FramePacket, ModuleResult, DecisionInput, BusinessEvent]:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))
    frame = Frame(index=4, time_sec=1.25, image=image)
    packet = FramePacket(
        camera_id="camera-7",
        frame=frame,
        pts=1.25,
        seq=4,
        width=8,
        height=6,
        decode_time_ms=2.5,
    )
    boxes = (
        BoundingBox(1, 1, 3, 5, 0.92),
        BoundingBox(4, 1, 7, 5, 0.88),
    )
    labels = (
        DetectionLabel("NORMAL", 0.92, False),
        DetectionLabel("FALL", 0.88, True),
    )
    keypoints = (
        tuple((index, index + 1, 0.9) for index in range(17)),
        tuple((index + 20, index + 21, 0.8) for index in range(17)),
    )
    track_ids = (31, 32)
    detection = DetectionResult(boxes=boxes, labels=labels, keypoints=keypoints)
    runner_result = DetectionRunnerResult(kind="detection", detections=detection)
    module_result = ModuleResult(
        module_name="mobility-v2",
        result=runner_result,
        elapsed_ms=4.5,
        output_adapter="pose",
    )
    normalized = FrameObservation.from_detection_result(runner_result.detections)
    observation = FrameObservation(
        detections=normalized.detections,
        poses=normalized.poses,
        regions=normalized.regions,
        track_ids=track_ids,
    )
    decision_input = DecisionInput(
        observation=observation,
        frame_width=packet.width,
        frame_height=packet.height,
        live_track_ids=track_ids,
        time_sec=packet.frame.time_sec,
        frame_index=packet.frame.index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )
    business_event = BusinessEvent(
        domain="bed_exit",
        event_type="bed-exit",
        identity=f"{track_ids[0]}:5",
        camera_id=packet.camera_id,
        facility_id="facility-a",
        time_sec=packet.frame.time_sec,
        probability=0.97,
        person_id=track_ids[0],
        bed_id=5,
    )
    return packet, module_result, decision_input, business_event


def test_worker_envelopes_have_exact_fields_and_authoritative_types() -> None:
    for envelope_type, expected_fields in _EXPECTED_FIELDS.items():
        assert tuple(item.name for item in fields(envelope_type)) == expected_fields
        assert tuple(envelope_type.__slots__) == expected_fields

    assert get_type_hints(FramePacket)["_frame"] is Frame
    assert get_type_hints(FramePacket)["lease"] is FrameLease
    assert get_type_hints(ModuleResult)["result"] == RunnerResult
    assert get_type_hints(ModuleResult)["output_adapter"] == str | None
    assert get_type_hints(DecisionInput)["observation"] is FrameObservation
    assert get_type_hints(DecisionInput)["bed_region"] is BedRegionDebugSnapshot
    assert get_type_hints(DecisionInput)["bed_pose_features"] is FrameBedPoseFeatures


def test_frame_key_includes_boot_camera_epoch_sequence_and_pts() -> None:
    packet = replace(
        _build_pipeline_values()[0],
        worker_boot_id="boot-7",
        stream_epoch=4,
    )

    assert packet.frame_key == FrameKey("boot-7", "camera-7", 4, 4, 1.25)
    assert packet.frame_key != replace(packet, stream_epoch=5).frame_key
    assert packet.frame_key != replace(packet, pts=1.5).frame_key


def test_worker_envelopes_support_value_equality_and_hashing() -> None:
    for value in _build_pipeline_values():
        equivalent = replace(value)

        assert equivalent == value
        assert hash(equivalent) == hash(value)


@pytest.mark.parametrize(
    ("position", "attribute", "replacement"),
    [
        (0, "camera_id", "camera-8"),
        (1, "module_name", "other-model"),
        (2, "frame_index", 99),
        (3, "probability", 0.1),
    ],
)
def test_worker_envelopes_reject_mutation(
    position: int,
    attribute: str,
    replacement: str | int | float,
) -> None:
    value = _build_pipeline_values()[position]

    with pytest.raises(FrozenInstanceError):
        setattr(value, attribute, replacement)


def test_worker_envelopes_have_no_mutable_default_aliases() -> None:
    mutable_default_types = (bytearray, dict, list, set)

    for envelope_type in _EXPECTED_FIELDS:
        for item in fields(envelope_type):
            assert not isinstance(item.default, mutable_default_types)
            assert item.default_factory is MISSING

    first_event = BusinessEvent("fall", "fall", 1, "camera-7", "facility-a", 1.0, 0.9)
    second_event = BusinessEvent("fall", "fall", 2, "camera-7", "facility-a", 2.0, 0.8)
    assert first_event.person_id is None
    assert second_event.person_id is None
    assert first_event.bed_id is None
    assert second_event.bed_id is None


def test_pipeline_values_preserve_box_keypoint_and_track_alignment_without_image_leak() -> None:
    packet, module_result, decision_input, business_event = _build_pipeline_values()
    runner_result = module_result.result
    assert isinstance(runner_result, DetectionRunnerResult)
    assert module_result.module_name == "mobility-v2"
    assert module_result.output_adapter == "pose"

    observation = decision_input.observation
    assert observation.boxes == runner_result.detections.boxes
    assert observation.keypoints == runner_result.detections.keypoints
    assert len(observation.boxes) == len(observation.keypoints) == len(observation.track_ids)
    assert decision_input.live_track_ids == observation.track_ids
    assert decision_input.frame_width == packet.width
    assert decision_input.frame_height == packet.height
    assert decision_input.frame_index == packet.frame.index
    assert decision_input.time_sec == packet.frame.time_sec
    assert business_event.person_id == observation.track_ids[0]
    assert business_event.camera_id == packet.camera_id
    assert not hasattr(module_result, "frame")
    assert not hasattr(decision_input, "frame")
    assert not hasattr(business_event, "frame")

    payload: EventPayload = {
        "domain": business_event.domain,
        "event_type": business_event.event_type,
        "identity": business_event.identity,
        "camera_id": business_event.camera_id,
        "facility_id": business_event.facility_id,
        "time_sec": business_event.time_sec,
        "probability": business_event.probability,
        "person_id": business_event.person_id,
        "bed_id": business_event.bed_id,
    }
    assert payload["identity"] == "31:5"


def test_worker_tree_defines_no_detection_result_class() -> None:
    declarations: list[tuple[Path, int]] = []

    for path in (_ROOT / "worker").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        declarations.extend(
            (path.relative_to(_ROOT), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "DetectionResult"
        )

    assert declarations == []
