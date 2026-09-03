from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from contracts.replay_trace import decode_jsonl
from worker.domains.fall.pose_bbox56 import pose_bbox56_row
from worker.native.deepstream.ipc import MetadataFrame
from worker.native.deepstream.metadata import LatestMetadataSlot, SourceBinding
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import SceneState
from worker.pipeline.trace.replay_trace_writer import ReplayTraceWriter
from worker.replay.inputs import replay_trace_to_decision_input
from worker.runtime.deepstream.native_policy_pump import NativePolicyContext, NativePolicyPump
from worker.types import (
    AssociationResult,
    BedRegionChannel,
    BusinessEvent,
    ChannelState,
    DecisionInput,
    HumanPoseChannel,
    Keypoint,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
)

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


@dataclass(slots=True)
class _Decider:
    input_value: DecisionInput | None = None

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        self.input_value = input_value
        return ()


class _Control:
    def snapshot(self, camera_id: str) -> None:
        raise AssertionError(f"unexpected snapshot for {camera_id}")


class _Sink:
    def emit_for_frame(self, event: object, trigger: object) -> None:
        raise AssertionError(f"unexpected event {event!r} {trigger!r}")


class _Diagnostics:
    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None:
        del camera_id, measured_fps

    def record_detection_completed(self, camera_id: str) -> None:
        del camera_id

    def record_native_detection_attempt(self, camera_id: str) -> None:
        del camera_id

    def record_track_id_switch(self, camera_id: str) -> None:
        del camera_id

    def record_bed_polygon_source(self, camera_id: str, source: str) -> None:
        del camera_id, source

    def record_replay_trace_write_failure(self, camera_id: str) -> None:
        del camera_id


def _metadata() -> MetadataFrame:
    identity = PerceptionFrameIdentity(str(_BOOT), "camera-a", 4, 11, 2_000_000_000)
    association = AssociationResult(
        "legacy-greedy-bbox-iou.v1", (7, 9), (0, 1), identity, live_track_ids=(7, 9)
    )
    poses = tuple(
        tuple(Keypoint(x + point, y + point, 0.9) for point in range(17))
        for x, y in ((20, 30), (400, 180))
    )
    return MetadataFrame(
        PerceptionFrameV1(
            identity,
            PersonBoxChannel(
                ChannelState.INFERRED,
                (PersonBox(10, 20, 40, 80, 0.8), PersonBox(380, 170, 500, 300, 0.7)),
            ),
            HumanPoseChannel(ChannelState.INFERRED, poses),
            BedRegionChannel(ChannelState.INFERRED_EMPTY),
            association,
        ),
        3,
        _CHILD,
        12,
        "seeon-perception-v1",
        640,
        360,
        2_100_000_000,
    )


def test_native_capture_and_replay_preserve_unit_observation_geometry(tmp_path: Path) -> None:
    binding = SourceBinding(str(_BOOT), str(_CHILD), "camera-a", 3, 4, "seeon-perception-v1")
    writer = ReplayTraceWriter(tmp_path, "camera-a")
    decider = _Decider()
    pump = NativePolicyPump(
        binding,
        NativePolicyContext(
            LatestMetadataSlot(),
            _Control(),  # pyright: ignore[reportArgumentType]
            SceneState("camera-a"),
            EventAggregator((decider,), IncidentManager(0.0, tmp_path / "events.jsonl")),
            _Sink(),  # pyright: ignore[reportArgumentType]
            AlertEvidenceAttacher({}),
            _Diagnostics(),
            90,
            replay_trace=writer,
        ),
    )

    pump._process(_metadata())  # noqa: SLF001

    assert decider.input_value is not None
    trace_path = tmp_path / f"{hashlib.sha256(b'camera-a').hexdigest()[:16]}.jsonl"
    _, rows = decode_jsonl(trace_path.read_text())
    assert rows[0].source_event == "open"
    row = rows[1]
    replayed = replay_trace_to_decision_input(row, seq=11)
    captured = decider.input_value.observation
    # Production pixel geometry round-trips exactly through unit coordinates + frame size.
    assert replayed.frame_width == 640 and replayed.frame_height == 360
    assert tuple(
        (box.x1, box.y1, box.x2, box.y2, box.confidence) for box in replayed.observation.boxes
    ) == tuple((box.x1, box.y1, box.x2, box.y2, box.confidence) for box in captured.boxes)
    assert replayed.live_track_ids == (7, 9)
    # The fall feature rows the decider consumes are byte-identical (float32 pose+bbox56).
    for track_index, (cap_pose, rep_pose) in enumerate(
        zip(captured.poses, replayed.observation.poses, strict=True)
    ):
        cap_box = captured.boxes[track_index]
        rep_box = replayed.observation.boxes[track_index]
        assert pose_bbox56_row(
            cap_pose, (cap_box.x1, cap_box.y1, cap_box.x2, cap_box.y2), 640, 360
        ) == pose_bbox56_row(rep_pose, (rep_box.x1, rep_box.y1, rep_box.x2, rep_box.y2), 640, 360)
