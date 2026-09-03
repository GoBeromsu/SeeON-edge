from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from contracts.observation import BoundingBox
from contracts.replay_trace import decode_jsonl
from worker.domains.bed_exit.detector import BedExitMonitor
from worker.domains.bed_exit.schema import BedExitConfig
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


def test_persisted_polygon_replay_matches_production_image_space_decision_input(
    tmp_path: Path,
) -> None:
    binding = SourceBinding(str(_BOOT), str(_CHILD), "camera-a", 3, 4, "seeon-perception-v1")
    writer = ReplayTraceWriter(tmp_path, "camera-a")
    captured = _Decider()
    scene = SceneState(
        "camera-a",
        persisted_bed_regions=(
            BoundingBox(960, 0, 1920, 1080, 1.0, ((960, 0), (1920, 0), (1920, 1080), (960, 1080))),
        ),
        bed_zone_image_width=1920,
        bed_zone_image_height=1080,
    )
    pump = NativePolicyPump(
        binding,
        NativePolicyContext(
            LatestMetadataSlot(),
            _Control(),
            scene,
            EventAggregator((captured,), IncidentManager(0.0, tmp_path / "events.jsonl")),
            _Sink(),
            AlertEvidenceAttacher({}),
            _Diagnostics(),
            90,
            replay_trace=writer,
        ),
    )
    pump._process(_metadata())  # noqa: SLF001
    assert captured.input_value is not None
    trace_path = tmp_path / f"{hashlib.sha256(b'camera-a').hexdigest()[:16]}.jsonl"
    _, rows = decode_jsonl(trace_path.read_text())
    replayed = replay_trace_to_decision_input(rows[1], seq=11)
    _assert_decision_inputs_equivalent(replayed, captured.input_value)
    clock = lambda: datetime(1970, 1, 1, tzinfo=UTC)  # noqa: E731
    production_monitor = BedExitMonitor(
        config=BedExitConfig("camera-a", "facility", hold_frames=1, grace_frames=1), clock=clock
    )
    replay_monitor = BedExitMonitor(
        config=BedExitConfig("camera-a", "facility", hold_frames=1, grace_frames=1), clock=clock
    )
    assert production_monitor.update(captured.input_value) == replay_monitor.update(replayed)


def _assert_decision_inputs_equivalent(replayed: object, production: object) -> None:
    """Exact equality on every integer/structural field; keypoint floats within 1e-6 px.

    Unit-coordinate round trips (px / W * W) can differ from the production
    float by one ulp; pose+bbox56 rows are float32 so that never changes a
    decision, and replay must not round production geometry to hide it.
    """
    from dataclasses import fields, replace

    rep_obs = replayed.observation  # type: ignore[attr-defined]
    prod_obs = production.observation  # type: ignore[attr-defined]
    assert len(rep_obs.poses) == len(prod_obs.poses)
    for rep_pose, prod_pose in zip(rep_obs.poses, prod_obs.poses, strict=True):
        for (rx, ry, rs), (px, py, ps) in zip(rep_pose, prod_pose, strict=True):
            assert math.isclose(rx, px, abs_tol=1e-6) and math.isclose(ry, py, abs_tol=1e-6)
            assert rs == ps
    rep_stripped = replace(replayed, observation=replace(rep_obs, poses=prod_obs.poses))  # type: ignore[arg-type]
    for field in fields(production):  # type: ignore[arg-type]
        if field.name == "bed_pose_features":
            rep_items = getattr(rep_stripped, field.name).items
            prod_items = getattr(production, field.name).items
            assert len(rep_items) == len(prod_items)
            for rep_item, prod_item in zip(rep_items, prod_items, strict=True):
                for item_field in fields(prod_item):
                    r = getattr(rep_item, item_field.name)
                    p = getattr(prod_item, item_field.name)
                    if isinstance(p, float):
                        assert math.isclose(r, p, rel_tol=1e-6, abs_tol=1e-6), item_field.name
                    else:
                        assert r == p, item_field.name
            continue
        assert getattr(rep_stripped, field.name) == getattr(production, field.name), field.name
