from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from worker.adapters.deepstream.metadata import association_pass, convert_frame
from worker.native.deepstream.metadata import LatestMetadataSlot, SourceBinding
from worker.types.perception_frame import PerceptionFrameIdentity


def _object(track_id: int, left: float, top: float, width: float, height: float) -> SimpleNamespace:
    return SimpleNamespace(
        object_id=track_id,
        confidence=0.9,
        rect_params=SimpleNamespace(left=left, top=top, width=width, height=height),
    )


def test_letterbox_inverse_is_exact_for_boxes_and_keypoints() -> None:
    binding = SourceBinding("boot", "child", "camera", 2, 3, "transform")
    rows = np.zeros((1, 57), dtype=np.float32)
    rows[0, 4] = 0.9
    # 1280x720 is padded by 140 pixels above and below in a 640 square.
    rows[0, :4] = (50, 190, 150, 290)
    rows[0, 6:9] = (100, 240, 0.7)
    meta = SimpleNamespace(
        buffer_pts=10,
        object_items=[_object(7, 100, 100, 200, 200)],
    )
    converted = convert_frame(
        meta,
        rows=rows,
        binding=binding,
        frame_w=1280,
        frame_h=720,
        publish_sequence=1,
        boot_id="boot",
    )
    assert converted.frame.person_box.boxes[0].x1 == 100
    assert converted.frame.person_box.boxes[0].y1 == 100
    assert converted.frame.person_box.boxes[0].x2 == 300
    assert converted.frame.person_box.boxes[0].y2 == 300
    assert converted.frame.human_pose.poses[0][0].x == 200
    assert converted.frame.human_pose.poses[0][0].y == 200


def test_association_blocks_row_reuse_and_marks_coasted_track_unmatched() -> None:
    rows = np.zeros((3, 57), dtype=np.float32)
    rows[:, 4] = (0.9, 0.8, 0.05)  # score threshold is strictly greater than .05
    rows[0, :4] = (64, 64, 128, 128)
    rows[1, :4] = (64, 64, 128, 128)
    meta = SimpleNamespace(
        buffer_pts=10,
        object_items=[_object(7, 10, 10, 10, 10), _object(8, 30, 30, 10, 10)],
    )
    identity = PerceptionFrameIdentity("boot", "camera", 3, 1, 10)
    observation = association_pass(meta, rows=rows, identity=identity, frame_w=100, frame_h=100)
    assert [track.pose_row for track in observation.observation.tracks] == [0, None]
    assert observation.observation.unmatched_tracks == 1
    assert observation.observation.rows_available == 2
    assert observation.matched_rows == (0,)


def test_slot_rejects_stale_source_generation() -> None:
    binding = SourceBinding("boot", "child", "camera", 2, 3, "transform")
    rows = np.zeros((1, 57), dtype=np.float32)
    rows[0, 4] = 0.9
    rows[0, :4] = (64, 64, 128, 128)
    meta = SimpleNamespace(buffer_pts=10, object_items=[_object(7, 10, 10, 10, 10)])
    converted = convert_frame(
        meta, rows=rows, binding=binding, frame_w=100, frame_h=100, publish_sequence=1, boot_id="boot"
    )
    slot = LatestMetadataSlot()
    slot.register_source(binding)
    assert slot.publish(converted)
    stale = SourceBinding("boot", "child", "camera", 1, 3, "transform")
    stale_frame = convert_frame(
        meta,
        rows=rows,
        binding=stale,
        frame_w=100,
        frame_h=100,
        publish_sequence=2,
        boot_id="boot",
    )
    assert not slot.publish(stale_frame)
