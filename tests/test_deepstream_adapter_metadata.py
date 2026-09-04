from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from worker.adapters.deepstream.metadata import _frame_box, _iou, association_pass, convert_frame
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
    # nvinfer letterboxes 1280x720 into the 640 square with the padding at the
    # right and bottom, so the inverse is the scale alone. Measured on hardware:
    # removing a centred pad instead shifted every box and matched nothing.
    rows[0, :4] = (50, 50, 150, 150)
    rows[0, 6:9] = (100, 100, 0.7)
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
        meta,
        rows=rows,
        binding=binding,
        frame_w=100,
        frame_h=100,
        publish_sequence=1,
        boot_id="boot",
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


def test_a_frame_without_pose_tensors_is_published_with_every_track_unmatched() -> None:
    """nvinfer can deliver a frame with no tensor metadata attached.

    The tracked objects are still real, so dropping the frame would hide it
    from the decision layer and make a streaming camera look silent. It must be
    published with no pose rows, leaving every track explicitly unmatched.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from worker.adapters.deepstream.service_maker import (
        DeepStreamMediaPlane,
        DeepStreamMediaPlaneConfig,
        _FlowHandle,
    )
    from worker.native.deepstream.metadata import LatestMetadataSlot

    config = DeepStreamMediaPlaneConfig("infer", "tracker", "lib", Path("/tmp"), 5, 640, 360)
    plane = DeepStreamMediaPlane(
        config,
        metadata_slot=LatestMetadataSlot(),
        flow_factory=lambda _: _FlowHandle(
            flow=object(),
            pipeline=object(),
            record_config=lambda **kwargs: kwargs,
            render_mode_discard="discard",
            make_probe=lambda name, probe: (name, probe),
        ),
        worker_boot_id="boot",
        child_instance_id="child",
    )
    plane.add_source("camera", "rtsp://one")
    frame = SimpleNamespace(
        pad_index=0,
        buffer_pts=1_000,
        tensor_items=[],
        object_items=[
            SimpleNamespace(
                object_id=7,
                confidence=0.9,
                rect_params=SimpleNamespace(left=10.0, top=10.0, width=20.0, height=40.0),
            )
        ],
    )

    plane.publish_frame(frame)

    assert plane.published_frames("camera") == 1, "the frame must count as published"
    assert "camera" in plane._live  # noqa: SLF001 - the camera is demonstrably alive


def test_the_letterbox_inverse_reproduces_boxes_nvinfer_and_nvtracker_actually_produced() -> None:
    """Ground truth, not a restatement of the implementation.

    The previous version of this test encoded the same wrong padding convention
    as the code, so both agreed while the worker matched nothing for an entire
    bring-up. This one loads raw 57-wide pose rows captured from nvinfer beside
    the frame-space boxes nvtracker attached to the same frames, and requires the
    inverse to land on them. Captured by
    scripts/qa/pyservicemaker-spike/live/capture_letterbox_fixture.py.
    """
    capture = json.loads(
        (Path(__file__).parent / "fixtures" / "nvinfer_letterbox_capture.json").read_text(
            encoding="utf-8"
        )
    )
    frame_w = int(capture["frame_width"])
    frame_h = int(capture["frame_height"])
    assert capture["samples"], "the capture must contain frames"

    for sample in capture["samples"]:
        tracked = [
            (
                float(box["left"]),
                float(box["top"]),
                float(box["left"]) + float(box["width"]),
                float(box["top"]) + float(box["height"]),
            )
            for box in sample["boxes"]
        ]
        assert tracked, "the capture must contain tracked boxes"
        for entry in sample["rows"]:
            row = np.asarray(entry["row"], dtype=np.float32)
            inverted = _frame_box(row, frame_w, frame_h)
            best = max(_iou(inverted, box) for box in tracked)
            assert best > 0.95, (
                f"the inverse of a real pose row landed at IoU {best:.3f} from every box "
                "nvtracker produced for that frame"
            )
