"""RED/GREEN contract for the DeepStream native parser/geometry parity spec.

`worker.native.deepstream.parity` is the executable specification the custom
`GstBaseTransform` implements: shipped preprocessing (BGR planes, FP32 ``/255``,
640x384 letterbox at pad value 114), the two separate box/keypoint inverse
rules, and the tensor row parsers. Every assertion here is hardware-free -- no
GPU, no DeepStream image and no TensorRT engine -- so this file runs unmarked in
CI alongside the ordinary suite.

Imports stay inside helpers so a missing module fails as an assertion, not as a
collection-time ImportError from a typo.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

_REQUIRED_PARITY_SYMBOLS = (
    "BOX_INVERSE_IDENTITY",
    "KEYPOINT_INVERSE_IDENTITY",
    "LETTERBOX_PAD_VALUE",
    "LETTERBOX_STRIDE",
    "NATIVE_CHANNEL_ORDER",
    "POSE_ROW_STRIDE",
    "TENSOR_PRECISION",
    "AffineMetadata",
    "ParityMismatch",
    "compare_bed",
    "compare_person",
    "compare_pose",
    "letterbox_affine",
    "parse_bed_rows",
    "parse_person_rows",
    "parse_pose_rows",
    "preprocess_tensor",
)


def _parity() -> Any:
    try:
        module = importlib.import_module("worker.native.deepstream.parity")
    except ImportError as exc:  # pragma: no cover - RED path only
        pytest.fail(f"missing module worker.native.deepstream.parity: {exc}")
    for name in _REQUIRED_PARITY_SYMBOLS:
        assert hasattr(module, name), f"parity spec is missing {name}"
    return module


def _rgb_frame(height: int, width: int) -> np.ndarray:
    """A true-RGB host frame with a distinct constant per channel."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 20
    frame[..., 2] = 30
    return frame


def _pose_row(
    *,
    score: float,
    box: tuple[float, float, float, float] = (300.0, 152.0, 340.0, 232.0),
    keypoint_origin: float = 100.0,
) -> list[float]:
    """One end-to-end pose row: tensor-space xyxy, score, class, then COCO-17."""
    row = [box[0], box[1], box[2], box[3], score, 0.0]
    for index in range(17):
        row.extend([keypoint_origin + index, keypoint_origin + index + 1, 0.9])
    return row


# --------------------------------------------------------------------------
# Shipped preprocessing: the network receives BGR planes.
# --------------------------------------------------------------------------


def test_preprocess_reproduces_shipped_bgr_plane_inversion() -> None:
    """The shipped chain double-flips true RGB, so plane 0 carries blue.

    Ingest emits true RGB (``cv2.COLOR_BGR2RGB``) and the adapter hands that
    array to ultralytics, whose predictor applies ``im = im[..., ::-1]`` a
    second time. Native preprocessing must reproduce the inversion; feeding
    true RGB to the exported model is "more correct" and destroys parity.
    """
    parity = _parity()
    assert parity.NATIVE_CHANNEL_ORDER == "bgr"
    tensor = parity.preprocess_tensor(_rgb_frame(360, 640))
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, 384, 640)
    center = [float(tensor[0, plane, 192, 320]) * 255.0 for plane in range(3)]
    assert center == pytest.approx([30.0, 20.0, 10.0])


def test_preprocess_scales_to_unit_range_and_pads_with_114() -> None:
    parity = _parity()
    assert parity.TENSOR_PRECISION == "fp32"
    assert parity.LETTERBOX_PAD_VALUE == 114
    tensor = parity.preprocess_tensor(_rgb_frame(360, 640))
    pad_value = float(tensor[0, 0, 0, 320]) * 255.0
    assert pad_value == pytest.approx(114.0)
    assert float(tensor.max()) <= 1.0


def test_preprocess_refuses_square_only_geometry() -> None:
    """RED fixture: a square-only 640x640 tensor is not the shipped geometry."""
    parity = _parity()
    tensor = parity.preprocess_tensor(_rgb_frame(360, 640))
    assert tensor.shape[2:] != (640, 640), "square-only preprocessing breaks parity"


# --------------------------------------------------------------------------
# Geometry: 640x360 content inside a 640x384 tensor, separate inverse rules.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("height", "width"), [(360, 640), (1080, 1920)])
def test_sixteen_by_nine_sources_share_one_tensor_geometry(height: int, width: int) -> None:
    parity = _parity()
    affine = parity.letterbox_affine(height, width)
    assert (affine.tensor_height, affine.tensor_width) == (384, 640)
    assert (affine.content_height, affine.content_width) == (360, 640)
    assert affine.pad_top == 12
    assert affine.pad_bottom == 12
    assert affine.pad_value == 114


def test_odd_fixture_separates_box_and_keypoint_inverse_rules() -> None:
    """The 100x102 fixture is where the two inverse rules diverge.

    ``ops.scale_boxes`` pads by ``round(pad/2 - 0.1)`` while ``ops.scale_coords``
    pads by the unrounded ``pad/2``. Collapsing them into one rule is a parity
    break the comparator must catch.
    """
    parity = _parity()
    affine = parity.letterbox_affine(102, 100)
    assert (affine.tensor_height, affine.tensor_width) == (640, 640)
    assert affine.box_pad_x == 6
    assert affine.keypoint_pad_x == pytest.approx(6.5)
    assert affine.box_pad_x != affine.keypoint_pad_x
    assert parity.BOX_INVERSE_IDENTITY != parity.KEYPOINT_INVERSE_IDENTITY


def test_inverse_mapping_truncates_boxes_and_clamps_to_source() -> None:
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    assert affine.invert_box((10.9, 24.9, 100.9, 200.9)) == (10, 12, 100, 188)
    assert affine.invert_box((-40.0, -40.0, 9000.0, 9000.0)) == (0, 0, 640, 360)
    assert affine.invert_keypoint((10.9, 24.9)) == (10, 12)


# --------------------------------------------------------------------------
# Pose parsing: strict score > 0.05, source order, no second NMS.
# --------------------------------------------------------------------------


def test_pose_parser_keeps_strict_threshold_and_source_order() -> None:
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    rows = np.asarray(
        [
            _pose_row(score=0.9, box=(480.0, 160.0, 520.0, 240.0)),
            _pose_row(score=0.2, box=(80.0, 60.0, 120.0, 140.0)),
            _pose_row(score=0.7, box=(280.0, 110.0, 320.0, 190.0)),
        ],
        dtype=np.float32,
    )
    parsed = parity.parse_pose_rows(rows, affine)
    assert [round(box[4], 4) for box in parsed.boxes] == [0.9, 0.2, 0.7], (
        "native parsing preserves source order and must not sort or re-NMS"
    )
    assert len(parsed.poses) == 3
    assert all(len(pose) == 17 for pose in parsed.poses)


def test_pose_parser_uses_strict_greater_than_not_greater_or_equal() -> None:
    """RED fixture: ``>=0.05`` admits a row the shipped threshold drops."""
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    rows = np.asarray(
        [_pose_row(score=0.05), _pose_row(score=0.0500001)],
        dtype=np.float64,
    )
    parsed = parity.parse_pose_rows(rows, affine)
    assert len(parsed.boxes) == 1, "score == 0.05 must be excluded by strict >"


def test_pose_parser_shape_contract() -> None:
    parity = _parity()
    assert parity.POSE_ROW_STRIDE == 57
    affine = parity.letterbox_affine(360, 640)
    with pytest.raises(ValueError, match="57"):
        parity.parse_pose_rows(np.zeros((2, 56), dtype=np.float32), affine)


# --------------------------------------------------------------------------
# Person and bed parsing.
# --------------------------------------------------------------------------


def test_person_parser_filters_class_and_confidence() -> None:
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    rows = np.asarray(
        [
            [300.0, 152.0, 340.0, 232.0, 0.9, 0.0],
            [300.0, 152.0, 340.0, 232.0, 0.9, 1.0],
            [300.0, 152.0, 340.0, 232.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    parsed = parity.parse_person_rows(rows, affine, confidence=0.25)
    assert len(parsed.boxes) == 1


def test_bed_parser_rejects_prototype_shape_drift() -> None:
    """RED fixture: prototypes must be ``[32, 160, 160]``."""
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    rows = np.zeros((1, 38), dtype=np.float32)
    with pytest.raises(ValueError, match="prototype"):
        parity.parse_bed_rows(
            rows,
            np.zeros((16, 160, 160), dtype=np.float32),
            affine,
            confidence=0.25,
        )


# --------------------------------------------------------------------------
# Comparator: binary, and it must fail for the right reason.
# --------------------------------------------------------------------------


def test_comparator_accepts_identical_output() -> None:
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    rows = np.asarray([_pose_row(score=0.9), _pose_row(score=0.3)], dtype=np.float32)
    parsed = parity.parse_pose_rows(rows, affine)
    assert parity.compare_pose(parsed, parsed) == ()


def test_comparator_reports_reordering_as_a_mismatch() -> None:
    """RED fixture: batch reordering must not silently pass."""
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    rows = np.asarray(
        [_pose_row(score=0.9, box=(480.0, 160.0, 520.0, 240.0)), _pose_row(score=0.3)],
        dtype=np.float32,
    )
    reference = parity.parse_pose_rows(rows, affine)
    reordered = parity.parse_pose_rows(rows[::-1], affine)
    mismatches = parity.compare_pose(reference, reordered)
    assert mismatches, "reordered rows must fail the comparator"
    assert all(isinstance(item, parity.ParityMismatch) for item in mismatches)


def test_comparator_reports_channel_inversion_as_a_mismatch() -> None:
    """RED fixture: an intentional channel inversion must fail the comparator."""
    parity = _parity()
    shipped = parity.preprocess_tensor(_rgb_frame(360, 640))
    inverted = parity.preprocess_tensor(_rgb_frame(360, 640), channel_order="rgb")
    assert not np.array_equal(shipped, inverted)
    center_shipped = [float(shipped[0, plane, 192, 320]) for plane in range(3)]
    center_inverted = [float(inverted[0, plane, 192, 320]) for plane in range(3)]
    assert center_shipped == list(reversed(center_inverted))


def test_comparator_reports_extra_nms_as_a_mismatch() -> None:
    """RED fixture: a second NMS drops overlapping rows the shipped path keeps."""
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    overlapping = np.asarray(
        [_pose_row(score=0.9), _pose_row(score=0.8)],
        dtype=np.float32,
    )
    reference = parity.parse_pose_rows(overlapping, affine)
    assert len(reference.boxes) == 2, "native parsing must not apply a second NMS"
    suppressed = parity.parse_pose_rows(overlapping[:1], affine)
    assert parity.compare_pose(reference, suppressed)


def test_comparator_reports_wrong_odd_padding_as_a_mismatch() -> None:
    """RED fixture: using the box pad rule for keypoints breaks the odd fixture."""
    parity = _parity()
    affine = parity.letterbox_affine(102, 100)
    correct = affine.invert_keypoint((320.0, 320.0))
    wrong = (
        int((320.0 - affine.box_pad_x) / affine.gain),
        int((320.0 - affine.box_pad_y) / affine.gain),
    )
    assert correct != wrong


def test_confidence_tolerance_cannot_mask_a_threshold_defect() -> None:
    """The score tolerance must stay far below every decision boundary.

    It exists only to absorb one FP32 ULP of forward-pass noise. If it ever grew
    near the strict pose cut (0.05) or the policy keypoint threshold (0.2) it
    could hide a real threshold defect, so both margins are pinned here.
    """
    parity = _parity()
    assert parity.CONFIDENCE_TOLERANCE < parity.POSE_SCORE_THRESHOLD / 1000
    assert parity.CONFIDENCE_TOLERANCE < 0.2 / 1000


def test_comparator_compares_coordinates_exactly() -> None:
    """Only scores are tolerant; a one-pixel box shift is still a mismatch."""
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    reference = parity.parse_person_rows(
        np.asarray([[300.0, 152.0, 340.0, 232.0, 0.9, 0.0]], dtype=np.float32),
        affine,
        confidence=0.25,
    )
    shifted = parity.parse_person_rows(
        np.asarray([[301.0, 152.0, 340.0, 232.0, 0.9, 0.0]], dtype=np.float32),
        affine,
        confidence=0.25,
    )
    mismatches = parity.compare_person(reference, shifted)
    assert [item.field for item in mismatches] == ["x1"]


def test_comparator_reports_person_and_bed_mismatches() -> None:
    parity = _parity()
    affine = parity.letterbox_affine(360, 640)
    person_rows = np.asarray([[300.0, 152.0, 340.0, 232.0, 0.9, 0.0]], dtype=np.float32)
    reference = parity.parse_person_rows(person_rows, affine, confidence=0.25)
    assert parity.compare_person(reference, reference) == ()
    empty = parity.parse_person_rows(np.zeros((0, 6), dtype=np.float32), affine, confidence=0.25)
    assert parity.compare_person(reference, empty)
    bed_rows = np.zeros((1, 38), dtype=np.float32)
    bed_rows[0, :6] = [300.0, 152.0, 340.0, 232.0, 0.9, 59.0]
    prototypes = np.zeros((32, 160, 160), dtype=np.float32)
    bed_reference = parity.parse_bed_rows(bed_rows, prototypes, affine, confidence=0.25)
    assert parity.compare_bed(bed_reference, bed_reference) == ()


# --------------------------------------------------------------------------
# Perception frame landing: parsed output uses the C1 named types.
# --------------------------------------------------------------------------


def test_parsed_output_lands_in_perception_frame_types() -> None:
    parity = _parity()
    types = importlib.import_module("worker.types.perception_frame")
    affine = parity.letterbox_affine(360, 640)
    rows = np.asarray([_pose_row(score=0.9)], dtype=np.float32)
    parsed = parity.parse_pose_rows(rows, affine)
    frame = parsed.human_pose_channel()
    assert isinstance(frame, types.HumanPoseChannel)
    assert frame.state is types.ChannelState.INFERRED
    assert all(isinstance(point, types.Keypoint) for point in frame.poses[0])
    assert not isinstance(frame.poses[0][0], tuple), "keypoints are the named type"


def test_empty_pose_output_reports_inferred_empty() -> None:
    parity = _parity()
    types = importlib.import_module("worker.types.perception_frame")
    affine = parity.letterbox_affine(360, 640)
    parsed = parity.parse_pose_rows(np.zeros((0, 57), dtype=np.float32), affine)
    assert parsed.human_pose_channel().state is types.ChannelState.INFERRED_EMPTY


# --------------------------------------------------------------------------
# Export manifest: FP32 only, FP16 rejected.
# --------------------------------------------------------------------------


def test_export_manifest_rejects_fp16() -> None:
    export = importlib.import_module("worker.native.deepstream.export")
    with pytest.raises(ValueError, match="fp16"):
        export.validate_precision("fp16")
    assert export.validate_precision("fp32") == "fp32"


def test_export_manifest_binds_source_identity() -> None:
    export = importlib.import_module("worker.native.deepstream.export")
    spec = export.MODEL_EXPORTS["pose"]
    assert spec.output_shape == (300, 57)
    assert export.MODEL_EXPORTS["person"].output_shape == (300, 6)
    assert export.MODEL_EXPORTS["bed"].output_shape == (300, 38)
    assert export.MODEL_EXPORTS["bed"].prototype_shape == (32, 160, 160)
