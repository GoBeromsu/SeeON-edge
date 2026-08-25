"""Executable specification for the native DeepStream parser and geometry.

This package is the reference the custom ``GstBaseTransform`` implements. It is
pure numeric Python with no GPU, DeepStream or TensorRT dependency, so the
parity suite runs on an ordinary CI runner while binding the exact shipped
behavior the native code must reproduce:

- ``preprocess``: BGR planes, FP32 ``/255``, 640x384 letterbox at pad value 114.
- ``geometry``: per-frame affine metadata with SEPARATE box and keypoint inverse
  rules.
- ``parse``: pose/person/bed tensor rows, strict ``score > 0.05``, source order,
  no second NMS.
- ``comparator``: binary parity verdicts naming the field that diverged.
"""

from __future__ import annotations

from worker.native.deepstream.parity.comparator import (
    CONFIDENCE_TOLERANCE,
    ParityMismatch,
    compare_bed,
    compare_person,
    compare_pose,
)
from worker.native.deepstream.parity.geometry import (
    BOX_INVERSE_IDENTITY,
    KEYPOINT_INVERSE_IDENTITY,
    LETTERBOX_PAD_VALUE,
    LETTERBOX_SIZE,
    LETTERBOX_STRIDE,
    AffineMetadata,
    letterbox_affine,
)
from worker.native.deepstream.parity.parse import (
    BED_PROTOTYPE_SHAPE,
    BED_ROW_STRIDE,
    PERSON_ROW_STRIDE,
    POSE_ROW_STRIDE,
    POSE_SCORE_THRESHOLD,
    ParsedBed,
    ParsedPerson,
    ParsedPose,
    parse_bed_rows,
    parse_person_rows,
    parse_pose_rows,
)
from worker.native.deepstream.parity.preprocess import (
    NATIVE_CHANNEL_ORDER,
    PREPROCESS_IDENTITY,
    TENSOR_PRECISION,
    ChannelOrder,
    preprocess_batch,
    preprocess_tensor,
)

__all__ = [
    "BED_PROTOTYPE_SHAPE",
    "BED_ROW_STRIDE",
    "BOX_INVERSE_IDENTITY",
    "CONFIDENCE_TOLERANCE",
    "KEYPOINT_INVERSE_IDENTITY",
    "LETTERBOX_PAD_VALUE",
    "LETTERBOX_SIZE",
    "LETTERBOX_STRIDE",
    "NATIVE_CHANNEL_ORDER",
    "PERSON_ROW_STRIDE",
    "POSE_ROW_STRIDE",
    "POSE_SCORE_THRESHOLD",
    "PREPROCESS_IDENTITY",
    "TENSOR_PRECISION",
    "AffineMetadata",
    "ChannelOrder",
    "ParityMismatch",
    "ParsedBed",
    "ParsedPerson",
    "ParsedPose",
    "compare_bed",
    "compare_person",
    "compare_pose",
    "letterbox_affine",
    "parse_bed_rows",
    "parse_person_rows",
    "parse_pose_rows",
    "preprocess_batch",
    "preprocess_tensor",
]
