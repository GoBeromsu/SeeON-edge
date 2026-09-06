"""Conversion from Flow metadata to worker perception envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from worker.interfaces.association import (
    NVDCF_STRATEGY,
    AssociationObservation,
    TrackedObject,
)
from worker.types.metadata import MetadataFrame, SourceBinding
from worker.types.perception_frame import (
    AssociationResult,
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    PerceptionFrameIdentity,
    PersonBox,
    PersonBoxChannel,
    assemble_perception_frame,
)

_SCORE_MIN = np.float32(0.05)
_NET_SIZE = 640.0
_IOU_GATE = 0.5


@dataclass(frozen=True, slots=True)
class AssociationPass:
    observation: AssociationObservation
    matched_rows: tuple[int, ...]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _frame_box(
    row: NDArray[np.float32], frame_w: int, frame_h: int
) -> tuple[float, float, float, float]:
    # nvinfer letterboxes with the padding at the right and bottom, not centred:
    # there is no offset to remove, only the scale to undo. Subtracting a centred
    # pad shifted every box by half the padding - 140 px for a 640x360 frame -
    # so no detection ever overlapped its tracked object and nothing matched.
    scale = min(_NET_SIZE / frame_w, _NET_SIZE / frame_h)
    return (
        float(row[0] / scale),
        float(row[1] / scale),
        float(row[2] / scale),
        float(row[3] / scale),
    )


def association_pass(
    frame_meta: Any,
    *,
    rows: NDArray[np.float32],
    identity: PerceptionFrameIdentity,
    frame_w: int,
    frame_h: int,
) -> AssociationPass:
    """Associate thresholded output rows to NvDCF objects exactly once."""
    candidates = [
        (index, _frame_box(row, frame_w, frame_h))
        for index, row in enumerate(rows)
        if row[4] > _SCORE_MIN
    ]
    objects: list[tuple[int, tuple[float, float, float, float], float]] = []
    for item in frame_meta.object_items:
        rect = item.rect_params
        objects.append(
            (
                int(item.object_id),
                (
                    float(rect.left),
                    float(rect.top),
                    float(rect.left + rect.width),
                    float(rect.top + rect.height),
                ),
                float(item.confidence),
            )
        )
    pairs = sorted(
        (
            (_iou(box, candidate_box), object_index, row_index)
            for object_index, (_, box, _) in enumerate(objects)
            for row_index, (_, candidate_box) in enumerate(candidates)
        ),
        key=lambda pair: (-pair[0], pair[1], pair[2]),
    )
    assigned: dict[int, int] = {}
    used_rows: set[int] = set()
    for score, object_index, candidate_index in pairs:
        if score < _IOU_GATE:
            break
        if object_index in assigned or candidate_index in used_rows:
            continue
        assigned[object_index] = candidate_index
        used_rows.add(candidate_index)
    tracks = tuple(
        TrackedObject(
            track_id=track_id,
            box=box,
            confidence=confidence,
            pose_row=None if index not in assigned else candidates[assigned[index]][0],
        )
        for index, (track_id, box, confidence) in enumerate(objects)
    )
    return AssociationPass(
        observation=AssociationObservation(
            identity=identity,
            strategy=NVDCF_STRATEGY,
            tracks=tracks,
            live_track_ids=tuple(track.track_id for track in tracks),
            unmatched_tracks=sum(track.pose_row is None for track in tracks),
            rows_available=len(candidates),
        ),
        matched_rows=tuple(candidates[assigned[index]][0] for index in sorted(assigned)),
    )


def convert_frame(
    frame_meta: Any,
    *,
    rows: NDArray[np.float32],
    binding: SourceBinding,
    frame_w: int,
    frame_h: int,
    publish_sequence: int,
    boot_id: str,
) -> MetadataFrame:
    """Build one accepted P1a-compatible frame from a Flow frame item."""
    identity = PerceptionFrameIdentity(
        worker_boot_id=boot_id,
        camera_id=binding.camera_id,
        stream_epoch=binding.stream_epoch,
        seq=publish_sequence,
        source_pts=int(frame_meta.buffer_pts),
    )
    linked = association_pass(
        frame_meta, rows=rows, identity=identity, frame_w=frame_w, frame_h=frame_h
    )
    matched = tuple(track for track in linked.observation.tracks if track.pose_row is not None)
    boxes = tuple(
        PersonBox(
            int(track.box[0]),
            int(track.box[1]),
            int(track.box[2]),
            int(track.box[3]),
            track.confidence,
        )
        for track in matched
    )
    # Same right/bottom letterbox as the boxes: undo the scale, no offset.
    scale = min(_NET_SIZE / frame_w, _NET_SIZE / frame_h)
    poses = tuple(
        tuple(
            Keypoint(
                int(max(0.0, min(float(frame_w), float(row[6 + point * 3] / scale)))),
                int(max(0.0, min(float(frame_h), float(row[7 + point * 3] / scale)))),
                float(row[8 + point * 3]),
            )
            for point in range(17)
        )
        for row in (rows[track.pose_row] for track in matched if track.pose_row is not None)
    )
    state = ChannelState.INFERRED if boxes else ChannelState.INFERRED_EMPTY
    frame = assemble_perception_frame(
        identity=identity,
        person_box=PersonBoxChannel(state, boxes),
        human_pose=HumanPoseChannel(state, poses),
        bed_region=BedRegionChannel(ChannelState.SKIPPED),
        association=AssociationResult(
            strategy=NVDCF_STRATEGY,
            track_ids=tuple(track.track_id for track in matched),
            selected_cue_indexes=tuple(range(len(matched))),
            identity=identity,
            live_track_ids=linked.observation.live_track_ids,
        ),
    )
    return MetadataFrame(
        frame=frame,
        source_generation=binding.source_generation,
        child_instance_id=binding.child_instance_id,
        native_publish_sequence=publish_sequence,
        transform_id=binding.transform_id,
        source_width=frame_w,
        source_height=frame_h,
        source_time_ns=int(frame_meta.buffer_pts),
    )


__all__ = ["AssociationPass", "association_pass", "convert_frame"]
