"""Best-effort overlays rendered from frozen frame-analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.observation import BoundingBox, DetectionLabel, FrameObservation
from worker.domains.bed_exit import BedExitDebugSnapshot
from worker.pipeline.output._overlay_primitives import (
    BED_DASHED_COLOR,
    FALL_LABEL_COLOR,
    NORMAL_LABEL_COLOR,
    PERSON_COLOR,
    POSE_DOT_COLOR,
    draw_box,
    draw_caption,
    draw_dashed_region,
    draw_label,
    draw_pose,
    draw_region,
)
from worker.types import FramePacket

MAX_SNAPSHOT_BYTES: Final = 200 * 1024

# ``none``: draw nothing (clean frames). ``bedexit``: teal dashed bed
# polygon(s) + person boxes/skeleton + per-bed occupancy label. ``fall``:
# person boxes/skeleton + per-track FALL/NORMAL label. No backward
# compatibility with the previous boolean ``show_pose`` toggle -- the
# frontend sends ``{"mode": ...}`` exclusively.
OverlayMode = Literal["none", "bedexit", "fall"]


class OverlayEncodingError(RuntimeError):
    """Raised when OpenCV cannot encode a rendered overlay."""


@dataclass(frozen=True, slots=True)
class OverlayRenderer:
    """Render frozen inference outputs without mutating the shared frame."""

    mode: OverlayMode = "none"

    def render(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> NDArray[np.uint8]:
        image = packet.frame.image.copy()
        if self.mode == "none":
            return image
        for box in observation.boxes:
            draw_box(image, box, PERSON_COLOR)
            draw_label(image, "person", box.x1, max(12, box.y1 - 4), PERSON_COLOR)
        for keypoints in observation.keypoints:
            draw_pose(image, keypoints, color=POSE_DOT_COLOR, dot_radius=2, skeleton=True)
        if self.mode == "bedexit":
            _draw_bedexit_beds(image, observation.bed_boxes, debug_snapshots)
        elif self.mode == "fall":
            _draw_fall_labels(image, observation)
        return image

    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> bytes:
        image = self.render(packet, observation, debug_snapshots)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise OverlayEncodingError("failed to encode overlay JPEG")
        return bytes(encoded)

    def encode_jpeg_bounded(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
        *,
        max_bytes: int = MAX_SNAPSHOT_BYTES,
    ) -> bytes | None:
        try:
            image = self.render(packet, observation, debug_snapshots)
            encoded = _encode_bounded_image(image, max_bytes=max_bytes)
            if encoded is not None:
                return encoded
            downscaled = np.asarray(
                cv2.resize(
                    image,
                    (0, 0),
                    fx=0.5,
                    fy=0.5,
                    interpolation=cv2.INTER_AREA,
                ),
                dtype=np.uint8,
            )
            return _encode_bounded_image(downscaled, max_bytes=max_bytes)
        except (cv2.error, OverlayEncodingError):
            return None


def _encode_bounded_image(
    image: NDArray[np.uint8],
    *,
    max_bytes: int,
) -> bytes | None:
    for quality in (85, 70, 55, 40, 25):
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if ok:
            jpeg = bytes(encoded)
            if len(jpeg) <= max_bytes:
                return jpeg
    return None


def _draw_bedexit_beds(
    image: NDArray[np.uint8],
    bed_boxes: tuple[BoundingBox, ...],
    debug_snapshots: tuple[BedExitDebugSnapshot, ...],
) -> None:
    """Teal dashed bed outline plus its occupancy label (empty/occupied/exit).

    Occupancy comes from the bed-exit domain's debug side-channel
    (``BedExitDebugSnapshot.statuses``), keyed by ``bed_id``.
    ``BedExitMonitor._update_frame`` builds that tuple by ``enumerate()``-ing
    this same ``observation.bed_boxes`` sequence
    (worker/domains/bed_exit/detector.py:145-159), so ``bed_id`` lines up
    with this loop's index without any extra join key.
    """
    occupancy_by_bed_id = {
        status.bed_id: status.occupancy
        for snapshot in debug_snapshots
        for status in snapshot.statuses
    }
    for bed_id, box in enumerate(bed_boxes):
        draw_dashed_region(image, box, BED_DASHED_COLOR)
        occupancy = occupancy_by_bed_id.get(bed_id)
        label = f"bed:{occupancy}" if occupancy is not None else "bed"
        draw_label(image, label, box.x1, max(12, box.y1 - 4), BED_DASHED_COLOR)


def _draw_fall_labels(image: NDArray[np.uint8], observation: FrameObservation) -> None:
    """Per-track FALL/NORMAL label: danger-red for FALL, neutral for NORMAL."""
    for index, label in _fall_labels_by_box_index(observation).items():
        box = observation.boxes[index]
        color = FALL_LABEL_COLOR if label.is_fall else NORMAL_LABEL_COLOR
        draw_label(image, label.text, box.x1, box.y2 + 14, color)


def _fall_labels_by_box_index(
    observation: FrameObservation,
) -> dict[int, DetectionLabel]:
    """Map each per-track FALL/NORMAL ``DetectionLabel`` back to its box index.

    ``FallWindowClassifier.classify()`` (worker/domains/fall/classifier.py:
    87-91) builds ``labels`` by walking ``track_ids`` in order and skipping
    any entry that is ``None`` or not currently live, so ``labels[k]`` is the
    label for the k-th non-``None`` track id -- not necessarily for
    ``boxes[k]``. Track ids stay positionally parallel to boxes from tracker
    assignment onward (worker/pipeline/analytics/composite.py:118-124), so
    replay that same walk here instead of assuming labels and boxes already
    line up index-for-index.
    """
    track_ids = observation.track_ids
    if len(track_ids) != len(observation.boxes):
        return {}
    mapping: dict[int, DetectionLabel] = {}
    label_iter = iter(observation.labels)
    for index, track_id in enumerate(track_ids):
        if track_id is None:
            continue
        label = next(label_iter, None)
        if label is None:
            break
        mapping[index] = label
    return mapping


__all__ = [
    "MAX_SNAPSHOT_BYTES",
    "OverlayEncodingError",
    "OverlayMode",
    "OverlayRenderer",
    "draw_box",
    "draw_caption",
    "draw_dashed_region",
    "draw_label",
    "draw_pose",
    "draw_region",
]
