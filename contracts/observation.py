from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

# DetectionResult is the documented raw-detection interchange type at the
# runner/observation-builder boundary. FrameObservation is the normalized
# per-frame contract used by downstream perception, demo, and api code.
FALL_LABEL_TEXT: Final = "FALL"
NORMAL_LABEL_TEXT: Final = "NORMAL"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    # Optional mask contour (issue #243): bed ROIs from instance segmentation
    # or an operator's persisted `bed_zone_polygon` carry the silhouette
    # polygon for shape-accurate rendering. None for plain detection boxes
    # (person/fall). x1..y2 stays the source of truth for person-tracker IoU
    # and remains the containment fallback when no polygon is present; when
    # a bed carries one, `bed_exit.geometry.containment_ratio` (#219) reads
    # it as the source of truth for bed containment instead of x1..y2, since
    # x1..y2 alone is only an axis-aligned bounding box around the polygon
    # and can measurably overstate a rotated/irregular bed's true area.
    polygon: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True, slots=True)
class DetectionLabel:
    text: str
    confidence: float
    is_fall: bool


Detections = tuple[tuple[BoundingBox, ...], tuple[DetectionLabel, ...]]
Regions = tuple[tuple[BoundingBox, ...], tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    boxes: tuple[BoundingBox, ...] = field(default_factory=tuple)
    labels: tuple[DetectionLabel, ...] = field(default_factory=tuple)
    # per-person COCO-17 keypoints; each kpt = (x:int, y:int, conf:float)
    keypoints: tuple[tuple[tuple[int, int, float], ...], ...] = field(default_factory=tuple)
    # static bed ROIs from a one-shot COCO detection at stream start;
    # cached once and carried per-frame so the per-frame path stays a single pose pass.
    bed_boxes: tuple[BoundingBox, ...] = field(default_factory=tuple)
    bed_exit_statuses: tuple[object, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """Per-frame perception observation payload.

    detections: boxes and labels.
    poses: per-person COCO-17 keypoints.
    regions: bed boxes and bed-exit statuses.
    """

    detections: Detections = field(default_factory=lambda: ((), ()))
    poses: tuple[tuple[tuple[int, int, float], ...], ...] = field(default_factory=tuple)
    regions: Regions = field(default_factory=lambda: ((), ()))
    track_ids: tuple[int | None, ...] = field(default_factory=tuple)

    @property
    def boxes(self) -> tuple[BoundingBox, ...]:
        return self.detections[0]

    @property
    def labels(self) -> tuple[DetectionLabel, ...]:
        return self.detections[1]

    @property
    def keypoints(self) -> tuple[tuple[tuple[int, int, float], ...], ...]:
        return self.poses

    @property
    def bed_boxes(self) -> tuple[BoundingBox, ...]:
        return self.regions[0]

    @property
    def bed_exit_statuses(self) -> tuple[object, ...]:
        return self.regions[1]

    @classmethod
    def from_detection_result(cls, result: DetectionResult) -> FrameObservation:
        return cls(
            detections=(result.boxes, result.labels),
            poses=result.keypoints,
            regions=(result.bed_boxes, result.bed_exit_statuses),
        )

    def to_detection_result(self) -> DetectionResult:
        boxes, labels = self.detections
        bed_boxes, bed_exit_statuses = self.regions
        return DetectionResult(
            boxes=boxes,
            labels=labels,
            keypoints=self.poses,
            bed_boxes=bed_boxes,
            bed_exit_statuses=bed_exit_statuses,
        )


class BedRegionCacheState(StrEnum):
    """Provenance of the bed ROI applied to a frame (dev overlay + ops telemetry)."""

    FRESH = "fresh"
    CACHED = "cached"
    EMPTY = "empty"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class BedRegionDebugSnapshot:
    """Per-frame bed-ROI cache provenance for the dev overlay and ops telemetry."""

    source: BedRegionCacheState
    age_frames: int | None = None
    empty_cycles: int = 0
    reset_reason: str | None = None
