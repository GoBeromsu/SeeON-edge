from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeAlias, final

import numpy as np
from numpy.typing import NDArray

from contracts.artifacts import bed_seg_weight_path
from contracts.runner import BedRunnerResult, Image, bed_result
from worker.adapters.model.artifact import artifact_digest
from worker.adapters.model.warmup import (
    DEFAULT_WARMUP_FRAME,
    WarmupFrameSpec,
    synthetic_rgb_frame,
)
from worker.adapters.model.yolo_api import (
    YoloModel,
    YoloOutputError,
    YoloPredictOptions,
    YoloResult,
    load_yolo_model,
    predict_one,
    to_float_array,
)

COCO_BED_CLASS_ID: Final = 59
BED_MODEL_CONFIDENCE: Final = 0.25
BED_NMS_IOU_THRESHOLD: Final = 0.5
BED_MERGE_IOU_THRESHOLD: Final = 0.5
BED_MASK_MAX_POINTS: Final = 48
BED_MODEL_PATH: Final = str(bed_seg_weight_path())
BED_PREPROCESSING_IDENTITY: Final = "rgb24-to-bed-regions.v1"

BedBox: TypeAlias = tuple[int, int, int, int, float]
BedPolygon: TypeAlias = tuple[tuple[int, int], ...]
BedInstance: TypeAlias = tuple[int, int, int, int, float, BedPolygon]


@final
class YoloBedSegRunner:
    def __init__(
        self,
        model_path: str = BED_MODEL_PATH,
        confidence: float = BED_MODEL_CONFIDENCE,
        max_points: int = BED_MASK_MAX_POINTS,
        device: str = "cpu",
        *,
        warmup_frame: WarmupFrameSpec = DEFAULT_WARMUP_FRAME,
        model: YoloModel | None = None,
    ) -> None:
        self.preprocessing_identity = BED_PREPROCESSING_IDENTITY
        self._model_path: Path = Path(model_path)
        self._artifact_digest: str | None = (
            artifact_digest(self._model_path) if self._model_path.is_file() else None
        )
        self._options: YoloPredictOptions = YoloPredictOptions(
            task="bed",
            confidence=confidence,
            device=device,
        )
        self._max_points: int = max_points
        self._warmup_frame: WarmupFrameSpec = warmup_frame
        self._model: YoloModel | None = model

    def detect_beds(self, frame: Image) -> BedRunnerResult:
        model = self._get_model()
        result = predict_one(model, frame, self._options)
        bed_class_id = _resolve_bed_class_id(model.names)
        if bed_class_id is None:
            return bed_result(())
        try:
            boxes = _extract_bed_instances(
                result,
                bed_class_id=bed_class_id,
                confidence=self._options.confidence,
                max_points=self._max_points,
            )
        except (AttributeError, IndexError, OverflowError, TypeError, ValueError) as exc:
            raise YoloOutputError(task="bed", detail=str(exc)) from exc
        return bed_result(boxes)

    def run(self, image: Image) -> BedRunnerResult:
        return self.detect_beds(image)

    def warmup(self) -> None:
        _ = self.run(synthetic_rgb_frame(self._warmup_frame))

    def _get_model(self) -> YoloModel:
        if self._model is None:
            self._model = load_yolo_model(self._model_path, "bed")
        return self._model


def _extract_bed_instances(
    result: YoloResult,
    *,
    bed_class_id: int,
    confidence: float,
    max_points: int,
) -> tuple[BedInstance, ...]:
    boxes = result.boxes
    masks = result.masks
    if boxes is None or len(boxes) == 0 or masks is None:
        return ()
    xyxy = to_float_array(boxes.xyxy)
    confidence_values = to_float_array(boxes.conf)
    classes = to_float_array(boxes.cls)
    return tuple(
        (
            int(box[0]),
            int(box[1]),
            int(box[2]),
            int(box[3]),
            float(score),
            _simplify_polygon(polygon, max_points),
        )
        for box, score, class_id, polygon in zip(
            xyxy,
            confidence_values,
            classes,
            masks.xy,
            strict=True,
        )
        if int(class_id) == bed_class_id and float(score) >= confidence
    )


def _simplify_polygon(
    points: NDArray[np.float32] | NDArray[np.float64],
    max_points: int,
) -> BedPolygon:
    values: NDArray[np.float64] = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return ()
    if values.shape[0] > max_points:
        indices: NDArray[np.int64] = np.linspace(
            0,
            values.shape[0] - 1,
            max_points,
        ).astype(np.int64)
        values = values[indices]
    return tuple(
        (
            round(float(values[index, 0])),
            round(float(values[index, 1])),
        )
        for index in range(values.shape[0])
    )


def dedupe_bed_boxes(
    boxes: tuple[BedBox, ...],
    *,
    max_beds: int | None = None,
) -> tuple[BedBox, ...]:
    if not boxes or (max_beds is not None and max_beds <= 0):
        return ()
    kept: list[BedBox] = []
    for box in sorted(boxes, key=lambda candidate: (-candidate[4], *candidate[:4])):
        if all(_box_iou(box, existing) < BED_NMS_IOU_THRESHOLD for existing in kept):
            kept.append(box)
    merged = _merge_overlapping_beds(tuple(kept))
    ordered = sorted(merged, key=lambda candidate: (*candidate[:4], -candidate[4]))
    return tuple(ordered if max_beds is None else ordered[:max_beds])


def _merge_overlapping_beds(boxes: tuple[BedBox, ...]) -> tuple[BedBox, ...]:
    clusters: list[list[BedBox]] = []
    for box in sorted(boxes, key=lambda candidate: (*candidate[:4], -candidate[4])):
        for cluster in clusters:
            if any(_box_iou(box, existing) >= BED_MERGE_IOU_THRESHOLD for existing in cluster):
                cluster.append(box)
                break
        else:
            clusters.append([box])
    return tuple(
        (
            min(box[0] for box in cluster),
            min(box[1] for box in cluster),
            max(box[2] for box in cluster),
            max(box[3] for box in cluster),
            max(box[4] for box in cluster),
        )
        for cluster in clusters
    )


def _box_iou(first: BedBox, second: BedBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _resolve_bed_class_id(names: Mapping[int, str]) -> int | None:
    if names.get(COCO_BED_CLASS_ID, "").casefold() == "bed":
        return COCO_BED_CLASS_ID
    return next(
        (class_id for class_id, name in names.items() if name.casefold() == "bed"),
        None,
    )


__all__ = [
    "BED_MASK_MAX_POINTS",
    "BED_MERGE_IOU_THRESHOLD",
    "BED_MODEL_CONFIDENCE",
    "BED_MODEL_PATH",
    "BED_NMS_IOU_THRESHOLD",
    "COCO_BED_CLASS_ID",
    "BedBox",
    "BedInstance",
    "BedPolygon",
    "YoloBedSegRunner",
    "dedupe_bed_boxes",
]
