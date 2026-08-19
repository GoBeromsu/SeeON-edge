from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypeAlias, final

from contracts.artifacts import pose_weight_path
from contracts.runner import Image, PoseRunnerResult, pose_result
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
from worker.adapters.model.yolo_batch import predict_many

PoseDetections: TypeAlias = tuple[tuple[tuple[int, int, float], ...], ...]
PoseBoxes: TypeAlias = tuple[tuple[int, int, int, int, float], ...]
POSE_MODEL_CONFIDENCE: Final = 0.05
COCO_KEYPOINT_COUNT: Final = 17
POSE_MODEL_PATH: Final = str(pose_weight_path("n"))
POSE_PREPROCESSING_IDENTITY: Final = "rgb24-to-coco17.v1"


@final
class YoloPoseRunner:
    def __init__(
        self,
        model_path: str = POSE_MODEL_PATH,
        confidence: float = POSE_MODEL_CONFIDENCE,
        device: str = "cpu",
        *,
        warmup_frame: WarmupFrameSpec = DEFAULT_WARMUP_FRAME,
        model: YoloModel | None = None,
    ) -> None:
        self.preprocessing_identity = POSE_PREPROCESSING_IDENTITY
        self._model_path: Path = Path(model_path)
        self._artifact_digest: str | None = (
            artifact_digest(self._model_path) if self._model_path.is_file() else None
        )
        self._options: YoloPredictOptions = YoloPredictOptions(
            task="pose",
            confidence=confidence,
            device=device,
        )
        self._warmup_frame: WarmupFrameSpec = warmup_frame
        self._model: YoloModel | None = model

    def predict_full(self, frame: Image) -> PoseRunnerResult:
        result = predict_one(self._get_model(), frame, self._options)
        try:
            return _pose_runner_result(result)
        except (AttributeError, IndexError, OverflowError, TypeError, ValueError) as exc:
            raise YoloOutputError(task="pose", detail=str(exc)) from exc

    def run(self, image: Image) -> PoseRunnerResult:
        return self.predict_full(image)

    def run_batch(self, images: Sequence[Image]) -> tuple[PoseRunnerResult, ...]:
        """One batched forward over ``images``, results in input order.

        The batch shares this runner's single model instance and predict
        options, so a row's result is the same result it would get alone
        (pinned by tests/test_serving_batch_parity.py).
        """
        if not images:
            return ()
        results = predict_many(self._get_model(), images, self._options)
        try:
            return tuple(_pose_runner_result(result) for result in results)
        except (AttributeError, IndexError, OverflowError, TypeError, ValueError) as exc:
            raise YoloOutputError(task="pose", detail=str(exc)) from exc

    def warmup(self) -> None:
        _ = self.run(synthetic_rgb_frame(self._warmup_frame))

    def _get_model(self) -> YoloModel:
        if self._model is None:
            self._model = load_yolo_model(self._model_path, "pose")
        return self._model


def _pose_runner_result(result: YoloResult) -> PoseRunnerResult:
    poses = _extract_poses(result)
    boxes = _extract_boxes(result)
    return _tag_legacy_pose_shape(poses, boxes)


def _tag_legacy_pose_shape(poses, boxes) -> PoseRunnerResult:
    return pose_result(poses, boxes)


def _extract_poses(result: YoloResult) -> PoseDetections:
    keypoints = result.keypoints
    if keypoints is None or keypoints.xy is None:
        return ()
    xy_values = to_float_array(keypoints.xy)
    confidence_tensor = keypoints.conf
    confidence_values = None if confidence_tensor is None else to_float_array(confidence_tensor)
    poses: list[tuple[tuple[int, int, float], ...]] = []
    for person_index, person_points in enumerate(xy_values):
        if len(person_points) != COCO_KEYPOINT_COUNT:
            raise YoloOutputError(
                task="pose",
                detail=(
                    f"expected {COCO_KEYPOINT_COUNT} COCO keypoints, "
                    f"received {len(person_points)}"
                ),
            )
        points: list[tuple[int, int, float]] = []
        for point_index, point in enumerate(person_points):
            confidence = (
                1.0
                if confidence_values is None
                else float(confidence_values[person_index][point_index])
            )
            points.append((int(point[0]), int(point[1]), confidence))
        poses.append(tuple(points))
    return tuple(poses)


def _extract_boxes(result: YoloResult) -> PoseBoxes:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()
    xyxy = to_float_array(boxes.xyxy)
    confidence_values = to_float_array(boxes.conf)
    return tuple(
        (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(confidence))
        for box, confidence in zip(xyxy, confidence_values, strict=True)
    )


__all__ = [
    "COCO_KEYPOINT_COUNT",
    "POSE_MODEL_CONFIDENCE",
    "POSE_MODEL_PATH",
    "PoseBoxes",
    "PoseDetections",
    "YoloPoseRunner",
]
