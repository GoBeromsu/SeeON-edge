from __future__ import annotations

from pathlib import Path
from typing import Final, TypeAlias, final

from contracts.runner import Image, PersonRunnerResult, person_result
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

PERSON_WEIGHTS_DIR: Final = Path(__file__).resolve().parents[3] / "models" / "person"
PERSON_MODEL_FILENAME: Final = "yolo26n.pt"
PERSON_MODEL_PATH: Final = str(PERSON_WEIGHTS_DIR / PERSON_MODEL_FILENAME)
PERSON_PREPROCESSING_IDENTITY: Final = "rgb24-to-person-boxes.v1"
PERSON_MODEL_CONFIDENCE: Final = 0.25
COCO_PERSON_CLASS_ID: Final = 0

PersonBoxes: TypeAlias = tuple[tuple[int, int, int, int, float], ...]


@final
class YoloPersonRunner:
    def __init__(
        self,
        model_path: str = PERSON_MODEL_PATH,
        confidence: float = PERSON_MODEL_CONFIDENCE,
        device: str = "cpu",
        *,
        warmup_frame: WarmupFrameSpec = DEFAULT_WARMUP_FRAME,
        model: YoloModel | None = None,
    ) -> None:
        self.preprocessing_identity = PERSON_PREPROCESSING_IDENTITY
        self._model_path: Path = Path(model_path)
        self._artifact_digest: str | None = (
            artifact_digest(self._model_path) if self._model_path.is_file() else None
        )
        self._options: YoloPredictOptions = YoloPredictOptions(
            task="person",
            confidence=confidence,
            device=device,
        )
        self._warmup_frame: WarmupFrameSpec = warmup_frame
        self._model: YoloModel | None = model

    def predict(self, frame: Image) -> PersonRunnerResult:
        result = predict_one(self._get_model(), frame, self._options)
        try:
            boxes = _extract_person_boxes(result, self._options.confidence)
        except (AttributeError, IndexError, OverflowError, TypeError, ValueError) as exc:
            raise YoloOutputError(task="person", detail=str(exc)) from exc
        return person_result(boxes)

    def run(self, image: Image) -> PersonRunnerResult:
        return self.predict(image)

    def warmup(self) -> None:
        _ = self.run(synthetic_rgb_frame(self._warmup_frame))

    def _get_model(self) -> YoloModel:
        if self._model is None:
            self._model = load_yolo_model(self._model_path, "person")
        return self._model


def _extract_person_boxes(result: YoloResult, confidence: float) -> PersonBoxes:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()
    xyxy = to_float_array(boxes.xyxy)
    confidence_values = to_float_array(boxes.conf)
    classes = to_float_array(boxes.cls)
    return tuple(
        (int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(score))
        for box, score, class_id in zip(xyxy, confidence_values, classes, strict=True)
        if int(class_id) == COCO_PERSON_CLASS_ID and float(score) >= confidence
    )


__all__ = [
    "COCO_PERSON_CLASS_ID",
    "PERSON_MODEL_CONFIDENCE",
    "PERSON_MODEL_FILENAME",
    "PERSON_MODEL_PATH",
    "PERSON_WEIGHTS_DIR",
    "PersonBoxes",
    "YoloPersonRunner",
]
