"""Ultralytics batch-forward adapter shared by batched model runners."""

from __future__ import annotations

from collections.abc import Sequence

from contracts.runner import Image
from worker.adapters.model.yolo_api import (
    YoloModel,
    YoloOutputError,
    YoloPredictOptions,
    YoloResult,
    classify_forward_error,
)


def predict_many(
    model: YoloModel,
    frames: Sequence[Image],
    options: YoloPredictOptions,
) -> tuple[YoloResult, ...]:
    """Run one list-source forward and preserve positional result mapping."""
    try:
        results = model.predict(
            source=list(frames),
            conf=options.confidence,
            verbose=False,
            device=options.device,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        classify_forward_error(exc, task=options.task, camera_id="")
    if len(results) != len(frames):
        raise YoloOutputError(
            task=options.task,
            detail=f"expected {len(frames)} batched results, received {len(results)}",
        )
    return tuple(results)


__all__ = ["predict_many"]
