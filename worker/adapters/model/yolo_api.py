from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    TypeAlias,
    assert_never,
    final,
    runtime_checkable,
)

import numpy as np
import torch
from numpy.typing import NDArray
from typing_extensions import override

from contracts.runner import Image
from worker.adapters.model.errors import FatalAcceleratorError

if TYPE_CHECKING:
    from ultralytics import YOLO

YoloTask: TypeAlias = Literal["pose", "person", "bed"]


@dataclass(slots=True)
class YoloArtifactError(RuntimeError):
    task: YoloTask
    path: Path

    @override
    def __str__(self) -> str:
        return f"{self.task} model artifact is missing: {self.path}"


@dataclass(slots=True)
class YoloLoadError(RuntimeError):
    task: YoloTask
    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.task} model load failed for {self.path}: {self.reason}"


@dataclass(slots=True)
class YoloForwardError(RuntimeError):
    task: YoloTask
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.task} model forward failed: {self.reason}"


@dataclass(slots=True)
class YoloOutputError(RuntimeError):
    task: YoloTask
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.task} model output is malformed: {self.detail}"


@runtime_checkable
class YoloTensor(Protocol):
    def cpu(self) -> YoloTensor: ...

    def numpy(self) -> NDArray[np.float32] | NDArray[np.float64]: ...


YoloArray: TypeAlias = (
    torch.Tensor | NDArray[np.float32] | NDArray[np.float64] | YoloTensor
)


class YoloBoxes(Protocol):
    @property
    def xyxy(self) -> YoloArray: ...

    @property
    def conf(self) -> YoloArray: ...

    @property
    def cls(self) -> YoloArray: ...

    def __len__(self) -> int: ...


class YoloKeypoints(Protocol):
    @property
    def xy(self) -> YoloArray | None: ...

    @property
    def conf(self) -> YoloArray | None: ...


class YoloMasks(Protocol):
    @property
    def xy(self) -> Sequence[NDArray[np.float32] | NDArray[np.float64]]: ...


class YoloResult(Protocol):
    @property
    def boxes(self) -> YoloBoxes | None: ...

    @property
    def keypoints(self) -> YoloKeypoints | None: ...

    @property
    def masks(self) -> YoloMasks | None: ...


class YoloModel(Protocol):
    @property
    def names(self) -> Mapping[int, str]: ...

    def predict(
        self,
        *,
        source: Image,
        conf: float,
        verbose: bool,
        device: str,
    ) -> Sequence[YoloResult]: ...


@dataclass(frozen=True, slots=True)
class YoloPredictOptions:
    task: YoloTask
    confidence: float
    device: str


@final
class _UltralyticsModel:
    def __init__(self, model: YOLO) -> None:
        self._model: YOLO = model

    @property
    def names(self) -> Mapping[int, str]:
        return self._model.names

    def predict(
        self,
        *,
        source: Image,
        conf: float,
        verbose: bool,
        device: str,
    ) -> Sequence[YoloResult]:
        return self._model.predict(
            source=source,
            conf=conf,
            verbose=verbose,
            device=device,
        )


def load_yolo_model(path: Path, task: YoloTask) -> YoloModel:
    if not path.is_file():
        raise YoloArtifactError(task=task, path=path)
    try:
        from ultralytics import YOLO

        model = _UltralyticsModel(YOLO(str(path)))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise YoloLoadError(task=task, path=path, reason=str(exc)) from exc
    return model


def to_float_array(tensor: YoloArray) -> NDArray[np.float64]:
    match tensor:
        case np.ndarray():
            values = tensor
        case torch.Tensor():
            values = tensor.detach().cpu().numpy()
        case YoloTensor():
            values = tensor.cpu().numpy()
        case unreachable:
            assert_never(unreachable)
    return np.asarray(values, dtype=np.float64)


def predict_one(
    model: YoloModel,
    frame: Image,
    options: YoloPredictOptions,
    *,
    camera_id: str = "",
) -> YoloResult:
    try:
        results = model.predict(
            source=frame,
            conf=options.confidence,
            verbose=False,
            device=options.device,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _classify_or_reraise(exc, task=options.task, camera_id=camera_id)
    if len(results) != 1:
        raise YoloOutputError(
            task=options.task,
            detail=f"expected one result, received {len(results)}",
        )
    return results[0]


_CUDA_KEYWORDS: tuple[str, ...] = (
    "cuda",
    "device-side assert",
    "illegal memory access",
    "device lost",
    "CUDA error",
    "cuLaunch",
    "cuDNN",
)


def _classify_or_reraise(exc: Exception, *, task: str, camera_id: str) -> None:
    """Re-raise exc as FatalAcceleratorError when it is a CUDA fault.

    Non-CUDA exceptions are wrapped in YoloForwardError (ordinary per-camera
    processing failure — isolated, never fatal).
    """
    msg = str(exc).lower()
    if any(kw.lower() in msg for kw in _CUDA_KEYWORDS):
        raise FatalAcceleratorError(
            f"{task} model CUDA fault: {exc}",
            camera_id=camera_id,
            task=task,
        ) from exc
    raise YoloForwardError(task=task, reason=str(exc)) from exc  # type: ignore[name-defined]


__all__ = [
    "YoloArtifactError",
    "YoloArray",
    "YoloBoxes",
    "YoloForwardError",
    "YoloKeypoints",
    "YoloLoadError",
    "YoloMasks",
    "YoloModel",
    "YoloOutputError",
    "YoloPredictOptions",
    "YoloResult",
    "YoloTask",
    "YoloTensor",
    "load_yolo_model",
    "predict_one",
    "to_float_array",
]
