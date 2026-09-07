"""CPU-only ONNX person-box inference for stored evidence clips."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from contracts.runner import Image
from worker.adapters.model.artifact import verify_artifact_digest
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_bed_seg import _read_digest_sidecar

_CPU_PROVIDER = ("CPUExecutionProvider",)
_NET_SIZE = 640


class _Input(Protocol):
    name: str


class _Session(Protocol):
    def get_inputs(self) -> Sequence[_Input]: ...

    def run(
        self, output_names: Sequence[str] | None, input_feed: dict[str, NDArray[np.float32]]
    ) -> Sequence[object]: ...


SessionFactory = Callable[[str, list[str]], _Session]


class OrtClipPoseRunner:
    """Verified YOLO26 pose model used only to publish person boxes."""

    def __init__(
        self, model_path: Path, threshold: float, *, session_factory: SessionFactory | None = None
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ModelLoadError("clip pose threshold must be in [0, 1]")
        self._model_path = model_path.expanduser().resolve()
        if not self._model_path.is_file():
            raise ModelLoadError("clip pose ONNX model does not exist")
        self.artifact_digest = verify_artifact_digest(
            self._model_path, _read_digest_sidecar(self._model_path)
        )
        if session_factory is None:
            session_factory = _onnxruntime_session_factory
        try:
            self._session = session_factory(str(self._model_path), list(_CPU_PROVIDER))
            self._input_name = self._session.get_inputs()[0].name
        except (IndexError, TypeError, ValueError) as exc:
            raise ModelLoadError("clip pose ONNX model input is invalid") from exc
        except Exception as exc:
            raise ModelLoadError("cannot load clip pose ONNX model") from exc
        self._threshold = threshold

    def detect_persons(self, image: Image) -> tuple[tuple[float, float, float, float, float], ...]:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ModelLoadError("clip pose image must be uint8 RGB")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ModelLoadError("clip pose image dimensions must be positive")
        scale = min(_NET_SIZE / width, _NET_SIZE / height)
        resized = _resize_rgb(image, round(width * scale), round(height * scale))
        canvas = np.zeros((_NET_SIZE, _NET_SIZE, 3), dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        tensor = np.transpose(canvas, (2, 0, 1))[None].astype(np.float32) / 255.0
        try:
            outputs = self._session.run(["output0"], {self._input_name: tensor})
            rows = np.asarray(outputs[0], dtype=np.float32)
        except Exception as exc:
            raise ModelLoadError("cannot run clip pose ONNX model") from exc
        if rows.shape != (1, 300, 57):
            raise ModelLoadError("clip pose ONNX output0 must have shape [1, 300, 57]")
        boxes: list[tuple[float, float, float, float, float]] = []
        for row in rows[0]:
            if row[5] != 0 or row[4] < self._threshold:
                continue
            x1 = max(0.0, min(float(width), float(row[0] / scale)))
            y1 = max(0.0, min(float(height), float(row[1] / scale)))
            x2 = max(0.0, min(float(width), float(row[2] / scale)))
            y2 = max(0.0, min(float(height), float(row[3] / scale)))
            if x1 < x2 and y1 < y2:
                boxes.append((x1, y1, x2, y2, float(row[4])))
        return tuple(boxes)


def _resize_rgb(image: Image, width: int, height: int) -> Image:
    source_height, source_width = image.shape[:2]
    y = np.minimum((np.arange(height) * source_height / height).astype(int), source_height - 1)
    x = np.minimum((np.arange(width) * source_width / width).astype(int), source_width - 1)
    return image[y[:, None], x]


def _onnxruntime_session_factory(model_path: str, providers: list[str]) -> _Session:
    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelLoadError("onnxruntime is required for clip pose ONNX model") from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(model_path, sess_options=options, providers=providers)


__all__ = ["OrtClipPoseRunner", "SessionFactory"]
