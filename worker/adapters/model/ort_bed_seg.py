"""CPU-only ONNX Runtime runner for the YOLO26 bed segmentation model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Protocol, final

import numpy as np

from contracts.artifacts import bed_seg_weight_path
from contracts.runner import BedRunnerResult, Image, bed_result
from worker.adapters.model.artifact import verify_artifact_digest
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.seg_postprocess import (
    BedInstance,
    decode_end_to_end_segmentation,
    letterbox_rgb,
)
from worker.adapters.model.warmup import (
    DEFAULT_WARMUP_FRAME,
    WarmupFrameSpec,
    synthetic_rgb_frame,
)

_CPU_PROVIDER: Final = ("CPUExecutionProvider",)
BED_ONNX_MODEL_PATH: Final = str(bed_seg_weight_path().with_suffix(".onnx"))
BED_MODEL_CONFIDENCE: Final = 0.25
BED_MASK_MAX_POINTS: Final = 48
BED_PREPROCESSING_IDENTITY: Final = "rgb24-to-bed-regions.v1"


class _OrtSession(Protocol):
    def run(
        self, output_names: Sequence[str] | None, input_feed: dict[str, np.ndarray]
    ) -> Sequence[object]: ...


SessionFactory = Callable[[str, list[str]], _OrtSession]


@final
class OrtBedSegRunner:
    """Verified end-to-end YOLO26 segmentation runner pinned to ORT CPU."""

    device: Final[str] = "cpu"

    def __init__(
        self,
        model_path: str = BED_ONNX_MODEL_PATH,
        confidence: float = BED_MODEL_CONFIDENCE,
        max_points: int = BED_MASK_MAX_POINTS,
        device: str = "cpu",
        *,
        providers: Sequence[str] = _CPU_PROVIDER,
        session_factory: SessionFactory | None = None,
        warmup_frame: WarmupFrameSpec = DEFAULT_WARMUP_FRAME,
    ) -> None:
        if device != self.device:
            raise ModelLoadError("bed segmentation ONNX model is pinned to cpu")
        if tuple(providers) != _CPU_PROVIDER:
            raise ModelLoadError("bed segmentation ONNX model requires CPUExecutionProvider only")
        if not 0.0 <= confidence <= 1.0:
            raise ModelLoadError("bed segmentation confidence must be in [0, 1]")
        if max_points <= 0:
            raise ModelLoadError("bed segmentation max_points must be positive")
        self.preprocessing_identity = BED_PREPROCESSING_IDENTITY
        self._model_path = Path(model_path).expanduser().resolve()
        if not self._model_path.is_file():
            raise ModelLoadError(f"bed segmentation ONNX model does not exist: {self._model_path}")
        self.artifact_digest = verify_artifact_digest(
            self._model_path, _read_digest_sidecar(self._model_path)
        )
        self._confidence = confidence
        self._max_points = max_points
        self._warmup_frame = warmup_frame
        if session_factory is None:
            session_factory = _onnxruntime_session_factory
        try:
            self._session = session_factory(str(self._model_path), list(_CPU_PROVIDER))
        except Exception as exc:
            raise ModelLoadError(f"cannot load bed segmentation ONNX model: {exc}") from exc

    def detect_beds(self, frame: Image) -> BedRunnerResult:
        tensor, letterbox = letterbox_rgb(frame)
        try:
            outputs = self._session.run(None, {"images": tensor})
        except Exception as exc:
            raise ModelLoadError(f"cannot run bed segmentation ONNX model: {exc}") from exc
        if len(outputs) != 2:
            raise ModelLoadError("bed segmentation ONNX model must return two outputs")
        try:
            boxes: tuple[BedInstance, ...] = decode_end_to_end_segmentation(
                outputs[0],
                outputs[1],
                letterbox,
                confidence=self._confidence,
                max_points=self._max_points,
            )
        except (TypeError, ValueError) as exc:
            raise ModelLoadError(f"invalid bed segmentation ONNX output: {exc}") from exc
        return bed_result(boxes)

    def run(self, image: Image) -> BedRunnerResult:
        return self.detect_beds(image)

    def warmup(self) -> None:
        self.run(synthetic_rgb_frame(self._warmup_frame))


def _read_digest_sidecar(model_path: Path) -> str:
    sidecar = model_path.with_suffix(model_path.suffix + ".sha256")
    try:
        digest = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise ModelLoadError(f"cannot read bed segmentation digest sidecar: {sidecar}") from exc
    if not digest.endswith("\n") or digest.count("\n") != 1:
        raise ModelLoadError("bed segmentation digest sidecar must contain one SHA-256 digest")
    return digest[:-1]


def _onnxruntime_session_factory(model_path: str, providers: list[str]) -> _OrtSession:
    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelLoadError("onnxruntime is required for bed segmentation ONNX model") from exc
    return onnxruntime.InferenceSession(model_path, providers=providers)


__all__ = ["BED_ONNX_MODEL_PATH", "OrtBedSegRunner", "SessionFactory"]
