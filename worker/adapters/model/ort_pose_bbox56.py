"""CPU ONNX Runtime runner for verified pose+bbox56 fall bundles."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from contracts.model_selection import POSE_BBOX56_PREPROCESSING_IDENTITY
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.pose_bbox56_bundle_support import (
    member_digest,
    read_json,
    verify_bundle,
)
from worker.interfaces.fall_model import FallV2Probabilities
from worker.types import FallModelInput

_SHAPE: Final = (30, 56)
_CPU_PROVIDER: Final = ("CPUExecutionProvider",)


class _OrtSession(Protocol):
    def run(
        self, output_names: Sequence[str] | None, input_feed: dict[str, np.ndarray]
    ) -> Sequence[object]: ...


SessionFactory = Callable[[str, list[str]], _OrtSession]


@dataclass(frozen=True)
class PackagedFallBundle:
    """The verified CPU runner and its published identity fields."""

    runner: OrtPoseBbox56Runner
    published_weights_digest: str
    preprocessing_identity: str


class OrtPoseBbox56Runner:
    """Verified ONNX pose+bbox56 binary proxy running exclusively on CPU."""

    device: Final[str] = "cpu"

    def __init__(
        self,
        session: _OrtSession,
        temperature: float,
        receipt_threshold: float | None,
        promotion_eligible: bool,
        artifact_digest: str,
        preprocessing_identity: str,
    ) -> None:
        self._session = session
        self._temperature = temperature
        self.receipt_threshold = receipt_threshold
        self.promotion_eligible = promotion_eligible
        self.artifact_digest = artifact_digest
        self.preprocessing_identity = preprocessing_identity

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        device: str = "cpu",
        *,
        providers: Sequence[str] = _CPU_PROVIDER,
        session_factory: SessionFactory | None = None,
    ) -> OrtPoseBbox56Runner:
        if device != "cpu":
            raise ModelLoadError("pose-bbox56 ONNX bundle is pinned to cpu")
        if tuple(providers) != _CPU_PROVIDER:
            raise ModelLoadError("pose-bbox56 ONNX bundle requires CPUExecutionProvider only")
        root = Path(artifact_dir).expanduser().resolve()
        manifest = read_json(root / "bundle-manifest.json")
        verify_bundle(root, manifest)
        artifact_digest = member_digest(manifest, "model.onnx")
        temperature, receipt_threshold, promotion_eligible = _bundle_metadata(root)
        if session_factory is None:
            session_factory = _onnxruntime_session_factory
        try:
            session = session_factory(str(root / "model.onnx"), list(_CPU_PROVIDER))
        except Exception as exc:
            raise ModelLoadError(f"cannot load pose-bbox56 ONNX model: {exc}") from exc
        runner = cls(
            session,
            temperature,
            receipt_threshold,
            promotion_eligible,
            artifact_digest,
            POSE_BBOX56_PREPROCESSING_IDENTITY,
        )
        runner.warmup()
        return runner

    def predict(self, features: FallModelInput) -> FallV2Probabilities:
        values = np.asarray(features, dtype=np.float32)
        if values.shape != _SHAPE or not np.isfinite(values).all():
            raise ModelLoadError("pose-bbox56 input must be finite shape (30, 56)")
        try:
            outputs = self._session.run(None, {"window": values[np.newaxis, ...]})
        except Exception as exc:
            raise ModelLoadError(f"cannot run pose-bbox56 ONNX model: {exc}") from exc
        if len(outputs) != 1:
            raise ModelLoadError("pose-bbox56 ONNX model must return one output")
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.shape != (1, 1) or not np.isfinite(logits).all():
            raise ModelLoadError("pose-bbox56 ONNX model must return finite shape (1, 1)")
        fall_transition = float(1.0 / (1.0 + np.exp(-logits[0, 0] / self._temperature)))
        return FallV2Probabilities(
            background=1.0 - fall_transition,
            fall_transition=fall_transition,
            fallen=0.0,
        )

    def warmup(self) -> None:
        self.predict(np.zeros(_SHAPE, dtype=np.float32))


def load_packaged_fall_bundle(artifact_dir: Path) -> PackagedFallBundle:
    """Load a packaged fall bundle and its published-weights identity."""
    root = artifact_dir.expanduser().resolve()
    runner = OrtPoseBbox56Runner.from_artifact_dir(root, device="cpu")
    manifest = read_json(root / "bundle-manifest.json")
    return PackagedFallBundle(
        runner,
        member_digest(manifest, "model.pt"),
        runner.preprocessing_identity,
    )


def _onnxruntime_session_factory(model_path: str, providers: list[str]) -> _OrtSession:
    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelLoadError("onnxruntime is required for pose-bbox56 ONNX bundles") from exc
    return onnxruntime.InferenceSession(model_path, providers=providers)


def _bundle_metadata(root: Path) -> tuple[float, float | None, bool]:
    calibration = read_json(root / "calibration.json")
    receipt = read_json(root / "evaluation-receipt.json")
    if not isinstance(calibration, dict):
        raise ModelLoadError("invalid calibration.json")
    temperature = calibration.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ModelLoadError("calibration temperature must be numeric")
    parsed_temperature = float(temperature)
    if not math.isfinite(parsed_temperature) or parsed_temperature <= 0:
        raise ModelLoadError("calibration temperature must be finite and positive")
    if not isinstance(receipt, dict):
        raise ModelLoadError("invalid evaluation-receipt.json")
    candidate_threshold = receipt.get("threshold", calibration.get("threshold"))
    receipt_threshold = (
        float(candidate_threshold)
        if isinstance(candidate_threshold, (int, float))
        and not isinstance(candidate_threshold, bool)
        and 0.0 <= float(candidate_threshold) <= 1.0
        else None
    )
    promotion_eligible = receipt.get("promotion_eligible", calibration.get("promotion_eligible"))
    if not isinstance(promotion_eligible, bool):
        raise ModelLoadError("receipt promotion_eligible must be boolean")
    return parsed_temperature, receipt_threshold, promotion_eligible


__all__ = [
    "OrtPoseBbox56Runner",
    "PackagedFallBundle",
    "SessionFactory",
    "load_packaged_fall_bundle",
]
