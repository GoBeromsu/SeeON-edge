from __future__ import annotations

import importlib
import json
import math
import pickle
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, final, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from worker.adapters.model.artifact import verify_artifact_digest
from worker.adapters.model.errors import ModelInputError, ModelLoadError
from worker.adapters.model.sklearn_metadata import (
    DEFAULT_OPERATING_THRESHOLD,
    EXPECTED_FEATURE_DIM,
    EXPECTED_STRIDE,
    EXPECTED_WINDOW,
    JsonValue,
    ModelMetadata,
)

MODELS_DIR: Final = Path(__file__).resolve().parents[3] / "models"


class _ProbabilityMatrix(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: tuple[int, int]) -> np.float64: ...


class _ProbabilityModel(Protocol):
    n_features_in_: int

    def predict_proba(self, values: NDArray[np.float32]) -> _ProbabilityMatrix: ...


@runtime_checkable
class _JoblibModule(Protocol):
    def load(self, filename: Path) -> _ProbabilityModel: ...


@runtime_checkable
class _JsonModule(Protocol):
    def loads(self, value: str) -> JsonValue: ...


@final
class FallDetector:
    def __init__(
        self,
        model_type: str = "random-forest",
        models_dir: Path | None = None,
        *,
        expected_artifact_digest: str | None = None,
    ) -> None:
        if model_type != "random-forest":
            raise ModelLoadError(f"unsupported model_type {model_type!r}")
        self.model_type = model_type
        root = MODELS_DIR if models_dir is None else models_dir
        self.artifact_dir = root / "fall" / model_type
        self.model_path = self.artifact_dir / "model.pkl"
        self.metadata_path = self.artifact_dir / "metadata.json"
        self.metadata = self._load_metadata()
        if not self.model_path.is_file():
            raise ModelLoadError(f"missing model.pkl at {self.model_path}")
        declared_digest = self.metadata.artifact_digest
        if (
            declared_digest is not None
            and expected_artifact_digest is not None
            and declared_digest != expected_artifact_digest
        ):
            raise ModelLoadError(
                "configured artifact digest does not match metadata artifact digest"
            )
        self.artifact_digest = verify_artifact_digest(
            self.model_path,
            expected_artifact_digest if expected_artifact_digest is not None else declared_digest,
        )
        self.model = _load_joblib_model(self.model_path)
        self._validate_model_shape()

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def operating_threshold(self) -> float:
        threshold = self.metadata.operating_threshold
        return DEFAULT_OPERATING_THRESHOLD if threshold is None else threshold

    def metadata_dict(self) -> dict[str, JsonValue]:
        metadata = self.metadata.asdict()
        metadata["artifact_digest"] = self.artifact_digest
        return metadata

    def predict(self, features: Sequence[float] | NDArray[np.float32]) -> float:
        values = np.asarray(features, dtype=np.float32)
        if values.size == 0:
            raise ModelInputError("feature vector must be non-empty")
        values = values.reshape(1, -1)
        if values.shape[1] != self.metadata.feature_dim:
            raise ModelInputError(
                f"expected {self.metadata.feature_dim} features, received {values.shape[1]}"
            )
        probabilities = self.model.predict_proba(values)
        if probabilities.shape != (1, 2):
            raise ModelInputError(
                f"predict_proba must return shape (1, 2), received {probabilities.shape}"
            )
        probability = float(probabilities[0, 1])
        if not math.isfinite(probability):
            raise ModelInputError("predict_proba returned a non-finite probability")
        return min(max(probability, 0.0), 1.0)

    def warmup(self) -> None:
        _ = self.predict(np.zeros((self.metadata.feature_dim,), dtype=np.float32))

    def _load_metadata(self) -> ModelMetadata:
        if not self.metadata_path.is_file():
            raise ModelLoadError(f"missing metadata.json at {self.metadata_path}")
        try:
            module: ModuleType = json
            if not isinstance(module, _JsonModule):
                raise ModelLoadError("json does not expose a compatible loads function")
            raw = module.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"cannot read metadata.json at {self.metadata_path}") from exc
        if not isinstance(raw, dict):
            raise ModelLoadError("metadata.json must contain an object")
        return ModelMetadata.from_dict(raw)

    def _validate_model_shape(self) -> None:
        if self.model.n_features_in_ != self.metadata.feature_dim:
            message = (
                f"model feature_dim mismatch: metadata={self.metadata.feature_dim} "
                f"artifact={self.model.n_features_in_}"
            )
            raise ModelLoadError(message)


def _load_joblib_model(path: Path) -> _ProbabilityModel:
    try:
        module: ModuleType = importlib.import_module("joblib")
        if not isinstance(module, _JoblibModule):
            raise ModelLoadError("joblib does not expose a compatible load function")
        return module.load(path)
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
        KeyError,
        pickle.UnpicklingError,
    ) as exc:
        raise ModelLoadError(f"cannot load model.pkl at {path}") from exc


__all__ = [
    "DEFAULT_OPERATING_THRESHOLD",
    "EXPECTED_FEATURE_DIM",
    "EXPECTED_STRIDE",
    "EXPECTED_WINDOW",
    "MODELS_DIR",
    "FallDetector",
    "ModelInputError",
    "ModelLoadError",
    "ModelMetadata",
]
