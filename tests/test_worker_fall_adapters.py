from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import final

import numpy as np
import pytest
from numpy.typing import NDArray

from worker.adapters.model import sklearn_fall
from worker.adapters.model.registry import UnknownModelTaskError, default_registry
from worker.adapters.model.sklearn_fall import (
    DEFAULT_OPERATING_THRESHOLD,
    EXPECTED_FEATURE_DIM,
    FallDetector,
    ModelInputError,
    ModelLoadError,
)


@final
class _ProbabilityModel:
    n_features_in_: int = EXPECTED_FEATURE_DIM

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def predict_proba(self, values: NDArray[np.float32]) -> NDArray[np.float64]:
        self.shapes.append(tuple(values.shape))
        return np.asarray(((0.25, 0.75),), dtype=np.float64)


def _write_bundle(
    path: Path,
    *,
    feature_dim: int = EXPECTED_FEATURE_DIM,
    operating_threshold: float | None = 0.37,
) -> Path:
    path.mkdir(parents=True)
    metadata: dict[str, str | int | float] = {
        "model_type": "random-forest",
        "framework": "sklearn",
        "window": 30,
        "stride": 5,
        "feature_dim": feature_dim,
        "name": "fall-detector",
        "version": "fixture-v1",
    }
    if operating_threshold is not None:
        metadata["operating_threshold"] = operating_threshold
    with (path / "model.pkl").open("wb") as model_file:
        pickle.dump(_ProbabilityModel(), model_file)
    metadata["artifact_digest"] = hashlib.sha256((path / "model.pkl").read_bytes()).hexdigest()
    _ = (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_default_registry_has_no_fall_fallback() -> None:
    # Given / When / Then
    # The fall model has no registry-backed fallback: it must be explicitly
    # configured (worker.runtime.worker.WorkerRuntime._create_fall_model), so
    # "fall" is not a registered task in the default registry at all.
    with pytest.raises(UnknownModelTaskError):
        default_registry().get_factory("fall")


def test_sklearn_adapter_loads_artifact_and_preserves_feature_provenance(
    tmp_path: Path,
) -> None:
    # Given
    artifact_dir = _write_bundle(tmp_path / "fall" / "random-forest")

    # When
    runner = FallDetector(models_dir=tmp_path)

    # Then
    assert runner.artifact_dir == artifact_dir
    assert runner.metadata.mode == "features"
    assert runner.operating_threshold == 0.37
    assert runner.name == "fall-detector"
    assert runner.version == "fixture-v1"
    assert (
        runner.artifact_digest
        == hashlib.sha256((artifact_dir / "model.pkl").read_bytes()).hexdigest()
    )


def test_sklearn_warmup_runs_one_forward_with_engineered_feature_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    artifact_dir = _write_bundle(tmp_path / "fall" / "random-forest")
    model = _ProbabilityModel()

    def load_model(_: Path) -> _ProbabilityModel:
        return model

    monkeypatch.setattr(sklearn_fall, "_load_joblib_model", load_model)
    runner = FallDetector(models_dir=tmp_path)

    # When
    runner.warmup()

    # Then
    assert artifact_dir.is_dir()
    assert model.shapes == [(1, EXPECTED_FEATURE_DIM)]


def test_sklearn_digest_mismatch_rejects_before_deserialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given
    _ = _write_bundle(tmp_path / "fall" / "random-forest")
    model_load_attempted = False

    def load_model(_: Path) -> _ProbabilityModel:
        nonlocal model_load_attempted
        model_load_attempted = True
        return _ProbabilityModel()

    monkeypatch.setattr(sklearn_fall, "_load_joblib_model", load_model)

    # When / Then
    with pytest.raises(ModelLoadError, match="artifact digest"):
        _ = FallDetector(models_dir=tmp_path, expected_artifact_digest="0" * 64)

    assert not model_load_attempted


def test_sklearn_metadata_rejects_wrong_engineered_feature_dimension(
    tmp_path: Path,
) -> None:
    # Given
    _ = _write_bundle(tmp_path / "fall" / "random-forest", feature_dim=44)

    # When / Then
    with pytest.raises(ModelLoadError, match="feature_dim"):
        _ = FallDetector(models_dir=tmp_path)


def test_sklearn_missing_weight_artifact_fails_during_construction(
    tmp_path: Path,
) -> None:
    # Given
    artifact_dir = _write_bundle(tmp_path / "fall" / "random-forest")
    (artifact_dir / "model.pkl").unlink()

    # When / Then
    with pytest.raises(ModelLoadError, match="missing model.pkl"):
        _ = FallDetector(models_dir=tmp_path)


def test_sklearn_default_threshold_and_typed_shape_error(tmp_path: Path) -> None:
    # Given
    _ = _write_bundle(
        tmp_path / "fall" / "random-forest",
        operating_threshold=None,
    )
    runner = FallDetector(models_dir=tmp_path)

    # When / Then
    assert runner.operating_threshold == DEFAULT_OPERATING_THRESHOLD
    with pytest.raises(ModelInputError, match="expected 45 features"):
        _ = runner.predict(np.zeros((44,), dtype=np.float32))
