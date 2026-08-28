from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from worker.adapters.model import torch_lstm_fall
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.torch_lstm_fall import (
    CURRENT_PREPROCESSING_IDENTITY,
    LEGACY_PREPROCESSING_IDENTITY,
    LstmFallManifest,
    LstmFallRunner,
    build_lstm_module,
)
from worker.runtime.config.worker_models import FallModelConfig


def _write_bundle(
    path: Path,
    *,
    schema_version: int | None = 2,
    preprocessing_identity: str | None = CURRENT_PREPROCESSING_IDENTITY,
) -> Path:
    path.mkdir()
    (path / "model.pt").write_bytes(b"placeholder")
    (path / "arch.json").write_text(
        json.dumps({"hidden": 4, "layers": 1, "dropout": 0.0}), encoding="utf-8"
    )
    manifest: dict[str, object] = {
        "type": "lstm",
        "framework": "pytorch",
        "mode": "sequence",
        "artifact_dir": str(path),
        "weights": "model.pt",
        "architecture": "arch.json",
        "metadata": "metadata.yaml",
        "window": 3,
        "stride": 1,
        "input_shape": [3, 51],
        "operating_threshold": 0.5,
    }
    if schema_version is not None:
        manifest["schema_version"] = schema_version
    if preprocessing_identity is not None:
        manifest["preprocessing_identity"] = preprocessing_identity
    (path / "metadata.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_current_bundle_loads_without_real_weight_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _write_bundle(tmp_path / "current")
    module = build_lstm_module(hidden=4, layers=1, dropout=0.0)
    monkeypatch.setattr(torch_lstm_fall, "_load_state_dict", lambda _: module.state_dict())

    runner = LstmFallRunner.from_artifact_dir(bundle)

    assert runner.schema_version == 2
    assert runner.preprocessing_identity == CURRENT_PREPROCESSING_IDENTITY


def test_worker_config_accepts_bundle_identity_fields(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "config")
    config = FallModelConfig.model_validate(
        {
            "type": "lstm",
            "framework": "pytorch",
            "mode": "sequence",
            "artifact_dir": bundle,
            "window": 3,
            "stride": 1,
            "input_shape": [3, 51],
            "operating_threshold": 0.5,
            "schema_version": 2,
            "preprocessing_identity": CURRENT_PREPROCESSING_IDENTITY,
        }
    )

    assert config.schema_version == 2
    assert config.preprocessing_identity == CURRENT_PREPROCESSING_IDENTITY


def test_configured_identity_mismatch_rejects_before_weight_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _write_bundle(
        tmp_path / "legacy-artifact",
        schema_version=1,
        preprocessing_identity=LEGACY_PREPROCESSING_IDENTITY,
    )
    weight_load_attempted = False

    def _weight_load_should_not_run(_: Path) -> object:
        nonlocal weight_load_attempted
        weight_load_attempted = True
        raise AssertionError("weights must not be read for an incompatible bundle")

    monkeypatch.setattr(torch_lstm_fall, "_load_state_dict", _weight_load_should_not_run)

    with pytest.raises(ModelLoadError, match="schema_version"):
        LstmFallRunner.from_artifact_dir(
            bundle,
            expected_schema_version=2,
            expected_preprocessing_identity=CURRENT_PREPROCESSING_IDENTITY,
        )

    assert not weight_load_attempted


def test_preprocessing_identity_mismatch_rejects_before_weight_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _write_bundle(
        tmp_path / "wrong-preprocessing",
        schema_version=2,
        preprocessing_identity=LEGACY_PREPROCESSING_IDENTITY,
    )
    weight_load_attempted = False

    def _weight_load_should_not_run(_: Path) -> object:
        nonlocal weight_load_attempted
        weight_load_attempted = True
        raise AssertionError("weights must not be read for an incompatible bundle")

    monkeypatch.setattr(torch_lstm_fall, "_load_state_dict", _weight_load_should_not_run)

    with pytest.raises(ModelLoadError, match="preprocessing_identity"):
        LstmFallRunner.from_artifact_dir(
            bundle,
            expected_schema_version=2,
            expected_preprocessing_identity=CURRENT_PREPROCESSING_IDENTITY,
        )

    assert not weight_load_attempted


def test_missing_schema_version_uses_legacy_identity(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path / "legacy-without-version",
        schema_version=None,
        preprocessing_identity=None,
    )

    manifest = LstmFallManifest.from_artifact_dir(bundle)

    assert manifest.schema_version == 1
    assert manifest.preprocessing_identity == LEGACY_PREPROCESSING_IDENTITY


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "invalid", schema_version=99)

    with pytest.raises(ModelLoadError, match="unsupported schema_version"):
        LstmFallManifest.from_artifact_dir(bundle)


def test_schema_v1_without_identity_uses_explicit_legacy_tag(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path / "legacy", schema_version=1, preprocessing_identity=None
    )

    manifest = LstmFallManifest.from_artifact_dir(bundle)

    assert manifest.preprocessing_identity == LEGACY_PREPROCESSING_IDENTITY
