"""Env-less packaged fall model default + #79 track 2 (aggregate boot-gate
error reporting) for ``worker.runtime.config.local_env``.

Nothing under ``models/`` is tracked: the packaged default is the published
pose+bbox56 bundle pinned in ``worker/tools/fetch_models/manifest.json`` and
provisioned by ``scripts/fetch-models.sh``. Tests that need the default to
actually *resolve* build a synthetic bundle in ``tmp_path`` through the
``packaged_fall_bundle`` fixture, so they are deterministic in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.local_env import (
    ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV,
    ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV,
    ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV,
    ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV,
    ML_WORKER_FALL_MODEL_STRIDE_ENV,
    ML_WORKER_FALL_MODEL_WINDOW_ENV,
    fall_model_config_from_environment,
    reject_retired_worker_environment,
)

_PACKAGED_DEFAULT_ARTIFACT_DIR = Path("models/fall/pose-bbox56-gru")
_PREPROCESSING_IDENTITY = "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"


def test_default_env_resolves_packaged_pose_bbox56_config(packaged_fall_bundle: Path) -> None:
    config = fall_model_config_from_environment({})

    assert config.type == "pose-bbox56-proxy-v0"
    assert config.artifact_dir == packaged_fall_bundle.resolve()
    assert config.window == 30
    assert config.stride == 5
    assert config.input_shape == (30, 56)
    assert config.schema_version == 2
    assert config.operating_threshold == 0.5
    assert config.preprocessing_identity == _PREPROCESSING_IDENTITY


def test_bundle_runner_loads_and_predicts_from_packaged_default(
    packaged_fall_bundle: Path,
) -> None:
    from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner

    runner = PoseBbox56BundleRunner.from_artifact_dir(packaged_fall_bundle)
    probabilities = runner.predict(np.zeros((30, 56), dtype=np.float32))

    assert runner.device == "cpu"
    assert 0.0 <= probabilities.fall_transition <= 1.0
    assert probabilities.fallen == 0.0


def test_default_env_missing_weights_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deterministic regardless of whether *this* repo checkout has fetched
    weights: chdir into an empty directory so the packaged default's
    relative path (``models/fall/pose-bbox56-gru``) resolves to nothing there, and
    assert the fail-closed error names the fetch script rather than silently
    booting without a fall model."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(WorkerConfigError, match="scripts/fetch-models.sh"):
        fall_model_config_from_environment({})


def _write_fake_packaged_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chdir into an empty directory and populate a packaged-default bundle
    there so ``fall_model_config_from_environment({})`` resolves
    deterministically -- independent of whether *this* checkout has fetched
    the real (gitignored) bundle via ``scripts/fetch-models.sh``."""
    monkeypatch.chdir(tmp_path)
    write_pose_bbox56_bundle(tmp_path / _PACKAGED_DEFAULT_ARTIFACT_DIR)


def test_default_env_with_no_overrides_resolves_packaged_manifest_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env absent (no ``ARTIFACT_DIR``, no window/stride/operating_threshold)
    must resolve to the packaged manifest's own defaults, unchanged."""
    _write_fake_packaged_default(tmp_path, monkeypatch)

    config = fall_model_config_from_environment({})

    assert config.window == 30
    assert config.stride == 5
    assert config.operating_threshold == 0.5


def test_model_policy_environment_keys_are_retired_explicitly() -> None:
    environ = {
        ML_WORKER_FALL_MODEL_WINDOW_ENV: "45",
        ML_WORKER_FALL_MODEL_STRIDE_ENV: "9",
        ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV: "0.5",
    }

    with pytest.raises(WorkerConfigError) as excinfo:
        reject_retired_worker_environment(environ)

    assert all(name in str(excinfo.value) for name in environ)
    assert "versioned worker config authority" in str(excinfo.value)


def test_default_env_with_no_threshold_override_logs_manifest_default_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_fake_packaged_default(tmp_path, monkeypatch)

    with caplog.at_level("INFO"):
        fall_model_config_from_environment({})

    assert any(
        "source: packaged manifest default" in record.getMessage() for record in caplog.records
    )


def test_retired_manifest_environment_keys_fail_instead_of_warning() -> None:
    environ = {
        ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV: "2",
        ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV: "some-other-identity",
    }

    with pytest.raises(WorkerConfigError) as excinfo:
        reject_retired_worker_environment(environ)

    assert all(name in str(excinfo.value) for name in environ)


def test_explicit_artifact_dir_aggregates_multiple_invalid_env_vars(tmp_path: Path) -> None:
    """Issue #79 (track 2): window/stride/operating_threshold are all
    malformed at once -- all three must be named in a single raised error
    instead of only the first one checked."""
    environ = {
        ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV: str(tmp_path),
        ML_WORKER_FALL_MODEL_WINDOW_ENV: "not-an-int",
        ML_WORKER_FALL_MODEL_STRIDE_ENV: "also-not-an-int",
        ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV: "not-a-float",
    }

    with pytest.raises(WorkerConfigError) as excinfo:
        fall_model_config_from_environment(environ)

    message = str(excinfo.value)
    assert "3 fall model environment variable(s) invalid" in message
    assert ML_WORKER_FALL_MODEL_WINDOW_ENV in message
    assert ML_WORKER_FALL_MODEL_STRIDE_ENV in message
    assert ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV in message
