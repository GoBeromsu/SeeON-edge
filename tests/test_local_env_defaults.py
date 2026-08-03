"""Issue #133 (env-less default fall model) + #79 track 2 (aggregate boot-gate
error reporting) for ``worker.runtime.config.local_env``.

``model.pt`` is gitignored (see .gitignore) so a fresh clone/CI checkout
never has it on disk; ``arch.json``/``metadata.yaml`` are tracked in git and
always present. Tests that need the packaged default to actually *resolve*
(not just fail closed) skip when the weights are absent -- run
``scripts/fetch-models.sh`` to provision them locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.local_env import (
    ML_WORKER_FALL_MODEL_ARTIFACT_DIR_ENV,
    ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV,
    ML_WORKER_FALL_MODEL_STRIDE_ENV,
    ML_WORKER_FALL_MODEL_WINDOW_ENV,
    fall_model_config_from_environment,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGED_DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "models" / "fall" / "lstm"
_WEIGHTS_PROVISIONED = (_PACKAGED_DEFAULT_ARTIFACT_DIR / "model.pt").exists()
_SKIP_NO_WEIGHTS_REASON = (
    "packaged default LSTM weights not provisioned locally "
    "(run scripts/fetch-models.sh)"
)


def test_packaged_default_sidecars_are_tracked_and_parseable() -> None:
    """arch.json/metadata.yaml are committed to git (unlike model.pt), so
    this always runs, even in a fresh CI checkout with no weights fetched."""
    arch = json.loads((_PACKAGED_DEFAULT_ARTIFACT_DIR / "arch.json").read_text())
    assert arch.keys() == {"hidden", "layers", "dropout"}
    assert isinstance(arch["hidden"], int) and arch["hidden"] > 0
    assert isinstance(arch["layers"], int) and arch["layers"] > 0

    metadata = yaml.safe_load((_PACKAGED_DEFAULT_ARTIFACT_DIR / "metadata.yaml").read_text())
    assert metadata["type"] == "lstm"
    assert metadata["window"] == 30
    assert metadata["stride"] == 5
    assert metadata["input_shape"] == [30, 51]
    assert metadata["schema_version"] == 1
    assert (
        metadata["preprocessing_identity"]
        == "legacy-coco17-xyc-frame-normalized-zero-fill-v1"
    )


@pytest.mark.skipif(not _WEIGHTS_PROVISIONED, reason=_SKIP_NO_WEIGHTS_REASON)
def test_default_env_resolves_packaged_lstm_config() -> None:
    config = fall_model_config_from_environment({})

    assert config.type == "lstm"
    assert config.artifact_dir == _PACKAGED_DEFAULT_ARTIFACT_DIR.resolve()
    assert config.window == 30
    assert config.stride == 5
    assert config.input_shape == (30, 51)
    assert config.schema_version == 1
    assert (
        config.preprocessing_identity == "legacy-coco17-xyc-frame-normalized-zero-fill-v1"
    )


@pytest.mark.skipif(not _WEIGHTS_PROVISIONED, reason=_SKIP_NO_WEIGHTS_REASON)
def test_lstm_runner_loads_and_predicts_from_packaged_default() -> None:
    import numpy as np

    from worker.adapters.model.torch_lstm_fall import LstmFallRunner

    runner = LstmFallRunner.from_artifact_dir(_PACKAGED_DEFAULT_ARTIFACT_DIR)
    probability = runner.predict(np.zeros((30, 51), dtype=np.float32))

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_default_env_missing_weights_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deterministic regardless of whether *this* repo checkout has fetched
    weights: chdir into an empty directory so the packaged default's
    relative path (``models/fall/lstm``) resolves to nothing there, and
    assert the fail-closed error names the fetch script rather than silently
    booting without a fall model."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(WorkerConfigError, match="scripts/fetch-models.sh"):
        fall_model_config_from_environment({})


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
