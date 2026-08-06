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
    ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV,
    ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV,
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


def _write_fake_packaged_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chdir into an empty directory and populate a fake packaged-default
    artifact there so ``fall_model_config_from_environment({})`` resolves
    deterministically -- independent of whether *this* checkout has fetched
    the real (gitignored) model.pt via ``scripts/fetch-models.sh``."""
    monkeypatch.chdir(tmp_path)
    artifact_dir = tmp_path / "models" / "fall" / "lstm"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.pt").write_bytes(b"placeholder")
    (artifact_dir / "arch.json").write_text(
        '{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8"
    )
    (artifact_dir / "metadata.yaml").write_text("type: lstm\n", encoding="utf-8")


def test_default_env_with_no_overrides_resolves_packaged_manifest_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env absent (no ``ARTIFACT_DIR``, no window/stride/operating_threshold)
    must resolve to the packaged manifest's own defaults, unchanged."""
    _write_fake_packaged_default(tmp_path, monkeypatch)

    config = fall_model_config_from_environment({})

    assert config.window == 30
    assert config.stride == 5
    assert config.operating_threshold == pytest.approx(0.0007872396381571889)


def test_default_env_respects_explicit_operating_threshold_without_artifact_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for issue #198: ``ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD``
    (and window/stride) must be honored even when
    ``ML_WORKER_FALL_MODEL_ARTIFACT_DIR`` is unset -- the packaged-default
    path the shipped edge topology actually takes. Before the fix, this env
    var was read only when ``ARTIFACT_DIR`` was also set, so an operator-set
    ``0.5`` was silently discarded in favor of the field default
    (``0.0007872396381571889``), producing 352 false fall events/hour in
    production."""
    _write_fake_packaged_default(tmp_path, monkeypatch)
    environ = {
        ML_WORKER_FALL_MODEL_WINDOW_ENV: "45",
        ML_WORKER_FALL_MODEL_STRIDE_ENV: "9",
        ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV: "0.5",
    }

    config = fall_model_config_from_environment(environ)

    assert config.window == 45
    assert config.stride == 9
    assert config.input_shape == (45, 51)
    assert config.operating_threshold == pytest.approx(0.5)


def test_default_env_logs_resolved_operating_threshold_and_its_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #198: the resolved operating_threshold -- and whether it came
    from env or the packaged manifest default -- must be logged once at
    boot, since previously the only way to discover the effective value was
    dumping an emitted event's ``audit`` blob out of the SQLite outbox."""
    _write_fake_packaged_default(tmp_path, monkeypatch)

    with caplog.at_level("INFO"):
        fall_model_config_from_environment(
            {ML_WORKER_FALL_MODEL_OPERATING_THRESHOLD_ENV: "0.5"}
        )

    assert any(
        record.getMessage()
        == "fall model operating_threshold resolved to 0.5 (source: env)"
        for record in caplog.records
    )


def test_default_env_with_no_threshold_override_logs_manifest_default_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_fake_packaged_default(tmp_path, monkeypatch)

    with caplog.at_level("INFO"):
        fall_model_config_from_environment({})

    assert any(
        "source: packaged manifest default" in record.getMessage()
        for record in caplog.records
    )


def test_default_env_warns_when_schema_version_env_is_set_but_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """General "set, documented, silently dead" guard (issue #198, matching
    #191): ``ML_WORKER_FALL_MODEL_SCHEMA_VERSION``/``..._PREPROCESSING_IDENTITY``
    still only apply once ``ARTIFACT_DIR`` is explicitly set (unlike
    window/stride/operating_threshold, they have no packaged-default
    fallback path), so setting either without ``ARTIFACT_DIR`` must warn
    instead of failing silently."""
    _write_fake_packaged_default(tmp_path, monkeypatch)
    environ = {
        ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV: "2",
        ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV: "some-other-identity",
    }

    with caplog.at_level("WARNING"):
        config = fall_model_config_from_environment(environ)

    # The env values are ignored -- packaged manifest defaults still apply.
    assert config.schema_version == 1
    assert config.preprocessing_identity == "legacy-coco17-xyc-frame-normalized-zero-fill-v1"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        ML_WORKER_FALL_MODEL_SCHEMA_VERSION_ENV in message and "ignored" in message
        for message in messages
    )
    assert any(
        ML_WORKER_FALL_MODEL_PREPROCESSING_IDENTITY_ENV in message and "ignored" in message
        for message in messages
    )


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
