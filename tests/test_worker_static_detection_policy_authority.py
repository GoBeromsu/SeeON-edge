from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import cast

import pytest
import yaml

import worker.__main__ as worker_main
from shared.detection_policies import default_policy_bundle
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.loader import load_worker_config
from worker.runtime.config.worker_models import WorkerConfig
from worker.runtime.worker import WorkerRuntime

_RETIRED_POLICY_ERROR = (
    "static detection_policies authority is retired; numeric detection policies "
    "must be pulled from central edge.sqlite3 via the versioned worker config"
)


def _yaml_payload(policy: object) -> dict[str, object]:
    return {
        "version": 1,
        "relay": {"url": "http://ml-api:8000", "token": "relay-token"},
        "cameras": [],
        "detection_policies": policy,
    }


def _fall_document(bundle: dict[str, object]) -> dict[str, object]:
    defaults = cast(dict[str, object], bundle["defaults"])
    return cast(dict[str, object], defaults["fall"])


def _policy_documents() -> tuple[object, ...]:
    valid = default_policy_bundle(("camera/opaque:alpha",)).as_dict()
    out_of_range = copy.deepcopy(valid)
    values = cast(dict[str, object], _fall_document(out_of_range)["values"])
    values["operating_threshold"] = 2.0
    forged = copy.deepcopy(valid)
    _fall_document(forged)["effective_policy_id"] = "0" * 64
    return valid, "malformed-policy", out_of_range, forged


def _write_yaml(path: Path, policy: object) -> Path:
    path.write_text(yaml.safe_dump(_yaml_payload(policy)), encoding="utf-8")
    return path


@pytest.mark.parametrize("policy", _policy_documents())
def test_direct_yaml_rejects_every_static_detection_policy_before_worker_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: object,
) -> None:
    path = _write_yaml(tmp_path / "ml-worker.yaml", policy)

    def _must_not_construct(*_args: object, **_kwargs: object) -> WorkerConfig:
        raise AssertionError("static detection policy reached WorkerConfig construction")

    monkeypatch.setattr(WorkerConfig, "model_validate", _must_not_construct)

    with pytest.raises(WorkerConfigError, match=_RETIRED_POLICY_ERROR):
        load_worker_config(path)


@pytest.mark.parametrize("check_only", [False, True])
def test_cli_config_rejects_static_detection_policy_before_pull_or_lkg_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    check_only: bool,
) -> None:
    path = _write_yaml(tmp_path / "ml-worker.yaml", _policy_documents()[3])

    def _must_not_fallback(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retired YAML policy reached relay/LKG fallback resolution")

    monkeypatch.setattr(worker_main, "resolve_startup_config", _must_not_fallback)
    monkeypatch.setattr(WorkerRuntime, "__init__", _must_not_fallback)
    args = ["--config", str(path)]
    if check_only:
        args.append("--check-config")

    with caplog.at_level(logging.ERROR):
        exit_code = worker_main.main(args)

    assert exit_code == worker_main.CONFIG_ERROR_EXIT_CODE
    assert any(
        record.exc_info is not None and _RETIRED_POLICY_ERROR in str(record.exc_info[1])
        for record in caplog.records
    )
