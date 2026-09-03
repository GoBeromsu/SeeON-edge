"""Selected bundles replace the packaged fall runner; no dark path exists."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import worker.runtime.worker as worker_module
from worker.adapters.model.fall_family_registry import (
    UnknownFallModelTypeError,
    default_fall_model_family_registry,
)
from worker.runtime.config import local_env
from worker.runtime.config.worker_models import WorkerModelsConfig
from worker.runtime.provenance.model_bundle import DesiredModelBundle
from worker.runtime.worker import WorkerRuntime


def _desired() -> DesiredModelBundle:
    return DesiredModelBundle(
        bundle_sha256="a" * 64,
        identities={
            "dataset": "1" * 64,
            "evaluation": "2" * 64,
            "field": "3" * 64,
            "calibration": "4" * 64,
            "conformance": "5" * 64,
            "class": "6" * 64,
            "input": "pose-bbox56.v1",
            "policy": "7" * 64,
            "members": "8" * 64,
        },
    )


def test_runtime_format_registry_is_active_and_fail_closed() -> None:
    registry = default_fall_model_family_registry()
    assert "torchscript-gru-pose-bbox" in registry.runtime_formats()
    with pytest.raises(UnknownFallModelTypeError):
        registry.create_bundle("unknown-format", Path("/bundle"), "cpu")


def test_selected_bundle_refuses_person_boxes() -> None:
    with pytest.raises(ValueError, match="requires box_source=pose"):
        WorkerModelsConfig(
            box_source="person",
            selected={"models_root": "/sealed/selected", "desired": _desired()},
        )


def test_selection_loads_without_image_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = tmp_path / "model-selection.json"
    selection.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        local_env, "desired_model_bundle_from_selection_document", lambda raw: _desired()
    )

    selected = local_env.selected_fall_bundle_config_from_environment(
        selection_path=selection,
        models_root=Path("/models"),
    )

    assert selected is not None
    assert selected.models_root == Path("/models")


def test_missing_selection_keeps_packaged_model_path(tmp_path: Path) -> None:
    assert (
        local_env.selected_fall_bundle_config_from_environment(
            selection_path=tmp_path / "missing-model-selection.json"
        )
        is None
    )


def test_selected_bundle_is_the_single_active_fall_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = SimpleNamespace(runtime_format="registered-format")
    selected = SimpleNamespace(
        models_root=Path("/models"),
        desired=SimpleNamespace(bundle_sha256="a" * 64, selection=selection),
    )
    calls: list[tuple[str, Path, str]] = []
    registry = SimpleNamespace(
        create_bundle=lambda runtime_format, artifact_dir, device: (
            calls.append((runtime_format, artifact_dir, device)) or "selected-runner"
        )
    )
    monkeypatch.setattr(worker_module, "DEFAULT_FALL_MODEL_FAMILY_REGISTRY", registry)
    runtime = object.__new__(WorkerRuntime)
    runtime.config = SimpleNamespace(models=SimpleNamespace(selected=selected, fall=None))

    assert runtime._create_fall_model("cpu") == "selected-runner"  # noqa: SLF001
    assert calls == [("registered-format", Path("/models/bundles") / ("a" * 64), "cpu")]


def test_no_selection_uses_the_packaged_fall_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object, str]] = []
    packaged = SimpleNamespace(type="lstm")
    registry = SimpleNamespace(
        create=lambda model_type, config, device: (
            calls.append((model_type, config, device)) or "packaged-lstm"
        )
    )
    monkeypatch.setattr(worker_module, "DEFAULT_FALL_MODEL_FAMILY_REGISTRY", registry)
    runtime = object.__new__(WorkerRuntime)
    runtime.config = SimpleNamespace(models=SimpleNamespace(selected=None, fall=packaged))

    assert runtime._create_fall_model("cpu") == "packaged-lstm"  # noqa: SLF001
    assert calls == [("lstm", packaged, "cpu")]
