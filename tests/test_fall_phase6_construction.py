"""Selected bundles replace the packaged fall runner; no dark path exists."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.runtime.config import local_env
from worker.runtime.config.worker_models import WorkerModelsConfig
from worker.runtime.provenance.model_bundle import DesiredModelBundle


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
