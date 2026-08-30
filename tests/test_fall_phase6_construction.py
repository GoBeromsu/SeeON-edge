"""Phase 6 remains a dark candidate: deployed LSTM behavior never changes."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.adapters.model.fall_family_registry import (
    DisabledFallModelTypeError,
    default_fall_model_family_registry,
)
from worker.runtime.config.worker_models import WorkerModelsConfig
from worker.runtime.provenance.model_bundle import DesiredModelBundle


def _desired() -> DesiredModelBundle:
    return DesiredModelBundle(
        bundle_sha256="a" * 64,
        identities={
            "dataset": "dataset.v1",
            "evaluation": "evaluation.v1",
            "field": "field.v1",
            "seed": "seed.v1",
            "rule": "rule.v1",
            "calibration": "calibration.v1",
            "conformance": "conformance.v1",
            "class": "class.v1",
            "input": "pose-bbox56.v1",
            "policy": "fall.policy.v2",
            "config": "config.v1",
            "restart": "restart.v1",
            "worker": "worker.v1",
        },
    )


def test_default_family_registry_keeps_gru_dark() -> None:
    registry = default_fall_model_family_registry()

    assert registry.types() == ("lstm",)
    with pytest.raises(DisabledFallModelTypeError):
        registry.create("gru", object(), "cpu")


def test_candidate_refuses_person_boxes_before_camera_activation() -> None:
    with pytest.raises(ValueError, match="requires box_source=pose"):
        WorkerModelsConfig(
            box_source="person",
            candidate={"models_root": str(Path("/sealed/candidate")), "desired": _desired()},
        )


def test_candidate_config_is_not_an_active_fall_selection() -> None:
    models = WorkerModelsConfig(
        candidate={"models_root": str(Path("/sealed/candidate")), "desired": _desired()},
    )

    assert models.fall is None
    assert models.candidate is not None
