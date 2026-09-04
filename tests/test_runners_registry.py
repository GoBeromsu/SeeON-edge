"""Residue of the legacy edge registry contract not already covered worker-side.

register/create/get_factory happy-path and unknown-task rejection are covered
by test_worker_model_serving.py (test_model_registry_registers_and_creates_with_
factory_options, test_model_registry_unknown_task_error_names_the_task). This
file keeps the remaining, still-uncovered pieces of the ModelRegistry contract:
empty-task rejection, per-task factory identity against worker's own runner
classes, and the models-directory resolution default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from worker.adapters.model.registry import EmptyModelTaskError, ModelRegistry, default_registry
from worker.adapters.model.sklearn_fall import MODELS_DIR


class FakeRunner:
    def __init__(self, *, value: int = 0) -> None:
        self.value = value


def test_model_registry_rejects_empty_task() -> None:
    registry = ModelRegistry()

    with pytest.raises(EmptyModelTaskError, match="task must be non-empty"):
        registry.register("", FakeRunner)


def test_default_registry_has_pose_bed_person_factories_without_loading_models() -> None:
    registry = default_registry()

    # "fall" is deliberately absent: it has no registry-backed fallback.
    assert registry.tasks() == ("bed", "person", "pose")
    # The ultralytics runners resolve lazily so a flow-profile process never
    # imports torch (P1b-AC7): building the registry must not import them.
    for task in ("pose", "bed", "person"):
        assert callable(registry.get_factory(task))
    assert "worker.adapters.model.yolo_pose" not in sys.modules
    assert "worker.adapters.model.yolo_bed_seg" not in sys.modules
    assert "worker.adapters.model.yolo_person" not in sys.modules


def test_sklearn_fall_default_models_dir_points_to_ml_models_root() -> None:
    assert Path(__file__).resolve().parents[1] / "models" == MODELS_DIR
