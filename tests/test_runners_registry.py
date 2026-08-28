"""Residue of the legacy edge registry contract not already covered worker-side.

register/create/get_factory happy-path and unknown-task rejection are covered
by test_worker_model_serving.py (test_model_registry_registers_and_creates_with_
factory_options, test_model_registry_unknown_task_error_names_the_task). This
file keeps the remaining, still-uncovered pieces of the ModelRegistry contract:
empty-task rejection, per-task factory identity against worker's own runner
classes, and the models-directory resolution default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.adapters.model.registry import EmptyModelTaskError, ModelRegistry, default_registry
from worker.adapters.model.sklearn_fall import MODELS_DIR
from worker.adapters.model.yolo_bed_seg import YoloBedSegRunner
from worker.adapters.model.yolo_person import YoloPersonRunner
from worker.adapters.model.yolo_pose import YoloPoseRunner


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
    assert registry.get_factory("pose") is YoloPoseRunner
    assert registry.get_factory("bed") is YoloBedSegRunner
    assert registry.get_factory("person") is YoloPersonRunner


def test_sklearn_fall_default_models_dir_points_to_ml_models_root() -> None:
    assert Path(__file__).resolve().parents[1] / "models" == MODELS_DIR
