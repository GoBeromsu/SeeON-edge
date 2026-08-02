from __future__ import annotations

from dataclasses import dataclass

import pytest

import worker.adapters.model.in_process as in_process_module
from contracts.runner import Image, RunnerProtocol, RunnerResult, pose_result
from worker.adapters.model import (
    InProcessServingClient,
    ModelRegistry,
    default_registry,
)
from worker.adapters.model.registry import UnknownModelTaskError
from worker.interfaces.serving import BatchServingClient, ServingClient

ServingOption = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class _Runner:
    value: int = 0

    def run(self, image: Image) -> RunnerResult:
        del image
        return pose_result((), ())


def test_default_registry_registers_the_production_task_set() -> None:
    registry = default_registry()

    # "fall" is deliberately absent: it has no registry-backed fallback (see
    # WorkerRuntime._create_fall_model, which fails closed instead).
    assert registry.tasks() == ("bed", "person", "pose")
    for task in registry.tasks():
        assert callable(registry.get_factory(task))
    with pytest.raises(UnknownModelTaskError):
        registry.get_factory("fall")


def test_model_registry_registers_and_creates_with_factory_options() -> None:
    registry = ModelRegistry()
    registry.register("fake", _Runner)

    runner = registry.create("fake", value=7)

    assert isinstance(runner, _Runner)
    assert runner.value == 7


def test_model_registry_unknown_task_error_names_the_task() -> None:
    registry = ModelRegistry()

    with pytest.raises(UnknownModelTaskError, match="unknown model task") as error:
        _ = registry.create("missing")

    assert error.value.task == "missing"


def test_in_process_client_satisfies_only_the_single_item_protocol() -> None:
    client = InProcessServingClient(ModelRegistry())

    assert isinstance(client, ServingClient)
    assert not isinstance(client, BatchServingClient)
    assert "infer_batch" not in InProcessServingClient.__dict__


def test_in_process_client_defaults_to_the_process_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelRegistry()
    sentinel = _Runner()

    def pose_factory() -> RunnerProtocol:
        return sentinel

    registry.register("pose", pose_factory)
    monkeypatch.setattr(in_process_module, "DEFAULT_REGISTRY", registry)
    client = InProcessServingClient()

    assert client.create("pose") is sentinel


def test_one_provisioned_model_object_is_shared_by_two_camera_contexts() -> None:
    registry = ModelRegistry()
    sentinel = _Runner()
    factory_calls: list[dict[str, ServingOption]] = []

    def pose_factory(**kwargs: ServingOption) -> RunnerProtocol:
        factory_calls.append(dict(kwargs))
        return sentinel

    registry.register("pose", pose_factory)
    client = InProcessServingClient(registry)
    shared_pose = client.create("pose", device="cpu")
    camera_one: dict[str, RunnerProtocol] = {"pose": shared_pose}
    camera_two: dict[str, RunnerProtocol] = {"pose": shared_pose}

    assert camera_one["pose"] is camera_two["pose"] is sentinel
    assert factory_calls == [{"device": "cpu"}]


def test_batch_serving_client_remains_a_pure_deferred_protocol() -> None:
    assert issubclass(BatchServingClient, ServingClient)
    assert BatchServingClient.__dict__["_is_protocol"] is True
    assert "infer_batch" in BatchServingClient.__dict__
