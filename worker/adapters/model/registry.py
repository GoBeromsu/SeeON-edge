"""Task-to-runner registry for in-process model execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Protocol, TypeAlias

from contracts.runner import RunnerProtocol
from worker.interfaces.fall_model import FallV2ModelProtocol

ModelOption: TypeAlias = str | int | float | bool | None


class FallModel(FallV2ModelProtocol, Protocol):
    def warmup(self) -> None: ...


class WarmupModel(Protocol):
    def warmup(self) -> None: ...


# This registry provisions only camera runners. Fall models use the separate
# fall-family registry; warmup is an optional runner capability, not a model kind.
ModelAdapter: TypeAlias = RunnerProtocol
RunnerFactory: TypeAlias = Callable[..., ModelAdapter]


class EmptyModelTaskError(ValueError):
    def __init__(self) -> None:
        super().__init__("task must be non-empty")


class UnknownModelTaskError(KeyError):
    task: str

    def __init__(self, task: str) -> None:
        self.task = task
        super().__init__(f"unknown model task {task!r}")


class ModelRegistry:
    """Map model task names to runner factories."""

    def __init__(self) -> None:
        self._factories: dict[str, RunnerFactory] = {}

    def register(self, task: str, factory: RunnerFactory) -> None:
        if not task:
            raise EmptyModelTaskError
        self._factories[task] = factory

    def create(self, task: str, **kwargs: ModelOption) -> ModelAdapter:
        return self.get_factory(task)(**kwargs)

    def get_factory(self, task: str) -> RunnerFactory:
        try:
            return self._factories[task]
        except KeyError as exc:
            raise UnknownModelTaskError(task) from exc

    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_registry() -> ModelRegistry:
    """Build the default registry.

    "fall" is deliberately absent: the fall model has no registry-backed
    fallback (see ``WorkerRuntime._create_fall_model``), so registering one
    here would just be dead code shadowing the fail-closed boot check.
    """
    registry = ModelRegistry()
    registry.register("pose", _yolo_pose)
    registry.register("person", _yolo_person)
    registry.register("bed", _yolo_bed_seg)
    return registry


# The ultralytics runners are the host/nvidia profiles' camera runners. They
# are resolved when a task is created, not when this module imports: under the
# flow profile DeepStream owns pose and ORT owns bed, and that process asserts
# torch/ultralytics are never imported (P1b-AC7).
def _yolo_pose(**kwargs: ModelOption) -> ModelAdapter:
    from worker.adapters.model.yolo_pose import YoloPoseRunner

    return YoloPoseRunner(**kwargs)


def _yolo_person(**kwargs: ModelOption) -> ModelAdapter:
    from worker.adapters.model.yolo_person import YoloPersonRunner

    return YoloPersonRunner(**kwargs)


def _yolo_bed_seg(**kwargs: ModelOption) -> ModelAdapter:
    from worker.adapters.model.yolo_bed_seg import YoloBedSegRunner

    return YoloBedSegRunner(**kwargs)


DEFAULT_REGISTRY: Final = default_registry()

__all__ = [
    "DEFAULT_REGISTRY",
    "EmptyModelTaskError",
    "FallModel",
    "ModelAdapter",
    "ModelOption",
    "ModelRegistry",
    "RunnerFactory",
    "UnknownModelTaskError",
    "WarmupModel",
    "default_registry",
]
