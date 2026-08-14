"""Task-to-runner registry for in-process model execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal, Protocol, TypeAlias

from contracts.runner import RunnerProtocol
from worker.adapters.model.yolo_bed_seg import YoloBedSegRunner
from worker.adapters.model.yolo_person import YoloPersonRunner
from worker.adapters.model.yolo_pose import YoloPoseRunner
from worker.types import FallModelInput

ModelOption: TypeAlias = str | int | float | bool | None


class FallModelMetadata(Protocol):
    @property
    def window(self) -> int: ...

    @property
    def stride(self) -> int: ...

    @property
    def mode(self) -> Literal["features", "sequence"]: ...


class FallModel(Protocol):
    @property
    def metadata(self) -> FallModelMetadata: ...

    @property
    def operating_threshold(self) -> float: ...

    def predict(self, features: FallModelInput) -> float: ...

    def warmup(self) -> None: ...


class WarmupModel(Protocol):
    def warmup(self) -> None: ...


# This registry provisions only camera runners. Fall models use the separate
# fall-family registry; warmup is an optional runner capability, not a model kind.
ModelAdapter: TypeAlias = RunnerProtocol
RunnerFactory: TypeAlias = Callable[..., ModelAdapter]

_MODEL_TASKS: Final = ("pose", "person", "bed", "fall")


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
    registry.register("pose", YoloPoseRunner)
    registry.register("person", YoloPersonRunner)
    registry.register("bed", YoloBedSegRunner)
    return registry


DEFAULT_REGISTRY: Final = default_registry()

__all__ = [
    "DEFAULT_REGISTRY",
    "EmptyModelTaskError",
    "FallModel",
    "FallModelMetadata",
    "ModelAdapter",
    "ModelOption",
    "ModelRegistry",
    "RunnerFactory",
    "UnknownModelTaskError",
    "WarmupModel",
    "default_registry",
]
