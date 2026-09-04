"""In-process implementation of the worker model-serving seams."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from typing import Protocol, runtime_checkable

from contracts.runner import Image, RunnerResult
from worker.adapters.model.batch_input import BatchInputError, validated_batch_images
from worker.adapters.model.registry import (
    DEFAULT_REGISTRY,
    ModelAdapter,
    ModelOption,
    ModelRegistry,
)
from worker.types import FramePacket

_BATCHED_TASKS: tuple[str, ...] = ("pose",)


@runtime_checkable
class _BatchRunner(Protocol):
    def run_batch(self, images: Sequence[Image]) -> tuple[RunnerResult, ...]: ...


class InProcessServingClient:
    """Provision one process-local runner per task/options for single frames."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry: ModelRegistry = DEFAULT_REGISTRY if registry is None else registry
        self._runners: dict[tuple[str, tuple[tuple[str, ModelOption], ...]], ModelAdapter] = {}
        self._runner_lock = Lock()
        self._batch_client: InProcessBatchServingClient | None = None

    def create(self, task: str, **kwargs: ModelOption) -> ModelAdapter:
        key = (task, tuple(sorted(kwargs.items())))
        with self._runner_lock:
            runner = self._runners.get(key)
            if runner is None:
                runner = self._registry.create(task, **kwargs)
                self._runners[key] = runner
            return runner

    @property
    def batch_serving_client(self) -> InProcessBatchServingClient:
        """A batch view sharing every runner/model provisioned by ``create``."""
        with self._runner_lock:
            if self._batch_client is None:
                self._batch_client = InProcessBatchServingClient(self)
            return self._batch_client


class InProcessBatchServingClient:
    """Batch facade over an ``InProcessServingClient`` with no model copies."""

    def __init__(self, serving_client: InProcessServingClient) -> None:
        self._serving_client = serving_client

    def create(self, task: str, **kwargs: ModelOption) -> ModelAdapter:
        return self._serving_client.create(task, **kwargs)

    def infer_batch(
        self,
        task: str,
        frames: Sequence[FramePacket],
        **kwargs: ModelOption,
    ) -> tuple[RunnerResult, ...]:
        """Issue one pose forward; result ``i`` belongs to frame ``i``."""
        ordered = tuple(frames)
        if task not in _BATCHED_TASKS:
            raise BatchInputError(
                task=task,
                camera_id=ordered[0].camera_id if ordered else "",
                detail=f"batched inference is available only for {_BATCHED_TASKS}",
            )
        if not ordered:
            return ()
        images = validated_batch_images(
            task,
            tuple((packet.camera_id, packet.borrow_host_frame().image) for packet in ordered),
        )
        runner = self.create(task, **kwargs)
        if not isinstance(runner, _BatchRunner):
            raise BatchInputError(
                task=task,
                camera_id=ordered[0].camera_id,
                detail=f"runner {type(runner).__name__} does not implement run_batch",
            )
        return runner.run_batch(images)


__all__ = ["InProcessBatchServingClient", "InProcessServingClient"]
