from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types import SceneRecord


@runtime_checkable
class SceneSink(Protocol):
    def append(self, record: SceneRecord) -> bool: ...


@runtime_checkable
class EpochRollingSceneSink(SceneSink, Protocol):
    def roll_epoch(self, camera_id: str, epoch: int) -> None: ...


__all__ = ["EpochRollingSceneSink", "SceneSink"]
