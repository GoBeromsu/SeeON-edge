from __future__ import annotations

from dataclasses import dataclass, field

from worker.types import CURRENT_TEMPORAL_PROFILE


@dataclass(frozen=True, slots=True)
class Scheduler:
    """Deterministic per-frame task scheduler."""

    task_intervals: dict[str, int] = field(default_factory=CURRENT_TEMPORAL_PROFILE.task_intervals)

    def tasks_for_frame(self, frame_index: int) -> tuple[str, ...]:
        due: list[str] = []
        for task, interval in self.task_intervals.items():
            if interval <= 0:
                continue
            if frame_index % interval == 0:
                due.append(task)
        return tuple(due)

    def should_run(self, task: str, frame_index: int) -> bool:
        return task in self.tasks_for_frame(frame_index)


__all__ = ["Scheduler"]
