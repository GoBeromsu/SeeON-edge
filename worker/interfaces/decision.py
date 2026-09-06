from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot


@runtime_checkable
class Decider(Protocol):
    """Update one domain's temporal state from image-free numeric input."""

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]: ...


@runtime_checkable
class TraceSnapshotProvider(Decider, Protocol):
    """A decision module that exposes its latest compiled trace state."""

    @property
    def last_trace_snapshots(self) -> tuple[DecisionTraceSnapshot, ...]: ...


__all__ = ["Decider", "TraceSnapshotProvider"]
