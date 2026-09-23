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


@runtime_checkable
class FreshnessProvider(TraceSnapshotProvider, Protocol):
    """A trace provider that says whether its snapshots came from the last update.

    ``last_update_evaluated`` is False after an update that coasted (no model
    evaluation, snapshots untouched). A provider without this protocol is
    assumed to refresh its snapshots on every update.
    """

    @property
    def last_update_evaluated(self) -> bool: ...


@runtime_checkable
class ShadowTraceProvider(TraceSnapshotProvider, Protocol):
    """A trace provider whose trailing ``last_shadow_trace_count`` snapshots are
    non-authoritative shadow evaluations (never ``triggered``, never a cause).

    A provider that does not implement this has no shadow path; every
    snapshot it exposes is authoritative.
    """

    @property
    def last_shadow_trace_count(self) -> int: ...


__all__ = [
    "Decider",
    "FreshnessProvider",
    "ShadowTraceProvider",
    "TraceSnapshotProvider",
]
