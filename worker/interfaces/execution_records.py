"""Non-blocking execution-record sink. Implementations live in pipeline/runtime."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutionRecordSink(Protocol):
    """Admit one observation into the Worker-side bounded buffer.

    Must be O(1) append-or-drop, never block on I/O, and never raise. ``True``
    means the record is queued; ``False`` means it was dropped (overflow or
    contract failure). The argument is a ``shared.events.execution_records.WireRecord``
    at runtime; this layer cannot import ``shared`` (import-linter).
    """

    def try_emit(self, record: object) -> bool: ...


__all__ = ["ExecutionRecordSink"]
