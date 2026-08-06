"""Typed, sanitized contracts for durable evidence export."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class DeliveryDisposition(StrEnum):
    RETRY = "RETRY"
    PERMANENT = "PERMANENT"
    COMPATIBILITY = "COMPATIBILITY"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    event_idempotency: Literal[1]
    clip_export: Literal[0, 1]


@dataclass(frozen=True, slots=True)
class DeliveryFailure:
    disposition: DeliveryDisposition
    code: str
    status_code: int | None = None
    retry_after_seconds: float | None = None
    # Set only on the transport-exception branch (no HTTP response at all):
    # "<ExceptionClassName>: <str(exc)>", bounded in length. Never derived
    # from a response body -- see shared/events/relay_failure_log.py.
    transport_error: str | None = None


@dataclass(frozen=True, slots=True)
class EventReceipt:
    status: Literal["accepted"]
    edge_event_id: str
    event_id: str


@dataclass(frozen=True, slots=True)
class ClipReceipt:
    clip_id: str
    state: Literal["READY", "UNAVAILABLE", "EXPIRED"]
    state_version: int
    sha256: str | None
    size_bytes: int | None


__all__ = [
    "BackendCapabilities",
    "ClipReceipt",
    "DeliveryDisposition",
    "DeliveryFailure",
    "EventReceipt",
]
