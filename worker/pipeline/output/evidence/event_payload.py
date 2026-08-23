"""Worker-local staging envelope for an admitted event carrying snapshot bytes.

``contracts.event.EventPayload`` is the vendored, ADR-0006 byte-identical L0
shape shared with the backend and dataset-ops; it stays a plain ``Mapping``
alias over scalars and nested scalar evidence, with no ``bytes`` member, so it
can cross the shared relay boundary without ever carrying raw media.

The worker's own evidence pipeline (event sink -> identity enrichment ->
durable stager, all under ``worker/pipeline/output/``) additionally needs a
strictly-typed *mutable* envelope that may carry an inline JPEG snapshot
in-process, before the durable stager base64-encodes it for relay
(``worker/pipeline/output/evidence/evidence_stager.py``). That shape is
worker-internal wiring, never a cross-instance contract, so it lives here per
``worker/AGENTS.md``'s vendored-contracts boundary rather than widening or
shadowing ``contracts.event.EventPayload``. It must never be imported outside
``worker/`` -- cross-repository/shared-boundary code
(``shared/events/edge_ingest_client.py``) stays on the canonical,
bytes-free ``contracts.event.EventPayload``.
"""

from __future__ import annotations

from typing import NotRequired, TypeAlias, TypedDict

from contracts.event import EventEvidence, EventScalar


class WorkerEventPayload(TypedDict):
    """A staged event envelope that may carry an inline JPEG snapshot."""

    edge_event_id: NotRequired[str]
    event_type: NotRequired[str]
    probability: NotRequired[float]
    confidence: NotRequired[float]
    detected_at: NotRequired[str]
    camera_id: NotRequired[str]
    facility_id: NotRequired[str]
    domain: NotRequired[str]
    identity: NotRequired[str | int]
    time_sec: NotRequired[float]
    person_id: NotRequired[int | None]
    bed_id: NotRequired[int | None]
    idempotency_key: NotRequired[str]
    event_id: NotRequired[str]
    clip_id: NotRequired[str]
    evidence: NotRequired[EventEvidence]
    audit: NotRequired[EventEvidence]
    snapshot_jpeg: NotRequired[bytes]
    snapshot: NotRequired[EventEvidence]


MutableWorkerEventPayload: TypeAlias = dict[str, EventScalar | bytes | dict[str, EventScalar]]


__all__ = ["MutableWorkerEventPayload", "WorkerEventPayload"]
