"""Regression for the vendor-drift remediation (ADR-0006).

A prior change on this branch mutated ``contracts.event.EventPayload`` from
its pinned vendored ``Mapping`` alias into a worker-staging ``TypedDict``
with a required ``bytes`` ``snapshot_jpeg`` field -- a cross-instance
contract drift caught by ``eldercare-dataset-ops``'s
``tests/test_vendor_drift.py``. This test pins ``contracts/event.py`` back to
its canonical byte-identical source and proves the worker-local staging
envelope that replaced the drifted shape (
``worker/pipeline/output/evidence/event_payload.WorkerEventPayload``) stays
inside ``worker/`` and never leaks its ``bytes`` field across the shared
relay boundary (``shared/events/edge_ingest_client.py``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import get_args, get_type_hints

from contracts.event import EventPayload, EventScalar, MutableEventPayload
from worker.pipeline.output.evidence.event_payload import (
    MutableWorkerEventPayload,
    WorkerEventPayload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# sha256 of contracts/event.py as committed at bb85a5f (the pinned canonical
# fall-ml-v2 commit dataset-ops's ml/contracts/event.py is vendored from).
CANONICAL_EVENT_SHA256 = "d6d0e5125d93722514e3c519340a7c0d01bb10d387ffed37117e7c74e69be237"


def test_canonical_contract_event_module_is_byte_identical_to_pinned_source() -> None:
    content = (REPO_ROOT / "contracts" / "event.py").read_bytes()

    assert hashlib.sha256(content).hexdigest() == CANONICAL_EVENT_SHA256


def test_canonical_event_payload_stays_a_bytes_free_mapping_alias() -> None:
    """The vendored contract must never regress into a TypedDict carrying bytes."""
    key_type, value_type = get_args(EventPayload)
    assert key_type is str
    assert bytes not in _flatten_union(value_type)

    mutable_value_type = get_args(MutableEventPayload)[1]
    assert bytes not in _flatten_union(mutable_value_type)
    assert bytes not in _flatten_union(EventScalar)


def _flatten_union(type_arg: object) -> tuple[object, ...]:
    args = get_args(type_arg)
    if not args:
        return (type_arg,)
    flattened: list[object] = []
    for arg in args:
        flattened.extend(_flatten_union(arg))
    return tuple(flattened)


def test_worker_local_staging_payload_requires_strict_bytes_snapshot_field() -> None:
    """The worker-local staging envelope types its snapshot field as ``bytes``,
    never ``Any`` and never a bare unannotated field, and accepts real bytes.
    """
    hints = get_type_hints(WorkerEventPayload, include_extras=True)
    snapshot_hint = hints["snapshot_jpeg"]

    # NotRequired[bytes] unwraps to bytes -- not Any, not bytes | None.
    assert get_args(snapshot_hint) == (bytes,)

    payload: WorkerEventPayload = {
        "edge_event_id": "event-1",
        "snapshot_jpeg": b"jpeg-bytes",
    }
    assert isinstance(payload["snapshot_jpeg"], bytes)

    mutable: MutableWorkerEventPayload = {}
    mutable["snapshot_jpeg"] = b"jpeg-bytes"
    assert isinstance(mutable["snapshot_jpeg"], bytes)

    # The worker envelope's non-bytes fields still satisfy the canonical
    # Mapping contract's key/value shape (no widening beyond scalars/bytes).
    assert isinstance(payload, Mapping)


def test_worker_local_staging_payload_does_not_leak_across_shared_boundary() -> None:
    """``shared/events/edge_ingest_client.py`` (the cross-repository relay
    client) must stay on the canonical, bytes-free ``contracts.event.EventPayload``
    and never import the worker-local staging envelope.
    """
    client_source = (REPO_ROOT / "shared" / "events" / "edge_ingest_client.py").read_text(
        encoding="utf-8"
    )

    assert "worker.pipeline.output.evidence.event_payload" not in client_source
    assert "WorkerEventPayload" not in client_source
    assert "from contracts.event import EventPayload" in client_source


def test_worker_local_staging_type_is_scoped_under_worker() -> None:
    module_path = Path("worker/pipeline/output/evidence/event_payload.py")

    assert (REPO_ROOT / module_path).is_file()
    assert WorkerEventPayload.__module__ == "worker.pipeline.output.evidence.event_payload"
