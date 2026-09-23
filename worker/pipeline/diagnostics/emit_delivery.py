"""event.delivery and backend.acceptance payloads."""

from __future__ import annotations

from shared.events.execution_records import PROCESS_SCOPE, WireRecord
from worker.pipeline.diagnostics.record_builder import (
    PRODUCER_BACKEND,
    PRODUCER_EVENT,
    make_record,
    monotonic_or,
)


def event_delivery_record(
    *,
    camera_id: str,
    worker_boot_id: str,
    source_generation: int,
    stream_epoch: int,
    frame_seq: int,
    source_pts_ns: int | None,
    edge_event_id: str,
    event_type: str,
    domain: str,
    admitted: bool,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    return make_record(
        record_kind="event.delivery",
        camera_id=camera_id,
        worker_boot_id=worker_boot_id,
        source_generation=source_generation,
        stream_epoch=stream_epoch,
        producer=PRODUCER_EVENT,
        observed_at_ns=monotonic_or(observed_at_ns),
        time_quality="monotonic",
        causal_unit_id=edge_event_id,
        outcome="admitted" if admitted else "refused",
        payload={
            "edge_event_id": edge_event_id,
            "event_type": event_type,
            "domain": domain,
            "queue": "delivery",
        },
        frame_seq=frame_seq,
        source_pts_ns=source_pts_ns,
    )


def backend_acceptance_record(
    *,
    camera_id: str,
    observing_boot_id: str,
    edge_event_id: str,
    status: str,
    hub_event_id: str,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    """Record that *this* boot observed the Backend receipt for ``edge_event_id``.

    The durable delivery queue can outlive the boot that staged the event, and
    a queue entry carries no origin boot/generation/epoch. backend.acceptance is
    therefore a *process-scoped* kind (declared in the wire contract): the row
    stamps the boot that observed the receipt with PROCESS_SCOPE for
    generation/epoch, and joins to the originating stream through
    ``causal_unit_id == edge_event_id`` (the event.delivery record carries the
    full origin identity). Origin fields are ``null`` in the payload, never a
    fabricated value.
    """
    if status == "accepted_local":
        outcome = "accepted_local"
    elif status == "accepted":
        outcome = "hub-accepted"
    else:
        outcome = status
    return make_record(
        record_kind="backend.acceptance",
        camera_id=camera_id,
        worker_boot_id=observing_boot_id,
        source_generation=PROCESS_SCOPE,
        stream_epoch=PROCESS_SCOPE,
        producer=PRODUCER_BACKEND,
        observed_at_ns=monotonic_or(observed_at_ns),
        time_quality="monotonic",
        causal_unit_id=edge_event_id,
        outcome=outcome,
        payload={
            "edge_event_id": edge_event_id,
            "status": status,
            "hub_event_id": hub_event_id,
            "accepted_local": status == "accepted_local",
            "hub_accepted": status == "accepted" and bool(hub_event_id),
            "observing_boot_id": observing_boot_id,
            # The queue entry does not carry these; join through causal_unit_id.
            "origin_boot_id": None,
            "origin_source_generation": None,
            "origin_stream_epoch": None,
        },
    )


__all__ = ["backend_acceptance_record", "event_delivery_record"]
