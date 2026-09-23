"""event.delivery and backend.acceptance payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from shared.events.execution_records import PROCESS_SCOPE, WireRecord
from worker.pipeline.diagnostics.record_builder import (
    PRODUCER_BACKEND,
    PRODUCER_EVENT,
    make_record,
    monotonic_or,
)

#: Closed vocabulary for the sender's own event.delivery dispositions.
#: These are not Hub acceptance; backend.acceptance is the only record that
#: says accepted_local / hub-accepted.
#:
#: retry-transient: DeliveryDisposition.RETRY / 5xx / unreachable; attempt
#:     budget is not consumed.
#: retry-counted: attempt consumed (send exception, PERMANENT non-4xx, or
#:     receipt edge_event_id mismatch).
#: refused-retained: PERMANENT 4xx retained in the dead-letter directory.
#: refused-retention-full: PERMANENT 4xx but the retention area is full; the
#:     entry stays queued.
#: exhausted-retained / exhausted-retention-full: attempt budget spent.
#: operator-blocked: CAMERA_MAPPING_MISSING. EvidenceSender.run_once applies
#:     that wait only to CLIP entries, so EVENT entries never take this
#:     outcome today; the builder still accepts it if an EVENT path hits it.
#: ack-removal-deferred: delivered but queue.acknowledge failed.
DELIVERY_ATTEMPT_OUTCOMES: Final = (
    "retry-transient",
    "retry-counted",
    "refused-retained",
    "refused-retention-full",
    "exhausted-retained",
    "exhausted-retention-full",
    "operator-blocked",
    "ack-removal-deferred",
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
    reason: str | None = None,
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    """Stream-scoped queue admission observed against the triggering frame.

    ``admitted`` must be the durable queue's proof (try_admit.accepted, or
    the queue's AdmissionResult.accepted being True). Never pass True without that proof.
    """
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
            "reason": reason,
        },
        frame_seq=frame_seq,
        source_pts_ns=source_pts_ns,
    )


def delivery_attempt_record(
    *,
    camera_id: str,
    observing_boot_id: str,
    edge_event_id: str,
    outcome: str,
    attempt: int,
    max_attempts: int,
    failure_class: str | None,
    status_code: int | None,
    retained: bool | None,
    dead_letter_dir: str | None = None,
    queue_kind: str = "EVENT",
    observed_at_ns: int | None = None,
) -> WireRecord | None:
    """Record that *this* boot observed a sender disposition for ``edge_event_id``.

    These are the sender's own dispositions — not Hub acceptance.
    ``backend.acceptance`` remains the only record that says accepted_local
    or hub-accepted.

    The durable delivery queue can outlive the boot that staged the event, and
    a queue entry carries no origin boot/generation/epoch. A sender
    event.delivery is therefore process-scoped in the same way as
    backend.acceptance: the row stamps the boot that observed the attempt with
    PROCESS_SCOPE for generation/epoch, and joins to the originating stream
    through ``causal_unit_id == edge_event_id`` (the stream-scoped admission
    event.delivery carries the full origin identity). ``dead_letter_dir`` is
    the directory name, never a full path.
    """
    if outcome not in DELIVERY_ATTEMPT_OUTCOMES:
        return None
    dir_name = None if dead_letter_dir is None else Path(dead_letter_dir).name
    return make_record(
        record_kind="event.delivery",
        camera_id=camera_id,
        worker_boot_id=observing_boot_id,
        source_generation=PROCESS_SCOPE,
        stream_epoch=PROCESS_SCOPE,
        producer=PRODUCER_EVENT,
        observed_at_ns=monotonic_or(observed_at_ns),
        time_quality="monotonic",
        causal_unit_id=edge_event_id,
        outcome=outcome,
        payload={
            "edge_event_id": edge_event_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "failure_class": failure_class,
            "status_code": status_code,
            "retained": retained,
            "dead_letter_dir": dir_name,
            "queue_kind": queue_kind,
        },
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


__all__ = [
    "DELIVERY_ATTEMPT_OUTCOMES",
    "backend_acceptance_record",
    "delivery_attempt_record",
    "event_delivery_record",
]
