"""Closed vocabulary and process-scope for sender event.delivery records."""

from __future__ import annotations

import pytest

from shared.events.execution_records import PROCESS_SCOPE
from worker.pipeline.diagnostics.emit_delivery import (
    DELIVERY_ATTEMPT_OUTCOMES,
    delivery_attempt_record,
    event_delivery_record,
)


def test_delivery_attempt_outcomes_are_closed() -> None:
    assert DELIVERY_ATTEMPT_OUTCOMES == (
        "retry-transient",
        "retry-counted",
        "refused-retained",
        "refused-retention-full",
        "exhausted-retained",
        "exhausted-retention-full",
        "operator-blocked",
        "ack-removal-deferred",
    )


@pytest.mark.parametrize(
    ("outcome", "attempt", "failure_class", "status_code", "retained"),
    [
        ("retry-transient", 0, "RETRY", 503, None),
        ("retry-counted", 1, "exception", None, None),
        ("refused-retained", 0, "PERMANENT", 422, True),
        ("refused-retention-full", 0, "PERMANENT", 422, False),
        ("exhausted-retained", 10, "exhausted", 599, True),
        ("exhausted-retention-full", 10, "exhausted", 599, False),
        ("operator-blocked", 0, "CAMERA_MAPPING_MISSING", 409, False),
        ("ack-removal-deferred", 0, "acknowledge", None, None),
    ],
)
def test_each_outcome_maps_to_process_scoped_payload(
    outcome: str,
    attempt: int,
    failure_class: str,
    status_code: int | None,
    retained: bool | None,
) -> None:
    record = delivery_attempt_record(
        camera_id="camera-a",
        observing_boot_id="boot-observer",
        edge_event_id="event-a",
        outcome=outcome,
        attempt=attempt,
        max_attempts=10,
        failure_class=failure_class,
        status_code=status_code,
        retained=retained,
        dead_letter_dir="/var/lib/seeon/queue-dead-letter",
        queue_kind="EVENT",
        observed_at_ns=7,
    )
    assert record is not None
    assert record.record_kind == "event.delivery"
    assert record.producer == "event"
    assert record.worker_boot_id == "boot-observer"
    assert record.source_generation == PROCESS_SCOPE
    assert record.stream_epoch == PROCESS_SCOPE
    assert record.causal_unit_id == "event-a"
    assert record.outcome == outcome
    assert record.payload == {
        "edge_event_id": "event-a",
        "attempt": attempt,
        "max_attempts": 10,
        "failure_class": failure_class,
        "status_code": status_code,
        "retained": retained,
        "dead_letter_dir": "queue-dead-letter",
        "queue_kind": "EVENT",
    }


def test_unknown_outcome_is_dropped() -> None:
    assert (
        delivery_attempt_record(
            camera_id="camera-a",
            observing_boot_id="boot-observer",
            edge_event_id="event-a",
            outcome="hub-accepted",
            attempt=1,
            max_attempts=10,
            failure_class=None,
            status_code=None,
            retained=None,
        )
        is None
    )


def test_admission_record_stays_stream_scoped() -> None:
    record = event_delivery_record(
        camera_id="camera-a",
        worker_boot_id="boot-origin",
        source_generation=2,
        stream_epoch=4,
        frame_seq=17,
        source_pts_ns=12_000,
        edge_event_id="event-a",
        event_type="fall.detected",
        domain="fall",
        admitted=True,
        reason=None,
        observed_at_ns=9,
    )
    assert record is not None
    assert record.worker_boot_id == "boot-origin"
    assert record.source_generation == 2
    assert record.stream_epoch == 4
    assert record.frame_seq == 17
    assert record.outcome == "admitted"
    assert record.causal_unit_id == "event-a"
