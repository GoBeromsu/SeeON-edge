from __future__ import annotations

import json
from base64 import b64decode
from pathlib import Path

from shared.events.delivery_queue import DeliveryQueue, EventEntry


def test_decision_basis_is_admitted_with_event_while_detail_drop_is_observable(
    tmp_path: Path,
) -> None:
    decision_trace = json.dumps({"reason": "fall-onset"}, sort_keys=True).encode()
    values = json.dumps({"fall_probability": 0.98}, sort_keys=True).encode()
    queue = DeliveryQueue(tmp_path / "delivery")

    admitted = queue.try_admit(
        EventEntry(
            edge_event_id="event-a",
            event_type="fall",
            detected_at="2026-08-21T00:00:00Z",
            camera_id="camera-a",
            facility_id="facility-a",
            decision_trace=decision_trace,
            values=values,
        )
    )

    assert admitted.accepted
    entry = next(queue.entries())
    assert b64decode(str(entry["decision_trace_b64"])) == decision_trace
    assert b64decode(str(entry["values_b64"])) == values
