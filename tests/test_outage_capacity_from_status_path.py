"""Outage capacity must be derived from what `GET /api/v1/status` reports.

The goal states it plainly: measure attachment- and disposition-aware capacity
through the real worker POST to runtime-status-store to `GET /api/v1/status`
path, and re-derive both the 13-camera and extrapolated 50-camera budgets;
execution may not declare capacity closed using any stale formula.

An earlier derivation computed the budgets from the live SQLite database
instead. The arithmetic was right, but a number obtained by a route nobody
operates is not a measurement of the system an operator can observe. This module
therefore drives real entries through the real status path and derives the
budgets from the values that endpoint actually returns, so the figures in the
cutover runbook are reproducible from a running deployment rather than from a
one-off query.

Incidence is the one input this cannot synthesise: 1143 live events over 34.5
hours across 13 cameras, giving 2.55 events per camera-hour. That is measured
observation, recorded here as a named constant so a re-measurement after the
lease-backpressure repair updates one place.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan
from shared.events.delivery_queue import (
    DeliveryQueue,
    EventEntry,
    SnapshotAttachmentEntry,
    SnapshotDispositionEntry,
)
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RelayRuntimeStatusTransport,
    RuntimeStatusSender,
)


def _client() -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_inventory = {
        "camera-1": {"camera_id": "camera-1", "facility_id": "facility-1"}
    }
    return TestClient(app)


def _post_queue_capacity(client: TestClient, queue: DeliveryQueue) -> dict[str, object]:
    """Drive the real chain: transport, relay route, store, GET /api/v1/status.

    `RelayRuntimeStatusTransport` is used rather than a fake so the URL it
    builds and the Authorization header it sets are exercised too; a capture
    stub skips exactly the parts a deployment gets wrong.
    """

    def _request(
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes,
        _timeout: float,
        _on_response: object = None,
    ) -> tuple[int, dict[str, str], bytes]:
        # The transport built this URL and these headers; hand them to the real
        # app rather than to a socket.
        assert method == "POST"
        assert headers["Authorization"] == "Bearer relay-token"
        response = client.post(
            url.replace("http://relay.test", ""),
            content=body,
            headers=headers,
        )
        return (response.status_code, dict(response.headers), response.content)

    transport = RelayRuntimeStatusTransport(
        "http://relay.test", "relay-token", request=_request
    )
    sender = RuntimeStatusSender(
        WorkerDiagnostics(), "facility-1", transport, delivery_queue=queue
    )
    assert sender.publish_once()

    body = client.get("/api/v1/status").json()
    facility = body["runtime"]["facilities"]["facility-1"]
    reported = facility["delivery_queue"]
    assert isinstance(reported, dict), "GET /api/v1/status did not report the queue"
    return reported

#: Measured from the live deployment: 1143 events / 34.5 h / 13 cameras.
OBSERVED_EVENTS_PER_CAMERA_HOUR = 2.55

#: The plan's outage-survival target.
TARGET_OUTAGE_HOURS = 72.0

_ROSTERS = (13, 50)


def _populated_queue(directory: Path, *, falls: int) -> DeliveryQueue:
    """One EVENT plus one ATTACHMENT and one DISPOSITION per fall.

    Three entries per fall is the conservative shape: `EvidenceEventSink` can
    admit an attachment and then add a disposition when its commit fails, so a
    two-entry assumption would overstate the horizon.
    """
    queue = DeliveryQueue(directory)
    for index in range(falls):
        event_id = f"event-{index}"
        assert queue.try_admit(
            EventEntry(
                edge_event_id=event_id,
                event_type="fall",
                detected_at="2026-08-22T00:00:00Z",
                camera_id="camera-1",
                facility_id="facility-1",
                decision_trace=b"t" * 4096,
                values=b"v" * 8192,
            )
        ).accepted
        assert queue.try_admit(
            SnapshotAttachmentEntry(
                event_id,
                f"snapshot-{index}",
                "a" * 64,
                f"snapshots/snapshot-{index}.jpg",
                184320,
                "image/jpeg",
            )
        ).accepted
        assert queue.try_admit(
            SnapshotDispositionEntry(event_id, f"snapshot-{index}", "unavailable", "stage_failed")
        ).accepted
    return queue


@pytest.fixture(name="reported")
def _reported(tmp_path: Path) -> dict[str, object]:
    """Capacity as `GET /api/v1/status` reports it, not as we computed it."""
    queue = _populated_queue(tmp_path / "delivery-queue", falls=8)
    return _post_queue_capacity(_client(), queue)


def test_the_status_path_reports_the_kind_mix_a_fall_actually_produces(
    reported: dict[str, object],
) -> None:
    """Attachment- and disposition-aware means the mix must be visible."""
    by_kind = reported["by_kind"]
    assert isinstance(by_kind, dict)

    assert by_kind["EVENT"] == 8
    assert by_kind["SNAPSHOT_ATTACHMENT"] == 8
    assert by_kind["SNAPSHOT_DISPOSITION"] == 8
    assert reported["accepted_count"] == 24


def test_capacity_is_derivable_from_the_endpoint_without_hardcoding_bounds(
    reported: dict[str, object],
) -> None:
    """A reader must be able to compute headroom from the response alone."""
    for field in ("accepted_count", "accepted_bytes", "max_accepted_entries", "max_accepted_bytes"):
        value = reported[field]
        assert isinstance(value, int) and value > 0, f"{field} is not usable for arithmetic"

    assert int(reported["accepted_count"]) <= int(reported["max_accepted_entries"])
    assert int(reported["accepted_bytes"]) <= int(reported["max_accepted_bytes"])


def test_the_seventy_two_hour_target_is_not_met_at_either_roster(
    reported: dict[str, object],
) -> None:
    """Derive the budgets from reported values and pin the negative result.

    This is the finding the runbook records. If a future change makes the target
    reachable -- a larger bound, a smaller envelope, fewer entries per fall --
    this test fails and the runbook must be corrected rather than silently
    drifting out of date.
    """
    accepted_count = int(reported["accepted_count"])
    accepted_bytes = int(reported["accepted_bytes"])
    max_entries = int(reported["max_accepted_entries"])
    max_bytes = int(reported["max_accepted_bytes"])

    falls = accepted_count // 3
    entries_per_fall = accepted_count / falls
    bytes_per_fall = accepted_bytes / falls

    for cameras in _ROSTERS:
        falls_per_hour = OBSERVED_EVENTS_PER_CAMERA_HOUR * cameras
        entry_hours = max_entries / (falls_per_hour * entries_per_fall)
        byte_hours = max_bytes / (falls_per_hour * bytes_per_fall)
        survivable = min(entry_hours, byte_hours)

        assert survivable < TARGET_OUTAGE_HOURS, (
            f"{cameras} cameras now survive {survivable:.1f}h, which meets the "
            f"{TARGET_OUTAGE_HOURS:.0f}h target. That is good news, but the cutover "
            f"runbook still records the shortfall -- update it and this test together."
        )


def test_the_entry_bound_binds_before_the_byte_bound(
    reported: dict[str, object],
) -> None:
    """The runbook says raising only the byte ceiling would buy nothing."""
    accepted_count = int(reported["accepted_count"])
    accepted_bytes = int(reported["accepted_bytes"])
    max_entries = int(reported["max_accepted_entries"])
    max_bytes = int(reported["max_accepted_bytes"])

    entry_headroom = max_entries / accepted_count
    byte_headroom = max_bytes / accepted_bytes

    assert entry_headroom < byte_headroom, (
        "the byte ceiling now binds before the entry ceiling, which inverts the "
        "runbook's capacity guidance; re-derive both and update the runbook"
    )


def test_retained_refused_evidence_reaches_the_operator_status_endpoint(
    tmp_path: Path,
) -> None:
    """Retention is only actionable if the deployment reports it.

    Dead-lettered evidence is refused, retained on disk, and needs an operator.
    Counting it inside the queue object is not enough: the operator reads
    `GET /api/v1/status`, so the count has to cross the wire model, the relay
    route and the store to get there.
    """
    queue = _populated_queue(tmp_path / "delivery-queue", falls=1)
    entry_id = str(next(iter(queue.entries()))["entry_id"])
    assert queue.dead_letter(entry_id, 422)

    with _client() as client:
        reported = _post_queue_capacity(client, queue)

    assert reported["dead_lettered_count"] == 1, (
        "refused evidence never reaches GET /api/v1/status, so nobody learns "
        "it is sitting on disk undelivered"
    )
    assert reported["dead_lettered_bytes"] > 0
