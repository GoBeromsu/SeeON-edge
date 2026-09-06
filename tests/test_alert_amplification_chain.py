"""WP4 measurement: one real same-E T -> E -> B -> I chain, media-free.

Composes the real product seams end to end for a single synthetic
``edge_event_id``:

* T -> E through the real ``IncidentManager`` admission and its durable
  ``EventIdentityStore`` (no frames, no RTSP, no model);
* E -> B through the real ``/api/v1/relay/alerts`` route, real
  ``EdgeIngestClient``/``BackendEvidenceClient`` and the contract-exact Hub
  fixture served over loopback HTTP;
* E -> I through the real ``DurableEvidenceStager`` -> ``EvidenceOutbox`` ->
  central incident staging and the authenticated ``GET /api/v1/incidents``
  projection.

B is captured from the actual fixture receipt; nothing in the join is
hard-coded. Model/policy attribution stays categorically ``판정 불가``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.main import create_app, no_lifespan
from tests_support.alert_amplification_harness import (
    DiagnosticOutcome,
    IncidentProjection,
    classify_rows,
    rows_from_relations,
)
from tests_support.alert_amplification_runtime import RELAY_TOKEN, ServedFixture, relay_client
from worker.pipeline.decision.incident_manager import IncidentManager
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.types import BusinessEvent

_DETECTED_AT = "2026-08-16T00:00:00.000Z"


def _transition(identity: str | int, *, time_sec: float = 100.0) -> BusinessEvent:
    return BusinessEvent(
        domain="fall",
        event_type="fall",
        identity=identity,
        camera_id="room-camera",
        facility_id="facility-1",
        time_sec=time_sec,
        probability=0.91,
    )


def _admit(identity_path: Path, event: BusinessEvent, *, now_sec: float) -> str:
    """Return the durable E minted by real product admission for this T."""

    manager = IncidentManager(cooldown_sec=0.0, identity_path=identity_path)
    admitted = manager.admit(event, now_sec=now_sec)
    assert admitted is not None
    return str(admitted.identity)


def _stage_incident(queue_directory: Path, edge_event_id: str) -> DurableEvidenceStager:
    stager = DurableEvidenceStager(
        queue_directory=queue_directory,
        camera_id="room-camera",
        facility_id="facility-1",
        resident_id=None,
        config_version=1,
        clock=lambda: 1.0,
    )
    stager.stage(
        {
            "edge_event_id": edge_event_id,
            "event_type": "fall",
            "probability": 0.91,
            "detected_at": _DETECTED_AT,
            "camera_id": "room-camera",
            "facility_id": "facility-1",
        }
    )
    return stager


def _projections(database: Path) -> list[IncidentProjection]:
    app = create_app(lifespan=no_lifespan)
    app.state.central_evidence_query = CentralEvidenceQuery(database)
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
            ).status_code
            == 204
        )
        first = client.get("/api/v1/incidents")
        second = client.get("/api/v1/incidents")
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        return [
            IncidentProjection(
                str(item["incident_id"]),
                str(item["edge_event_id"]),
                str(item["detected_at"]),
                str(item["lifecycle_state"]),
                None if item["event_delivery_state"] is None else str(item["event_delivery_state"]),
                None,
            )
            for item in first.json()["incidents"]
        ]


def _deliver(client: TestClient, stager: DurableEvidenceStager) -> str:
    [entry] = tuple(stager.queue.entries())
    payload = json.loads(base64.b64decode(str(entry["values_b64"])))
    response = client.post(
        "/api/v1/relay/alerts",
        json=payload,
        headers={"X-Edge-Relay-Token": RELAY_TOKEN},
    )
    assert response.status_code == 202, response.text
    return str(response.json()["event_id"])


def test_one_transition_yields_one_edge_backend_and_incident_identity(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "identities.jsonl"
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)

    edge_event_id = _admit(identity_path, _transition("onset-1"), now_sec=100.0)
    stager = _stage_incident(tmp_path / "delivery-queue", edge_event_id)

    with ServedFixture() as served:
        relay = relay_client(served.origin, tmp_path, database=database)
        first_backend = _deliver(relay, stager)
        retry_backend = _deliver(relay, stager)

        accepted = served.fixture.accepted_event_ids(edge_event_id)

    projections = _projections(database)

    assert first_backend == retry_backend
    assert len(set(accepted)) == 1
    assert [item.edge_event_id for item in projections] == [edge_event_id]

    rows = rows_from_relations(
        transitions={edge_event_id: "onset-1"},
        attempts={edge_event_id: [1, 2]},
        backend_event_ids={edge_event_id: list(set(accepted))},
        incidents=projections,
        terminal_states={edge_event_id: "ACKED"},
        clock_order_valid=True,
    )
    result = classify_rows(rows)

    assert rows[0].backend_event_ids == (first_backend,)
    assert rows[0].incident_ids == (projections[0].incident_id,)
    assert result.outcome is DiagnosticOutcome.TRANSPORT_RETRY
    assert result.model_policy_cause == "판정 불가"


def test_durable_identity_survives_restart_without_minting_a_second_edge_id(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "identities.jsonl"

    first = _admit(identity_path, _transition("onset-1"), now_sec=100.0)
    # A fresh IncidentManager models a worker restart against the same durable
    # identity store: the same source transition must not mint a new E.
    after_restart = _admit(identity_path, _transition("onset-1"), now_sec=200.0)

    assert first == after_restart


def test_refire_fault_produces_two_edge_ids_for_one_physical_onset(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "identities.jsonl"
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)

    # Test-only refire fault: the same physical onset is admitted under two
    # distinct source identities, which is exactly what worker refire looks
    # like to the durable identity store.
    first_edge = _admit(identity_path, _transition("onset-1"), now_sec=100.0)
    second_edge = _admit(identity_path, _transition("onset-1-refire"), now_sec=101.0)
    assert first_edge != second_edge
    first_stager = _stage_incident(tmp_path / "first-queue", first_edge)
    second_stager = _stage_incident(tmp_path / "second-queue", second_edge)

    with ServedFixture() as served:
        relay = relay_client(served.origin, tmp_path, database=database)
        first_backend = _deliver(relay, first_stager)
        second_backend = _deliver(relay, second_stager)

    assert first_backend != second_backend
    projections = _projections(database)
    assert len(projections) == 2

    rows = rows_from_relations(
        transitions={first_edge: "onset-1", second_edge: "onset-1"},
        attempts={first_edge: [1], second_edge: [1]},
        backend_event_ids={
            first_edge: [first_backend],
            second_edge: [second_backend],
        },
        incidents=projections,
        terminal_states={first_edge: "ACKED", second_edge: "ACKED"},
        clock_order_valid=True,
    )
    result = classify_rows(rows)

    assert result.outcome is DiagnosticOutcome.WORKER_REFIRE
    assert result.model_policy_cause == "판정 불가"


def test_cooldown_collapses_a_repeat_within_the_window(tmp_path: Path) -> None:
    identity_path = tmp_path / "identities.jsonl"
    manager = IncidentManager(cooldown_sec=30.0, identity_path=identity_path)

    admitted = manager.admit(_transition("onset-1"), now_sec=100.0)
    suppressed = manager.admit(_transition("onset-1"), now_sec=110.0)

    assert admitted is not None
    assert suppressed is None


def test_a_failing_identity_journal_still_admits_the_alert(tmp_path: Path) -> None:
    """Durability must never suppress a resident alert.

    The identity journal exists so a restart reuses the same edge event id and
    the backend can deduplicate. It is a durability aid, not the decision. It
    was called unguarded, so any journal I/O failure -- a full disk, a
    permission change, an fsync error -- propagated out of `admit()` and the
    event was never admitted, never queued and never delivered.

    A fresh identity risks a duplicate alert after a restart, which the backend
    already deduplicates. A missing alert is the accident this system exists to
    prevent.
    """
    identity_path = tmp_path / "identity.jsonl"
    manager = IncidentManager(cooldown_sec=0.0, identity_path=identity_path)

    def _explode(_source_key: str) -> str:
        raise OSError("journal device is full")

    manager._identities.resolve = _explode  # type: ignore[method-assign]  # noqa: SLF001

    admitted = manager.admit(_transition("onset-journal-failure"), now_sec=100.0)

    assert admitted is not None, "the alert was suppressed by an identity-journal write failure"
    assert str(admitted.identity), "the admitted alert carries no identity"
    assert manager.identity_journal_failures == 1, (
        "the degradation is not counted, so a silently failing journal looks healthy"
    )


def test_a_malformed_identity_journal_still_lets_the_camera_detect(
    tmp_path: Path,
) -> None:
    """A journal left corrupt by an earlier crash must not disable a camera.

    `EventIdentityStore` loads and validates on construction, and that raised
    out of `IncidentManager.__post_init__`. A single malformed journal file
    therefore stopped that camera activating at all: it detected nothing until
    somebody noticed and deleted the file by hand. Losing the stored identities
    costs deduplication across this restart; losing the camera costs every fall
    it would have seen.
    """
    identity_path = tmp_path / "identity.jsonl"
    corrupt = b"{ this is not valid json\n"
    identity_path.write_bytes(corrupt)

    manager = IncidentManager(cooldown_sec=0.0, identity_path=identity_path)
    admitted = manager.admit(_transition("onset-after-corrupt"), now_sec=100.0)

    assert admitted is not None, "the camera could not admit an event at all"
    assert str(admitted.identity)
    assert manager.identity_journal_failures == 1
    assert identity_path.read_bytes() == corrupt, (
        "the unusable journal was rewritten; an operator can no longer inspect it"
    )


def test_the_identity_fallback_survives_a_full_disk(tmp_path: Path) -> None:
    """The fallback must not fail for the same reason the journal did.

    An earlier version of this fallback created a scratch journal in a temporary
    directory. That reintroduced the exact defect it was fixing: a full disk is
    a very likely reason the real journal failed in the first place, and then
    the fallback's own `mkdtemp` raises too and the camera still never
    activates. `EventIdentityStore(None)` performs no I/O at all.
    """
    import tempfile
    from unittest.mock import patch

    identity_path = tmp_path / "identity.jsonl"
    identity_path.write_bytes(b"{ not valid json\n")

    with patch.object(tempfile, "mkdtemp", side_effect=OSError("ENOSPC")):
        manager = IncidentManager(cooldown_sec=0.0, identity_path=identity_path)
        admitted = manager.admit(_transition("onset-full-disk"), now_sec=100.0)

    assert admitted is not None, (
        "the camera could not activate because the identity fallback needed the "
        "same disk that had already failed"
    )
    assert str(admitted.identity)
    assert manager.identity_journal_failures == 1
