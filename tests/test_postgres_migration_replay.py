"""A migrated incident replayed from the worker queue is accepted once, not refused.

Oracles, none of them the code under test:

- ``TARGET_DECISIONS["event_outbox"]`` in the migration mapping: the old runtime's
  pending set is the worker file queue, retained on its volume and replayed, so the
  target outbox starts empty and the replay fills it.
- ADR 0009, recovery: no second event ID or external delivery for a replayed event.
- A minimal relay input and its original identity/content, admitted through the
  public worker delivery queue before migration.
- The old runtime's row for a relayed alert (``relay_projection`` on main): incident
  ``incident:<edge_event_id>`` holding the alert's identity values verbatim.

The SQLite source, worker volume and PostgreSQL target are case-local. The migration
runs through its public calls (export, import, fence, reconcile, transfer). The replay
goes through the real relay route on the migrated root, as the runtime role, to the
contract-exact Hub fixture over loopback HTTP.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from backend.app.edge_db.migration.load import import_snapshot
from backend.app.edge_db.migration.reconcile import delivery_state, reconcile
from backend.app.edge_db.migration.snapshot import export_snapshot
from backend.app.edge_db.migration.sqlite_fence import fence_sqlite
from backend.app.edge_db.migration.transfer import transfer
from backend.app.edge_db.migration.worker_state import queue_digest
from backend.app.edge_db.postgres import PoolBudget
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.postgres_root import (
    API_POSTGRES_AUTHORITY_FILE_ENV,
    API_POSTGRES_DSN_FILE_ENV,
    API_POSTGRES_SCHEMA_ENV,
    close_postgres_database,
    open_postgres_root,
)
from shared.events.delivery_queue import DeliveryQueue, EventEntry
from shared.events.evidence_export_contract import EventReceipt
from shared.events.evidence_http_transport import encode_json, parse_event_result
from tests_support.alert_amplification_runtime import ServedFixture, hub_client
from tests_support.postgres_migration import (
    NOW,
    MigrationTarget,
    authority_file_token,
    insert_row,
    open_source_writer,
)
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import RELAY_HEADERS, relay_postgres_app
from tests_support.sqlite_source import create_schema19_source

pytest_plugins = ("tests_support.postgres_migration",)

_IDENTITY = ("facility_id", "camera_id", "event_type", "probability", "detected_at")
_ROOT_BUDGET = PoolBudget(
    max_connections=2,
    max_waiting=2,
    acquire_timeout_sec=5.0,
    statement_timeout_ms=5000,
    lock_timeout_ms=5000,
    startup_timeout_sec=5.0,
)


@dataclass(frozen=True, slots=True)
class _Wire:
    alert: dict[str, Any]
    body: bytes
    entry: EventEntry


def _worker_wire(event_type: str) -> _Wire:
    alert = {
        "edge_event_id": "11111111-1111-4111-8111-111111111111",
        "event_type": event_type,
        "probability": 0.87,
        "detected_at": NOW,
        "camera_id": "camera-hub",
        "facility_id": "facility-1",
    }
    body = encode_json(alert)
    return _Wire(
        alert=alert,
        body=body,
        entry=EventEntry(
            edge_event_id=alert["edge_event_id"],
            event_type=event_type,
            detected_at=alert["detected_at"],
            camera_id=alert["camera_id"],
            facility_id=alert["facility_id"],
            decision_trace=b"{}",
            values=body,
        ),
    )


def _relayed_source(root: Path, alert: dict[str, Any]) -> tuple[Path, Path]:
    """The old runtime after it committed the alert's incident and before the Hub took it.

    Its audit log is empty: the relay path writes no audit record, and the API verifies
    the migrated chain before it admits the replay.
    """
    source = create_schema19_source(root / "state" / "edge.sqlite3")
    snapshots = root / "snapshots"
    snapshots.mkdir()
    with closing(open_source_writer(source)) as writer:
        insert_row(writer, "edge_site", {"id": 1, "updated_at": NOW})
        insert_row(
            writer,
            "cameras",
            {
                "camera_id": "camera-replay",
                "label": "camera",
                "rtsp_url": "rtsp://camera.invalid/replay",
                "normalized_stream_identity": "stream-replay",
                "backend_camera_id": alert["camera_id"],
                "mapping_state": "MAPPED",
                "never_connected": 1,
                "revision": 1,
                "created_at": NOW,
                "updated_at": NOW,
            },
        )
        insert_row(
            writer,
            "incidents",
            {
                "incident_id": f"incident:{alert['edge_event_id']}",
                "edge_event_id": alert["edge_event_id"],
                **{key: alert[key] for key in _IDENTITY},
                "lifecycle_state": "OPEN",
                "provenance_state": "MISSING",
                "provenance_missing_reason": "NOT_RECORDED",
                "review_version": 0,
                "revision": 1,
                "created_at": alert["detected_at"],
                "updated_at": alert["detected_at"],
            },
        )
    return source, snapshots / "edge.snapshot.sqlite3"


def _worker_volume(root: Path, wire: _Wire) -> Path:
    """A stopped old worker whose only pending delivery is the alert."""
    state = root / "worker-state"
    queue = DeliveryQueue(state / "delivery-queue")
    (state / "delivery-queue-dead-letter").mkdir()
    (state / ".gpu.lease").write_bytes(b"")
    assert queue.try_admit(wire.entry).accepted
    return state


def _migrate(target: MigrationTarget, root: Path, wire: _Wire) -> DeliveryQueue:
    source, destination = _relayed_source(root, wire.alert)
    state = _worker_volume(root, wire)
    entry_path = state / "delivery-queue" / f"{wire.entry.entry_id}.json"
    original_entry = entry_path.read_bytes()
    snapshot = export_snapshot(source, destination).path
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)
    receipts = root / "receipts"
    receipts.mkdir(mode=0o700)
    receipt = receipts / "fence.json"
    generation, _ = authority_file_token(target.authority_path)
    fence_sqlite(source, snapshot=snapshot, generation=generation, receipt=receipt)
    report = reconcile(
        target.database,
        schema=target.schema,
        snapshot_path=snapshot,
        source_path=source,
        worker_state_dir=state,
        expected_queue_sha256=queue_digest(state).sha256,
        fence_receipt=receipt,
    )
    assert (report["result"], report["failures"]) == ("PASS", [])
    assert report["delivery_queue"]["queued"] == 1
    token = transfer(
        target.database, target.authority_path, schema=target.schema, worker_state_dir=state
    )
    assert token.generation == generation + 1
    assert entry_path.read_bytes() == original_entry
    queue = DeliveryQueue(state / "delivery-queue")
    assert queue.accepted_count == 1
    retained = queue.try_admit(wire.entry)
    assert retained.accepted and retained.already_admitted
    return queue


def _root_environ(target: MigrationTarget, root: Path) -> dict[str, str]:
    dsn_path = root / "api-postgres.dsn"
    dsn_path.write_text(
        make_conninfo(target.dsn, options=f"-c role={target.runtime_role}"), encoding="utf-8"
    )
    dsn_path.chmod(0o600)
    return {
        API_POSTGRES_DSN_FILE_ENV: str(dsn_path),
        API_POSTGRES_AUTHORITY_FILE_ENV: str(target.authority_path),
        API_POSTGRES_SCHEMA_ENV: target.schema,
    }


def _rows(target: MigrationTarget, table: str, columns: str, edge_event_id: str) -> list[Any]:
    return target.admin.execute(
        sql.SQL("SELECT {} FROM {} WHERE edge_event_id = %s").format(
            sql.SQL(columns), sql.Identifier(target.schema, table)
        ),
        (edge_event_id,),
    ).fetchall()


def _hub_sends(hub: ServedFixture) -> int:
    return sum(
        1
        for route in hub.fixture.route_ledger
        if (route.method, route.path) == ("POST", "/api/v1/events")
    )


@pytest.mark.parametrize("event_type", ("fall", "bed-exit"))
def test_migrated_pending_alert_replays_to_one_accepted_delivery(
    migration_target: MigrationTarget, tmp_path: Path, event_type: str
) -> None:
    target = migration_target
    wire = _worker_wire(event_type)
    edge_event_id = wire.alert["edge_event_id"]
    identity = (f"incident:{edge_event_id}", *(wire.alert[key] for key in _IDENTITY))
    queue = _migrate(target, tmp_path, wire)
    assert delivery_state(target.admin, target.schema)["outbox_states"] == {}

    root = open_postgres_root(_root_environ(target, tmp_path), budget=_ROOT_BUDGET)
    try:
        sandbox = ProductSandbox(
            target.admin, root.database, root.authority, target.schema, target.dsn
        )
        audit = PostgresAuditRuntime(
            PostgresAuditStore(root.database, root.authority),
            maximum_snapshot_age_sec=10,
            clock=lambda: 0.0,
        )
        assert audit.verify_once() and audit.start_session_once()
        with ServedFixture() as hub:
            # No test camera: the relay resolves the alert's camera from the migrated row.
            app = relay_postgres_app(sandbox, audit, client=hub_client(hub.origin), camera_id=None)
            headers = {**RELAY_HEADERS, "Content-Type": "application/json"}
            with TestClient(app) as client:
                first = client.post("/api/v1/relay/alerts", content=wire.body, headers=headers)
                assert first.status_code == 202, first.text
                hub_event = hub.fixture.event_for_edge_id(edge_event_id)
                assert hub_event is not None
                assert first.json() == {
                    "status": "accepted",
                    "edge_event_id": edge_event_id,
                    "event_id": hub_event.event_id,
                }
                assert _hub_sends(hub) == 1
                assert queue.accepted_count == 1

                # A retry after a lost response lands on the same delivery.
                retried = client.post("/api/v1/relay/alerts", content=wire.body, headers=headers)
                assert (retried.status_code, retried.json()) == (202, first.json())
                assert _hub_sends(hub) == 1
                assert parse_event_result(
                    (retried.status_code, retried.headers, retried.content), edge_event_id
                ) == EventReceipt("accepted", edge_event_id, hub_event.event_id)
                assert queue.acknowledge(wire.entry.entry_id)
                assert queue.accepted_count == 0
    finally:
        close_postgres_database(root.database)

    assert _rows(target, "incidents", "incident_id, " + ", ".join(_IDENTITY), edge_event_id) == [
        identity
    ]
    assert _rows(target, "event_outbox", "state, backend_camera_id", edge_event_id) == [
        ("SENT", wire.alert["camera_id"])
    ]
    delivery = delivery_state(target.admin, target.schema)
    assert delivery["outbox_states"] == {"SENT": 1}
    assert delivery["active_leases"] == 0
