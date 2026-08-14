from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from backend.app.features.evidence.record_store import (
    CentralEvidenceQuery,
    CentralEvidenceReviewStore,
    EvidenceReviewConflictError,
    ReviewDisposition,
)
from backend.app.main import create_app, no_lifespan
from shared.edge_db.connection import RuntimeActor, open_runtime_database
from shared.edge_db.importer import LegacyDatabasePaths, import_legacy_databases
from shared.edge_db.migrator import migrate_database
from shared.edge_db.schema import MIGRATIONS
from worker.pipeline.output.evidence.evidence_records import EvidenceRecordStore

INCIDENT_ID = "incident:opaque/review"
EVENT_ID = "event:opaque/review"
CLIP_ID = "clip:opaque/review"
REVIEWED_AT = "2026-08-13T12:00:00Z"


def _insert_incident_relation(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO evidence_events "
        "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at) "
        "VALUES (?, ?, '{}', 'STAGED', 1, 1)",
        (EVENT_ID, REVIEWED_AT),
    )
    connection.execute(
        "INSERT INTO evidence_clips (clip_id, local_state, state_version) "
        "VALUES (?, 'VERIFIED', 1)",
        (CLIP_ID,),
    )
    connection.execute(
        """
        INSERT INTO evidence_incidents (
            incident_id, edge_event_id, camera_id, event_type, detected_at,
            provenance_missing_reason, primary_clip_id, lifecycle_state,
            created_at, updated_at
        ) VALUES (?, ?, 'camera:opaque', 'fall', ?, 'NOT_RECORDED', ?,
                  'STAGING', ?, ?)
        """,
        (INCIDENT_ID, EVENT_ID, REVIEWED_AT, CLIP_ID, REVIEWED_AT, REVIEWED_AT),
    )
    connection.execute(
        """
        INSERT INTO evidence_primary_clips (
            incident_id, clip_id, source_packet_preserved, source_missing_reason,
            truncation_json, unavailable_reason, created_at
        ) VALUES (?, ?, 0, 'NOT_RECORDED', '[]', 'MISSING', ?)
        """,
        (INCIDENT_ID, CLIP_ID, REVIEWED_AT),
    )


def _migrated_with_incident(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _insert_incident_relation(connection)
        connection.commit()
    return database


def test_review_rows_require_existing_central_incident_and_clip_foreign_keys(
    tmp_path: Path,
) -> None:
    database = _migrated_with_incident(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        statement = """
            INSERT INTO control_evidence_review_revisions (
                review_id, incident_id, clip_id, review_version, actor_id,
                reviewed_at, disposition, notes
            ) VALUES (?, ?, ?, 1, 'operator:opaque', ?, 'TRUE_POSITIVE', NULL)
        """
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY|central evidence relation"):
            connection.execute(
                statement,
                ("review:missing-incident", "incident:missing", CLIP_ID, REVIEWED_AT),
            )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY|central evidence relation"):
            connection.execute(
                statement,
                ("review:missing-clip", INCIDENT_ID, "clip:missing", REVIEWED_AT),
            )

    api_connection = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            api_connection.execute(
                "ALTER TABLE control_evidence_review_state ADD COLUMN forbidden TEXT"
            )
    finally:
        api_connection.close()

    service = CentralEvidenceReviewStore(database)
    with pytest.raises(ValueError, match="central evidence relation"):
        service.update(
            incident_id="incident:missing",
            clip_id=CLIP_ID,
            expected_version=0,
            actor_id="operator:opaque",
            reviewed_at=REVIEWED_AT,
            disposition=ReviewDisposition.FALSE_POSITIVE,
            notes=None,
        )


def test_review_updates_are_versioned_immutable_and_optimistically_concurrent(
    tmp_path: Path,
) -> None:
    database = _migrated_with_incident(tmp_path)
    service = CentralEvidenceReviewStore(database)
    first = service.update(
        incident_id=INCIDENT_ID,
        clip_id=CLIP_ID,
        expected_version=0,
        actor_id="operator:first",
        reviewed_at=REVIEWED_AT,
        disposition=ReviewDisposition.TRUE_POSITIVE,
        notes="bounded operator note",
    )
    assert first.version == 1

    barrier = Barrier(2)

    def revise(actor_id: str) -> object:
        barrier.wait()
        try:
            return service.update(
                incident_id=INCIDENT_ID,
                clip_id=CLIP_ID,
                expected_version=1,
                actor_id=actor_id,
                reviewed_at="2026-08-13T12:01:00Z",
                disposition=ReviewDisposition.FALSE_POSITIVE,
                notes=None,
            )
        except EvidenceReviewConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(revise, ("operator:a", "operator:b")))

    assert sum(not isinstance(value, Exception) for value in outcomes) == 1
    assert sum(isinstance(value, EvidenceReviewConflictError) for value in outcomes) == 1
    with sqlite3.connect(database) as connection:
        history = connection.execute(
            "SELECT review_version, actor_id, disposition, notes "
            "FROM control_evidence_review_revisions ORDER BY review_version"
        ).fetchall()
        assert history[0] == (
            1,
            "operator:first",
            "TRUE_POSITIVE",
            "bounded operator note",
        )
        assert history[1][0] == 2
        assert history[1][2:] == ("FALSE_POSITIVE", None)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE control_evidence_review_revisions SET notes = 'rewrite' "
                "WHERE review_version = 1"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE evidence_incidents SET camera_id = 'rewritten', revision = revision + 1 "
                "WHERE incident_id = ?",
                (INCIDENT_ID,),
            )


def test_review_contract_bounds_actor_time_disposition_and_notes(tmp_path: Path) -> None:
    database = _migrated_with_incident(tmp_path)
    service = CentralEvidenceReviewStore(database)
    common = {
        "incident_id": INCIDENT_ID,
        "clip_id": CLIP_ID,
        "expected_version": 0,
        "reviewed_at": REVIEWED_AT,
        "disposition": ReviewDisposition.TRUE_POSITIVE,
        "notes": None,
    }
    with pytest.raises(ValueError, match="actor_id"):
        service.update(actor_id="x" * 129, **common)
    with pytest.raises(ValueError, match="reviewed_at"):
        service.update(actor_id="operator", **(common | {"reviewed_at": "not-a-time"}))
    with pytest.raises(ValueError, match="notes"):
        service.update(actor_id="operator", **(common | {"notes": "x" * 1001}))
    with pytest.raises(ValueError, match="notes"):
        service.update(actor_id="operator", **(common | {"notes": "unsafe\x00note"}))


def test_review_relation_is_projected_by_worker_and_backend_and_survives_tombstone(
    tmp_path: Path,
) -> None:
    database = _migrated_with_incident(tmp_path)
    review = CentralEvidenceReviewStore(database).update(
        incident_id=INCIDENT_ID,
        clip_id=CLIP_ID,
        expected_version=0,
        actor_id="operator:opaque",
        reviewed_at=REVIEWED_AT,
        disposition=ReviewDisposition.FALSE_POSITIVE,
        notes="No resident identifiers.",
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_retention_states "
            "(clip_id, state, revision, requested_at, updated_at) "
            "VALUES (?, 'PURGED', 1, ?, ?)",
            (CLIP_ID, REVIEWED_AT, REVIEWED_AT),
        )
        connection.commit()

    worker_record = EvidenceRecordStore(database).get(INCIDENT_ID)
    backend_record = CentralEvidenceQuery(database).get(INCIDENT_ID)
    assert worker_record is not None and worker_record.review is not None
    assert backend_record is not None and backend_record.review is not None
    assert worker_record.review.review_id == review.review_id
    assert worker_record.review.version == 1
    assert backend_record.review.disposition is ReviewDisposition.FALSE_POSITIVE
    assert backend_record.review.notes == "No resident identifiers."
    assert backend_record.retention_state == "PURGED"
    assert not hasattr(backend_record.review, "payload_json")


def test_operator_incident_api_projects_lifecycle_and_uses_review_cas(tmp_path: Path) -> None:
    database = _migrated_with_incident(tmp_path)
    app = create_app(lifespan=no_lifespan)
    app.state.central_evidence_query = CentralEvidenceQuery(database)
    app.state.central_evidence_review_store = CentralEvidenceReviewStore(database)
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
            ).status_code
            == 204
        )
        listed = client.get("/api/v1/incidents", params={"limit": 10})
        assert listed.status_code == 200
        body = listed.json()
        assert body["pagination"] == {
            "limit": 10,
            "next_cursor": None,
            "has_more": False,
        }
        incident = body["incidents"][0]
        assert incident["incident_id"] == INCIDENT_ID
        assert incident["primary_artifact_state"] is None
        bad_cursor = client.get("/api/v1/incidents", params={"cursor": "not-a-cursor"})
        assert bad_cursor.status_code == 400
        reviewed = client.put(
            f"/api/v1/incident-reviews/{INCIDENT_ID}",
            json={"expected_version": 0, "disposition": "TRUE_POSITIVE", "notes": None},
        )
        conflict = client.put(
            f"/api/v1/incident-reviews/{INCIDENT_ID}",
            json={"expected_version": 0, "disposition": "FALSE_POSITIVE", "notes": None},
        )
    assert reviewed.status_code == 200
    assert reviewed.json()["review"]["version"] == 1
    assert conflict.status_code == 409


def test_v9_migration_promotes_legitimate_label_and_classifies_every_orphan(
    tmp_path: Path,
) -> None:
    database = tmp_path / "upgrade.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:9])
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _insert_incident_relation(connection)
        connection.execute("INSERT INTO evidence_clips (clip_id) VALUES ('clip:no-incident')")
        labels = (
            (CLIP_ID, "TRUE_POSITIVE", "legacy:operator", REVIEWED_AT),
            ("clip:no-incident", "FALSE_POSITIVE", "legacy:operator", REVIEWED_AT),
            ("clip:missing", "FALSE_POSITIVE", "legacy:operator", REVIEWED_AT),
            ("clip:unsupported", None, "legacy:operator", REVIEWED_AT),
        )
        for clip_id, label, reviewer, reviewed_at in labels:
            connection.execute(
                "INSERT INTO labels (clip_id, label, reviewer, reviewed_at, payload_json) "
                "VALUES (?, ?, ?, ?, '{}')",
                (clip_id, label, reviewer, reviewed_at),
            )
        connection.commit()

    migrate_database(database)

    with sqlite3.connect(database) as connection:
        migrated = connection.execute(
            "SELECT incident_id, clip_id, review_version, actor_id, disposition "
            "FROM control_evidence_review_revisions"
        ).fetchall()
        classifications = dict(
            connection.execute(
                "SELECT source_clip_id, classification FROM control_legacy_label_migrations"
            ).fetchall()
        )
    assert migrated == [(INCIDENT_ID, CLIP_ID, 1, "legacy:operator", "TRUE_POSITIVE")]
    assert classifications == {
        CLIP_ID: "MIGRATED",
        "clip:no-incident": "ORPHAN_INCIDENT",
        "clip:missing": "ORPHAN_CLIP",
        "clip:unsupported": "UNSUPPORTED_DISPOSITION",
    }


def test_imported_legacy_label_without_central_incident_is_explicitly_classified(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "legacy-catalog.sqlite3"
    worker = tmp_path / "legacy-worker.sqlite3"
    missing_connection = tmp_path / "missing-connection.sqlite3"
    with sqlite3.connect(catalog) as connection:
        connection.execute("PRAGMA user_version = 3")
        connection.execute(
            "CREATE TABLE labels (clip_id TEXT PRIMARY KEY, label TEXT, reviewer TEXT, "
            "reviewed_at TEXT, payload_json TEXT NOT NULL) STRICT"
        )
        connection.execute(
            "INSERT INTO labels VALUES (?, 'TRUE_POSITIVE', 'legacy:operator', ?, '{}')",
            (CLIP_ID, REVIEWED_AT),
        )
    with sqlite3.connect(worker) as connection:
        connection.execute("PRAGMA user_version = 8")
        connection.execute("CREATE TABLE evidence_clips (clip_id TEXT PRIMARY KEY) STRICT")
        connection.execute("INSERT INTO evidence_clips VALUES (?)", (CLIP_ID,))

    target = tmp_path / "edge" / "edge.sqlite3"
    import_legacy_databases(
        target,
        LegacyDatabasePaths(catalog, missing_connection, worker),
    )

    with sqlite3.connect(target) as connection:
        assert connection.execute(
            "SELECT classification FROM control_legacy_label_migrations WHERE source_clip_id = ?",
            (CLIP_ID,),
        ).fetchone() == ("ORPHAN_INCIDENT",)
        assert connection.execute(
            "SELECT count(*) FROM control_evidence_review_revisions"
        ).fetchone() == (0,)
