from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from fastapi.testclient import TestClient

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.evidence.record_store import (
    CentralEvidenceQuery,
    CentralEvidenceReviewStore,
    EvidenceReviewConflictError,
    ReviewDisposition,
)
from backend.app.main import create_app, no_lifespan

INCIDENT_ID = "incident-review"
EVENT_ID = "event-review"
CLIP_ID = "clip-review"
REVIEWED_AT = "2026-08-13T12:00:00Z"
HASH = "ab" * 32


def _insert_incident(
    connection: sqlite3.Connection,
    *,
    artifact_state: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO incidents (
            incident_id, edge_event_id, facility_id, camera_id, event_type,
            probability, detected_at, lifecycle_state, provenance_state,
            provenance_missing_reason, review_version, revision, created_at, updated_at
        ) VALUES (?, ?, 'facility-1', 'camera-1', 'fall', 0.8, ?, 'OPEN',
                  'MISSING', 'NOT_RECORDED', 0, 1, ?, ?)
        """,
        (INCIDENT_ID, EVENT_ID, REVIEWED_AT, REVIEWED_AT, REVIEWED_AT),
    )
    if artifact_state is None:
        return
    connection.execute(
        """
        INSERT INTO clips (
            clip_id, camera_id, event_facet, started_at, manifest_relpath,
            media_relpath, manifest_sha256, media_sha256, manifest_size_bytes,
            media_size_bytes, local_state, publish_state, retention_state,
            revision, created_at, updated_at
        ) VALUES (?, 'camera-1', 'fall', ?, 'clips/clip-review/manifest.json',
                  'clips/clip-review/clip.mp4', ?, ?, 10, 10, 'AVAILABLE',
                  'WAITING', 'RETAINED', 1, ?, ?)
        """,
        (CLIP_ID, REVIEWED_AT, HASH, HASH, REVIEWED_AT, REVIEWED_AT),
    )
    if artifact_state == "AVAILABLE":
        connection.execute(
            """
            INSERT INTO artifacts (
                incident_id, kind, artifact_id, clip_id, state, contained_relpath,
                content_sha256, size_bytes, mime_type, codec, revision, created_at, updated_at
            ) VALUES (?, 'PRIMARY_CLIP', 'artifact-review', ?, 'AVAILABLE',
                      'clips/clip-review/clip.mp4', ?, 10, 'video/mp4', 'h264', 1, ?, ?)
            """,
            (INCIDENT_ID, CLIP_ID, HASH, REVIEWED_AT, REVIEWED_AT),
        )
    else:
        connection.execute(
            """
            INSERT INTO artifacts (
                incident_id, kind, artifact_id, clip_id, state, reason,
                revision, created_at, updated_at
            ) VALUES (?, 'PRIMARY_CLIP', 'artifact-review', ?, 'PURGED',
                      'OPERATOR_DELETE', 1, ?, ?)
            """,
            (INCIDENT_ID, CLIP_ID, REVIEWED_AT, REVIEWED_AT),
        )


def _database(tmp_path: Path, *, artifact_state: str | None = None) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _insert_incident(connection, artifact_state=artifact_state)
    return database


def test_review_is_incident_local_when_primary_clip_is_missing(tmp_path: Path) -> None:
    # Given: an incident with no primary artifact row.
    store = CentralEvidenceReviewStore(_database(tmp_path))

    # When: an operator submits review version zero.
    review = store.update(
        incident_id=INCIDENT_ID,
        expected_version=0,
        actor_id="operator-1",
        reviewed_at=REVIEWED_AT,
        disposition=ReviewDisposition.TRUE_POSITIVE,
        notes=None,
    )

    # Then: the incident-local CAS advances without requiring media.
    assert review.version == 1
    assert review.clip_id is None


def test_review_updates_are_optimistically_concurrent(tmp_path: Path) -> None:
    # Given: one committed review and two writers subscribed at the same version.
    database = _database(tmp_path, artifact_state="AVAILABLE")
    store = CentralEvidenceReviewStore(database)
    store.update(
        incident_id=INCIDENT_ID,
        expected_version=0,
        actor_id="operator-first",
        reviewed_at=REVIEWED_AT,
        disposition=ReviewDisposition.TRUE_POSITIVE,
        notes="first",
    )
    barrier = Barrier(2)

    def revise(actor_id: str) -> object:
        barrier.wait()
        try:
            return store.update(
                incident_id=INCIDENT_ID,
                expected_version=1,
                actor_id=actor_id,
                reviewed_at="2026-08-13T12:01:00Z",
                disposition=ReviewDisposition.FALSE_POSITIVE,
                notes=None,
            )
        except EvidenceReviewConflictError as error:
            return error

    # When: both writers perform the same compare-and-swap.
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(revise, ("operator-a", "operator-b")))

    # Then: exactly one revision wins and identity fields remain unchanged.
    assert sum(not isinstance(value, Exception) for value in outcomes) == 1
    assert sum(isinstance(value, EvidenceReviewConflictError) for value in outcomes) == 1
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT review_version, review_disposition, revision FROM incidents"
        ).fetchone()
    assert row == (2, "FP", 3)


def test_review_contract_bounds_actor_time_and_notes(tmp_path: Path) -> None:
    # Given: a compact incident review store.
    store = CentralEvidenceReviewStore(_database(tmp_path))

    # When/Then: malformed boundary values are rejected before SQL.
    for actor, reviewed_at, notes in (
        ("x" * 129, REVIEWED_AT, None),
        ("operator", "not-a-time", None),
        ("operator", REVIEWED_AT, "x" * 1001),
    ):
        try:
            store.update(
                incident_id=INCIDENT_ID,
                expected_version=0,
                actor_id=actor,
                reviewed_at=reviewed_at,
                disposition=ReviewDisposition.TRUE_POSITIVE,
                notes=notes,
            )
        except ValueError:
            continue
        raise AssertionError("invalid review input was accepted")


def test_primary_clip_is_projected_from_purged_artifact(tmp_path: Path) -> None:
    # Given: a PURGED primary artifact retaining its clip identity.
    database = _database(tmp_path, artifact_state="PURGED")

    # When: the incident projection is read.
    summary = CentralEvidenceQuery(database).get(INCIDENT_ID)

    # Then: primary identity/state are derived from artifacts and review remains allowed.
    assert summary is not None
    assert summary.primary_clip_id == CLIP_ID
    assert summary.primary_artifact_state == "PURGED"
    review = CentralEvidenceReviewStore(database).update(
        incident_id=INCIDENT_ID,
        expected_version=0,
        actor_id="operator",
        reviewed_at=REVIEWED_AT,
        disposition=ReviewDisposition.FALSE_POSITIVE,
        notes=None,
    )
    assert review.version == 1


def test_operator_incident_api_reviews_without_available_primary_clip(tmp_path: Path) -> None:
    # Given: a schema-18 incident without a primary clip.
    database = _database(tmp_path)
    app = create_app(lifespan=no_lifespan)
    app.state.central_evidence_query = CentralEvidenceQuery(database)
    app.state.central_evidence_review_store = CentralEvidenceReviewStore(database)

    # When: the dashboard lists and reviews the incident, then retries stale version zero.
    with TestClient(app) as client:
        assert client.post(
            "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
        ).status_code == 204
        listed = client.get("/api/v1/incidents", params={"limit": 10})
        reviewed = client.put(
            f"/api/v1/incident-reviews/{INCIDENT_ID}",
            json={"expected_version": 0, "disposition": "TRUE_POSITIVE", "notes": None},
        )
        conflict = client.put(
            f"/api/v1/incident-reviews/{INCIDENT_ID}",
            json={"expected_version": 0, "disposition": "FALSE_POSITIVE", "notes": None},
        )
        malformed = client.get("/api/v1/incidents", params={"cursor": "not-a-cursor"})

    # Then: v0 is null initially, review succeeds without media, and stale/malformed are conflicts.
    assert listed.status_code == 200
    assert listed.json()["incidents"][0]["review"] is None
    assert reviewed.status_code == 200
    assert reviewed.json()["review"]["version"] == 1
    assert conflict.status_code == 409
    assert malformed.status_code == 400


def test_incident_keyset_does_not_skip_equal_timestamps(tmp_path: Path) -> None:
    # Given: three incidents sharing one detected_at value.
    database = _database(tmp_path)
    with sqlite3.connect(database) as connection:
        for suffix in ("a", "b"):
            connection.execute(
                """
                INSERT INTO incidents (
                    incident_id, edge_event_id, facility_id, camera_id, event_type,
                    detected_at, lifecycle_state, provenance_state,
                    provenance_missing_reason, review_version, revision, created_at, updated_at
                ) VALUES (?, ?, 'facility-1', 'camera-1', 'fall', ?, 'OPEN',
                          'MISSING', 'NOT_RECORDED', 0, 1, ?, ?)
                """,
                (f"incident-{suffix}", f"event-{suffix}", REVIEWED_AT, REVIEWED_AT, REVIEWED_AT),
            )
    query = CentralEvidenceQuery(database)

    # When: all one-row keyset pages are traversed.
    seen: list[str] = []
    cursor: str | None = None
    while True:
        page, cursor = query.list(limit=1, cursor=cursor)
        seen.extend(item.incident_id for item in page)
        if cursor is None:
            break

    # Then: the unique tie-breaker yields each incident exactly once.
    assert seen == ["incident-review", "incident-b", "incident-a"]
