from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from backend.app.features.evidence.record_store import CentralEvidenceQuery
from shared.edge_db.connection import RuntimeActor, open_runtime_database
from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.evidence.evidence_outbox import (
    ClipId,
    ClipLocalState,
    ClipOutcome,
    EvidenceOutbox,
    EvidenceReasonCode,
)
from worker.pipeline.output.evidence.evidence_records import (
    ArtifactState,
    EvidenceLifecycle,
    EvidenceRecordConflictError,
    EvidenceRecordStore,
)


def test_central_evidence_schema_has_owned_strict_records_and_integrity_guards(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)

    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type = 'table' "
                "AND (name LIKE 'evidence_%' OR name LIKE 'derivative_%')"
            )
        }
        assert {
            "evidence_incidents",
            "evidence_media_objects",
            "evidence_artifact_slots",
            "evidence_primary_clips",
            "evidence_incident_snapshots",
            "evidence_retention_states",
            "derivative_evidence_slots",
        } <= tables.keys()
        assert all(
            "STRICT" in tables[name]
            for name in (
                "evidence_incidents",
                "evidence_media_objects",
                "evidence_artifact_slots",
                "evidence_primary_clips",
                "evidence_incident_snapshots",
                "evidence_retention_states",
                "derivative_evidence_slots",
            )
        )
        foreign_key_failures = connection.execute("PRAGMA foreign_key_check").fetchall()
        assert foreign_key_failures == []

    worker = open_runtime_database(database, actor=RuntimeActor.WORKER)
    api = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            api.execute(
                "INSERT INTO evidence_incidents "
                "(incident_id, edge_event_id, camera_id, event_type, detected_at, "
                "lifecycle_state, created_at, updated_at) "
                "VALUES ('i','e','c','fall','2026-08-13T00:00:00Z',"
                "'STAGING','2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')"
            )
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            worker.execute("ALTER TABLE evidence_incidents ADD COLUMN forbidden TEXT")
    finally:
        worker.close()
        api.close()


def test_every_declared_unavailable_reason_is_accepted_and_loaded_by_central_store(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)

    with EvidenceOutbox.open(database) as outbox:
        for index, reason in enumerate(EvidenceReasonCode):
            clip_id = ClipId(f"reason:{index}")
            outbox.record_clip_outcome(
                ClipOutcome(
                    clip_id=clip_id,
                    local_state=ClipLocalState.UNAVAILABLE,
                    manifest_path=None,
                    state_version=2,
                    unavailable_reason=reason,
                )
            )
            loaded = outbox.clip_outcome(clip_id)
            assert loaded is not None
            assert loaded.unavailable_reason is reason


def test_backend_central_evidence_query_is_privacy_bounded(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at) "
            "VALUES ('event:query','2026-08-13T00:00:00Z',?, 'STAGED',1,1)",
            ('{"facility_id":"private","snapshot_jpeg_base64":"private"}',),
        )
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                provenance_missing_reason, lifecycle_state, created_at, updated_at
            ) VALUES ('incident:query','event:query','camera:opaque','fall',
                      '2026-08-13T00:00:00Z','NOT_RECORDED','STAGING',
                      '2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')
            """
        )
        connection.commit()

    summary = CentralEvidenceQuery(database).get("event:query")

    assert summary is not None
    assert summary.incident_id == "incident:query"
    assert summary.camera_id == "camera:opaque"
    assert summary.lifecycle_state == "STAGING"
    assert "private" not in repr(summary)
    assert not hasattr(summary, "payload_json")
    assert not hasattr(summary, "operator_only")


def test_incident_lifecycle_is_optimistically_concurrent_and_legally_ordered(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at) "
            "VALUES ('event:opaque','2026-08-13T00:00:00Z','{}','STAGED',1,1)"
        )
        connection.execute(
            "INSERT INTO evidence_clips "
            "(clip_id, local_state, state_version) VALUES ('clip:opaque','VERIFIED',2)"
        )
        connection.execute(
            """
            INSERT INTO evidence_media_objects (
                media_id, content_sha256, size_bytes, mime_type,
                contained_relpath, basename, created_at
            ) VALUES ('media:opaque', ?, 1, 'video/mp4',
                      'clips/clip:opaque/clip.mp4', 'clip.mp4',
                      '2026-08-13T00:00:00Z')
            """,
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO evidence_incidents "
            "(incident_id, edge_event_id, camera_id, event_type, detected_at, "
            "provenance_missing_reason, primary_clip_id, lifecycle_state, "
            "created_at, updated_at) "
            "VALUES ('incident:opaque','event:opaque','camera:byte-exact','fall',"
            "'2026-08-13T00:00:00Z','NOT_RECORDED','clip:opaque','STAGING',"
            "'2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')"
        )
        connection.execute(
            """
            INSERT INTO evidence_artifact_slots (
                incident_id, slot_name, state, media_id, created_at, updated_at
            ) VALUES ('incident:opaque','PRIMARY_CLIP','AVAILABLE','media:opaque',
                      '2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')
            """
        )
        connection.execute(
            """
            INSERT INTO evidence_primary_clips (
                incident_id, clip_id, manifest_relpath, manifest_sha256,
                manifest_size_bytes, media_id, source_packet_preserved,
                source_media_json, truncation_json, created_at
            ) VALUES ('incident:opaque','clip:opaque',
                      'clips/clip:opaque/manifest.json', ?, 1, 'media:opaque',
                      1, '{}', '[]', '2026-08-13T00:00:00Z')
            """,
            ("b" * 64,),
        )
        connection.commit()

    store = EvidenceRecordStore(database)
    barrier = Barrier(2)

    def advance() -> object:
        barrier.wait()
        try:
            return store.transition(
                "incident:opaque",
                expected_revision=1,
                target=EvidenceLifecycle.MEDIA_READY,
                updated_at="2026-08-13T00:00:01Z",
            )
        except EvidenceRecordConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: advance(), range(2)))

    assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, EvidenceRecordConflictError) for outcome in outcomes) == 1
    record = store.get("incident:opaque")
    assert record is not None
    assert record.lifecycle is EvidenceLifecycle.MEDIA_READY
    assert record.revision == 2

    with pytest.raises(EvidenceRecordConflictError, match="evidence lifecycle"):
        store.transition(
            "incident:opaque",
            expected_revision=2,
            target=EvidenceLifecycle.COMPLETE,
            updated_at="2026-08-13T00:00:02Z",
        )

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence_events SET state = 'ACKED', delivery_state = 'ACKED' "
            "WHERE edge_event_id = 'event:opaque'"
        )
        connection.execute(
            "UPDATE evidence_clips SET publish_state = 'PUBLISHED' WHERE clip_id = 'clip:opaque'"
        )
        connection.commit()

    published = store.transition(
        "incident:opaque",
        expected_revision=2,
        target=EvidenceLifecycle.PUBLISHED,
        updated_at="2026-08-13T00:00:03Z",
    )
    pending = store.request_annotated_derivative(
        "incident:opaque",
        expected_revision=published.revision,
        updated_at="2026-08-13T00:00:04Z",
    )
    replay = store.request_annotated_derivative(
        "incident:opaque",
        expected_revision=published.revision,
        updated_at="2026-08-13T00:00:04Z",
    )
    assert pending.lifecycle is EvidenceLifecycle.DERIVATIVE_PENDING
    assert pending.derivative_state is ArtifactState.PENDING
    assert replay == pending
