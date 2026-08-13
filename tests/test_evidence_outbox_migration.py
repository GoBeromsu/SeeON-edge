"""Forward-only migration coverage for durable worker state."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from worker.pipeline.output.evidence.evidence_outbox import (
    EvidenceOutbox,
    NewerSchemaVersionError,
)
from worker.pipeline.output.evidence.evidence_outbox_schema import MIGRATIONS, SCHEMA_VERSION

ORDINARY_EVENT_ID = "00000000-0000-4000-8000-000000000001"
PENDING_EVENT_ID = "00000000-0000-4000-8000-000000000002"
REMOVED_EVENT_ID = "00000000-0000-4000-8000-000000000099"
CLIP_ID = "clip-preserved"


def _create_schema(database: Path, version: int) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    for migration in MIGRATIONS[:version]:
        for statement in migration:
            connection.execute(statement)
    connection.execute(f"PRAGMA user_version = {version}")
    return connection


def _seed_v6_rows(connection: sqlite3.Connection) -> None:
    ordinary_payload = json.dumps(
        {
            "edge_event_id": ORDINARY_EVENT_ID,
            "event_type": "fall",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "audit": {"config_version": 11},
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    connection.execute(
        """INSERT INTO evidence_events (
               edge_event_id, detected_at, payload_json, state, queued_at,
               next_attempt_at, attempt_count, delivery_state, backend_event_id
           ) VALUES (?, ?, ?, 'ACKED', 1.0, 2.0, 3, 'ACKED', 'backend-event-1')""",
        (ORDINARY_EVENT_ID, "2026-08-12T00:00:00Z", ordinary_payload),
    )
    connection.execute(
        """INSERT INTO evidence_events (
               edge_event_id, detected_at, payload_json, state, queued_at,
               next_attempt_at, attempt_count, delivery_state, last_error_code
           ) VALUES (?, ?, ?, 'READY', 2.0, 8.0, 2, 'PENDING', 'NETWORK')""",
        (
            PENDING_EVENT_ID,
            "2026-08-12T00:00:01Z",
            '{"edge_event_id":"' + PENDING_EVENT_ID + '","event_type":"bed-exit"}',
        ),
    )
    connection.execute(
        """INSERT INTO evidence_clips (
               clip_id, local_state, manifest_path, state_version, media_relpath,
               sha256, size_bytes, mime_type, codec, duration_ms, clip_start_at,
               clip_end_at, finalized_at, publish_state, publish_attempt_count,
               publish_next_attempt_at, remote_state, backend_ack_at
           ) VALUES (
               ?, 'VERIFIED', 'clips/clip-preserved/manifest.json', 4,
               'clips/clip-preserved/clip.mp4', ?, 123, 'video/mp4', 'h264', 1000,
               '2026-08-12T00:00:00Z', '2026-08-12T00:00:01Z',
               '2026-08-12T00:00:02Z', 'PUBLISHED', 2, 3.0, 'READY', 4.0
           )""",
        (CLIP_ID, "a" * 64),
    )
    connection.execute(
        "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, 0)",
        (CLIP_ID, ORDINARY_EVENT_ID),
    )
    config_payload = '{"generation":7,"version":11}'
    connection.execute(
        """INSERT INTO config_current (
               id, generation, config_version, registry_version, payload_json, saved_at
           ) VALUES (1, 7, 11, 13, ?, 5.0)""",
        (config_payload,),
    )
    connection.execute(
        """INSERT INTO config_history (
               config_version, generation, registry_version, payload_json, saved_at
           ) VALUES (11, 7, 13, ?, 5.0)""",
        (config_payload,),
    )
    connection.execute(
        """INSERT INTO faults (
               id, pid, boot_time_iso, profile, task, stage, camera_id, frame_index,
               pts, frame_shape_json, frame_hash_sha256, model_artifact_digest,
               invocation_seq, exception_type, exception_message, exit_code,
               action, fault_time_iso
           ) VALUES (
               1, 42, '2026-08-12T00:00:00Z', 'prod', 'fall', 'inference',
               'camera-1', 9, 1.5, '[720,1280,3]', ?, ?, 17,
               'RuntimeError', 'accelerator fault', 4, 'exit',
               '2026-08-12T00:00:03Z'
           )""",
        ("b" * 64, "model-digest"),
    )


def _seed_removed_v7_row(connection: sqlite3.Connection) -> None:
    connection.execute(
        """INSERT INTO evidence_events (
               edge_event_id, detected_at, payload_json, state, queued_at,
               next_attempt_at, delivery_state, operator_only
           ) VALUES (?, '2026-08-12T00:00:04Z', ?, 'READY', 6.0, 6.0, 'PENDING', 1)""",
        (
            REMOVED_EVENT_ID,
            '{"type":"SYSTEM_TEST","validation_run_id":'
            '"0197f671-3a31-7a6c-a6e4-83ed412de80f"}',
        ),
    )


def _ordinary_snapshot(connection: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    tables = (
        "evidence_events",
        "evidence_clips",
        "clip_events",
        "config_current",
        "config_history",
        "faults",
    )
    return {
        table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()  # noqa: S608
        for table in tables
    }


def test_concurrent_first_open_applies_migrations_once(tmp_path: Path) -> None:
    database = tmp_path / "worker-state.sqlite3"
    barrier = Barrier(2)

    def open_once(_index: int) -> int:
        barrier.wait()
        with EvidenceOutbox.open(database):
            pass
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = tuple(executor.map(open_once, range(2)))

    assert versions == (SCHEMA_VERSION, SCHEMA_VERSION)


def test_open_migrates_v1_database_to_current_clip_manifest_schema(tmp_path: Path) -> None:
    database = tmp_path / "evidence.sqlite3"
    with _create_schema(database, 1):
        pass

    with EvidenceOutbox.open(database):
        pass

    with sqlite3.connect(database) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(evidence_clips)").fetchall()
        }
    assert version == (SCHEMA_VERSION,)
    assert {
        "local_state",
        "manifest_path",
        "state_version",
        "media_relpath",
        "sha256",
        "size_bytes",
        "unavailable_reason",
    } <= columns


@pytest.mark.parametrize("source_version", (6, 7, 8))
def test_schema_6_7_8_upgrade_preserves_ordinary_state_and_removes_feature_state(
    tmp_path: Path,
    source_version: int,
) -> None:
    database = tmp_path / f"worker-state-v{source_version}.sqlite3"
    with _create_schema(database, source_version) as connection:
        _seed_v6_rows(connection)
        if source_version >= 7:
            _seed_removed_v7_row(connection)
        if source_version >= 8:
            connection.execute(
                "INSERT INTO system_test_runs (validation_run_id, edge_event_id) VALUES (?, ?)",
                ("0197f671-3a31-7a6c-a6e4-83ed412de80f", REMOVED_EVENT_ID),
            )
        before = _ordinary_snapshot(connection)

    with EvidenceOutbox.open(database) as outbox:
        assert outbox.pending_count() == 1

    with sqlite3.connect(database) as connection:
        after = _ordinary_snapshot(connection)
        event_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(evidence_events)").fetchall()
        }
        objects = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type IN ('table', 'index')"
            ).fetchall()
        }
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        version = connection.execute("PRAGMA user_version").fetchone()

    expected = dict(before)
    expected["evidence_events"] = [
        row for row in before["evidence_events"] if row[0] != REMOVED_EVENT_ID
    ]
    if source_version >= 7:
        expected["evidence_events"] = [
            tuple(row[:-1]) for row in expected["evidence_events"]
        ]
    assert after == expected
    assert "operator_only" not in event_columns
    assert "system_test_runs" not in objects
    assert "evidence_events_operator_claim_idx" not in objects
    assert integrity == ("ok",)
    assert foreign_keys == []
    assert version == (SCHEMA_VERSION,)


def test_schema_8_upgrade_removes_feature_relation_without_deleting_clip(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-state-v8-relation.sqlite3"
    with _create_schema(database, 8) as connection:
        _seed_v6_rows(connection)
        _seed_removed_v7_row(connection)
        connection.execute(
            "INSERT INTO evidence_clips (clip_id) VALUES ('feature-shared-clip')"
        )
        connection.execute(
            "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, 0)",
            ("feature-shared-clip", REMOVED_EVENT_ID),
        )
        connection.execute(
            "INSERT INTO system_test_runs (validation_run_id, edge_event_id) VALUES (?, ?)",
            ("0197f671-3a31-7a6c-a6e4-83ed412de80f", REMOVED_EVENT_ID),
        )

    with EvidenceOutbox.open(database):
        pass

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT clip_id FROM evidence_clips WHERE clip_id = 'feature-shared-clip'"
        ).fetchone() == ("feature-shared-clip",)
        assert connection.execute(
            "SELECT edge_event_id FROM clip_events WHERE edge_event_id = ?",
            (REMOVED_EVENT_ID,),
        ).fetchone() is None


def test_schema_9_is_a_forward_boundary_for_schema_6_7_8_images(tmp_path: Path) -> None:
    database = tmp_path / "worker-state.sqlite3"
    with EvidenceOutbox.open(database):
        pass
    before = database.read_bytes()

    for supported in (6, 7, 8):
        with pytest.raises(NewerSchemaVersionError) as raised:
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                found = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if found > supported:
                raise NewerSchemaVersionError(found=found, supported=supported)
        assert (raised.value.found, raised.value.supported) == (SCHEMA_VERSION, supported)
        assert database.read_bytes() == before
