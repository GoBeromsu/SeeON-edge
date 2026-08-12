"""Schema migration coverage for durable clip finalization state."""

from __future__ import annotations

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


def test_concurrent_first_open_applies_migrations_once(tmp_path: Path) -> None:
    # Given: two process-equivalent connections target one absent state database.
    database = tmp_path / "worker-state.sqlite3"
    barrier = Barrier(2)

    def open_once(_index: int) -> int:
        barrier.wait()
        with EvidenceOutbox.open(database):
            pass
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    # When: both perform their first open concurrently.
    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = tuple(executor.map(open_once, range(2)))

    # Then: migration converges without replaying DDL or losing the final schema.
    assert versions == (SCHEMA_VERSION, SCHEMA_VERSION)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM system_test_runs").fetchone() == (0,)


def test_open_migrates_v1_database_to_clip_manifest_schema(tmp_path: Path) -> None:
    # Given: a durable outbox created by the previous release.
    database = tmp_path / "evidence.sqlite3"
    connection = sqlite3.connect(database)
    for statement in MIGRATIONS[0]:
        connection.execute(statement)
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    # When: the current worker opens the existing database.
    with EvidenceOutbox.open(database):
        pass

    # Then: schema-v2 finalization fields exist and the migration is durable.
    connection = sqlite3.connect(database)
    version = connection.execute("PRAGMA user_version").fetchone()
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(evidence_clips)").fetchall()
    }
    connection.close()
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


def _create_v6_database_with_pending_event(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        for migration in MIGRATIONS[:6]:
            for statement in migration:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 6")
        connection.execute(
            """INSERT INTO evidence_events (
                   edge_event_id, detected_at, payload_json, state,
                   queued_at, next_attempt_at, delivery_state
               ) VALUES (?, ?, ?, 'READY', 1.0, 1.0, 'PENDING')""",
            (
                "00000000-0000-4000-8000-000000000001",
                "2026-08-12T00:00:00Z",
                '{"type":"fall"}',
            ),
        )


def test_v6_upgrade_preserves_queue_and_supports_fix_forward_reopen(
    tmp_path: Path,
) -> None:
    # Given: the previous approved schema has one queued ordinary event.
    database = tmp_path / "worker-state.sqlite3"
    _create_v6_database_with_pending_event(database)

    # When: the SYSTEM_TEST-capable image migrates and a compatible image reopens it.
    with EvidenceOutbox.open(database):
        pass
    with EvidenceOutbox.open(database) as fix_forward:
        assert fix_forward.pending_count() == 1

    # Then: the queue remains byte-identical and the idempotency mapping is additive.
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT edge_event_id, payload_json, operator_only FROM evidence_events"
        ).fetchone() == (
            "00000000-0000-4000-8000-000000000001",
            '{"type":"fall"}',
            0,
        )
        assert connection.execute("SELECT COUNT(*) FROM system_test_runs").fetchone() == (0,)


@pytest.mark.parametrize("previous_schema", (6, 7))
def test_previous_image_refuses_migrated_state_without_mutation(
    tmp_path: Path,
    previous_schema: int,
) -> None:
    # Given: the forward-only open has completed and preserved its queued evidence.
    database = tmp_path / "worker-state.sqlite3"
    _create_v6_database_with_pending_event(database)
    with EvidenceOutbox.open(database):
        pass
    before = database.read_bytes()

    # When: the previous image's schema-6 startup gate sees that database.
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        found = int(connection.execute("PRAGMA user_version").fetchone()[0])
    with pytest.raises(NewerSchemaVersionError) as raised:
        if found > previous_schema:
            raise NewerSchemaVersionError(found=found, supported=previous_schema)

    # Then: binary rollback is refused and no queue bytes are rewritten.
    assert (raised.value.found, raised.value.supported) == (
        SCHEMA_VERSION,
        previous_schema,
    )
    assert database.read_bytes() == before


def test_v7_fix_forward_migration_preserves_duplicates_and_maps_earliest(
    tmp_path: Path,
) -> None:
    # Given: the first PR image produced two rows for one validation run in schema 7.
    database = tmp_path / "worker-state.sqlite3"
    validation_run_id = "0197f671-3a31-7a6c-a6e4-83ed412de80f"
    with sqlite3.connect(database) as connection:
        for migration in MIGRATIONS[:7]:
            for statement in migration:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 7")
        connection.execute(
            """INSERT INTO evidence_events (
                   edge_event_id, detected_at, payload_json, state, queued_at,
                   next_attempt_at, delivery_state, operator_only
               ) VALUES (?, ?, ?, 'READY', ?, ?, 'PENDING', 1)""",
            (
                "00000000-0000-4000-8000-000000000002",
                "2026-08-12T00:00:02Z",
                '{"type":"SYSTEM_TEST","validation_run_id":"'
                + validation_run_id
                + '"}',
                2.0,
                2.0,
            ),
        )
        connection.execute(
            """INSERT INTO evidence_events (
                   edge_event_id, detected_at, payload_json, state, queued_at,
                   next_attempt_at, delivery_state, operator_only, backend_event_id
               ) VALUES (?, ?, ?, 'ACKED', ?, ?, 'ACKED', 1, 'backend-first')""",
            (
                "00000000-0000-4000-8000-000000000001",
                "2026-08-12T00:00:01Z",
                '{"type":"SYSTEM_TEST","validation_run_id":"'
                + validation_run_id
                + '"}',
                1.0,
                1.0,
            ),
        )

    # When: the fix-forward image opens the prior v7 state twice.
    with EvidenceOutbox.open(database):
        pass
    with EvidenceOutbox.open(database):
        pass

    # Then: both immutable rows survive and future recovery selects the earliest one.
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute("SELECT COUNT(*) FROM evidence_events").fetchone() == (2,)
        assert connection.execute(
            "SELECT validation_run_id, edge_event_id FROM system_test_runs"
        ).fetchone() == (
            validation_run_id,
            "00000000-0000-4000-8000-000000000001",
        )
