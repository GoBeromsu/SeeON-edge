from __future__ import annotations

import multiprocessing
import os
import sqlite3
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from backend.app.edge_db.compatibility import (
    COMPATIBILITY_MATRIX,
    CURRENT_SCHEMA_RANGE,
    CompatibilityDisposition,
    MigrationRequiredError,
    NewerSchemaError,
    SchemaCompatibility,
    SchemaLedgerError,
    classify_schema,
)
from backend.app.edge_db.connection import (
    BusyPolicy,
    RuntimeActor,
    open_runtime_database,
    write_transaction,
)
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.ownership import APPLICATION_LEGACY_TABLES, writer_for_table
from backend.app.edge_db.schema import MIGRATIONS, SCHEMA_VERSION, Migration

_INTERRUPTED_MIGRATION = Migration(
    version=SCHEMA_VERSION + 1,
    name="interrupted_process_migration",
    statements=(
        "CREATE TABLE runtime_interrupted_a (id INTEGER PRIMARY KEY) STRICT",
        "CREATE TABLE runtime_interrupted_b (id INTEGER PRIMARY KEY) STRICT",
    ),
)


def _migrate_until_barrier(database: str, channel: Connection) -> None:
    def on_statement_applied(version: int, statement: int) -> None:
        if (version, statement) == (SCHEMA_VERSION + 1, 1):
            channel.send("FIRST_STATEMENT_APPLIED")
            channel.recv()

    migrate_database(
        Path(database),
        migrations=(*MIGRATIONS, _INTERRUPTED_MIGRATION),
        on_statement_applied=on_statement_applied,
    )


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "edge-state" / "edge.sqlite3"


def test_fresh_migration_records_schema_ledger_and_secure_local_files(
    database_path: Path,
) -> None:
    result = migrate_database(database_path)

    assert result.previous_version == 0
    assert result.current_version == CURRENT_SCHEMA_RANGE.maximum
    assert database_path.parent.stat().st_mode & 0o777 == 0o700
    assert database_path.stat().st_mode & 0o777 == 0o600

    connection = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (
            CURRENT_SCHEMA_RANGE.maximum,
        )
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        application_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert writer_for_table("schema_migrations") is not None
        assert {
            table: None if writer is None else writer.value
            for table in application_tables
            if table != "schema_migrations"
            for writer in (writer_for_table(table),)
        } == {
            table: "api"
            for table in application_tables
            if table != "schema_migrations"
        }
        assert connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall() == [(migration.version, migration.name) for migration in MIGRATIONS]
        assert application_tables == {
            "artifacts",
            "audit_events",
            "cameras",
            "clips",
            "credentials",
            "edge_site",
            "incidents",
            "locations",
            "policies",
            "schema_migrations",
        }
        create_sql = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_schema "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert all(" STRICT" in sql.upper() for sql in create_sql.values())
        assert "system_test_runs" not in APPLICATION_LEGACY_TABLES
        assert writer_for_table("system_test_runs") is None

        # WAL and SHM are colocated inside the one local state directory while open.
        names = {path.name for path in database_path.parent.iterdir()}
        assert {"edge.sqlite3", "edge.sqlite3-wal", "edge.sqlite3-shm"} <= names
    finally:
        connection.close()


def test_pre_v9_events_and_finalized_clips_backfill_authoritative_central_relations(
    database_path: Path,
) -> None:
    migrate_database(database_path, migrations=MIGRATIONS[:8])
    payload = (
        '{"camera_id":"camera:legacy","event_type":"fall",'
        '"audit":{"runtime_manifest_sha256":"not-resolvable"}}'
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at) "
            "VALUES ('event:legacy','2026-08-13T00:00:00Z',?,'READY',1,1)",
            (payload,),
        )
        connection.execute(
            """
            INSERT INTO evidence_clips (
                clip_id, local_state, state_version, unavailable_reason,
                clip_start_at, clip_end_at, finalized_at
            ) VALUES ('clip:legacy','UNAVAILABLE',2,'NO_FRAMES',
                      '2026-08-12T23:59:59Z','2026-08-13T00:00:01Z',
                      '2026-08-13T00:00:02Z')
            """
        )
        connection.execute("INSERT INTO clip_events VALUES ('clip:legacy','event:legacy',0)")
        connection.commit()

    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE evidence_events SET state = 'ACKED'")
        connection.commit()
    migrate_database(database_path, migrations=MIGRATIONS[:17])

    with sqlite3.connect(database_path) as connection:
        incident = connection.execute(
            "SELECT camera_id,event_type,provenance_state,provenance_missing_reason,"
            "primary_clip_id,lifecycle_state,failure_reason FROM evidence_incidents "
            "WHERE edge_event_id='event:legacy'"
        ).fetchone()
        primary = connection.execute(
            "SELECT clip_id,source_packet_preserved,source_missing_reason,"
            "unavailable_reason FROM evidence_primary_clips "
            "WHERE incident_id='event:legacy'"
        ).fetchone()
        slots = connection.execute(
            "SELECT slot_name,state,reason FROM evidence_artifact_slots "
            "WHERE incident_id='event:legacy' ORDER BY slot_name"
        ).fetchall()
    assert incident == (
        "camera:legacy",
        "fall",
        "MISSING",
        "LEGACY_PROVENANCE_NOT_RECORDED",
        "clip:legacy",
        "FAILED",
        "UNAVAILABLE",
    )
    assert primary == (
        "clip:legacy",
        0,
        "LEGACY_SOURCE_FACTS_NOT_RECORDED",
        "NO_FRAMES",
    )
    assert slots == [
        ("PRIMARY_CLIP", "UNAVAILABLE", "NO_FRAMES"),
        ("SNAPSHOT", "UNAVAILABLE", "LEGACY_SNAPSHOT_NOT_RECORDED"),
    ]


def test_v11_forward_migration_retires_operator_only_state_and_keeps_ordinary_evidence(
    database_path: Path,
) -> None:
    migrate_database(database_path, migrations=MIGRATIONS[:11])
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            "INSERT INTO evidence_events "
            "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at,"
            "operator_only) "
            "VALUES (?, '2026-08-13T00:00:00Z', ?, 'STAGED', 1, 1, ?)",
            (
                ("event:ordinary", '{"event_type":"fall"}', 0),
                ("event:system-test", '{"type":"SYSTEM_TEST"}', 1),
            ),
        )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id,local_state,state_version) "
            "VALUES ('clip:shared','VERIFIED',1)"
        )
        connection.executemany(
            "INSERT INTO clip_events (clip_id,edge_event_id,ordinal) VALUES ('clip:shared',?,?)",
            (("event:ordinary", 0), ("event:system-test", 1)),
        )
        connection.executemany(
            "INSERT INTO evidence_incidents "
            "(incident_id,edge_event_id,camera_id,event_type,detected_at,"
            "provenance_missing_reason,primary_clip_id,lifecycle_state,created_at,updated_at) "
            "VALUES (?,?,'camera:one',?,'2026-08-13T00:00:00Z','NOT_RECORDED',"
            "'clip:shared','STAGING','2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')",
            (
                ("incident:ordinary", "event:ordinary", "fall"),
                ("incident:system-test", "event:system-test", "SYSTEM_TEST"),
            ),
        )
        connection.executemany(
            "INSERT INTO evidence_primary_clips "
            "(incident_id,clip_id,source_packet_preserved,source_missing_reason,"
            "truncation_json,unavailable_reason,created_at) "
            "VALUES (?,'clip:shared',0,'NOT_RECORDED','[]','MISSING',"
            "'2026-08-13T00:00:00Z')",
            (("incident:ordinary",), ("incident:system-test",)),
        )
        connection.execute(
            "INSERT INTO evidence_media_objects "
            "(media_id,content_sha256,size_bytes,mime_type,contained_relpath,basename,created_at) "
            "VALUES ('media:system-test',? ,1,'image/jpeg','snapshots/system-test.jpg',"
            "'system-test.jpg','2026-08-13T00:00:00Z')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO evidence_incident_snapshots "
            "VALUES ('incident:system-test','snapshot:system-test','media:system-test',"
            "'2026-08-13T00:00:00Z','camera:one','2026-08-13T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO control_evidence_review_revisions "
            "VALUES ('review:system-test','incident:system-test','clip:shared',1,"
            "'operator','2026-08-13T00:00:00Z','FALSE_POSITIVE',NULL)"
        )
        connection.execute(
            "INSERT INTO control_evidence_review_state "
            "VALUES ('incident:system-test','clip:shared',1)"
        )
        manifest_sha256 = "b" * 64
        connection.execute(
            "INSERT INTO runtime_manifest_contents VALUES (?,1,'{}','2026-08-13T00:00:00Z')",
            (manifest_sha256,),
        )
        connection.execute(
            "INSERT INTO runtime_manifest_boots VALUES ('boot:migration',?,'2026-08-13T00:00:00Z')",
            (manifest_sha256,),
        )
        connection.execute(
            "INSERT INTO runtime_manifest_cameras VALUES "
            "('boot:migration','camera:one',?,'2026-08-13T00:00:00Z')",
            (manifest_sha256,),
        )
        for ordinal, trace_id in enumerate(("c" * 64, "d" * 64)):
            analysis_id = ("e" if ordinal == 0 else "f") * 64
            connection.execute(
                "INSERT INTO runtime_analysis_traces "
                "(trace_id,trace_schema_version,worker_boot_id,camera_id,stream_epoch,"
                "frame_seq,pts,source_time_sec,frame_width,frame_height,"
                "bed_region_provenance,storage_bytes) "
                "VALUES (?,1,'boot:migration','camera:one',1,?,1,1,16,16,'fresh',1)",
                (analysis_id, ordinal),
            )
            connection.execute(
                "INSERT INTO evidence_decision_traces "
                "(trace_id,trace_schema_version,analysis_trace_id,module_qualified_id,"
                "policy_qualified_id,effective_policy_id,runtime_manifest_sha256,reason,"
                "previous_state,current_state,triggered,track_missing_reason,"
                "bed_missing_reason) "
                "VALUES (?,1,?,'fall.v1','fall.policy.v1',?,?,'reason','clear',"
                "'triggered',1,'not-applicable','not-applicable')",
                (trace_id, analysis_id, "a" * 64, manifest_sha256),
            )
        connection.executemany(
            "INSERT INTO evidence_event_trace_refs VALUES (?,?)",
            (("event:ordinary", "c" * 64), ("event:system-test", "d" * 64)),
        )
        connection.executemany(
            "INSERT INTO evidence_clip_trace_refs VALUES ('clip:shared',?,?)",
            (
                ("event:ordinary", "c" * 64),
                ("event:system-test", "d" * 64),
            ),
        )
        connection.execute(
            "INSERT INTO evidence_decision_values VALUES (?, 'probability', 0.5, NULL)",
            ("d" * 64,),
        )
        connection.execute(
            "INSERT INTO system_test_runs VALUES ('run:system-test','event:system-test')"
        )
        connection.commit()

    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE evidence_events SET state = 'ACKED'")
        connection.commit()
    result = migrate_database(database_path, migrations=MIGRATIONS[:17])

    assert result.previous_version == 11
    assert result.current_version == 17
    with sqlite3.connect(database_path) as connection:
        objects = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type IN ('table','index')"
            )
        }
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(evidence_events)")}
        assert connection.execute(
            "SELECT edge_event_id FROM evidence_events ORDER BY edge_event_id"
        ).fetchall() == [("event:ordinary",)]
        assert connection.execute(
            "SELECT incident_id FROM evidence_incidents ORDER BY incident_id"
        ).fetchall() == [("incident:ordinary",)]
        assert connection.execute(
            "SELECT clip_id,edge_event_id,ordinal FROM clip_events ORDER BY edge_event_id"
        ).fetchall() == [("clip:shared", "event:ordinary", 0)]
        assert connection.execute("SELECT clip_id FROM evidence_clips").fetchall() == [
            ("clip:shared",)
        ]
        assert connection.execute(
            "SELECT edge_event_id,decision_trace_id FROM evidence_event_trace_refs "
            "ORDER BY edge_event_id"
        ).fetchall() == [("event:ordinary", "c" * 64)]
        assert connection.execute(
            "SELECT clip_id,edge_event_id,decision_trace_id FROM evidence_clip_trace_refs "
            "ORDER BY edge_event_id"
        ).fetchall() == [("clip:shared", "event:ordinary", "c" * 64)]
        # Orphan media/trace rows may remain; ordinary evidence and clips stay.
        assert connection.execute("SELECT count(*) FROM evidence_media_objects").fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM control_evidence_review_state"
        ).fetchone() == (0,)
        assert "operator_only" not in columns
        assert "system_test_runs" not in objects
        assert "evidence_events_operator_claim_idx" not in objects
        assert connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'system_test_state_policy'"
        ).fetchone() == ("retired",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_migration_is_idempotent_and_runtime_open_does_not_mutate_schema(
    database_path: Path,
) -> None:
    first = migrate_database(database_path)
    second = migrate_database(database_path)
    assert first.current_version == second.current_version
    assert second.previous_version == second.current_version

    before = sqlite3.connect(database_path)
    try:
        schema_before = before.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
    finally:
        before.close()

    runtime = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        assert runtime.total_changes == 0
    finally:
        runtime.close()

    after = sqlite3.connect(database_path)
    try:
        assert (
            after.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
            == schema_before
        )
    finally:
        after.close()


def test_forward_backward_compatibility_matrix_is_explicit() -> None:
    assert COMPATIBILITY_MATRIX == (
        ("database_version < minimum", CompatibilityDisposition.MIGRATION_REQUIRED),
        ("minimum <= database_version <= maximum", CompatibilityDisposition.COMPATIBLE),
        ("database_version > maximum", CompatibilityDisposition.NEWER_SCHEMA),
    )
    supported = SchemaCompatibility(minimum=3, maximum=4)
    assert classify_schema(2, supported) is CompatibilityDisposition.MIGRATION_REQUIRED
    assert classify_schema(3, supported) is CompatibilityDisposition.COMPATIBLE
    assert classify_schema(4, supported) is CompatibilityDisposition.COMPATIBLE
    assert classify_schema(5, supported) is CompatibilityDisposition.NEWER_SCHEMA


def test_runtime_refuses_unmigrated_older_and_newer_schemas(database_path: Path) -> None:
    with pytest.raises(MigrationRequiredError):
        open_runtime_database(database_path, actor=RuntimeActor.API)

    migrate_database(database_path)
    with pytest.raises(MigrationRequiredError):
        open_runtime_database(
            database_path,
            actor=RuntimeActor.API,
            compatibility=SchemaCompatibility(
                minimum=SCHEMA_VERSION + 1, maximum=SCHEMA_VERSION + 1
            ),
        )
    with pytest.raises(NewerSchemaError):
        open_runtime_database(
            database_path,
            actor=RuntimeActor.API,
            compatibility=SchemaCompatibility(minimum=0, maximum=0),
        )


def test_runtime_denies_ddl_and_schema_ledger_writes(database_path: Path) -> None:
    migrate_database(database_path)

    api = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        with write_transaction(api):
            api.execute(
                "INSERT INTO locations "
                "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
                "created_at, updated_at) VALUES "
                "('floor-1', 'FLOOR', NULL, NULL, 'Floor 1', 0, "
                "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
            )
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            api.execute("UPDATE schema_migrations SET name = 'forged' WHERE version = 1")
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            api.execute("CREATE TABLE runtime_illegal (id INTEGER)")
    finally:
        api.close()


def test_migrator_allows_declared_ddl_for_quoted_table_family_names(
    database_path: Path,
) -> None:
    migration = Migration(
        version=SCHEMA_VERSION + 1,
        name="quoted_family_ddl",
        statements=(
            'CREATE TABLE "control_quoted" (id INTEGER PRIMARY KEY) STRICT',
            "CREATE TABLE [qa_quoted] (id INTEGER PRIMARY KEY) STRICT",
            "CREATE TABLE `runtime_quoted` (id INTEGER PRIMARY KEY) STRICT",
            'CREATE TABLE "evidence_quoted" (id INTEGER PRIMARY KEY) STRICT',
            'CREATE TABLE "derivative_quoted" (id INTEGER PRIMARY KEY) STRICT',
            'CREATE INDEX "runtime_quoted_id_idx" ON `runtime_quoted` (id)',
        ),
    )

    result = migrate_database(database_path, migrations=(*MIGRATIONS, migration))

    assert result.current_version == SCHEMA_VERSION + 1
    connection = sqlite3.connect(database_path)
    try:
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE name GLOB '*_quoted'"
            )
        } == {
            "control_quoted",
            "qa_quoted",
            "runtime_quoted",
            "evidence_quoted",
            "derivative_quoted",
        }
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name = 'runtime_quoted_id_idx'"
        ).fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "statement",
    [
        'CREATE TABLE "control_ctas" AS SELECT 1 AS value',
        "CREATE TABLE [runtime_ctas] AS WITH payload(value) AS (VALUES (41), (42)) "
        "SELECT value FROM payload",
        "CREATE TABLE `evidence_ctas` AS VALUES (1), (2)",
        'CREATE TABLE "derivative_ctas" AS SELECT column1 FROM (VALUES (1), (2))',
        "CREATE TABLE [qa_ctas] AS WITH payload(value) AS (VALUES (1)) SELECT value FROM payload",
    ],
)
def test_migrator_denies_application_family_create_table_as_select_and_rolls_back(
    database_path: Path,
    statement: str,
) -> None:
    migrate_database(database_path)
    migration = Migration(
        version=SCHEMA_VERSION + 1, name="forbidden_application_ctas", statements=(statement,)
    )

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        migrate_database(database_path, migrations=(*MIGRATIONS, migration))

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name GLOB '*_ctas'"
        ).fetchone() == (0,)
        assert connection.execute(
            f"SELECT count(*) FROM schema_migrations WHERE version = {SCHEMA_VERSION + 1}"
        ).fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "statement",
    [
        'INSERT INTO "control_guard" VALUES (1)',
        "WITH candidate(id) AS (VALUES (1)) INSERT INTO [runtime_guard] SELECT id FROM candidate",
        "UPDATE `evidence_guard` SET id = 2 WHERE id = 1",
        'DELETE FROM "derivative_guard"',
        'INSERT INTO "qa_guard" VALUES (1)',
    ],
)
def test_migrator_denies_application_family_dml_even_when_quoted_or_cte(
    database_path: Path,
    statement: str,
) -> None:
    migrate_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE control_guard (id INTEGER PRIMARY KEY) STRICT;
            CREATE TABLE qa_guard (id INTEGER PRIMARY KEY) STRICT;
            CREATE TABLE runtime_guard (id INTEGER PRIMARY KEY) STRICT;
            CREATE TABLE evidence_guard (id INTEGER PRIMARY KEY) STRICT;
            CREATE TABLE derivative_guard (id INTEGER PRIMARY KEY) STRICT;
            INSERT INTO evidence_guard VALUES (1);
            """
        )
    finally:
        connection.close()
    migration = Migration(
        version=SCHEMA_VERSION + 1, name="forbidden_application_dml", statements=(statement,)
    )

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        migrate_database(database_path, migrations=(*MIGRATIONS, migration))

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            f"SELECT count(*) FROM schema_migrations WHERE version = {SCHEMA_VERSION + 1}"
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_migrator_denies_trigger_mediated_cross_family_write(database_path: Path) -> None:
    migrate_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("CREATE TABLE evidence_trigger_target (id INTEGER PRIMARY KEY) STRICT")
    finally:
        connection.close()
    migration = Migration(
        version=SCHEMA_VERSION + 1,
        name="forbidden_trigger_write",
        statements=(
            """
            CREATE TRIGGER schema_cross_family_trigger
            AFTER UPDATE ON schema_migrations
            BEGIN
                INSERT INTO "evidence_trigger_target" VALUES (1);
            END
            """,
            "UPDATE schema_migrations SET name = name WHERE version = 1",
        ),
    )

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        migrate_database(database_path, migrations=(*MIGRATIONS, migration))

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("SELECT count(*) FROM evidence_trigger_target").fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name = 'schema_cross_family_trigger'"
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_forged_migration_name_and_checksum_fail_runtime_and_migrator_consistently(
    database_path: Path,
) -> None:
    migrate_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "UPDATE schema_migrations SET name = 'forged', checksum = ? WHERE version = 1",
            ("f" * 64,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SchemaLedgerError) as runtime_error:
        open_runtime_database(database_path, actor=RuntimeActor.API)
    with pytest.raises(SchemaLedgerError) as migrator_error:
        migrate_database(database_path)
    assert str(runtime_error.value) == str(migrator_error.value)
    assert str(runtime_error.value) == "applied migration ledger differs from this migrator"


def test_process_kill_during_migration_rolls_back_the_in_progress_version(
    database_path: Path,
) -> None:
    migrate_database(database_path)
    context = multiprocessing.get_context("spawn")
    parent_channel, child_channel = context.Pipe()
    process = context.Process(
        target=_migrate_until_barrier,
        args=(os.fspath(database_path), child_channel),
    )
    process.start()
    assert parent_channel.poll(10), "migrator did not reach the deterministic DDL barrier"
    assert parent_channel.recv() == "FIRST_STATEMENT_APPLIED"
    process.kill()
    process.join(10)
    assert process.exitcode is not None and process.exitcode != 0

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name LIKE 'runtime_interrupted_%'"
        ).fetchone() == (0,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        connection.close()

    result = migrate_database(
        database_path,
        migrations=(*MIGRATIONS, _INTERRUPTED_MIGRATION),
    )
    assert result.previous_version == SCHEMA_VERSION
    assert result.current_version == SCHEMA_VERSION + 1


def test_failed_migration_rolls_back_ddl_and_version(database_path: Path) -> None:
    broken = Migration(
        version=SCHEMA_VERSION + 1,
        name="interrupted",
        statements=(
            "CREATE TABLE runtime_interrupted (id INTEGER PRIMARY KEY) STRICT",
            "THIS IS NOT SQL",
        ),
    )
    with pytest.raises(sqlite3.Error):
        migrate_database(database_path, migrations=(*MIGRATIONS, broken))

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name = 'runtime_interrupted'"
        ).fetchone() == (0,)
        assert connection.execute(
            f"SELECT count(*) FROM schema_migrations WHERE version = {SCHEMA_VERSION + 1}"
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_migrator_cli_is_the_only_schema_entrypoint(database_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.edge_db",
            "--database",
            os.fspath(database_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        f"EDGE_DB_MIGRATION_OK path={database_path} "
        f"previous=0 current={CURRENT_SCHEMA_RANGE.maximum}\n"
    )


@pytest.mark.parametrize(
    ("corrupt_sql", "expected_error"),
    [
        (
            "UPDATE schema_migrations SET checksum = '" + ("0" * 64) + "' WHERE version = 1",
            "ledger",
        ),
        (
            "CREATE TABLE extra_table (id INTEGER PRIMARY KEY) STRICT",
            "application table",
        ),
    ],
)
def test_migrator_cli_refuses_corrupt_format_or_ownership_without_success_marker(
    database_path: Path,
    corrupt_sql: str,
    expected_error: str,
) -> None:
    migrate_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(corrupt_sql)
        connection.commit()
    finally:
        connection.close()

    completed = subprocess.run(
        [sys.executable, "-m", "backend.app.edge_db", "--database", os.fspath(database_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "EDGE_DB_MIGRATION_OK" not in completed.stdout
    assert "EDGE_DB_MIGRATION_FAILED" in completed.stderr
    assert expected_error in completed.stderr.lower()


def test_zero_wait_policy_is_explicit(database_path: Path) -> None:
    migrate_database(database_path)
    connection = open_runtime_database(
        database_path,
        actor=RuntimeActor.API,
        busy_policy=BusyPolicy.ZERO_WAIT,
    )
    try:
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (0,)
    finally:
        connection.close()
