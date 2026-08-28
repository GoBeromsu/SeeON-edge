from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.compatibility import SchemaLedgerError
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.edge_db.schema18_manifest import (
    compile_schema18_manifest,
    read_schema18_manifest,
)


def _fresh(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    return database


def test_compiled_schema18_manifest_matches_fresh_database(tmp_path: Path) -> None:
    database = _fresh(tmp_path)
    with sqlite3.connect(database) as connection:
        actual = read_schema18_manifest(connection)
    assert actual == compile_schema18_manifest()
    assert actual.diff(compile_schema18_manifest()) == ()


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: c.execute("DROP INDEX clips_started_at_idx"),
            id="missing_index",
        ),
        pytest.param(
            lambda c: c.execute("DROP TRIGGER audit_events_immutable_update"),
            id="missing_trigger",
        ),
        pytest.param(
            lambda c: c.executescript(
                """
                PRAGMA foreign_keys = OFF;
                CREATE TABLE artifacts_new (
                    incident_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    artifact_id TEXT,
                    clip_id TEXT,
                    state TEXT NOT NULL,
                    reason TEXT,
                    contained_relpath TEXT,
                    content_sha256 TEXT,
                    size_bytes INTEGER,
                    mime_type TEXT,
                    codec TEXT,
                    captured_at TEXT,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (incident_id, kind),
                    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
                ) STRICT;
                INSERT INTO artifacts_new SELECT * FROM artifacts;
                DROP TABLE artifacts;
                ALTER TABLE artifacts_new RENAME TO artifacts;
                """
            ),
            id="missing_fk",
        ),
        pytest.param(
            lambda c: c.executescript(
                """
                CREATE TABLE credentials_new (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    username TEXT,
                    algorithm TEXT NOT NULL CHECK (algorithm = 'scrypt'),
                    salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                ) STRICT;
                INSERT INTO credentials_new SELECT * FROM credentials;
                DROP TABLE credentials;
                ALTER TABLE credentials_new RENAME TO credentials;
                """
            ),
            id="altered_nullability",
        ),
        pytest.param(
            lambda c: c.executescript(
                """
                CREATE TABLE credentials_new (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    username TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                ) STRICT;
                INSERT INTO credentials_new SELECT * FROM credentials;
                DROP TABLE credentials;
                ALTER TABLE credentials_new RENAME TO credentials;
                """
            ),
            id="missing_check",
        ),
        pytest.param(
            lambda c: c.executescript(
                """
                CREATE TABLE schema_migrations_new (
                    version INTEGER PRIMARY KEY CHECK (version >= 0),
                    name TEXT NOT NULL UNIQUE,
                    checksum TEXT NOT NULL CHECK (length(checksum) = 64),
                    applied_at TEXT NOT NULL,
                    source_schema_version INTEGER,
                    source_db_sha256 TEXT,
                    reconciliation_sha256 TEXT
                ) STRICT;
                INSERT INTO schema_migrations_new SELECT * FROM schema_migrations;
                DROP TABLE schema_migrations;
                ALTER TABLE schema_migrations_new RENAME TO schema_migrations;
                """
            ),
            id="schema_migrations_check_drift",
        ),
    ],
)
def test_runtime_rejects_schema18_structural_mutations(
    tmp_path: Path,
    mutate: object,
) -> None:
    database = _fresh(tmp_path)
    with sqlite3.connect(database) as connection:
        mutate(connection)
        connection.commit()

    with pytest.raises(SchemaLedgerError, match="schema 18"):
        open_runtime_database(database, actor=RuntimeActor.API)
