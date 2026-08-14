"""Canonical fingerprints for worker-state and central edge evidence tables."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from typing import Final

from shared.edge_db.compatibility import CURRENT_SCHEMA_RANGE
from worker.pipeline.output.evidence.evidence_outbox_schema import (
    MIGRATIONS,
)
from worker.pipeline.output.evidence.evidence_outbox_schema import (
    SCHEMA_VERSION as WORKER_SCHEMA_VERSION,
)

EVIDENCE_TABLES: Final = ("evidence_events", "evidence_clips", "clip_events")


@lru_cache(maxsize=1)
def canonical_worker_schema_fingerprint() -> tuple[object, ...]:
    """Full sqlite_master fingerprint for the legacy/worker-state lineage."""
    connection = _memory_worker_schema()
    try:
        return schema_fingerprint(connection)
    finally:
        connection.close()


# Upstream PR #292 named this for schema 9; keep the alias for call sites/tests.
canonical_schema9_fingerprint = canonical_worker_schema_fingerprint


@lru_cache(maxsize=1)
def canonical_evidence_relation_fingerprint() -> tuple[object, ...]:
    """Pragma-level fingerprint of the three relation tables shared by both DBs."""
    connection = _memory_worker_schema()
    try:
        return evidence_relation_fingerprint(connection)
    finally:
        connection.close()


def schema_fingerprint(connection: sqlite3.Connection) -> tuple[object, ...]:
    master = tuple(
        (str(row[0]), str(row[1]), str(row[2]), _normalize_sql(row[3]))
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "ORDER BY type, name, tbl_name"
        )
    )
    tables = tuple(row[1] for row in master if row[0] == "table")
    return master, _table_details(connection, tables)


def evidence_relation_fingerprint(connection: sqlite3.Connection) -> tuple[object, ...]:
    """Compare only pragma surfaces so worker-state and edge.sqlite3 agree."""
    present = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = tuple(table for table in EVIDENCE_TABLES if table not in present)
    if missing:
        return ("missing", missing)
    # Include index *names* attached to the evidence tables so required-index
    # drift still fails without depending on CREATE SQL whitespace.
    index_names = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name IN ({placeholders}) "
            "AND name NOT LIKE 'sqlite_autoindex%' "
            "ORDER BY name".format(
                placeholders=",".join("?" for _ in EVIDENCE_TABLES)
            ),
            EVIDENCE_TABLES,
        )
    )
    return index_names, _table_details(connection, EVIDENCE_TABLES)


def is_supported_schema_version(version: int) -> bool:
    if version == WORKER_SCHEMA_VERSION:
        return True
    return CURRENT_SCHEMA_RANGE.minimum <= version <= CURRENT_SCHEMA_RANGE.maximum


def is_worker_state_schema(version: int) -> bool:
    return version == WORKER_SCHEMA_VERSION


def _memory_worker_schema() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    for target, statements in enumerate(MIGRATIONS, start=1):
        for statement in statements:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {target}")
    return connection


def _table_details(
    connection: sqlite3.Connection, tables: tuple[str, ...]
) -> tuple[object, ...]:
    table_details: list[object] = []
    for table in tables:
        columns = tuple(
            tuple(row) for row in connection.execute(f"PRAGMA table_xinfo({table})")
        )
        foreign_keys = tuple(
            tuple(row)
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        indexes = tuple(
            tuple(row) for row in connection.execute(f"PRAGMA index_list({table})")
        )
        index_details = tuple(
            (
                str(index[1]),
                tuple(
                    tuple(row)
                    for row in connection.execute(f"PRAGMA index_xinfo({index[1]})")
                ),
            )
            for index in indexes
        )
        table_details.append((table, columns, foreign_keys, indexes, index_details))
    return tuple(table_details)


def _normalize_sql(value: object) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())


__all__ = [
    "EVIDENCE_TABLES",
    "canonical_evidence_relation_fingerprint",
    "canonical_schema9_fingerprint",
    "canonical_worker_schema_fingerprint",
    "evidence_relation_fingerprint",
    "is_supported_schema_version",
    "is_worker_state_schema",
    "schema_fingerprint",
]
