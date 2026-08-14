"""Canonical schema-9 fingerprint derived from the shipped migration history."""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from worker.pipeline.output.evidence.evidence_outbox_schema import MIGRATIONS


@lru_cache(maxsize=1)
def canonical_schema9_fingerprint() -> tuple[object, ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for target, statements in enumerate(MIGRATIONS, start=1):
            for statement in statements:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {target}")
        return schema_fingerprint(connection)
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
                    for row in connection.execute(
                        f"PRAGMA index_xinfo({index[1]})"
                    )
                ),
            )
            for index in indexes
        )
        table_details.append((table, columns, foreign_keys, indexes, index_details))
    return master, tuple(table_details)


def _normalize_sql(value: object) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())


__all__ = ["canonical_schema9_fingerprint", "schema_fingerprint"]
