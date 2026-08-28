"""Resumable table receipts for the released legacy database adapter."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeAlias

from backend.app.edge_db.legacy_import_snapshot import LegacyImportError, SourceSnapshot

SqliteValue: TypeAlias = int | float | str | bytes | None
ImportProgress = Callable[[str, str], None]
_RETIRED_LEGACY_TABLES: Final = frozenset({"system_test_runs"})
_TABLE_PRIORITY: Final = {
    "camera_topology_floors": 10,
    "camera_topology_rooms": 11,
    "camera_topology_cameras": 12,
    "evidence_events": 20,
    "evidence_clips": 21,
    "clip_events": 22,
}


def record_backup_receipt(
    target: sqlite3.Connection,
    snapshot: SourceSnapshot,
    on_receipt: ImportProgress | None,
) -> None:
    existing = target.execute(
        "SELECT digest,row_count FROM schema_import_receipts "
        "WHERE source_name=? AND barrier='backup'",
        (snapshot.name,),
    ).fetchone()
    expected = (snapshot.sha256, snapshot.size_bytes)
    if existing is not None and existing != expected:
        raise LegacyImportError(f"{snapshot.name} changed after import receipt")
    if existing is None:
        target.execute("BEGIN IMMEDIATE")
        try:
            target.execute(
                "INSERT INTO schema_import_receipts "
                "VALUES (?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                (snapshot.name, "backup", snapshot.schema, snapshot.sha256, snapshot.size_bytes),
            )
            target.commit()
        except sqlite3.Error:
            target.rollback()
            raise
        if on_receipt is not None:
            on_receipt(snapshot.name, "backup")


def import_source_tables(
    target: sqlite3.Connection,
    snapshot: SourceSnapshot,
    source_path: Path,
    *,
    on_receipt: ImportProgress | None,
) -> None:
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    try:
        tables = [
            str(row[0])
            for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        tables.sort(key=lambda table: (_TABLE_PRIORITY.get(table, 15), table))
        total_rows = 0
        operator_only_event_ids = _operator_only_event_ids(source)
        for table in tables:
            source_columns = [
                str(row[1]) for row in source.execute(f'PRAGMA table_info("{table}")')
            ]
            source_rows = source.execute(
                f"SELECT {','.join(_quote(column) for column in source_columns)} "
                f"FROM {_quote(table)} "
                f"ORDER BY {','.join(str(index + 1) for index in range(len(source_columns)))}"
            ).fetchall()
            if table in _RETIRED_LEGACY_TABLES:
                # Retired SYSTEM_TEST mapping tables are not projected into edge.sqlite3.
                total_rows += len(source_rows)
                continue
            columns, rows = _filter_retired_operator_rows(
                table,
                source_columns,
                source_rows,
                operator_only_event_ids=operator_only_event_ids,
            )
            total_rows += len(source_rows)
            if (
                target.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()
                is None
            ):
                raise LegacyImportError(f"legacy {snapshot.name} owns unsupported table {table}")
            digest = _rows_digest(columns, rows)
            barrier = f"table:{table}"
            existing = target.execute(
                "SELECT digest,row_count FROM schema_import_receipts "
                "WHERE source_name=? AND barrier=?",
                (snapshot.name, barrier),
            ).fetchone()
            if existing is not None:
                if existing != (digest, len(rows)) or _target_projection(
                    target, table, columns
                ) != (
                    digest,
                    len(rows),
                ):
                    raise LegacyImportError(f"{snapshot.name}.{table} changed after import receipt")
                continue
            placeholders = ",".join("?" for _ in columns)
            target.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    column_list = ",".join(_quote(column) for column in columns)
                    target.execute(
                        f"INSERT OR IGNORE INTO {_quote(table)} "
                        f"({column_list}) VALUES ({placeholders})",
                        row,
                    )
                _validate_target_projection(
                    target,
                    table,
                    columns,
                    digest,
                    len(rows),
                    source_name=snapshot.name,
                )
                target.execute(
                    "INSERT INTO schema_import_receipts "
                    "VALUES (?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                    (snapshot.name, barrier, snapshot.schema, digest, len(rows)),
                )
                target.commit()
            except (sqlite3.Error, LegacyImportError):
                target.rollback()
                raise
            if on_receipt is not None:
                on_receipt(snapshot.name, barrier)
        existing_source = target.execute(
            "SELECT source_schema,source_sha256,source_size_bytes,table_count,row_count "
            "FROM schema_import_sources WHERE source_name=?",
            (snapshot.name,),
        ).fetchone()
        identity = (snapshot.schema, snapshot.sha256, snapshot.size_bytes, len(tables), total_rows)
        if existing_source is not None:
            if existing_source != identity:
                raise LegacyImportError(f"{snapshot.name} changed after import receipt")
            return
        target.execute("BEGIN IMMEDIATE")
        try:
            target.execute(
                "INSERT INTO schema_import_sources "
                "VALUES (?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                (snapshot.name, *identity),
            )
            target.commit()
        except sqlite3.Error:
            target.rollback()
            raise
        if on_receipt is not None:
            on_receipt(snapshot.name, "complete")
    finally:
        source.close()


def _operator_only_event_ids(source: sqlite3.Connection) -> frozenset[str]:
    columns = {str(row[1]) for row in source.execute("PRAGMA table_info(evidence_events)")}
    if "operator_only" not in columns:
        return frozenset()
    rows = source.execute(
        "SELECT edge_event_id FROM evidence_events WHERE operator_only = 1"
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def _filter_retired_operator_rows(
    table: str,
    columns: list[str],
    rows: list[tuple[SqliteValue, ...]],
    *,
    operator_only_event_ids: frozenset[str],
) -> tuple[list[str], list[tuple[SqliteValue, ...]]]:
    """Drop temporary SYSTEM_TEST authority while importing ordinary evidence."""
    if table == "evidence_events" and "operator_only" in columns:
        operator_index = columns.index("operator_only")

        def _is_ordinary(row: tuple[SqliteValue, ...]) -> bool:
            flag = row[operator_index]
            if flag is None:
                return True
            if isinstance(flag, bool):
                return not flag
            if isinstance(flag, int):
                return flag == 0
            if isinstance(flag, str) and flag.isdigit():
                return int(flag) == 0
            return True

        kept = [row for row in rows if _is_ordinary(row)]
        kept_columns = [column for column in columns if column != "operator_only"]
        kept_rows = [
            tuple(value for index, value in enumerate(row) if index != operator_index)
            for row in kept
        ]
        return kept_columns, kept_rows
    if table == "clip_events" and operator_only_event_ids and "edge_event_id" in columns:
        event_index = columns.index("edge_event_id")
        kept_rows = [row for row in rows if str(row[event_index]) not in operator_only_event_ids]
        return columns, kept_rows
    return columns, rows


def _validate_target_projection(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    digest: str,
    row_count: int,
    *,
    source_name: str,
) -> None:
    if _target_projection(connection, table, columns) != (digest, row_count):
        raise LegacyImportError(f"conflicting pre-existing target data for {source_name}.{table}")


def _target_projection(
    connection: sqlite3.Connection, table: str, columns: list[str]
) -> tuple[str, int]:
    rows = connection.execute(
        f"SELECT {','.join(_quote(column) for column in columns)} FROM {_quote(table)} "
        f"ORDER BY {','.join(str(index + 1) for index in range(len(columns)))}"
    ).fetchall()
    return _rows_digest(columns, rows), len(rows)


def _rows_digest(columns: list[str], rows: list[tuple[SqliteValue, ...]]) -> str:
    digest = hashlib.sha256()
    for column in columns:
        encoded = column.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
    for row in rows:
        for value in row:
            encoded = _encode_value(value)
            digest.update(struct.pack(">I", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


def _encode_value(value: SqliteValue) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bytes):
        return b"b" + value
    if isinstance(value, str):
        return b"t" + value.encode("utf-8")
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"f" + struct.pack(">d", value)
    raise LegacyImportError(f"unsupported SQLite value {type(value)!r}")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


__all__ = ["ImportProgress", "import_source_tables", "record_backup_receipt"]
