"""Stopped-runtime, receipt-based import of the three released edge databases."""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import struct
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backend.app.edge_db.evidence_backfill import backfill_legacy_evidence
from backend.app.edge_db.migrator import deployment_lock, migrate_database
from backend.app.edge_db.paths import EDGE_DATABASE_PATH, secure_database_files
from backend.app.edge_db.review_migration import classify_legacy_labels
from backend.app.edge_db.schema import MIGRATIONS

ImportProgress = Callable[[str, str], None]
_ALLOWED_WORKER_SCHEMAS: Final = {6, 7, 8, 9, 10}
_RETIRED_LEGACY_TABLES: Final = frozenset({"system_test_runs"})
_TABLE_PRIORITY: Final = {
    "camera_topology_floors": 10,
    "camera_topology_rooms": 11,
    "camera_topology_cameras": 12,
    "evidence_events": 20,
    "evidence_clips": 21,
    "clip_events": 22,
}


@dataclass(frozen=True, slots=True)
class LegacyDatabasePaths:
    catalog: Path
    connection: Path
    worker: Path

    @classmethod
    def production(cls) -> LegacyDatabasePaths:
        return cls(
            catalog=Path("/var/lib/legacy-api-state/catalog.sqlite3"),
            connection=Path("/var/lib/legacy-api-state/connection-settings.sqlite3"),
            worker=Path("/var/lib/legacy-worker-state/worker-state.sqlite3"),
        )


@dataclass(frozen=True, slots=True)
class ImportResult:
    path: Path
    imported_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    name: str
    path: Path
    backup: Path
    schema: str
    sha256: str
    size_bytes: int


def import_legacy_databases(
    target: Path = EDGE_DATABASE_PATH,
    sources: LegacyDatabasePaths | None = None,
    *,
    on_receipt: ImportProgress | None = None,
) -> ImportResult:
    """Import each existing legacy database under one exclusive deployment lock."""
    resolved = sources or LegacyDatabasePaths.production()
    imported: list[str] = []
    with deployment_lock(target.parent) as lock:
        # Build the transitional schema first, then import before applying v17.
        # Its preflight can therefore reject undelivered legacy evidence rather
        # than stamping a database that contains it as schema 17.
        if _target_schema_version(target) < len(MIGRATIONS):
            migrate_database(target, migrations=MIGRATIONS[:-1], lock=lock)
        target_connection = sqlite3.connect(target, isolation_level=None)
        try:
            target_connection.execute("PRAGMA foreign_keys = ON")
            for name, path in (
                ("catalog", resolved.catalog),
                ("connection", resolved.connection),
                ("worker", resolved.worker),
            ):
                if not path.is_file():
                    continue
                snapshot = _snapshot_source(name, path, target.parent)
                _record_backup_receipt(target_connection, snapshot, on_receipt)
                _import_source_tables(
                    target_connection,
                    snapshot,
                    snapshot.backup,
                    on_receipt=on_receipt,
                )
                imported.append(name)
            backfill_legacy_evidence(target_connection)
            classify_legacy_labels(target_connection)
            integrity = target_connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise sqlite3.DatabaseError(f"edge import integrity check failed: {integrity!r}")
        finally:
            target_connection.close()
            secure_database_files(target)
        migrate_database(target, lock=lock)
    return ImportResult(target, tuple(imported))


def _target_schema_version(path: Path) -> int:
    if not path.is_file():
        return 0
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


def _snapshot_source(name: str, path: Path, state_directory: Path) -> _SourceSnapshot:
    source = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
        timeout=5.0,
        isolation_level=None,
    )
    try:
        if source.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise sqlite3.DatabaseError(f"{name} integrity check failed")
        schema = _source_schema(name, source)
        backup_directory = state_directory / "legacy-backups"
        backup_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, raw = tempfile.mkstemp(prefix=f".{name}-", suffix=".tmp", dir=backup_directory)
        os.close(descriptor)
        temporary = Path(raw)
        try:
            destination = sqlite3.connect(temporary)
            try:
                source.backup(destination)
                if destination.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise sqlite3.DatabaseError(f"{name} backup integrity check failed")
            finally:
                destination.close()
            encoded = temporary.read_bytes()
            digest = hashlib.sha256(encoded).hexdigest()
            backup = backup_directory / f"{name}-schema-{schema}-{digest}.sqlite3"
            if backup.exists():
                _validate_existing_backup(backup, digest, name)
                temporary.unlink()
            else:
                os.replace(temporary, backup)
                backup.chmod(0o600)
            return _SourceSnapshot(name, path, backup, schema, digest, backup.stat().st_size)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    finally:
        source.close()


def _validate_existing_backup(backup: Path, digest: str, source_name: str) -> None:
    if hashlib.sha256(backup.read_bytes()).hexdigest() != digest:
        raise sqlite3.DatabaseError(f"{source_name} backup digest collision")


def _source_schema(name: str, connection: sqlite3.Connection) -> str:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if name == "catalog":
        if version != 3:
            raise ValueError(f"unsupported catalog schema {version}; expected 3")
        return str(version)
    if name == "worker":
        if version not in _ALLOWED_WORKER_SCHEMAS:
            raise ValueError(
                f"unsupported worker outbox schema {version}; expected 6, 7, 8, 9, or 10"
            )
        return str(version)
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(connection_settings)")}
    required = {
        "facility_code",
        "client_installation_ref",
        "edge_installation_id",
        "enrollment_generation",
    }
    if not required <= columns:
        raise ValueError("unsupported connection schema; run released connection migration first")
    return "connection-v2"


def _record_backup_receipt(
    target: sqlite3.Connection,
    snapshot: _SourceSnapshot,
    on_receipt: ImportProgress | None,
) -> None:
    existing = target.execute(
        "SELECT digest,row_count FROM schema_import_receipts "
        "WHERE source_name=? AND barrier='backup'",
        (snapshot.name,),
    ).fetchone()
    expected = (snapshot.sha256, snapshot.size_bytes)
    if existing is not None and existing != expected:
        raise ValueError(f"{snapshot.name} changed after import receipt")
    if existing is None:
        target.execute("BEGIN IMMEDIATE")
        try:
            target.execute(
                "INSERT INTO schema_import_receipts "
                "VALUES (?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                (snapshot.name, "backup", snapshot.schema, snapshot.sha256, snapshot.size_bytes),
            )
            target.commit()
        except BaseException:
            target.rollback()
            raise
        if on_receipt is not None:
            on_receipt(snapshot.name, "backup")


def _import_source_tables(
    target: sqlite3.Connection,
    snapshot: _SourceSnapshot,
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
                raise ValueError(f"legacy {snapshot.name} owns unsupported table {table}")
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
                    raise ValueError(f"{snapshot.name}.{table} changed after import receipt")
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
            except BaseException:
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
                raise ValueError(f"{snapshot.name} changed after import receipt")
            return
        target.execute("BEGIN IMMEDIATE")
        try:
            target.execute(
                "INSERT INTO schema_import_sources "
                "VALUES (?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                (snapshot.name, *identity),
            )
            target.commit()
        except BaseException:
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
    rows: list[tuple[object, ...]],
    *,
    operator_only_event_ids: frozenset[str],
) -> tuple[list[str], list[tuple[object, ...]]]:
    """Drop temporary SYSTEM_TEST authority while importing ordinary evidence."""
    if table == "evidence_events" and "operator_only" in columns:
        operator_index = columns.index("operator_only")

        def _is_ordinary(row: tuple[object, ...]) -> bool:
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
        raise ValueError(f"conflicting pre-existing target data for {source_name}.{table}")


def _target_projection(
    connection: sqlite3.Connection, table: str, columns: list[str]
) -> tuple[str, int]:
    rows = connection.execute(
        f"SELECT {','.join(_quote(column) for column in columns)} FROM {_quote(table)} "
        f"ORDER BY {','.join(str(index + 1) for index in range(len(columns)))}"
    ).fetchall()
    return _rows_digest(columns, rows), len(rows)


def _rows_digest(columns: list[str], rows: list[tuple[object, ...]]) -> str:
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


def _encode_value(value: object) -> bytes:
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
    raise TypeError(f"unsupported SQLite value {type(value)!r}")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate and import local edge state")
    parser.add_argument("--database", type=Path, default=EDGE_DATABASE_PATH)
    parser.add_argument("--catalog", type=Path, default=LegacyDatabasePaths.production().catalog)
    parser.add_argument(
        "--connection", type=Path, default=LegacyDatabasePaths.production().connection
    )
    parser.add_argument("--worker", type=Path, default=LegacyDatabasePaths.production().worker)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = import_legacy_databases(
            args.database,
            LegacyDatabasePaths(args.catalog, args.connection, args.worker),
        )
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as error:
        print(f"EDGE_DB_IMPORT_FAILED: {error}", file=sys.stderr)
        return 1
    sources = ",".join(result.imported_sources) or "fresh"
    print(f"EDGE_DB_IMPORT_OK path={result.path} sources={sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ImportResult",
    "LegacyDatabasePaths",
    "deployment_lock",
    "import_legacy_databases",
    "main",
]
