"""Stopped-runtime, receipt-based import of the three released edge databases."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db.evidence_backfill import backfill_legacy_evidence
from backend.app.edge_db.legacy_import_snapshot import snapshot_source
from backend.app.edge_db.legacy_import_tables import (
    ImportProgress,
    import_source_tables,
    record_backup_receipt,
)
from backend.app.edge_db.migrator import deployment_lock, migrate_database
from backend.app.edge_db.paths import EDGE_DATABASE_PATH, secure_database_files
from backend.app.edge_db.review_migration import classify_legacy_labels
from backend.app.edge_db.schema import MIGRATIONS


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
        if _target_schema_version(target) < 17:
            migrate_database(target, migrations=MIGRATIONS[:-2], lock=lock)
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
                snapshot = snapshot_source(name, path, target.parent)
                record_backup_receipt(target_connection, snapshot, on_receipt)
                import_source_tables(
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
        # This adapter ends at the lossless v17 staging contract. Schema 18 is
        # candidate-only and must be applied by compact_cutover after exhaustive
        # source-row and filesystem reconciliation.
        migrate_database(target, migrations=MIGRATIONS[:-1], lock=lock)
    return ImportResult(target, tuple(imported))


def _target_schema_version(path: Path) -> int:
    if not path.is_file():
        return 0
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


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
