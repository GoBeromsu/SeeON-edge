"""Create-or-extend schema-19 bootstrap: the sole DDL owner of the local edge database.

On an empty database this module creates schema 19 in one BEGIN IMMEDIATE
transaction under the exclusive deployment lock (ledger row 19 only). On an
exact schema-18 database it EXTENDS: take a consistent sqlite3 backup, then in
one BEGIN IMMEDIATE create the six execution-record tables and indexes, insert
ledger row 19 with source_schema_version=18 and source_db_sha256 of the backup,
and set PRAGMA user_version=19. Schema-18 tables and rows are never ALTER/DROP
rewritten. Any other version is refused.

Rolling back to a schema-18 image after extension requires stopping the stack
and restoring ``<database name>.schema18-backup.sqlite3`` over the live
database (remove -wal/-shm). That restore discards every application write
made after the extension. A schema-18 image refuses a schema-19 database by
design. There is no in-process downgrade path.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backend.app.edge_db.compact_schema import SCHEMA_19_STATEMENTS
from backend.app.edge_db.compatibility import (
    SCHEMA_19_IDENTITY,
    EdgeDatabaseError,
    NewerSchemaError,
    SchemaLedgerError,
    verify_runtime_schema,
    verify_schema18_contract,
)
from backend.app.edge_db.execution_records_ddl import EXECUTION_RECORD_CREATE_STATEMENTS
from backend.app.edge_db.functions import register_edge_db_functions
from backend.app.edge_db.paths import (
    DIAGNOSTICS_DATABASE_FILENAME,
    EDGE_DATABASE_PATH,
    prepare_database_path,
    schema18_backup_path,
    secure_database_files,
)
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION

BOOTSTRAP_BUSY_TIMEOUT_MS: Final = 5_000
DEPLOYMENT_LOCK_NAME: Final = "deployment.lock"
SCHEMA_18_VERSION: Final = 18

# The standalone diagnostics database has no migration history to preserve --
# unlike edge.sqlite3 it carries no schema_migrations ledger table at all, only
# this flat PRAGMA user_version stamp (#579/#580, S1/S2).
DIAGNOSTICS_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    path: Path
    created: bool
    extended: bool
    schema_version: int


@dataclass(frozen=True, slots=True)
class DiagnosticsBootstrapResult:
    path: Path
    created: bool
    schema_version: int


class DeploymentLockError(EdgeDatabaseError):
    """The exclusive deployment lock could not be acquired or does not cover the path."""


@dataclass(slots=True)
class UnsupportedSchemaError(EdgeDatabaseError):
    """The database exists at a schema other than 18 or 19; there is no migration path."""

    found: int

    def __str__(self) -> str:
        return (
            f"edge database schema {self.found} is not schema "
            f"{EDGE_DATABASE_SCHEMA_VERSION}; bootstrap creates 19 or extends 18 "
            "and never migrates any other version"
        )


@dataclass(slots=True)
class UnsupportedDiagnosticsSchemaError(EdgeDatabaseError):
    """The diagnostics database exists at a user_version other than the current one."""

    found: int

    def __str__(self) -> str:
        return (
            f"diagnostics database schema {self.found} is not schema "
            f"{DIAGNOSTICS_SCHEMA_VERSION}; bootstrap only ever creates or verifies it"
        )


@dataclass(slots=True)
class DeploymentLock:
    """Proof that this process currently holds the exclusive deployment lock."""

    state_directory: Path
    _descriptor: int
    _active: bool = True

    def require_for(self, database: Path) -> None:
        if not self._active:
            raise DeploymentLockError("edge deployment lock has already been released")
        try:
            descriptor_stat = os.fstat(self._descriptor)
            lock_stat = (self.state_directory / DEPLOYMENT_LOCK_NAME).stat()
        except OSError as error:
            raise DeploymentLockError("edge deployment lock file is gone") from error
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
            raise DeploymentLockError("edge deployment lock file was replaced")
        if database.parent.resolve() != self.state_directory:
            raise DeploymentLockError("edge deployment lock does not cover the database path")


@contextmanager
def deployment_lock(state_directory: Path, *, blocking: bool = False) -> Iterator[DeploymentLock]:
    """Hold the exclusive deployment lock that runtime connections take shared."""
    state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_directory = state_directory.resolve()
    descriptor = os.open(resolved_directory / DEPLOYMENT_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
    ownership: DeploymentLock | None = None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise DeploymentLockError(
                "edge deployment lock is held by a running runtime"
            ) from error
        ownership = DeploymentLock(resolved_directory, descriptor)
        yield ownership
    finally:
        if ownership is not None:
            ownership._active = False
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _user_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return 0 if row is None else int(row[0])


def _has_any_table(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE type = 'table')"
    ).fetchone()
    return row == (1,)


def _enable_wal(connection: sqlite3.Connection) -> None:
    # Serialize the first-open boundary before changing the persistent journal mode.
    connection.execute("BEGIN IMMEDIATE")
    connection.commit()
    row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
    if row != ("wal",):
        raise SchemaLedgerError("edge database could not enter WAL mode")


def _require_still_empty(connection: sqlite3.Connection) -> None:
    if _user_version(connection) != 0 or _has_any_table(connection):
        raise SchemaLedgerError("edge database changed underneath the bootstrap")


def _create_schema_19(connection: sqlite3.Connection) -> None:
    """Create every schema-19 object and ledger row 19 in one transaction."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        _require_still_empty(connection)
        for statement in SCHEMA_19_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO schema_migrations (version, name, checksum, applied_at)
            VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            SCHEMA_19_IDENTITY,
        )
        connection.execute(f"PRAGMA user_version = {EDGE_DATABASE_SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _backup_schema18(source: sqlite3.Connection, backup_path: Path) -> str:
    """Take a consistent SQLite backup; refuse if the destination already exists."""
    if backup_path.exists():
        raise SchemaLedgerError(
            f"schema-18 backup already exists at {backup_path}; move it aside before extending"
        )
    descriptor = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(descriptor)
    backup_path.chmod(0o600)
    destination = sqlite3.connect(backup_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
    backup_path.chmod(0o600)
    return _sha256_file(backup_path)


def _require_still_schema_18(connection: sqlite3.Connection) -> None:
    if _user_version(connection) != SCHEMA_18_VERSION:
        raise SchemaLedgerError("edge database changed underneath the schema-18 extension")


def _extend_schema_18_to_19(connection: sqlite3.Connection, database: Path) -> None:
    """Verify exact schema 18, backup, then add the six execution tables in one txn."""
    verify_schema18_contract(connection)
    backup_path = schema18_backup_path(database)
    source_sha256 = _backup_schema18(connection, backup_path)
    connection.execute("BEGIN IMMEDIATE")
    try:
        _require_still_schema_18(connection)
        for statement in EXECUTION_RECORD_CREATE_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO schema_migrations (
                version, name, checksum, applied_at,
                source_schema_version, source_db_sha256
            )
            VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?)
            """,
            (*SCHEMA_19_IDENTITY, SCHEMA_18_VERSION, source_sha256),
        )
        connection.execute(f"PRAGMA user_version = {EDGE_DATABASE_SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def bootstrap_database(
    path: Path = EDGE_DATABASE_PATH,
    *,
    lock: DeploymentLock | None = None,
) -> BootstrapResult:
    """Create schema 19 on an empty database, extend exact schema 18, or verify 19.

    Raises ``NewerSchemaError`` when ``user_version`` is greater than 19,
    ``UnsupportedSchemaError`` for any other non-18 version marker, and
    ``SchemaLedgerError`` for a version-less database that already holds tables.

    Rolling back to a schema-18 image after extension requires stopping the
    stack and restoring the schema18-backup file over the live database
    (remove -wal/-shm). That restore discards every application write made
    after the extension. A schema-18 image refuses a schema-19 database by
    design. There is no in-process downgrade path.
    """
    if lock is None:
        with deployment_lock(path.parent) as ownership:
            return bootstrap_database(path, lock=ownership)
    lock.require_for(path)
    prepare_database_path(path)
    connection = sqlite3.connect(
        path,
        timeout=BOOTSTRAP_BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,
    )
    try:
        connection.execute(f"PRAGMA busy_timeout = {BOOTSTRAP_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        _enable_wal(connection)
        connection.execute("PRAGMA synchronous = FULL")
        register_edge_db_functions(connection)
        version = _user_version(connection)
        created = False
        extended = False
        if version == 0:
            if _has_any_table(connection):
                raise SchemaLedgerError(
                    "edge database has tables but no schema version; refusing to bootstrap over it"
                )
            _create_schema_19(connection)
            created = True
        elif version > EDGE_DATABASE_SCHEMA_VERSION:
            raise NewerSchemaError(found=version, maximum=EDGE_DATABASE_SCHEMA_VERSION)
        elif version == SCHEMA_18_VERSION:
            _extend_schema_18_to_19(connection, path)
            extended = True
        elif version != EDGE_DATABASE_SCHEMA_VERSION:
            raise UnsupportedSchemaError(found=version)
        current = verify_runtime_schema(connection)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise SchemaLedgerError(f"edge database integrity check failed: {integrity!r}")
        return BootstrapResult(
            path=path,
            created=created,
            extended=extended,
            schema_version=current,
        )
    finally:
        connection.close()
        secure_database_files(path)


def _create_diagnostics_schema(connection: sqlite3.Connection) -> None:
    """Create the six execution-record tables and stamp user_version, one txn."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        _require_still_empty(connection)
        for statement in EXECUTION_RECORD_CREATE_STATEMENTS:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {DIAGNOSTICS_SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def bootstrap_diagnostics_database(
    path: Path,
    *,
    lock: DeploymentLock,
) -> DiagnosticsBootstrapResult:
    """Create the standalone execution-record database, or verify it as-is.

    Sibling of ``edge.sqlite3``: same one-shot bootstrap, same exclusive
    ``deployment.lock`` (``DeploymentLock.require_for`` only checks the
    containing directory, so one lock legitimately covers both files), same
    bootstrap-once/verify-only runtime split -- but a flat ``PRAGMA
    user_version`` is its only schema ledger, with no migration path and no
    ``schema_migrations`` table (see ``backend/app/edge_db/diagnostics_connection.py``,
    which now only opens and verifies it). (#579/#580, S1/S2/N1)
    """
    lock.require_for(path)
    prepare_database_path(path)
    connection = sqlite3.connect(
        path,
        timeout=BOOTSTRAP_BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,
    )
    try:
        connection.execute(f"PRAGMA busy_timeout = {BOOTSTRAP_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        _enable_wal(connection)
        connection.execute("PRAGMA synchronous = FULL")
        version = _user_version(connection)
        created = False
        if version == 0:
            if _has_any_table(connection):
                raise SchemaLedgerError(
                    "diagnostics database has tables but no schema version; "
                    "refusing to bootstrap over it"
                )
            _create_diagnostics_schema(connection)
            created = True
            version = DIAGNOSTICS_SCHEMA_VERSION
        elif version != DIAGNOSTICS_SCHEMA_VERSION:
            raise UnsupportedDiagnosticsSchemaError(found=version)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise SchemaLedgerError(f"diagnostics database integrity check failed: {integrity!r}")
        return DiagnosticsBootstrapResult(path=path, created=created, schema_version=version)
    finally:
        connection.close()
        secure_database_files(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create schema 19 on an empty SeeON edge database, extend an exact "
            "schema-18 database, or verify an existing schema-19 database"
        )
    )
    parser.add_argument("--database", type=Path, default=EDGE_DATABASE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # One shared exclusive lock covers both files (same directory), so the
        # diagnostics sibling is created/verified in the same one-shot run as
        # edge.sqlite3 -- never bootstrapped lazily at runtime (#579/#580).
        with deployment_lock(args.database.parent) as lock:
            result = bootstrap_database(args.database, lock=lock)
            diagnostics_result = bootstrap_diagnostics_database(
                args.database.parent / DIAGNOSTICS_DATABASE_FILENAME, lock=lock
            )
    except (OSError, sqlite3.Error, EdgeDatabaseError) as error:
        print(f"EDGE_DB_BOOTSTRAP_FAILED: {error}", file=sys.stderr)
        return 1
    print(
        f"EDGE_DB_BOOTSTRAP_OK path={result.path} "
        f"schema={result.schema_version} created={str(result.created).lower()} "
        f"extended={str(result.extended).lower()}"
    )
    print(
        f"EDGE_DIAGNOSTICS_DB_BOOTSTRAP_OK path={diagnostics_result.path} "
        f"schema={diagnostics_result.schema_version} "
        f"created={str(diagnostics_result.created).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BootstrapResult",
    "DeploymentLock",
    "DeploymentLockError",
    "DiagnosticsBootstrapResult",
    "UnsupportedDiagnosticsSchemaError",
    "UnsupportedSchemaError",
    "bootstrap_database",
    "bootstrap_diagnostics_database",
    "deployment_lock",
    "main",
]
