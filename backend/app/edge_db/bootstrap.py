"""Create-only schema-18 bootstrap: the sole DDL owner of the local edge database.

There is exactly one schema (18, the compact ten-table contract) and no
migration ledger. On an empty database this module creates schema 18 in one
transaction under the exclusive deployment lock; on an existing database it
verifies that the schema already *is* 18 and refuses anything else. Nothing is
ever upgraded, drained, imported, or repaired here.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backend.app.edge_db.compact_schema import SCHEMA_18_STATEMENTS
from backend.app.edge_db.compatibility import (
    SCHEMA_18_IDENTITY,
    EdgeDatabaseError,
    NewerSchemaError,
    SchemaLedgerError,
    verify_runtime_schema,
)
from backend.app.edge_db.functions import register_edge_db_functions
from backend.app.edge_db.paths import (
    EDGE_DATABASE_PATH,
    prepare_database_path,
    secure_database_files,
)
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION

BOOTSTRAP_BUSY_TIMEOUT_MS: Final = 5_000
DEPLOYMENT_LOCK_NAME: Final = "deployment.lock"


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    path: Path
    created: bool
    schema_version: int


class DeploymentLockError(EdgeDatabaseError):
    """The exclusive deployment lock could not be acquired or does not cover the path."""


@dataclass(slots=True)
class UnsupportedSchemaError(EdgeDatabaseError):
    """The database exists at a schema other than 18; there is no migration path."""

    found: int

    def __str__(self) -> str:
        return (
            f"edge database schema {self.found} is not schema "
            f"{EDGE_DATABASE_SCHEMA_VERSION}; bootstrap is create-only and never migrates"
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


def _create_schema_18(connection: sqlite3.Connection) -> None:
    """Create every schema-18 object and the ledger row in one transaction."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        _require_still_empty(connection)
        for statement in SCHEMA_18_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO schema_migrations (version, name, checksum, applied_at)
            VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            SCHEMA_18_IDENTITY,
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
    """Create schema 18 on an empty database, or verify an existing one is schema 18.

    Raises ``UnsupportedSchemaError`` for any other version marker and
    ``SchemaLedgerError`` for a version-less database that already holds tables.
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
        if version == 0:
            if _has_any_table(connection):
                raise SchemaLedgerError(
                    "edge database has tables but no schema version; refusing to bootstrap over it"
                )
            _create_schema_18(connection)
            created = True
        elif version > EDGE_DATABASE_SCHEMA_VERSION:
            raise NewerSchemaError(found=version, maximum=EDGE_DATABASE_SCHEMA_VERSION)
        elif version != EDGE_DATABASE_SCHEMA_VERSION:
            raise UnsupportedSchemaError(found=version)
        current = verify_runtime_schema(connection)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise SchemaLedgerError(f"edge database integrity check failed: {integrity!r}")
        return BootstrapResult(path=path, created=created, schema_version=current)
    finally:
        connection.close()
        secure_database_files(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create schema 18 on an empty SeeON edge database or verify an existing one"
    )
    parser.add_argument("--database", type=Path, default=EDGE_DATABASE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = bootstrap_database(args.database)
    except (OSError, sqlite3.Error, EdgeDatabaseError) as error:
        print(f"EDGE_DB_BOOTSTRAP_FAILED: {error}", file=sys.stderr)
        return 1
    print(
        f"EDGE_DB_BOOTSTRAP_OK path={result.path} "
        f"schema={result.schema_version} created={str(result.created).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BootstrapResult",
    "DeploymentLock",
    "DeploymentLockError",
    "UnsupportedSchemaError",
    "bootstrap_database",
    "deployment_lock",
    "main",
]
