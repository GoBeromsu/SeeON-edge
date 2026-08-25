"""The sole one-shot DDL owner for the local edge database."""

from __future__ import annotations

import argparse
import fcntl
import os
import sqlite3
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from backend.app.edge_db.compatibility import (
    CANONICAL_MIGRATION_LEDGER,
    EdgeDatabaseError,
    NewerSchemaError,
    SchemaCompatibility,
    SchemaLedgerError,
    verify_runtime_schema,
)
from backend.app.edge_db.cutover_authorization import CompactCutoverRequiredError
from backend.app.edge_db.functions import register_edge_db_functions
from backend.app.edge_db.ownership import Writer, writer_for_table
from backend.app.edge_db.paths import (
    EDGE_DATABASE_PATH,
    prepare_database_path,
    secure_database_files,
)
from backend.app.edge_db.schema import MIGRATIONS, Migration

if TYPE_CHECKING:
    from backend.app.edge_db.cutover_authorization import CompactCutoverAuthorization

MIGRATION_BUSY_TIMEOUT_MS: Final = 5_000
MigrationProgress = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class MigrationResult:
    path: Path
    previous_version: int
    current_version: int


class DeploymentLockError(EdgeDatabaseError):
    """An exclusive migration lock could not be acquired."""


class _LockProof:
    """Inode-bound proof minted only after a successful flock."""

    __slots__ = ("_dev", "_ino")

    def __init__(self, descriptor: int) -> None:
        stat = os.fstat(descriptor)
        self._dev = stat.st_dev
        self._ino = stat.st_ino


_LIVE_LOCKS: dict[int, tuple[int, int, int]] = {}


@dataclass(slots=True)
class DeploymentLock:
    """Proof that this process currently owns one deployment lock."""

    state_directory: Path
    _descriptor: int
    _active: bool = True
    _proof: object = None

    def require_for(self, database: Path) -> None:
        registered = _LIVE_LOCKS.get(id(self))
        if registered is None:
            if isinstance(self._proof, _LockProof) and not self._active:
                raise DeploymentLockError("EXPIRED_LOCK")
            raise DeploymentLockError("FORGED_LOCK")
        registered_descriptor, registered_dev, registered_ino = registered
        if registered_descriptor != self._descriptor:
            raise DeploymentLockError("FORGED_LOCK")
        if not isinstance(self._proof, _LockProof):
            raise DeploymentLockError("FORGED_LOCK")
        if not self._active:
            raise DeploymentLockError("EXPIRED_LOCK")
        try:
            descriptor_stat = os.fstat(self._descriptor)
            lock_stat = (self.state_directory / "deployment.lock").stat()
        except OSError as error:
            raise DeploymentLockError("FORGED_LOCK") from error
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            registered_dev,
            registered_ino,
        ):
            raise DeploymentLockError("FORGED_LOCK")
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            self._proof._dev,
            self._proof._ino,
        ):
            raise DeploymentLockError("FORGED_LOCK")
        if (lock_stat.st_dev, lock_stat.st_ino) != (self._proof._dev, self._proof._ino):
            raise DeploymentLockError("FORGED_LOCK")
        if database.parent.resolve() != self.state_directory:
            raise DeploymentLockError("edge deployment lock does not cover the database path")


@contextmanager
def deployment_lock(
    state_directory: Path,
    *,
    blocking: bool = False,
) -> Iterator[DeploymentLock]:
    """Hold the exclusive deployment lock shared with both runtimes."""
    state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_directory = state_directory.resolve()
    descriptor = os.open(resolved_directory / "deployment.lock", os.O_CREAT | os.O_RDWR, 0o600)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    ownership: DeploymentLock | None = None
    try:
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as error:
            raise DeploymentLockError(
                "edge deployment lock is held by a running runtime"
            ) from error
        ownership = DeploymentLock(
            resolved_directory, descriptor, _proof=_LockProof(descriptor)
        )
        proof_stat = os.fstat(descriptor)
        _LIVE_LOCKS[id(ownership)] = (descriptor, proof_stat.st_dev, proof_stat.st_ino)
        yield ownership
    finally:
        if ownership is not None:
            _LIVE_LOCKS.pop(id(ownership), None)
            ownership._active = False
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _migration_identities(migrations: Sequence[Migration]) -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (migration.version, migration.name, migration.checksum) for migration in migrations
    )


def _validate_ledger(migrations: Sequence[Migration]) -> None:
    versions = [migration.version for migration in migrations]
    if versions != list(range(1, len(migrations) + 1)):
        raise SchemaLedgerError("migration versions must be contiguous and start at one")
    if len({migration.name for migration in migrations}) != len(migrations):
        raise SchemaLedgerError("migration names must be unique")
    identities = _migration_identities(migrations)
    canonical_prefix = CANONICAL_MIGRATION_LEDGER[: len(identities)]
    if identities[: len(canonical_prefix)] != canonical_prefix:
        raise SchemaLedgerError("compiled migration registry differs from canonical identities")


def _user_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return 0 if row is None else int(row[0])


def _peek_user_version(path: Path) -> int:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return _user_version(connection)
    finally:
        connection.close()


def _run_schema18_preflight(path: Path, migrations: Sequence[Migration]) -> None:
    for migration in migrations:
        if migration.version == 18 and migration.preflight is not None:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                migration.preflight(connection)
            finally:
                connection.close()
            return


def _enable_wal(connection: sqlite3.Connection) -> None:
    # Serialize the first-open boundary before changing the persistent journal mode.
    connection.execute("BEGIN IMMEDIATE")
    connection.commit()
    row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
    if row != ("wal",):
        raise SchemaLedgerError("edge database could not enter WAL mode")


def _verify_schema(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration],
    version: int,
) -> None:
    if version == 0:
        return
    verify_runtime_schema(
        connection,
        SchemaCompatibility(minimum=version, maximum=version),
        expected_migrations=_migration_identities(migrations),
    )


class _MigrationAuthorizer:
    def __init__(self) -> None:
        self._creates_application_table = False
        self._altered_application_table: str | None = None
        self._writable_tables: frozenset[str] = frozenset()

    def begin_statement(self, writable_tables: frozenset[str] = frozenset()) -> None:
        self._creates_application_table = False
        self._altered_application_table = None
        self._writable_tables = writable_tables

    def __call__(
        self,
        action: int,
        argument_one: str | None,
        argument_two: str | None,
        database: str | None,
        source: str | None,
    ) -> int:
        del database, source
        if action == sqlite3.SQLITE_CREATE_TABLE and argument_one is not None:
            writer = writer_for_table(argument_one)
            self._creates_application_table = writer is Writer.API
        if action == sqlite3.SQLITE_ALTER_TABLE and argument_two is not None:
            if writer_for_table(argument_two) in {Writer.API, Writer.MIGRATOR}:
                self._altered_application_table = argument_two
        if action == sqlite3.SQLITE_SELECT and self._creates_application_table:
            # SQLite reports CREATE TABLE before SELECT for every CTAS form,
            # including quoted identifiers, WITH clauses, and bare VALUES.
            return sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            if argument_one in {"sqlite_master", "sqlite_schema"}:
                return sqlite3.SQLITE_OK
            if self._altered_application_table is not None and argument_one in {
                "sqlite_temp_master",
                "sqlite_temp_schema",
            }:
                # ALTER TABLE RENAME updates SQLite's temporary schema cache as
                # part of the same authorized DDL statement.
                return sqlite3.SQLITE_OK
            if argument_one in self._writable_tables:
                return sqlite3.SQLITE_OK
            if argument_one is not None and writer_for_table(argument_one) is Writer.MIGRATOR:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:
            pragma = "" if argument_one is None else argument_one.lower()
            if pragma in {
                "foreign_key_list",
                "index_info",
                "index_list",
                "index_xinfo",
                "integrity_check",
                "quick_check",
                "table_info",
                "table_xinfo",
            } and argument_two is None:
                return sqlite3.SQLITE_OK
            if pragma in {
                "foreign_key_list",
                "index_info",
                "index_list",
                "index_xinfo",
                "table_info",
                "table_xinfo",
            }:
                return sqlite3.SQLITE_OK
            if pragma == "quick_check" and argument_two == self._altered_application_table:
                # SQLite implements ALTER TABLE with an internal table-scoped
                # quick_check. Admit only the table named by this same statement.
                return sqlite3.SQLITE_OK
            if pragma != "user_version":
                return sqlite3.SQLITE_DENY
            if argument_two is not None and not argument_two.isdecimal():
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK


def _require_next_version(current: int, migration: Migration) -> None:
    if current != migration.version - 1:
        raise SchemaLedgerError("concurrent migrator produced a schema version gap")


def migrate_database(
    path: Path = EDGE_DATABASE_PATH,
    *,
    migrations: Sequence[Migration] = MIGRATIONS,
    on_statement_applied: MigrationProgress | None = None,
    lock: DeploymentLock | None = None,
    cutover: CompactCutoverAuthorization | None = None,
) -> MigrationResult:
    """Apply each pending migration atomically under the deployment lock."""
    if lock is None:
        with deployment_lock(path.parent) as ownership:
            return migrate_database(
                path,
                migrations=migrations,
                on_statement_applied=on_statement_applied,
                lock=ownership,
                cutover=cutover,
            )
    try:
        lock.require_for(path)
    except DeploymentLockError as error:
        if cutover is not None:
            raise CompactCutoverRequiredError(str(error)) from error
        raise
    _validate_ledger(migrations)
    cutover_source = None
    if path.is_file() and any(migration.version == 18 for migration in migrations):
        peeked = _peek_user_version(path)
        if peeked == 17:
            _run_schema18_preflight(path, migrations)
            if cutover is None:
                raise CompactCutoverRequiredError
            cutover_source = cutover.redeem(lock, path)
    prepare_database_path(path)
    connection = sqlite3.connect(
        path,
        timeout=MIGRATION_BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,
    )
    previous = 0
    try:
        connection.execute(f"PRAGMA busy_timeout = {MIGRATION_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        _enable_wal(connection)
        connection.execute("PRAGMA synchronous = FULL")
        previous = _user_version(connection)
        target = len(migrations)
        if previous > target:
            raise NewerSchemaError(found=previous, maximum=target)
        register_edge_db_functions(connection)
        _verify_schema(connection, migrations, previous)
        authorizer = _MigrationAuthorizer()
        connection.set_authorizer(authorizer)

        for migration in migrations:
            if migration.version <= _user_version(connection):
                continue
            if migration.version == 18 and previous not in {0, 17}:
                continue
            if migration.preflight is not None:
                migration.preflight(connection)
            if migration.version == 18 and previous == 17 and cutover_source is None:
                if cutover is None:
                    raise CompactCutoverRequiredError
                cutover_source = cutover.redeem(lock, path)
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = _user_version(connection)
                if current >= migration.version:
                    connection.commit()
                    continue
                _require_next_version(current, migration)
                _verify_schema(connection, migrations, current)
                for statement_index, statement in enumerate(migration.statements, start=1):
                    authorizer.begin_statement(migration.writable_tables)
                    connection.execute(statement)
                    if on_statement_applied is not None:
                        on_statement_applied(migration.version, statement_index)
                authorizer.begin_statement()
                if migration.version == 18 and cutover_source is not None:
                    connection.execute(
                        """
                        INSERT INTO schema_migrations (
                            version, name, checksum, applied_at,
                            source_schema_version, source_db_sha256, reconciliation_sha256
                        )
                        VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?, ?)
                        """,
                        (
                            migration.version,
                            migration.name,
                            migration.checksum,
                            cutover_source.source_schema_version,
                            cutover_source.source_db_sha256,
                            cutover_source.reconciliation_sha256,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO schema_migrations (version, name, checksum, applied_at)
                        VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                        """,
                        (migration.version, migration.name, migration.checksum),
                    )
                authorizer.begin_statement()
                connection.execute(f"PRAGMA user_version = {migration.version}")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        authorizer.begin_statement()
        current = _user_version(connection)
        _verify_schema(connection, migrations, current)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise SchemaLedgerError(f"edge database integrity check failed: {integrity!r}")
        return MigrationResult(path=path, previous_version=previous, current_version=current)
    finally:
        connection.close()
        secure_database_files(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate the local SeeON edge database")
    parser.add_argument("--database", type=Path, default=EDGE_DATABASE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = migrate_database(args.database)
    except (OSError, sqlite3.Error, EdgeDatabaseError) as error:
        print(f"EDGE_DB_MIGRATION_FAILED: {error}", file=sys.stderr)
        return 1
    print(
        f"EDGE_DB_MIGRATION_OK path={result.path} "
        f"previous={result.previous_version} current={result.current_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DeploymentLock",
    "DeploymentLockError",
    "MigrationResult",
    "deployment_lock",
    "main",
    "migrate_database",
]
