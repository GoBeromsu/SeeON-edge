"""DDL-free runtime connections and bounded write transactions."""

from __future__ import annotations

import fcntl
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Final

from shared.edge_db.compatibility import (
    CURRENT_SCHEMA_RANGE,
    EdgeDatabaseError,
    MigrationRequiredError,
    SchemaCompatibility,
    verify_runtime_schema,
)
from shared.edge_db.ownership import Writer, writer_for_table
from shared.edge_db.paths import secure_database_files

NORMAL_BUSY_TIMEOUT_MS: Final = 5_000


class RuntimeActor(StrEnum):
    API = "api"
    WORKER = "worker"


class BusyPolicy(StrEnum):
    BOUNDED_WAIT = "bounded_wait"
    ZERO_WAIT = "zero_wait"


class NestedTransactionError(EdgeDatabaseError):
    """A write transaction was requested while another is already active."""


_DDL_ACTION_NAMES: Final = (
    "SQLITE_ALTER_TABLE",
    "SQLITE_ANALYZE",
    "SQLITE_ATTACH",
    "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TEMP_INDEX",
    "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_TRIGGER",
    "SQLITE_CREATE_TEMP_VIEW",
    "SQLITE_CREATE_TRIGGER",
    "SQLITE_CREATE_VIEW",
    "SQLITE_CREATE_VTABLE",
    "SQLITE_DETACH",
    "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_TEMP_TRIGGER",
    "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_TRIGGER",
    "SQLITE_DROP_VIEW",
    "SQLITE_DROP_VTABLE",
    "SQLITE_REINDEX",
)
_DDL_ACTIONS: Final = frozenset(
    getattr(sqlite3, name) for name in _DDL_ACTION_NAMES if hasattr(sqlite3, name)
)
_ARGUMENT_READ_PRAGMAS: Final = frozenset(
    {"foreign_key_list", "index_info", "index_list", "table_info", "table_xinfo"}
)
_READ_PRAGMAS: Final = frozenset(
    {
        "busy_timeout",
        "compile_options",
        "foreign_key_list",
        "foreign_keys",
        "index_info",
        "index_list",
        "integrity_check",
        "journal_mode",
        "quick_check",
        "synchronous",
        "table_info",
        "table_xinfo",
        "user_version",
    }
)


Authorizer = Callable[[int, str | None, str | None, str | None, str | None], int]


class _RuntimeConnection(sqlite3.Connection):
    _deployment_lock_descriptor: int | None = None

    def close(self) -> None:
        descriptor = self._deployment_lock_descriptor
        self._deployment_lock_descriptor = None
        try:
            super().close()
        finally:
            if descriptor is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def _acquire_runtime_lock(path: Path) -> int:
    descriptor = os.open(path.parent / "deployment.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _runtime_authorizer(actor: RuntimeActor) -> Authorizer:
    writer = Writer(actor.value)

    def authorize(
        action: int,
        argument_one: str | None,
        argument_two: str | None,
        database: str | None,
        source: str | None,
    ) -> int:
        del database, source
        if action in _DDL_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            return (
                sqlite3.SQLITE_OK
                if argument_one is not None and writer_for_table(argument_one) is writer
                else sqlite3.SQLITE_DENY
            )
        if action == sqlite3.SQLITE_PRAGMA:
            pragma = "" if argument_one is None else argument_one.lower()
            if pragma not in _READ_PRAGMAS or (
                argument_two is not None and pragma not in _ARGUMENT_READ_PRAGMAS
            ):
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return authorize


def _require_wal(journal_row: tuple[object, ...] | None) -> None:
    if journal_row != ("wal",):
        raise EdgeDatabaseError("edge database must be migrated into WAL mode before runtime")


def open_runtime_database(
    path: Path,
    *,
    actor: RuntimeActor,
    compatibility: SchemaCompatibility = CURRENT_SCHEMA_RANGE,
    busy_policy: BusyPolicy = BusyPolicy.BOUNDED_WAIT,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open an already-migrated database without creating or migrating schema."""
    if not path.is_file():
        raise MigrationRequiredError(found=0, minimum=compatibility.minimum)
    timeout_ms = 0 if busy_policy is BusyPolicy.ZERO_WAIT else NORMAL_BUSY_TIMEOUT_MS
    try:
        lock_descriptor = _acquire_runtime_lock(path)
    except BlockingIOError as error:
        raise EdgeDatabaseError("edge deployment migration is in progress") from error
    try:
        connection = sqlite3.connect(
            path,
            timeout=timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=check_same_thread,
            factory=_RuntimeConnection,
        )
    except BaseException:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
        raise
    connection._deployment_lock_descriptor = lock_descriptor
    try:
        connection.execute(f"PRAGMA busy_timeout = {timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        journal_row = connection.execute("PRAGMA journal_mode").fetchone()
        _require_wal(journal_row)
        verify_runtime_schema(connection, compatibility)
        connection.set_authorizer(_runtime_authorizer(actor))
        secure_database_files(path)
    except (OSError, sqlite3.Error, EdgeDatabaseError):
        connection.close()
        raise
    return connection


@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Run one short write unit under SQLite's fixed bounded lock policy."""
    if connection.in_transaction:
        raise NestedTransactionError("nested edge database write transactions are forbidden")
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


def best_effort_zero_wait_write(
    path: Path,
    *,
    actor: RuntimeActor,
    write: Callable[[sqlite3.Connection], None],
    compatibility: SchemaCompatibility = CURRENT_SCHEMA_RANGE,
) -> bool:
    """Attempt a fatal-path write once; storage contention/failure returns ``False``."""
    connection: sqlite3.Connection | None = None
    try:
        connection = open_runtime_database(
            path,
            actor=actor,
            compatibility=compatibility,
            busy_policy=BusyPolicy.ZERO_WAIT,
        )
        with write_transaction(connection):
            write(connection)
    except (OSError, sqlite3.Error, EdgeDatabaseError):
        return False
    finally:
        if connection is not None:
            connection.close()
    return True


__all__ = [
    "NORMAL_BUSY_TIMEOUT_MS",
    "BusyPolicy",
    "NestedTransactionError",
    "RuntimeActor",
    "best_effort_zero_wait_write",
    "open_runtime_database",
    "write_transaction",
]
