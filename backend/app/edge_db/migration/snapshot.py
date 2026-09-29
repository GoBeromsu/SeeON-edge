"""Consistent read-only snapshot of the fenced schema-19 SQLite database."""

from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backend.app.edge_db.migration.authority_file import discard, fsync_directory
from backend.app.edge_db.migration.compatibility import verify_runtime_schema
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.mapping import verify_source_tables
from backend.app.edge_db.migration.sqlite_functions import register_edge_db_functions

# The old runtime holds this lock shared while any connection is open.
DEPLOYMENT_LOCK_NAME: Final = "deployment.lock"
_CHUNK: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    sha256: str
    schema_version: int


def export_snapshot(source: Path, destination: Path) -> Snapshot:
    """Copy the source through the backup API; the source is opened read-only."""
    if destination.exists() or destination.is_symlink():
        raise MigrationError("snapshot destination already exists")
    if not destination.parent.is_dir():
        raise MigrationError("snapshot directory does not exist")
    temp = temporary_path(destination)
    try:
        # Owner-only: it holds resident data.
        create_private_file(temp)
        with open_fenced_source(source) as origin:
            copy_database(origin, temp)
        fsync_file(temp)
        snapshot = _verify(temp)
        try:
            os.link(temp, destination)
        except FileExistsError as error:
            raise MigrationError("snapshot destination already exists") from error
        temp.unlink()
        fsync_directory(destination.parent)
    finally:
        discard_database(temp)
    return Snapshot(
        path=destination, sha256=snapshot.sha256, schema_version=snapshot.schema_version
    )


def temporary_path(path: Path) -> Path:
    """A unique hidden sibling, so a finished file is published by link or rename."""
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


def create_private_file(path: Path) -> None:
    """Create an empty owner-only file; SQLite gives its -wal and -shm the same mode."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def copy_database(origin: sqlite3.Connection, path: Path) -> None:
    """Copy through the backup API into an empty file, left checkpointed in WAL mode."""
    with closing(sqlite3.connect(path)) as copy:
        origin.backup(copy, progress=_refuse_busy)
        copy.execute("PRAGMA journal_mode = WAL").fetchone()
        copy.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()


def sidecar_paths(path: Path) -> tuple[Path, Path, Path]:
    """The -wal, -shm and -journal files SQLite keeps beside a database."""
    wal, shm, journal = (path.with_name(f"{path.name}-{end}") for end in ("wal", "shm", "journal"))
    return wal, shm, journal


def discard_database(path: Path) -> None:
    for member in (path, *sidecar_paths(path)):
        discard(member)


@contextmanager
def open_fenced_source(source: Path) -> Iterator[sqlite3.Connection]:
    """Open the live source read-only while no runtime connection can exist."""
    if source.is_symlink() or not source.is_file():
        raise MigrationError("source database is not a regular file")
    with exclusive_deployment_lock(source.parent):
        connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
        with closing(connection):
            register_edge_db_functions(connection)
            verify_runtime_schema(connection)
            verify_source_tables(connection)
            yield connection


def open_snapshot(path: Path) -> sqlite3.Connection:
    """Open a snapshot immutable and read-only, with the edge SQL functions."""
    if path.is_symlink() or not path.is_file():
        raise MigrationError("snapshot is not a regular file")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        register_edge_db_functions(connection)
        verify_runtime_schema(connection)
        verify_source_tables(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path) -> Snapshot:
    with closing(open_snapshot(path)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if [tuple(row) for row in integrity] != [("ok",)]:
            raise MigrationError("snapshot failed the integrity check")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise MigrationError("snapshot has foreign-key violations")
        version = verify_runtime_schema(connection)
    return Snapshot(path=path, sha256=snapshot_sha256(path), schema_version=version)


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_deployment_lock(state_directory: Path) -> Iterator[None]:
    """Exclude an old runtime that takes the lock; one that ignores it is not excluded."""
    descriptor = os.open(state_directory / DEPLOYMENT_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MigrationError("source database is in use by a running runtime") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _refuse_busy(status: int, remaining: int, total: int) -> None:
    # CPython retries a busy or locked step forever; a held source must fail instead.
    if status not in (sqlite3.SQLITE_OK, sqlite3.SQLITE_DONE):
        raise MigrationError("source database is locked")


__all__ = [
    "DEPLOYMENT_LOCK_NAME",
    "Snapshot",
    "copy_database",
    "create_private_file",
    "discard_database",
    "exclusive_deployment_lock",
    "export_snapshot",
    "fsync_file",
    "open_fenced_source",
    "open_snapshot",
    "sidecar_paths",
    "snapshot_sha256",
    "temporary_path",
]
