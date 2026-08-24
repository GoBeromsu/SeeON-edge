"""Stopped-runtime schema-18 candidate cutover command."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from backend.app.edge_db.compact_projection import (
    filesystem_inventory_sha256,
    project_compact_data,
    require_filesystem_drain,
    verify_manifest_projection,
)
from backend.app.edge_db.compact_receipt_verification import (
    verify_candidate_contract,
    verify_receipts,
)
from backend.app.edge_db.compact_receipts import write_or_verify_receipts
from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.cutover_authorization import issue_compact_cutover_authorization
from backend.app.edge_db.migrator import deployment_lock, migrate_database
from backend.app.edge_db.schema import MIGRATIONS
from backend.app.edge_db.sqlite_runtime import SqliteVersion, require_supported_sqlite


class CutoverPhase(StrEnum):
    PREFLIGHT = "preflight"
    ARCHIVED = "archived"
    RECEIPT_SYNCED = "receipt_synced"
    CANDIDATE_MIGRATED = "candidate_migrated"
    CANDIDATE_SYNCED = "candidate_synced"
    REPLACED = "replaced"


CutoverProgress = Callable[[CutoverPhase], None]


@dataclass(frozen=True, slots=True)
class CompactCutoverRequest:
    source: Path
    live: Path
    archive: Path
    candidate: Path
    receipt: Path
    clip_store: Path
    worker_state: Path
    expected_source_sha256: str | None = None
    sqlite_version: SqliteVersion | None = None


@dataclass(frozen=True, slots=True)
class CompactCutoverResult:
    live: Path
    archive: Path
    receipt: Path
    source_sha256: str
    receipt_sha256: str
    source_rows: int


class CompactCutoverError(EdgeDatabaseError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)

    def __str__(self) -> str:
        return self.reason


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _emit(progress: CutoverProgress | None, phase: CutoverPhase) -> None:
    if progress is not None:
        progress(phase)


def _require_regular(path: Path, *, reason: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise CompactCutoverError(f"{reason}_MISSING") from error
    if stat.S_ISLNK(mode):
        raise CompactCutoverError("EDGE_DB_CUTOVER_SYMLINK")
    if not stat.S_ISREG(mode):
        raise CompactCutoverError(f"{reason}_NOT_REGULAR")


def _require_paths(request: CompactCutoverRequest) -> None:
    _require_regular(request.source, reason="EDGE_DB_CUTOVER_SOURCE")
    _require_regular(request.live, reason="EDGE_DB_CUTOVER_LIVE")
    if request.source.samefile(request.live):
        raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_IS_LIVE")
    live_wal = Path(f"{request.live}-wal")
    if live_wal.exists() and live_wal.stat().st_size > 0:
        raise CompactCutoverError("EDGE_DB_CUTOVER_LIVE_SIDECAR")
    for directory in (request.clip_store, request.worker_state, request.live.parent):
        mode = directory.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise CompactCutoverError("EDGE_DB_CUTOVER_SYMLINK")
        if not stat.S_ISDIR(mode):
            raise CompactCutoverError("EDGE_DB_CUTOVER_DIRECTORY_INVALID")
    parent = request.live.parent.resolve()
    if request.source.parent.resolve() != parent:
        raise CompactCutoverError("EDGE_DB_CUTOVER_CROSS_FILESYSTEM")
    for output in (request.archive, request.candidate, request.receipt):
        if output.parent.resolve() != parent:
            raise CompactCutoverError("EDGE_DB_CUTOVER_CROSS_FILESYSTEM")
        if output.is_symlink():
            raise CompactCutoverError("EDGE_DB_CUTOVER_SYMLINK")
    devices = {
        request.source.stat().st_dev,
        request.live.stat().st_dev,
        request.clip_store.stat().st_dev,
        request.worker_state.stat().st_dev,
        parent.stat().st_dev,
    }
    if len(devices) != 1:
        raise CompactCutoverError("EDGE_DB_CUTOVER_CROSS_FILESYSTEM")
    if request.source.stat().st_uid != request.live.stat().st_uid:
        raise CompactCutoverError("EDGE_DB_CUTOVER_OWNERSHIP")
    if request.source.stat().st_uid != os.geteuid():
        raise CompactCutoverError("EDGE_DB_CUTOVER_OWNERSHIP")


def _schema_version(path: Path) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        row = connection.execute("PRAGMA user_version").fetchone()
    return -1 if row is None else int(row[0])


def _checkpoint_and_preflight(source: Path) -> None:
    connection = sqlite3.connect(source, timeout=5.0, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout=5000")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise CompactCutoverError("EDGE_DB_CUTOVER_CHECKPOINT_BUSY")
    finally:
        connection.close()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as readonly:
        version = readonly.execute("PRAGMA user_version").fetchone()
        if version != (17,):
            raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_VERSION")
        preflight = MIGRATIONS[17].preflight
        assert preflight is not None
        preflight(readonly)


def _copy_or_verify(source: Path, destination: Path, stale_reason: str) -> None:
    digest = _sha256(source)
    if destination.exists():
        _require_regular(destination, reason=stale_reason)
        if _sha256(destination) != digest:
            raise CompactCutoverError(stale_reason)
        return
    shutil.copyfile(source, destination)
    destination.chmod(0o400 if "ARCHIVE" in stale_reason else 0o600)
    _fsync_file(destination)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_compact_cutover(
    request: CompactCutoverRequest,
    *,
    on_phase: CutoverProgress | None = None,
) -> CompactCutoverResult:
    """Build, reconcile, and atomically install a schema-18 candidate."""
    require_supported_sqlite(request.sqlite_version or sqlite3.sqlite_version_info[:3])
    _require_paths(request)
    with deployment_lock(request.live.parent) as lock:
        _checkpoint_and_preflight(request.source)
        try:
            require_filesystem_drain(request.clip_store, request.worker_state)
        except sqlite3.DatabaseError as error:
            raise CompactCutoverError("EDGE_DB_CUTOVER_FILESYSTEM_DRAIN") from error
        source_hash = _sha256(request.source)
        if request.expected_source_sha256 not in {None, source_hash}:
            raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_CHANGED")
        live_version = _schema_version(request.live)
        required_bytes = request.source.stat().st_size * 3
        if live_version == 17 and shutil.disk_usage(request.live.parent).free < required_bytes:
            raise CompactCutoverError("EDGE_DB_CUTOVER_INSUFFICIENT_SPACE")
        inventory_hash = filesystem_inventory_sha256(request.clip_store, request.worker_state)
        _emit(on_phase, CutoverPhase.PREFLIGHT)
        _copy_or_verify(request.source, request.archive, "EDGE_DB_CUTOVER_STALE_ARCHIVE")
        _emit(on_phase, CutoverPhase.ARCHIVED)
        try:
            source_rows, receipt_hash = write_or_verify_receipts(
                request.source, inventory_hash, request.receipt
            )
        except sqlite3.DatabaseError as error:
            raise CompactCutoverError("EDGE_DB_CUTOVER_STALE_RECEIPT") from error
        _emit(on_phase, CutoverPhase.RECEIPT_SYNCED)
        if live_version == 18:
            verify_receipts(request.live, request.receipt, source_rows)
            verify_manifest_projection(request.clip_store, request.live)
            verify_candidate_contract(request.live, request.receipt)
            return CompactCutoverResult(
                live=request.live,
                archive=request.archive,
                receipt=request.receipt,
                source_sha256=source_hash,
                receipt_sha256=receipt_hash,
                source_rows=source_rows,
            )
        if live_version != 17 or _sha256(request.live) != source_hash:
            raise CompactCutoverError("EDGE_DB_CUTOVER_LIVE_CHANGED")
        candidate_version = _schema_version(request.candidate) if request.candidate.exists() else 17
        if not request.candidate.exists() or candidate_version == 17:
            _copy_or_verify(request.archive, request.candidate, "EDGE_DB_CUTOVER_STALE_CANDIDATE")
            authorization = issue_compact_cutover_authorization(
                lock,
                source=request.archive,
                candidate=request.candidate,
                reconciliation=request.receipt,
            )
            migrate_database(request.candidate, lock=lock, cutover=authorization)
            _emit(on_phase, CutoverPhase.CANDIDATE_MIGRATED)
            project_compact_data(request.source, request.candidate, request.clip_store)
        elif candidate_version != 18:
            raise CompactCutoverError("EDGE_DB_CUTOVER_STALE_CANDIDATE")
        verify_receipts(request.candidate, request.receipt, source_rows)
        verify_manifest_projection(request.clip_store, request.candidate)
        verify_candidate_contract(request.candidate, request.receipt)
        _fsync_file(request.candidate)
        _fsync_directory(request.live.parent)
        _emit(on_phase, CutoverPhase.CANDIDATE_SYNCED)
        if _sha256(request.source) != source_hash or _sha256(request.archive) != source_hash:
            raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_CHANGED")
        if _sha256(request.live) != source_hash:
            raise CompactCutoverError("EDGE_DB_CUTOVER_LIVE_CHANGED")
        os.replace(request.candidate, request.live)
        _fsync_directory(request.live.parent)
        _emit(on_phase, CutoverPhase.REPLACED)
    return CompactCutoverResult(
        live=request.live,
        archive=request.archive,
        receipt=request.receipt,
        source_sha256=source_hash,
        receipt_sha256=receipt_hash,
        source_rows=source_rows,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI boundary without loading argparse for library callers."""
    from backend.app.edge_db.compact_cutover_cli import main as command_main

    return command_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompactCutoverError",
    "CompactCutoverRequest",
    "CompactCutoverResult",
    "CutoverPhase",
    "main",
    "run_compact_cutover",
]
