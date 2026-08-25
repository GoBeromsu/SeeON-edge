"""Stopped-runtime schema-18 candidate cutover command."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db.compatibility import EdgeDatabaseError, verify_runtime_schema
from backend.app.edge_db.cutover_authorization import issue_compact_cutover_authorization
from backend.app.edge_db.inventory import check_filesystem_drain
from backend.app.edge_db.migrator import deployment_lock, migrate_database
from backend.app.edge_db.sqlite_runtime import (
    SqliteRuntimeError,
    SqliteVersion,
    require_supported_sqlite,
)
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION

DRAIN_SQL = """
SELECT EXISTS(SELECT 1 FROM evidence_events WHERE state IN ('STAGED','READY','IN_FLIGHT'))
    OR EXISTS(SELECT 1 FROM evidence_clips WHERE local_state = 'AWAITING_FINALIZE')
    OR EXISTS(SELECT 1 FROM evidence_clips WHERE publish_state = 'IN_FLIGHT')
    OR EXISTS(SELECT 1 FROM derivative_jobs WHERE state IN ('PENDING', 'RUNNING'))
    OR EXISTS(SELECT 1 FROM derivative_evidence_slots WHERE state = 'PENDING')
    OR EXISTS(SELECT 1 FROM evidence_retention_states WHERE state = 'PENDING')
"""
AUTHORITY_TABLES = frozenset(
    {
        "audit",
        "audit_events",
        "camera_registry",
        "cameras",
        "clips",
        "connection_settings",
        "control_detection_policy_state",
        "credentials",
        "detection_settings",
        "edge_site",
        "events",
        "evidence_clips",
        "evidence_events",
        "evidence_incidents",
        "locations",
        "policies",
    }
)
COMPACT_WRITE_TABLES = frozenset(
    {
        "artifacts",
        "audit_events",
        "cameras",
        "clips",
        "credentials",
        "edge_site",
        "incidents",
        "locations",
        "policies",
    }
)


class CompactCutoverError(EdgeDatabaseError):
    """A schema-18 cutover refused before live replacement."""


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


@dataclass(frozen=True, slots=True)
class CompactCutoverResult:
    live: Path
    archive: Path
    current_version: int
    source_sha256: str | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _user_version(path: Path) -> int:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = connection.execute("PRAGMA user_version").fetchone()
    finally:
        connection.close()
    return 0 if row is None else int(row[0])


def _copy_exclusive(source: Path, dest: Path, mode: int) -> None:
    if dest.exists() or dest.is_symlink():
        raise CompactCutoverError("EDGE_DB_CUTOVER_TARGET_EXISTS")
    descriptor = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as incoming:
            while True:
                chunk = incoming.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    os.chmod(dest, mode)


def _checkpoint(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
    finally:
        connection.close()


def _count_named_tables(path: Path, names: frozenset[str]) -> int:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        existing = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        total = 0
        for table in sorted(names & existing):
            total += int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    finally:
        connection.close()
    return total


def _require_drained(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        blocked = connection.execute(DRAIN_SQL).fetchone()
    finally:
        connection.close()
    if blocked == (1,):
        raise CompactCutoverError("EDGE_DB_DRAIN_INCOMPLETE")


def _require_same_filesystem(request: CompactCutoverRequest) -> None:
    if any(path.is_symlink() for path in (request.live, request.archive, request.candidate)):
        raise CompactCutoverError("EDGE_DB_CUTOVER_SYMLINK")
    live_dev = request.live.parent.resolve().stat().st_dev
    candidate_dev = request.candidate.parent.resolve().stat().st_dev
    if live_dev != candidate_dev:
        raise CompactCutoverError("EDGE_DB_CUTOVER_CROSS_FILESYSTEM")


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.write_text(encoded, encoding="utf-8")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _restore_archive(archive: Path, live: Path) -> None:
    restore = live.with_name(f"{live.name}.v17-restore")
    if restore.exists() or restore.is_symlink():
        restore.unlink()
    _copy_exclusive(archive, restore, 0o600)
    os.replace(restore, live)


def _rollback(request: CompactCutoverRequest) -> CompactCutoverResult:
    if not request.live.is_file() or not request.archive.is_file():
        raise CompactCutoverError("EDGE_DB_CUTOVER_ROLLBACK_UNAVAILABLE")
    live_version = _user_version(request.live)
    if live_version == 17:
        return CompactCutoverResult(request.live, request.archive, 17, _sha256(request.archive))
    if live_version != 18:
        raise CompactCutoverError("EDGE_DB_CUTOVER_ROLLBACK_UNAVAILABLE")
    if _count_named_tables(request.live, COMPACT_WRITE_TABLES):
        raise CompactCutoverError("EDGE_DB_CUTOVER_FORWARD_ONLY")
    _restore_archive(request.archive, request.live)
    return CompactCutoverResult(request.live, request.archive, 17, _sha256(request.archive))


def run_compact_cutover(
    request: CompactCutoverRequest,
    *,
    rollback: bool = False,
    sqlite_version: SqliteVersion | None = None,
) -> CompactCutoverResult:
    """Build, validate, and atomically install a schema-18 candidate."""
    try:
        require_supported_sqlite(
            sqlite3.sqlite_version_info[:3] if sqlite_version is None else sqlite_version
        )
    except SqliteRuntimeError as error:
        raise CompactCutoverError(str(error)) from error
    if rollback:
        return _rollback(request)
    if not request.live.exists() and not request.source.exists():
        migrate_database(request.live)
        return CompactCutoverResult(request.live, request.archive, 18, None)
    source = request.source if request.source.exists() else request.live
    _require_same_filesystem(request)
    passed, message = check_filesystem_drain(source, request.worker_state, request.clip_store)
    if not passed:
        raise CompactCutoverError(message)
    live_version = _user_version(request.live)
    if live_version == EDGE_DATABASE_SCHEMA_VERSION:
        connection = sqlite3.connect(f"file:{request.live}?mode=ro", uri=True)
        try:
            verify_runtime_schema(connection)
        finally:
            connection.close()
        return CompactCutoverResult(request.live, request.archive, 18, _sha256(request.live))
    if live_version != 17:
        raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_INVALID")
    _checkpoint(source)
    _require_drained(source)
    if _count_named_tables(source, AUTHORITY_TABLES):
        raise CompactCutoverError("EDGE_DB_CUTOVER_PROJECTION_REQUIRED")
    source_hash = _sha256(source)
    if request.expected_source_sha256 not in {None, source_hash}:
        raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_CHANGED")
    if request.archive.exists():
        if _sha256(request.archive) != source_hash:
            raise CompactCutoverError("EDGE_DB_CUTOVER_STALE_ARCHIVE")
    else:
        _copy_exclusive(source, request.archive, 0o400)
    if request.candidate.exists():
        request.candidate.unlink()
    _copy_exclusive(request.archive, request.candidate, 0o600)
    with deployment_lock(request.candidate.parent) as lock:
        authorization = issue_compact_cutover_authorization(
            lock,
            source=request.archive,
            candidate=request.candidate,
            reconciliation=source_hash.encode("ascii"),
        )
        migrate_database(request.candidate, lock=lock, cutover=authorization)
    if _sha256(source) != source_hash or _sha256(request.archive) != source_hash:
        raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_CHANGED")
    os.replace(request.candidate, request.live)
    _write_receipt(
        request.receipt,
        {
            "action": "CUTOVER",
            "archive_sha256": source_hash,
            "source_schema_version": 17,
            "target_schema_version": 18,
        },
    )
    return CompactCutoverResult(request.live, request.archive, 18, source_hash)


def main(argv: list[str] | None = None) -> int:
    from backend.app.edge_db.compact_cutover_cli import main as command_main

    return command_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompactCutoverError",
    "CompactCutoverRequest",
    "CompactCutoverResult",
    "main",
    "run_compact_cutover",
]
