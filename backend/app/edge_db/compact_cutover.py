"""Stopped-runtime schema-18 candidate cutover command."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from backend.app.edge_db.compact_cutover_files import (
    copy_exclusive,
    discard_sqlite_artifact,
    file_sha256,
    fsync_directory,
    fsync_file,
)
from backend.app.edge_db.compact_cutover_preflight import (
    checkpoint_and_preflight,
    require_paths,
    schema_version,
)
from backend.app.edge_db.compact_cutover_types import (
    CompactCutoverError,
    CompactCutoverRequest,
    CompactCutoverResult,
    CutoverPhase,
    CutoverProgress,
)
from backend.app.edge_db.compact_inventory import (
    filesystem_inventory_sha256,
    require_filesystem_drain,
)
from backend.app.edge_db.compact_projection import (
    project_compact_data,
    rebuilt_clip_ids,
    verify_manifest_projection,
)
from backend.app.edge_db.compact_receipt_verification import (
    ReceiptVerification,
    verify_candidate_contract,
    verify_receipts,
)
from backend.app.edge_db.compact_receipts import write_or_verify_receipts
from backend.app.edge_db.cutover_authorization import issue_compact_cutover_authorization
from backend.app.edge_db.migrator import deployment_lock, migrate_database
from backend.app.edge_db.sqlite_runtime import SqliteVersion, require_supported_sqlite


def _runtime_sqlite_version() -> tuple[int, int, int]:
    return sqlite3.sqlite_version_info[:3]


def _emit(progress: CutoverProgress | None, phase: CutoverPhase) -> None:
    if progress is not None:
        progress(phase)


def installed_marker(live: Path) -> Path:
    return live.with_name(f"{live.name}.v18-installed.sha256")


def _receipt_hash(request: CompactCutoverRequest, fallback: str) -> str:
    if request.receipt.exists():
        return file_sha256(request.receipt)
    return fallback


def _live_fingerprint(live: Path) -> str:
    digest = hashlib.sha256()
    for path in (live, Path(f"{live}-wal"), Path(f"{live}-shm")):
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def rollback_compact_cutover(request: CompactCutoverRequest) -> CompactCutoverResult:
    """Restore the read-only v17 archive only before the first post-cutover write."""
    live_version = schema_version(request.live)
    archive_hash = file_sha256(request.archive)
    if live_version == 17:
        return CompactCutoverResult(
            live=request.live,
            archive=request.archive,
            receipt=request.receipt,
            source_sha256=archive_hash,
            receipt_sha256=_receipt_hash(request, archive_hash),
            source_rows=0,
        )
    if live_version != 18:
        raise CompactCutoverError("EDGE_DB_CUTOVER_ROLLBACK_UNAVAILABLE")
    marker = installed_marker(request.live)
    installed = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    if not installed or _live_fingerprint(request.live) != installed:
        raise CompactCutoverError("EDGE_DB_CUTOVER_FORWARD_ONLY")
    restore = request.live.with_name(f"{request.live.name}.v17-restore")
    discard_sqlite_artifact(restore)
    copy_exclusive(request.archive, restore, mode=0o600)
    os.replace(restore, request.live)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{request.live}{suffix}")
        if sidecar.exists() or sidecar.is_symlink():
            sidecar.unlink()
    discard_sqlite_artifact(marker)
    return CompactCutoverResult(
        live=request.live,
        archive=request.archive,
        receipt=request.receipt,
        source_sha256=archive_hash,
        receipt_sha256=_receipt_hash(request, archive_hash),
        source_rows=0,
    )


def run_compact_cutover(
    request: CompactCutoverRequest,
    *,
    on_phase: CutoverProgress | None = None,
    sqlite_version: SqliteVersion | None = None,
) -> CompactCutoverResult:
    """Build, reconcile, and atomically install a schema-18 candidate."""
    require_supported_sqlite(
        _runtime_sqlite_version() if sqlite_version is None else sqlite_version
    )
    with deployment_lock(request.live.parent) as lock:
        require_paths(request)
        _emit(on_phase, CutoverPhase.BEFORE_CHECKPOINT)
        checkpoint_and_preflight(request.source)
        _emit(on_phase, CutoverPhase.AFTER_CHECKPOINT)
        try:
            require_filesystem_drain(request.clip_store, request.worker_state)
        except sqlite3.DatabaseError as error:
            raise CompactCutoverError("EDGE_DB_CUTOVER_FILESYSTEM_DRAIN") from error
        source_hash = file_sha256(request.source)
        if request.expected_source_sha256 not in {None, source_hash}:
            raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_CHANGED")
        live_version = schema_version(request.live)
        required_bytes = request.source.stat().st_size * 3
        if live_version == 17 and shutil.disk_usage(request.live.parent).free < required_bytes:
            raise CompactCutoverError("EDGE_DB_CUTOVER_INSUFFICIENT_SPACE")
        inventory_hash = filesystem_inventory_sha256(request.clip_store, request.worker_state)
        rebuilt = rebuilt_clip_ids(request.clip_store)
        if request.archive.exists():
            if file_sha256(request.archive) != source_hash:
                raise CompactCutoverError("EDGE_DB_CUTOVER_STALE_ARCHIVE")
            fsync_file(request.archive)
        else:
            copy_exclusive(
                request.source,
                request.archive,
                mode=0o400,
                on_written=lambda: _emit(on_phase, CutoverPhase.ARCHIVE_WRITTEN),
            )
            fsync_directory(request.live.parent)
        _emit(on_phase, CutoverPhase.ARCHIVE_SYNCED)
        try:
            source_rows, receipt_hash = write_or_verify_receipts(
                request.source,
                inventory_hash,
                request.receipt,
                rebuilt,
                on_written=lambda: _emit(on_phase, CutoverPhase.RECEIPT_WRITTEN),
            )
        except sqlite3.DatabaseError as error:
            raise CompactCutoverError("EDGE_DB_CUTOVER_STALE_RECEIPT") from error
        fsync_directory(request.live.parent)
        _emit(on_phase, CutoverPhase.RECEIPT_SYNCED)
        if live_version == 18:
            verify_receipts(
                ReceiptVerification(
                    request.source,
                    request.live,
                    request.receipt,
                    source_rows,
                    inventory_hash,
                    rebuilt,
                )
            )
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
        if live_version != 17 or file_sha256(request.live) != source_hash:
            raise CompactCutoverError("EDGE_DB_CUTOVER_LIVE_CHANGED")
        candidate_ready = False
        if request.candidate.exists():
            candidate_version = schema_version(request.candidate)
            if candidate_version == 18:
                try:
                    verify_receipts(
                        ReceiptVerification(
                            request.source,
                            request.candidate,
                            request.receipt,
                            source_rows,
                            inventory_hash,
                            rebuilt,
                        )
                    )
                    verify_manifest_projection(request.clip_store, request.candidate)
                    verify_candidate_contract(request.candidate, request.receipt)
                except (sqlite3.Error, ValueError):
                    discard_sqlite_artifact(request.candidate)
                else:
                    candidate_ready = True
            elif candidate_version == 17 and file_sha256(request.candidate) == source_hash:
                discard_sqlite_artifact(request.candidate)
            else:
                raise CompactCutoverError("EDGE_DB_CUTOVER_STALE_CANDIDATE")
        if not candidate_ready:
            copy_exclusive(
                request.archive,
                request.candidate,
                mode=0o600,
                on_written=lambda: _emit(on_phase, CutoverPhase.CANDIDATE_WRITTEN),
            )
            fsync_directory(request.live.parent)
            _emit(on_phase, CutoverPhase.CANDIDATE_SYNCED)
            _emit(on_phase, CutoverPhase.BEFORE_V18_TRANSACTION)
            authorization = issue_compact_cutover_authorization(
                lock,
                source=request.archive,
                candidate=request.candidate,
                reconciliation=request.receipt,
            )
            migrate_database(request.candidate, lock=lock, cutover=authorization)
            _emit(on_phase, CutoverPhase.V18_COMMITTED)
            project_compact_data(request.source, request.candidate, request.clip_store)
        _emit(on_phase, CutoverPhase.BEFORE_RECONCILIATION)
        verify_receipts(
            ReceiptVerification(
                request.source,
                request.candidate,
                request.receipt,
                source_rows,
                inventory_hash,
                rebuilt,
            )
        )
        verify_manifest_projection(request.clip_store, request.candidate)
        verify_candidate_contract(request.candidate, request.receipt)
        _emit(on_phase, CutoverPhase.RECONCILED)
        fsync_file(request.candidate)
        _emit(on_phase, CutoverPhase.CANDIDATE_FILE_SYNCED)
        _emit(on_phase, CutoverPhase.BEFORE_PRE_RENAME_DIRECTORY_SYNC)
        fsync_directory(request.live.parent)
        _emit(on_phase, CutoverPhase.PRE_RENAME_DIRECTORY_SYNCED)
        if (
            file_sha256(request.source) != source_hash
            or file_sha256(request.archive) != source_hash
        ):
            raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_CHANGED")
        if file_sha256(request.live) != source_hash:
            raise CompactCutoverError("EDGE_DB_CUTOVER_LIVE_CHANGED")
        verify_receipts(
            ReceiptVerification(
                request.source,
                request.candidate,
                request.receipt,
                source_rows,
                inventory_hash,
                rebuilt,
            )
        )
        verify_candidate_contract(request.candidate, request.receipt)
        os.replace(request.candidate, request.live)
        installed_marker(request.live).write_text(
            f"{_live_fingerprint(request.live)}\n", encoding="utf-8"
        )
        _emit(on_phase, CutoverPhase.RENAMED)
        fsync_directory(request.live.parent)
        _emit(on_phase, CutoverPhase.FINAL_DIRECTORY_SYNCED)
        _emit(on_phase, CutoverPhase.BEFORE_MANIFEST_VERIFY)
        verify_manifest_projection(request.clip_store, request.live)
        _emit(on_phase, CutoverPhase.MANIFEST_VERIFIED)
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
    "installed_marker",
    "main",
    "rollback_compact_cutover",
    "run_compact_cutover",
]
