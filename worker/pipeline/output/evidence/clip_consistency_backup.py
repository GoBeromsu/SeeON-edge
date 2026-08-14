"""Gap-free SQLite online backup and strict prebackup receipt handling."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_backup_receipt import (
    RECEIPT_VERSION,
    database_state_sha256,
    parse_backup_receipt,
    source_identity,
    verify_backup,
)
from worker.pipeline.output.evidence.clip_consistency_io import (
    FaultHook,
    atomic_write_json,
    checkpoint,
    ensure_secure_subdirectory,
    fsync_directory,
    sha256_regular,
    validate_under_root,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    BackupReceipt,
    ClipConsistencyError,
)


def create_verified_backup(
    source: Path,
    snapshot: sqlite3.Connection,
    *,
    maintenance_root: Path,
    expected_uid: int,
    schema_version: int,
    fault_hook: FaultHook | None = None,
) -> BackupReceipt:
    backup_root = ensure_secure_subdirectory(
        maintenance_root,
        "backups",
        expected_uid=expected_uid,
    )
    identity = source_identity(source, snapshot, expected_uid)
    stem = datetime.now(UTC).strftime("worker-state-%Y%m%dT%H%M%S.%fZ")
    backup_path = backup_root / f"{stem}.sqlite3"
    receipt_path = backup_root / f"{stem}.receipt.json"
    destination: sqlite3.Connection | None = None
    backup_source: sqlite3.Connection | None = None
    try:
        descriptor = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(backup_path, 0o600)
        destination = sqlite3.connect(backup_path)
        backup_source = sqlite3.connect(
            f"file:{source}?mode=ro",
            uri=True,
            isolation_level=None,
        )
        backup_source.execute("BEGIN")
        backup_source.backup(destination)
        backup_source.rollback()
        backup_source.close()
        backup_source = None
        destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        destination.commit()
        destination.close()
        destination = None
        _fsync_backup(backup_path, backup_root, fault_hook)
        backup_size = backup_path.stat().st_size
        backup_file_sha256 = sha256_regular(backup_path)
        backup_state_sha256 = database_state_sha256(backup_path)
        _require_equal_state(backup_state_sha256, identity.source_state_sha256)
        _require_unchanged_source(
            identity,
            source_identity(source, snapshot, expected_uid),
        )
        receipt = BackupReceipt(
            format_version=RECEIPT_VERSION,
            schema_version=schema_version,
            owner_uid=expected_uid,
            source_path=identity.source_path,
            source_mode=identity.source_mode,
            source_size=identity.source_size,
            source_file_sha256=identity.source_file_sha256,
            source_wal_path=identity.source_wal_path,
            source_wal_present=identity.source_wal_present,
            source_wal_size=identity.source_wal_size,
            source_wal_sha256=identity.source_wal_sha256,
            source_state_sha256=identity.source_state_sha256,
            source_identity_sha256=identity.source_identity_sha256,
            backup_path=str(backup_path.resolve(strict=True)),
            backup_mode=0o600,
            backup_size=backup_size,
            backup_file_sha256=backup_file_sha256,
            backup_state_sha256=backup_state_sha256,
            receipt_path=str(receipt_path.resolve(strict=False)),
        )
        atomic_write_json(
            receipt_path,
            receipt.to_dict(),
            root=maintenance_root,
            expected_uid=expected_uid,
            hook=fault_hook,
            stage="backup_receipt",
        )
    except BaseException:
        if backup_source is not None:
            backup_source.close()
        if destination is not None:
            destination.close()
        receipt_path.unlink(missing_ok=True)
        backup_path.unlink(missing_ok=True)
        fsync_directory(backup_root)
        raise
    return receipt


def ensure_prebackup(
    source: Path,
    snapshot: sqlite3.Connection,
    *,
    receipt_path: Path | None,
    maintenance_root: Path,
    expected_uid: int,
    schema_version: int,
    hook: FaultHook | None = None,
) -> BackupReceipt:
    if receipt_path is None:
        return create_verified_backup(
            source,
            snapshot,
            maintenance_root=maintenance_root,
            expected_uid=expected_uid,
            schema_version=schema_version,
            fault_hook=hook,
        )
    receipt = verify_backup_receipt_for_resume(
        receipt_path,
        maintenance_root=maintenance_root,
        expected_uid=expected_uid,
    )
    current = source_identity(source, snapshot, expected_uid)
    if not current.matches_reusable_source(receipt):
        raise ClipConsistencyError("backup_receipt_stale", "source identity differs")
    return receipt


def verify_backup_receipt_for_resume(
    path: Path,
    *,
    maintenance_root: Path,
    expected_uid: int,
) -> BackupReceipt:
    receipt = parse_backup_receipt(
        path,
        maintenance_root=maintenance_root,
        expected_uid=expected_uid,
    )
    validate_under_root(
        Path(receipt.backup_path),
        maintenance_root,
        allow_missing_leaf=False,
    )
    verify_backup(receipt, expected_uid)
    return receipt


def _fsync_backup(path: Path, root: Path, hook: FaultHook | None) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    checkpoint(hook, "backup:file_fsynced")
    fsync_directory(root)
    checkpoint(hook, "backup:directory_fsynced")


def _require_equal_state(backup_sha256: str, source_sha256: str) -> None:
    if backup_sha256 != source_sha256:
        raise ClipConsistencyError("backup_verification_failed", "snapshot state differs")


def _require_unchanged_source(before: object, after: object) -> None:
    if before != after:
        raise ClipConsistencyError("source_changed", "database changed during backup")


__all__ = [
    "create_verified_backup",
    "ensure_prebackup",
    "verify_backup_receipt_for_resume",
]
