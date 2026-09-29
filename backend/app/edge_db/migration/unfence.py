"""Put the pre-fence SQLite bytes back once a rollback check has allowed it."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Final

from backend.app.edge_db.migration.authority_file import discard, fsync_directory
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.rollback import ALLOW, REPORT_FORMAT
from backend.app.edge_db.migration.snapshot import (
    create_private_file,
    exclusive_deployment_lock,
    sidecar_paths,
    snapshot_sha256,
    temporary_path,
)
from backend.app.edge_db.migration.sqlite_fence import (
    FenceReceipt,
    descriptor_blocks,
    inspect_fence,
    preserved_path,
    read_fence_receipt,
)

UNFENCE_REPORT_FORMAT: Final = "seeon-edge-sqlite-unfence/1"
RESTORED: Final = "RESTORED"


def unfence_sqlite(source: Path, *, receipt: Path, rollback_report: Path) -> dict[str, object]:
    """Replace the fenced file with the preserved copy, byte for byte.

    SQLite never opens either file: the fenced bytes must still be exactly the
    ones the receipt names, and the restored bytes exactly the ones it preserved.
    Neither the snapshot nor the preserved copy is removed.
    """
    fence = read_fence_receipt(receipt)
    if not fence.source_present:
        raise MigrationError(
            "fence receipt records no SQLite source; PostgreSQL stays authoritative"
        )
    _require_allowed(_read_rollback_report(rollback_report), fence)
    preserved = preserved_path(receipt)
    with exclusive_deployment_lock(source.parent):
        if not _restored(source, fence):
            _, reasons = inspect_fence(source, fence)
            if reasons:
                raise MigrationError(f"fenced source changed: {','.join(reasons)}")
            if (
                preserved.is_symlink()
                or not preserved.is_file()
                or snapshot_sha256(preserved) != fence.pre_fence_sha256
            ):
                raise MigrationError("preserved source copy does not match the fence receipt")
            _restore(preserved, source)
        fsync_directory(source.parent)
        if snapshot_sha256(source) != fence.pre_fence_sha256:
            raise MigrationError("restored source does not match the preserved copy")
    return {
        "format": UNFENCE_REPORT_FORMAT,
        "result": RESTORED,
        "generation": fence.generation,
        "snapshot_sha256": fence.snapshot_sha256,
        "fenced_sha256": fence.fenced_sha256,
        "restored_sha256": fence.pre_fence_sha256,
    }


def _read_rollback_report(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise MigrationError("rollback report is unreadable") from error
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or payload.get("format") != REPORT_FORMAT:
        raise MigrationError("rollback report is malformed")
    return payload


def _require_allowed(report: dict[str, object], fence: FenceReceipt) -> None:
    if report.get("result") != ALLOW or report.get("reasons") != []:
        raise MigrationError("rollback check did not ALLOW")
    checked = report.get("sqlite")
    if not isinstance(checked, dict):
        raise MigrationError("rollback report did not check the fenced source")
    snapshot = report.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("sha256") != fence.snapshot_sha256:
        raise MigrationError("rollback report checked a different snapshot")
    if (checked.get("generation"), checked.get("fenced_sha256")) != (
        fence.generation,
        fence.fenced_sha256,
    ):
        raise MigrationError("rollback report checked a different fence")


def _restored(source: Path, fence: FenceReceipt) -> bool:
    """A rerun after the replace landed: the preserved bytes are live and untouched."""
    wal, shm, journal = sidecar_paths(source)
    return (
        not source.is_symlink()
        and source.is_file()
        and not any(_has_content(path) for path in (wal, shm))
        and not journal.exists()
        and snapshot_sha256(source) == fence.pre_fence_sha256
    )


def _restore(preserved: Path, source: Path) -> None:
    fenced = source.lstat()
    temp = temporary_path(source)
    try:
        create_private_file(temp)
        descriptor = os.open(preserved, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with temp.open("r+b") as handle:
                for block in descriptor_blocks(descriptor):
                    handle.write(block)
                handle.flush()
                # The runtime opens the restored file as it opened the fenced one.
                if (fenced.st_uid, fenced.st_gid) != (os.getuid(), os.getgid()):
                    os.fchown(handle.fileno(), fenced.st_uid, fenced.st_gid)
                os.fchmod(handle.fileno(), stat.S_IMODE(fenced.st_mode))
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        if snapshot_sha256(temp) != snapshot_sha256(preserved):
            raise MigrationError("restored source does not match the preserved copy")
        os.replace(temp, source)
    finally:
        discard(temp)


def _has_content(path: Path) -> bool:
    try:
        return path.lstat().st_size > 0
    except FileNotFoundError:
        return False


__all__ = ["RESTORED", "UNFENCE_REPORT_FORMAT", "unfence_sqlite"]
