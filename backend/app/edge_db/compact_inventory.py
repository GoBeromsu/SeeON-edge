"""Read-only filesystem inventory and drain gate."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filesystem_inventory_sha256(clip_store: Path, worker_state: Path) -> str:
    digest = hashlib.sha256()
    roots = (
        clip_store,
        worker_state / "delivery-queue",
        worker_state / "delivery-queue-dead-letter",
    )
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise sqlite3.DatabaseError(f"unsafe cutover inventory root: {root}")
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            metadata = path.lstat()
            if path.is_symlink() or metadata.st_nlink != 1:
                raise sqlite3.DatabaseError(f"unsafe cutover inventory file: {path}")
            relative = f"{root.name}/{path.relative_to(root).as_posix()}".encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(metadata.st_size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def require_filesystem_drain(clip_store: Path, worker_state: Path) -> None:
    pending: list[Path] = []
    for directory in (
        worker_state / "delivery-queue",
        worker_state / "delivery-queue-dead-letter",
    ):
        if directory.exists():
            pending.extend(path for path in directory.rglob("*") if path.is_file())
    for directory in (clip_store / "clips" / ".staging", clip_store / ".snapshot-staging"):
        if directory.exists():
            pending.extend(path for path in directory.rglob("*") if path.is_file() or path.is_dir())
    if pending:
        raise sqlite3.DatabaseError("EDGE_DB_FILESYSTEM_DRAIN_INCOMPLETE")


__all__ = ["filesystem_inventory_sha256", "require_filesystem_drain"]
