"""Inode-safe durable file operations for compact cutover."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from pathlib import Path

from backend.app.edge_db.compact_cutover_types import CompactCutoverError

_COPY_BLOCK = 1024 * 1024


def file_sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        while block := os.read(descriptor, _COPY_BLOCK):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def require_regular_single_link(path: Path, reason: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise CompactCutoverError(f"{reason}_MISSING") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise CompactCutoverError(f"{reason}_SYMLINK_OR_NOT_REGULAR")
    if metadata.st_nlink != 1:
        raise CompactCutoverError(f"{reason}_HARDLINK_LINK_COUNT")
    return metadata


def require_distinct_inodes(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    identities: set[tuple[int, int]] = set()
    for path in existing:
        metadata = require_regular_single_link(path, "EDGE_DB_CUTOVER_OUTPUT")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in identities:
            raise CompactCutoverError("EDGE_DB_CUTOVER_INODE_ALIAS")
        identities.add(identity)


def copy_exclusive(
    source: Path,
    destination: Path,
    *,
    mode: int,
    on_written: Callable[[], None] | None = None,
) -> os.stat_result:
    """Copy through no-follow descriptors and durably create one unaliased inode."""
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        source_before = os.fstat(source_fd)
        if source_before.st_nlink != 1 or not stat.S_ISREG(source_before.st_mode):
            raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_HARDLINK")
        destination_fd = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            mode,
        )
        try:
            while block := os.read(source_fd, _COPY_BLOCK):
                view = memoryview(block)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            if on_written is not None:
                on_written()
            os.fsync(destination_fd)
            destination_stat = os.fstat(destination_fd)
        finally:
            os.close(destination_fd)
        source_after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
    if (source_before.st_dev, source_before.st_ino) != (
        source_after.st_dev,
        source_after.st_ino,
    ) or source_before.st_size != source_after.st_size:
        raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_CHANGED")
    if destination_stat.st_nlink != 1:
        raise CompactCutoverError("EDGE_DB_CUTOVER_OUTPUT_HARDLINK")
    if (source_after.st_dev, source_after.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise CompactCutoverError("EDGE_DB_CUTOVER_INODE_ALIAS")
    return destination_stat


def discard_sqlite_artifact(path: Path) -> None:
    """Remove a rejected database and validated private WAL/SHM sidecars durably."""
    paths = (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
    existing = tuple(item for item in paths if item.exists() or item.is_symlink())
    for item in existing:
        require_regular_single_link(item, "EDGE_DB_CUTOVER_CANDIDATE_SIDECAR")
    for item in existing:
        os.unlink(item)
    fsync_directory(path.parent)


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "copy_exclusive",
    "discard_sqlite_artifact",
    "file_sha256",
    "fsync_directory",
    "fsync_file",
    "require_distinct_inodes",
    "require_regular_single_link",
]
