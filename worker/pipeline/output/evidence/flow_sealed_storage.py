"""Bounded, publish-once file primitives for Flow sealed-clip sidecars."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import Final
from uuid import uuid4

from worker.pipeline.output.evidence.durability import fsync_directory

MAX_SIDECAR_BYTES: Final = 1 << 20
_READ_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_CREATE_FLAGS: Final = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_MODE: Final = 0o644


class FlowSealedConflictError(RuntimeError):
    """A committed sidecar holds different bytes; it is preserved and refused."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"sealed Flow sidecar contradicts committed evidence path={path}")
        self.path = path


class FlowSealedUnreadableError(RuntimeError):
    """A sidecar is a symlink, not regular, empty, or over the read bound."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"sealed Flow sidecar is not a bounded regular file path={path}")
        self.path = path


def read_bounded(path: Path, *, sync: bool = False) -> bytes | None:
    """Read one committed sidecar without following a final symlink.

    Returns ``None`` only when the name is absent. Shape violations raise
    :class:`FlowSealedUnreadableError`; any other I/O failure propagates.
    """
    try:
        descriptor = os.open(path, _READ_FLAGS)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise FlowSealedUnreadableError(path) from exc
        raise
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_SIDECAR_BYTES:
            raise FlowSealedUnreadableError(path)
        chunks: list[bytes] = []
        remaining = MAX_SIDECAR_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if not data or len(data) > MAX_SIDECAR_BYTES:
            raise FlowSealedUnreadableError(path)
        if sync:
            os.fsync(descriptor)
        return data
    finally:
        os.close(descriptor)


def publish_once(target: Path, payload: bytes) -> None:
    """Atomically commit ``payload`` at ``target`` without replacing any entry.

    Identical committed bytes are accepted and made durable again; different
    or unreadable committed evidence is preserved and refused.
    """
    if not 0 < len(payload) <= MAX_SIDECAR_BYTES:
        raise ValueError("sealed Flow sidecar payload is outside the size bound")
    if _reconcile(target, payload):
        return
    directory = target.parent
    temporary = directory / f".{target.name}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, _CREATE_FLAGS, _FILE_MODE)
    linked = False
    try:
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(descriptor, view) :]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, target, follow_symlinks=False)
            linked = True
        except FileExistsError:
            linked = False
        fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
        fsync_directory(directory)
    if not linked and not _reconcile(target, payload):
        raise FileExistsError(errno.EEXIST, "sealed Flow sidecar changed during publication")


def remove_durable(path: Path) -> None:
    path.unlink(missing_ok=True)
    fsync_directory(path.parent)


def sidecar_names(directory: Path) -> tuple[str, ...]:
    """Committed sidecar names in code-point order; temporaries are hidden."""
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return ()
    return tuple(sorted(name for name in names if name.endswith(".json") and name[0] != "."))


def _reconcile(target: Path, payload: bytes) -> bool:
    existing = read_bounded(target, sync=True)
    if existing is None:
        return False
    if existing != payload:
        raise FlowSealedConflictError(target)
    fsync_directory(target.parent)
    return True


__all__ = [
    "MAX_SIDECAR_BYTES",
    "FlowSealedConflictError",
    "FlowSealedUnreadableError",
    "publish_once",
    "read_bounded",
    "remove_durable",
    "sidecar_names",
]
