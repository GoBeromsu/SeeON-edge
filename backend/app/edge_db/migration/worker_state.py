"""Old-worker liveness probe and delivery-queue digest for the cutover window."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backend.app.edge_db.migration.errors import MigrationError

# The old worker holds this lease exclusively for its whole lifetime.
WORKER_LEASE_NAME: Final = ".gpu.lease"
QUEUE_DIRECTORY_NAME: Final = "delivery-queue"
DEAD_LETTER_DIRECTORY_NAME: Final = "delivery-queue-dead-letter"
QUEUE_LOCK_NAME: Final = ".delivery-queue.lock"
_CHUNK: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class QueueDigest:
    queued: int
    temporary: int
    dead_lettered: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "queued": self.queued,
            "temporary": self.temporary,
            "dead_lettered": self.dead_lettered,
            "sha256": self.sha256,
        }


@contextmanager
def worker_stopped(state_directory: Path) -> Iterator[None]:
    """Hold the worker lease shared and the queue lock, proving no sender runs."""
    if state_directory.is_symlink() or not state_directory.is_dir():
        raise MigrationError("worker state directory does not exist")
    with ExitStack() as stack:
        lease = state_directory / WORKER_LEASE_NAME
        if not _try_lock(stack, lease, fcntl.LOCK_SH, "the old worker's runtime lease"):
            raise MigrationError("the old worker holds its runtime lease; stop it first")
        # A worker makes its queue only after it boots, so no queue means nothing was sent.
        queue = _queue_directory(state_directory, QUEUE_DIRECTORY_NAME)
        if queue is not None and not _try_lock(
            stack, queue / QUEUE_LOCK_NAME, fcntl.LOCK_EX, "the delivery queue lock"
        ):
            raise MigrationError("the delivery queue is locked by another process")
        yield


def queue_digest(state_directory: Path) -> QueueDigest:
    """Digest every retained queue file by relative name and content, never by payload."""
    with worker_stopped(state_directory):
        entries: list[tuple[str, str]] = []
        for name in (QUEUE_DIRECTORY_NAME, DEAD_LETTER_DIRECTORY_NAME):
            root = _queue_directory(state_directory, name)
            if root is not None:
                entries.extend(_walk(root, name))
    entries.sort()
    digest = hashlib.sha256()
    queued = temporary = dead_lettered = 0
    for relative, content in entries:
        digest.update(f"{relative}\0{content}\n".encode())
        directory, _, leaf = relative.partition("/")
        if directory == DEAD_LETTER_DIRECTORY_NAME:
            dead_lettered += 1
        elif leaf.startswith(".") and leaf.endswith(".tmp"):
            temporary += 1
        else:
            queued += 1
    return QueueDigest(
        queued=queued, temporary=temporary, dead_lettered=dead_lettered, sha256=digest.hexdigest()
    )


def _queue_directory(state_directory: Path, name: str) -> Path | None:
    """A queue directory the old worker made, or None when it never made one."""
    path = state_directory / name
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(mode):
        raise MigrationError("delivery queue directory is a symlink")
    if not stat.S_ISDIR(mode):
        raise MigrationError("delivery queue directory is not a directory")
    return path


def _walk(root: Path, prefix: str) -> Iterator[tuple[str, str]]:
    with os.scandir(root) as iterator:
        children = sorted(iterator, key=lambda entry: entry.name)
    for entry in children:
        relative = f"{prefix}/{entry.name}"
        mode = entry.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            yield from _walk(Path(entry.path), relative)
        elif not stat.S_ISREG(mode):
            raise MigrationError("delivery queue holds a non-regular entry")
        elif relative != f"{QUEUE_DIRECTORY_NAME}/{QUEUE_LOCK_NAME}":
            yield relative, _file_sha256(Path(entry.path))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _try_lock(stack: ExitStack, path: Path, operation: int, name: str) -> bool:
    """Lock a file the old worker left without creating it; a missing one proves nothing."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise MigrationError(f"{name} is missing") from None
    except OSError as error:
        if error.errno != errno.ELOOP:
            raise
        raise MigrationError(f"{name} is not a regular file") from None
    stack.callback(os.close, descriptor)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise MigrationError(f"{name} is not a regular file")
    try:
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    stack.callback(fcntl.flock, descriptor, fcntl.LOCK_UN)
    return True


__all__ = [
    "DEAD_LETTER_DIRECTORY_NAME",
    "QUEUE_DIRECTORY_NAME",
    "QUEUE_LOCK_NAME",
    "WORKER_LEASE_NAME",
    "QueueDigest",
    "queue_digest",
    "worker_stopped",
]
