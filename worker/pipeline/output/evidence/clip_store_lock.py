"""Lifetime advisory lock for the shared clip store."""

from __future__ import annotations

import errno
import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self, final

from typing_extensions import override


@dataclass(slots=True)
class ClipStoreLockedError(Exception):
    lock_path: Path

    @override
    def __str__(self) -> str:
        return f"clip store already has an active worker lock: {self.lock_path.name}"


@final
class ClipStoreLock:
    """Own one non-blocking flock until explicitly closed."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle: BinaryIO | None = handle

    @classmethod
    def acquire(cls, store_dir: Path) -> Self:
        store_dir.mkdir(parents=True, exist_ok=True)
        lock_path = store_dir / ".worker.lock"
        descriptor = os.open(
            lock_path,
            os.O_CLOEXEC | os.O_CREAT | os.O_RDWR,
            0o600,
        )
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ClipStoreLockedError(lock_path) from None
            raise
        return cls(handle)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


__all__ = ["ClipStoreLock", "ClipStoreLockedError"]
