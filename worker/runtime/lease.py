"""Repo-wide advisory GPU lease (todo 26).

One non-blocking ``flock`` under the resolved worker state directory
(``worker/runtime/state_dir.py``, ``~/.local/state/ml-worker``, no env
override) serializes GPU access across every process in this repository: the
worker itself plus the smoke/replay commands.  Two workers sharing one GPU is
the failure mode this removes — the
second entrant loses the lease and exits *before* it touches CUDA, NVDEC, or
constructs a model, rather than discovering the overlap as an out-of-memory or
context-corruption fault later.

The lease is *advisory*, in the POSIX ``flock`` sense: it binds cooperating
processes in this repository and is released automatically when a holder dies.
It is not a driver-level reservation and makes no claim about processes outside
this repository.  Acquire it before any accelerator construction and release it
on clean exit.

The lock file inode is deliberately never unlinked.  Deleting it would let a
second process create a fresh inode and take an independent lock on it while the
first holder still owns the old one — the classic advisory-lock defeat.  A
leftover unlocked file is harmless and is reused by the next process.
"""

from __future__ import annotations

import errno
import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final, Self, final

from typing_extensions import override

from worker.runtime.state_dir import resolve_state_dir as _default_state_dir

GPU_LEASE_FILENAME: Final = ".gpu.lease"


@dataclass(slots=True)
class GpuLeaseUnavailableError(RuntimeError):
    """Another process in this repository already holds the GPU lease."""

    lease_path: Path

    @override
    def __str__(self) -> str:
        return (
            f"GPU lease is already held by another process: {self.lease_path.name}; "
            "refusing to start before any CUDA/NVDEC/model construction"
        )


def resolve_state_dir(state_dir: Path | None = None) -> Path:
    """Resolve the lease directory, honouring an explicit override."""
    if state_dir is not None:
        return state_dir
    return _default_state_dir()


@final
class GpuLease:
    """Own one non-blocking advisory GPU lease until explicitly closed."""

    def __init__(self, handle: BinaryIO, lease_path: Path) -> None:
        self._handle: BinaryIO | None = handle
        self._lease_path = lease_path

    @property
    def lease_path(self) -> Path:
        return self._lease_path

    @property
    def held(self) -> bool:
        return self._handle is not None

    @classmethod
    def acquire(cls, state_dir: Path | None = None) -> Self:
        """Take the lease, or raise :class:`GpuLeaseUnavailableError` immediately.

        Never blocks: contention is a refuse-to-start condition, not something to
        wait out, because the competing holder owns the GPU for its lifetime.
        """
        directory = resolve_state_dir(state_dir)
        directory.mkdir(parents=True, exist_ok=True)
        lease_path = directory / GPU_LEASE_FILENAME
        descriptor = os.open(
            lease_path,
            os.O_CLOEXEC | os.O_CREAT | os.O_RDWR,
            0o600,
        )
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise GpuLeaseUnavailableError(lease_path) from None
            raise
        _record_owner(handle)
        return cls(handle, lease_path)

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
        """Release the lease.  Idempotent; safe to call on an already-closed lease."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _record_owner(handle: BinaryIO) -> None:
    """Stamp the holder PID into the lease file for operator diagnostics."""
    try:
        _ = handle.seek(0)
        handle.truncate(0)
        _ = handle.write(f"{os.getpid()}\n".encode())
        handle.flush()
    except OSError:
        # Diagnostics only.  The flock is the lease; a failed stamp must never
        # invalidate a lease we already hold.
        return


__all__ = [
    "GPU_LEASE_FILENAME",
    "GpuLease",
    "GpuLeaseUnavailableError",
    "resolve_state_dir",
]
