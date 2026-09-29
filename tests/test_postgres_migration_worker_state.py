"""Prove the old worker is stopped from the lock files it leaves, never from their absence.

The worker's own ``GpuLease`` and ``DeliveryQueue`` build every state directory
here, so the names and lock modes are the ones a real worker uses. A new lease
that the worker cannot take, and a queue lock its sender cannot take, are the
oracle that the probe holds what it claims to hold.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.worker_state import (
    DEAD_LETTER_DIRECTORY_NAME,
    QUEUE_DIRECTORY_NAME,
    QUEUE_LOCK_NAME,
    WORKER_LEASE_NAME,
    QueueDigest,
    queue_digest,
    worker_stopped,
)
from shared.events.delivery_queue import DeliveryQueue
from worker.runtime.lease import GpuLease, GpuLeaseUnavailableError

LEASE_MISSING = "^the old worker's runtime lease is missing$"
LEASE_NOT_REGULAR = "^the old worker's runtime lease is not a regular file$"
QUEUE_LOCK_MISSING = "^the delivery queue lock is missing$"
QUEUE_LOCK_NOT_REGULAR = "^the delivery queue lock is not a regular file$"
QUEUE_SYMLINK = "^delivery queue directory is a symlink$"
QUEUE_NOT_DIRECTORY = "^delivery queue directory is not a directory$"
EMPTY_DIGEST = QueueDigest(
    queued=0, temporary=0, dead_lettered=0, sha256=hashlib.sha256(b"").hexdigest()
)


def _booted_worker(root: Path) -> tuple[Path, DeliveryQueue]:
    """A worker that booted, made its queue, and stopped: both locks released."""
    state = root / "worker-state"
    with GpuLease.acquire(state):
        queue = DeliveryQueue(state / QUEUE_DIRECTORY_NAME)
    return state, queue


def _lease_refused_worker(root: Path) -> Path:
    """A worker refused before its queue: the lease file exists, the queue does not."""
    state = root / "worker-state"
    with GpuLease.acquire(state):
        pass
    return state


def _probe(state: Path) -> None:
    with worker_stopped(state):
        pass


def _assert_lease_free(state: Path) -> None:
    with GpuLease.acquire(state):
        pass


def test_a_stopped_worker_passes_and_cannot_restart_while_the_probe_holds(
    tmp_path: Path,
) -> None:
    state, queue = _booted_worker(tmp_path)

    with worker_stopped(state):
        with pytest.raises(GpuLeaseUnavailableError):
            GpuLease.acquire(state)
        with queue._try_locked() as sender_locked:
            assert sender_locked is False

    with GpuLease.acquire(state), queue._try_locked() as sender_locked:
        assert sender_locked is True
    assert queue_digest(state) == EMPTY_DIGEST


def test_a_running_worker_holding_its_lease_is_refused(tmp_path: Path) -> None:
    state, _ = _booted_worker(tmp_path)

    with (
        GpuLease.acquire(state),
        pytest.raises(
            MigrationError, match="^the old worker holds its runtime lease; stop it first$"
        ),
    ):
        _probe(state)


def test_a_sender_holding_the_queue_lock_is_refused_and_the_lease_is_released(
    tmp_path: Path,
) -> None:
    state, queue = _booted_worker(tmp_path)

    with (
        queue._locked(),
        pytest.raises(MigrationError, match="^the delivery queue is locked by another process$"),
    ):
        _probe(state)

    _assert_lease_free(state)


def test_a_worker_refused_before_its_queue_existed_passes(tmp_path: Path) -> None:
    state = _lease_refused_worker(tmp_path)

    _probe(state)

    assert not (state / QUEUE_DIRECTORY_NAME).exists()
    assert queue_digest(state) == EMPTY_DIGEST


def _unlink_lease(state: Path) -> None:
    (state / WORKER_LEASE_NAME).unlink()


def _unlink_queue_lock(state: Path) -> None:
    (state / QUEUE_DIRECTORY_NAME / QUEUE_LOCK_NAME).unlink()


def _replace_with_directory(relative: str) -> Callable[[Path], None]:
    def replace(state: Path) -> None:
        (state / relative).unlink()
        (state / relative).mkdir()

    return replace


def _replace_with_fifo(relative: str) -> Callable[[Path], None]:
    def replace(state: Path) -> None:
        (state / relative).unlink()
        os.mkfifo(state / relative)

    return replace


def _replace_with_symlink_to_a_free_lock(relative: str) -> Callable[[Path], None]:
    def replace(state: Path) -> None:
        free = _lease_refused_worker(state.parent / "elsewhere") / WORKER_LEASE_NAME
        (state / relative).unlink()
        (state / relative).symlink_to(free)

    return replace


QUEUE_LOCK = f"{QUEUE_DIRECTORY_NAME}/{QUEUE_LOCK_NAME}"
LOCK_FILE_CASES = [
    pytest.param(_unlink_lease, LEASE_MISSING, id="lease-missing"),
    pytest.param(_unlink_queue_lock, QUEUE_LOCK_MISSING, id="queue-lock-missing"),
    pytest.param(_replace_with_directory(WORKER_LEASE_NAME), LEASE_NOT_REGULAR, id="lease-dir"),
    pytest.param(_replace_with_fifo(WORKER_LEASE_NAME), LEASE_NOT_REGULAR, id="lease-fifo"),
    pytest.param(
        _replace_with_symlink_to_a_free_lock(WORKER_LEASE_NAME),
        LEASE_NOT_REGULAR,
        id="lease-symlink",
    ),
    pytest.param(_replace_with_directory(QUEUE_LOCK), QUEUE_LOCK_NOT_REGULAR, id="queue-dir"),
    pytest.param(
        _replace_with_symlink_to_a_free_lock(QUEUE_LOCK),
        QUEUE_LOCK_NOT_REGULAR,
        id="queue-symlink",
    ),
]


@pytest.mark.parametrize(("arrange", "message"), LOCK_FILE_CASES)
def test_a_lock_file_that_proves_nothing_is_refused(
    tmp_path: Path, arrange: Callable[[Path], None], message: str
) -> None:
    state, _ = _booted_worker(tmp_path)
    arrange(state)

    with pytest.raises(MigrationError, match=message):
        _probe(state)
    with pytest.raises(MigrationError, match=message):
        queue_digest(state)

    if (state / WORKER_LEASE_NAME).is_file():
        _assert_lease_free(state)


def _symlink_directory(name: str) -> Callable[[Path], None]:
    def arrange(state: Path) -> None:
        real = state.parent / f"real-{name}"
        real.mkdir()
        (state / name).symlink_to(real, target_is_directory=True)

    return arrange


def _regular_file(name: str) -> Callable[[Path], None]:
    def arrange(state: Path) -> None:
        (state / name).write_bytes(b"")

    return arrange


@pytest.mark.parametrize(
    ("arrange", "message"),
    [
        pytest.param(_symlink_directory(QUEUE_DIRECTORY_NAME), QUEUE_SYMLINK, id="queue-symlink"),
        pytest.param(_regular_file(QUEUE_DIRECTORY_NAME), QUEUE_NOT_DIRECTORY, id="queue-file"),
    ],
)
def test_a_queue_directory_that_is_not_the_workers_is_refused(
    tmp_path: Path, arrange: Callable[[Path], None], message: str
) -> None:
    state = _lease_refused_worker(tmp_path)
    arrange(state)

    with pytest.raises(MigrationError, match=message):
        _probe(state)

    _assert_lease_free(state)


@pytest.mark.parametrize(
    ("arrange", "message"),
    [
        pytest.param(
            _symlink_directory(DEAD_LETTER_DIRECTORY_NAME), QUEUE_SYMLINK, id="dead-letter-symlink"
        ),
        pytest.param(
            _regular_file(DEAD_LETTER_DIRECTORY_NAME), QUEUE_NOT_DIRECTORY, id="dead-letter-file"
        ),
    ],
)
def test_a_dead_letter_directory_that_is_not_the_workers_fails_the_digest(
    tmp_path: Path, arrange: Callable[[Path], None], message: str
) -> None:
    state, _ = _booted_worker(tmp_path)
    arrange(state)

    _probe(state)
    with pytest.raises(MigrationError, match=message):
        queue_digest(state)
