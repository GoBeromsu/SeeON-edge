"""First-fault record persistence (todo 25).

Writes exactly one record to the ``faults`` table on the first
FatalAcceleratorError. Production resolves the central ``edge.sqlite3`` path
(``EDGE_DATABASE_PATH``); legacy tests and alternate state dirs still use
``worker-state.sqlite3`` under the supplied state directory. Subsequent calls
within the same process are no-ops (the module-level ``_written`` flag).
Because the write is a fixed-id upsert (``faults.id = 1``), a later process's
crash overwrites whatever the table already holds -- the same "most recent
first fault wins" behavior the prior first_fault.json's unconditional
tmp-then-rename already had, not a permanent cross-restart log.

**Never blocks or delays process exit.** This is the hard constraint the
JSON-file predecessor already honored (best-effort write, swallow all
exceptions) and the SQLite replacement must honor just as strictly, including
under lock contention: the evidence outbox writer and other worker-owned
writers share the same database file and may hold its write lock when a fault
fires. Reusing the normal 5-second busy timeout would make the fault path
*wait up to 5 seconds* before degrading -- a real delay to a hard-exit
boundary that a watchdog-driven trip may itself be racing against a deadline
for.

Production ``edge.sqlite3`` therefore goes through
``best_effort_zero_wait_write`` / ``BusyPolicy.ZERO_WAIT`` so a conflicting
writer surfaces immediately with no wait. Legacy ``worker-state.sqlite3``
opens with ``busy_timeout_ms=0`` for the same fail-fast contract. Any storage
failure is caught and logged as a warning; the caller always proceeds to exit
regardless of whether the record was written.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from shared.edge_db import EDGE_DATABASE_PATH
from shared.edge_db.connection import RuntimeActor, best_effort_zero_wait_write
from worker.pipeline.output.evidence.evidence_outbox_database import open_connection
from worker.pipeline.output.evidence.evidence_outbox_types import NewerSchemaVersionError
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction
from worker.runtime.state_dir import resolve_state_dir

LOGGER: Final = logging.getLogger(__name__)

WORKER_STATE_DB_FILENAME: Final = "worker-state.sqlite3"
_EDGE_DATABASE_FILENAME: Final = "edge.sqlite3"

# Fault writes must never wait out a lock held by a concurrent writer: a zero
# busy-timeout makes a conflicting writer surface as an immediate
# sqlite3.OperationalError instead of blocking the hard-exit path for up to
# open_connection's / open_runtime_database's normal 5s.
_FAULT_WRITE_BUSY_TIMEOUT_MS: Final = 0

# Same failure modes WorkerConfigLkgStore (lkg_store.py's
# _STORE_UNAVAILABLE_ERRORS) treats as "storage unavailable, degrade
# silently rather than raise": OSError, sqlite3.Error (this is where the
# zero-timeout SQLITE_BUSY above surfaces), and NewerSchemaVersionError.
_STORE_UNAVAILABLE_ERRORS: Final = (OSError, sqlite3.Error, NewerSchemaVersionError)

# Bound on the stored exception message so one pathological fault message
# cannot bloat the row; mirrors the JSON predecessor's MAX_RECORD_BYTES
# truncation intent at a per-field granularity that fits typed columns.
MAX_EXCEPTION_MESSAGE_CHARS: Final = 4_096

_write_lock = threading.Lock()
_written = False


@dataclass(slots=True)
class FirstFaultRecord:
    """Immutable snapshot of the process state at the moment of the first CUDA fault."""

    # Process/boot identity
    pid: int
    boot_time_iso: str
    profile: str

    # Fault location
    task: str
    stage: str
    camera_id: str

    # Frame context (all optional — may be unknown at the time of the fault)
    frame_index: int | None
    pts: float | None
    frame_shape: tuple[int, ...] | None
    frame_hash_sha256: str | None

    # Model identity
    model_artifact_digest: str | None
    invocation_seq: int

    # Exception details
    exception_type: str
    exception_message: str
    exit_code: int
    action: str = "exit_process_retire_context"

    # Timestamp
    fault_time_iso: str = field(default_factory=lambda: _iso_now())


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _frame_hash(image: object) -> str | None:
    """SHA-256 of the raw image bytes, or None if the image is not a numpy array."""
    try:
        import hashlib

        import numpy as np

        if isinstance(image, np.ndarray):
            return hashlib.sha256(image.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 S110
        pass
    return None


def _truncate_message(message: str) -> str:
    if len(message) <= MAX_EXCEPTION_MESSAGE_CHARS:
        return message
    return message[:MAX_EXCEPTION_MESSAGE_CHARS] + "...[truncated]"


def _worker_state_db_path(state_dir: Path) -> Path:
    return (
        EDGE_DATABASE_PATH
        if state_dir == resolve_state_dir()
        else state_dir / WORKER_STATE_DB_FILENAME
    )


def _upsert_fault_row(connection: sqlite3.Connection, record: FirstFaultRecord) -> None:
    connection.execute(
        """
        INSERT INTO faults (
            id, pid, boot_time_iso, profile, task, stage, camera_id,
            frame_index, pts, frame_shape_json, frame_hash_sha256,
            model_artifact_digest, invocation_seq, exception_type,
            exception_message, exit_code, action, fault_time_iso
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            pid = excluded.pid,
            boot_time_iso = excluded.boot_time_iso,
            profile = excluded.profile,
            task = excluded.task,
            stage = excluded.stage,
            camera_id = excluded.camera_id,
            frame_index = excluded.frame_index,
            pts = excluded.pts,
            frame_shape_json = excluded.frame_shape_json,
            frame_hash_sha256 = excluded.frame_hash_sha256,
            model_artifact_digest = excluded.model_artifact_digest,
            invocation_seq = excluded.invocation_seq,
            exception_type = excluded.exception_type,
            exception_message = excluded.exception_message,
            exit_code = excluded.exit_code,
            action = excluded.action,
            fault_time_iso = excluded.fault_time_iso
        """,
        (
            record.pid,
            record.boot_time_iso,
            record.profile,
            record.task,
            record.stage,
            record.camera_id,
            record.frame_index,
            record.pts,
            None if record.frame_shape is None else json.dumps(list(record.frame_shape)),
            record.frame_hash_sha256,
            record.model_artifact_digest,
            record.invocation_seq,
            record.exception_type,
            _truncate_message(record.exception_message),
            record.exit_code,
            record.action,
            record.fault_time_iso,
        ),
    )


def _persist_edge_first_fault(database_path: Path, record: FirstFaultRecord) -> bool:
    """Production central-DB path: ZERO_WAIT once, never the default 5s bound."""
    wrote = best_effort_zero_wait_write(
        database_path,
        actor=RuntimeActor.WORKER,
        write=lambda connection: _upsert_fault_row(connection, record),
    )
    if not wrote:
        LOGGER.warning(
            "first-fault record unavailable at %s: zero-wait write failed",
            database_path,
        )
    return wrote


def _persist_legacy_first_fault(database_path: Path, record: FirstFaultRecord) -> bool:
    """Legacy worker-state.sqlite3 path: open with busy_timeout_ms=0."""
    connection: sqlite3.Connection | None = None
    try:
        connection = open_connection(database_path, busy_timeout_ms=_FAULT_WRITE_BUSY_TIMEOUT_MS)
        with ImmediateTransaction(connection):
            _upsert_fault_row(connection, record)
    except _STORE_UNAVAILABLE_ERRORS as error:
        LOGGER.warning("first-fault record unavailable at %s: %s", database_path, error)
        return False
    else:
        return True
    finally:
        if connection is not None:
            connection.close()


def persist_first_fault(
    record: FirstFaultRecord,
    *,
    state_dir: Path | None = None,
) -> bool:
    """Upsert *record* into the ``faults`` table. Returns True if this call
    wrote the record, False if a record was already written this process
    lifetime (no-op) or if the write failed for any reason.

    Best-effort and non-blocking: storage failures -- including a concurrent
    writer holding the database -- degrade silently to False rather than
    raising or waiting. The caller (``FaultHandler.handle``) must always
    proceed to stop camera loops and exit regardless of this return value.
    """
    global _written  # noqa: PLW0603
    with _write_lock:
        if _written:
            return False
        _written = True

    if state_dir is None:
        state_dir = resolve_state_dir()
    database_path = _worker_state_db_path(state_dir)

    if database_path.name == _EDGE_DATABASE_FILENAME:
        return _persist_edge_first_fault(database_path, record)
    return _persist_legacy_first_fault(database_path, record)


def make_fault_record(
    exc: Exception,
    *,
    profile: str,
    task: str,
    stage: str,
    camera_id: str,
    frame_index: int | None = None,
    pts: float | None = None,
    image: object = None,
    model_artifact_digest: str | None = None,
    invocation_seq: int = 0,
    exit_code: int = 4,
) -> FirstFaultRecord:
    """Build a FirstFaultRecord from a caught FatalAcceleratorError."""
    frame_shape: tuple[int, ...] | None = None
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            frame_shape = tuple(image.shape)
    except Exception:  # noqa: BLE001 S110
        pass

    return FirstFaultRecord(
        pid=os.getpid(),
        boot_time_iso=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.monotonic() - time.process_time())
        ),
        profile=profile,
        task=task,
        stage=stage,
        camera_id=camera_id,
        frame_index=frame_index,
        pts=pts,
        frame_shape=frame_shape,
        frame_hash_sha256=_frame_hash(image),
        model_artifact_digest=model_artifact_digest,
        invocation_seq=invocation_seq,
        exception_type=type(exc).__qualname__,
        exception_message=str(exc),
        exit_code=exit_code,
    )


__all__ = [
    "WORKER_STATE_DB_FILENAME",
    "FirstFaultRecord",
    "make_fault_record",
    "persist_first_fault",
]
