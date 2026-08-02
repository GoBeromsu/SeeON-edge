"""First-fault record persistence (todo 25).

Writes exactly one bounded, fsynced JSON record under the resolved worker
state directory (``worker/runtime/state_dir.py``, ``~/.local/state/ml-worker``,
no env override) on the first FatalAcceleratorError.  Subsequent calls are
no-ops (the record is written at most once per process lifetime).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

from worker.runtime.state_dir import resolve_state_dir

FIRST_FAULT_FILENAME: Final = "first_fault.json"
# Max record size: 64 KiB.  Enough for any realistic fault record.
MAX_RECORD_BYTES: Final = 65_536

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
        import numpy as np

        if isinstance(image, np.ndarray):
            return hashlib.sha256(image.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 S110
        pass
    return None


def persist_first_fault(
    record: FirstFaultRecord,
    *,
    state_dir: Path | None = None,
) -> Path | None:
    """Write *record* to disk atomically.  Returns the path written, or None if
    a record was already written (subsequent calls are silently ignored).

    The write is atomic: we write to a temp file in the same directory, fsync
    it, then rename it into place (rename is atomic on POSIX).  We also fsync
    the directory after the rename so the entry survives a crash.
    """
    global _written  # noqa: PLW0603
    with _write_lock:
        if _written:
            return None
        _written = True

    if state_dir is None:
        state_dir = resolve_state_dir()

    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        dest = state_dir / FIRST_FAULT_FILENAME
        tmp = state_dir / f"{FIRST_FAULT_FILENAME}.tmp"

        payload = json.dumps(asdict(record), default=str, separators=(",", ":"))
        if len(payload.encode()) > MAX_RECORD_BYTES:
            # Truncate the message to fit — we must not silently drop the record.
            truncated = record.__class__(
                **{
                    **asdict(record),
                    "exception_message": record.exception_message[:256] + "...[truncated]",
                },
            )
            payload = json.dumps(asdict(truncated), default=str, separators=(",", ":"))

        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())

        tmp.rename(dest)

        # fsync the directory so the rename is durable.
        dir_fd = os.open(str(state_dir), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:  # noqa: BLE001
        # Persistence failure must never prevent the process from exiting.
        return None
    else:
        return dest


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
    "FIRST_FAULT_FILENAME",
    "FirstFaultRecord",
    "make_fault_record",
    "persist_first_fault",
]
