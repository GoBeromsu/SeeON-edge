"""Best-effort first-fault publication on the durable delivery queue."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

from shared.events.delivery_queue import DeliveryQueue, EventEntry
from worker.runtime.state_dir import resolve_state_dir

LOGGER: Final = logging.getLogger(__name__)
MAX_EXCEPTION_MESSAGE_CHARS: Final = 4_096
WORKER_STATE_DB_FILENAME: Final = "delivery-queue"

_write_lock = threading.Lock()
_written = False


@dataclass(slots=True)
class FirstFaultRecord:
    pid: int
    boot_time_iso: str
    profile: str
    task: str
    stage: str
    camera_id: str
    frame_index: int | None
    pts: float | None
    frame_shape: tuple[int, ...] | None
    frame_hash_sha256: str | None
    model_artifact_digest: str | None
    invocation_seq: int
    exception_type: str
    exception_message: str
    exit_code: int
    action: str = "exit_process_retire_context"
    fault_time_iso: str = field(default_factory=lambda: _iso_now())


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _frame_hash(image: object) -> str | None:
    try:
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


def persist_first_fault(record: FirstFaultRecord, *, state_dir: Path | None = None) -> bool:
    """Schedule a first-fault queue admission without delaying fatal exit."""
    global _written  # noqa: PLW0603
    with _write_lock:
        if _written:
            return False
        _written = True

    directory = (resolve_state_dir() if state_dir is None else state_dir) / "delivery-queue"
    try:
        entry = _fault_entry(record)
    except (TypeError, ValueError) as error:
        LOGGER.warning("first-fault record unavailable: %s", error)
        return False

    try:
        result = DeliveryQueue(directory, recover=False).try_admit_nonblocking(entry)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("first-fault record unavailable: %s", error)
        return False
    if not result.accepted:
        LOGGER.warning("first-fault record unavailable: queue admission %s", result.fault)
        return False
    return True


def _fault_entry(record: FirstFaultRecord) -> EventEntry:
    values = asdict(record)
    values["frame_shape"] = list(record.frame_shape) if record.frame_shape is not None else None
    values["exception_message"] = _truncate_message(record.exception_message)
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    identity = hashlib.sha256(payload).hexdigest()
    return EventEntry(
        edge_event_id=f"fault-{identity}",
        event_type="runtime.fault",
        detected_at=_safe_text(record.fault_time_iso),
        camera_id=_safe_text(record.camera_id),
        facility_id="local",
        decision_trace=b"",
        values=payload,
    )


def _safe_text(value: str) -> str:
    if value.isascii() and value.isprintable():
        return value
    return hashlib.sha256(value.encode()).hexdigest()


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
