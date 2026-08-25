"""Synchronous durable first-fault receipt for the isolated dark runner."""

from __future__ import annotations

import json
import os
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.path_security import validate_private_directory


@dataclass(frozen=True, slots=True)
class DarkFirstFault:
    schema_version: int
    profile: str
    stage: str
    category: str
    exit_code: int
    worker_boot_id: str
    child_instance_id: str
    fault_time_iso: str
    action: str = "exit_container_retire_context"


def _read_existing(path: Path) -> DarkFirstFault:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != 0o600
        ):
            raise OSError("first-fault file trust validation failed")
        with os.fdopen(descriptor, "rb", closefd=False) as existing:
            return TypeAdapter(DarkFirstFault).validate_json(existing.read())
    finally:
        os.close(descriptor)


def persist_first_fault(
    path: Path,
    *,
    category: str,
    exit_code: int,
    worker_boot_id: uuid.UUID,
    child_instance_id: uuid.UUID,
) -> DarkFirstFault:
    validate_private_directory(path.parent)
    record = DarkFirstFault(
        schema_version=1,
        profile="nvidia",
        stage="deepstream_child",
        category=category,
        exit_code=exit_code,
        worker_boot_id=str(worker_boot_id),
        child_instance_id=str(child_instance_id),
        fault_time_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    encoded = json.dumps(asdict(record), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except FileExistsError:
        return _read_existing(path)
    with os.fdopen(descriptor, "wb") as fault_file:
        _ = fault_file.write(encoded)
        fault_file.flush()
        _ = os.fsync(fault_file.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _ = os.fsync(directory)
    finally:
        os.close(directory)
    return _read_existing(path)


def persist_child_fault(config: ChildConfig, category: str) -> DarkFirstFault:
    return persist_first_fault(
        config.first_fault_path,
        category=category,
        exit_code=4,
        worker_boot_id=config.worker_boot_id,
        child_instance_id=config.child_instance_id,
    )


__all__ = ["DarkFirstFault", "persist_child_fault", "persist_first_fault"]
