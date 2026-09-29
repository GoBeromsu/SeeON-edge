"""Owner-only JSON reports and receipts the cutover steps leave behind."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from backend.app.edge_db.migration.authority_file import discard, fsync_directory


def write_report(path: Path, report: dict[str, object]) -> None:
    """Atomically replace an owner-only JSON report."""
    body = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(body.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        fsync_directory(path.parent)
    finally:
        discard(temp)


__all__ = ["write_report"]
