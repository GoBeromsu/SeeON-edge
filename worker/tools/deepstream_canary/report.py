"""Canonical content-addressed report and receipt-manifest persistence."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def canonical_json(value: Mapping[str, JsonValue]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_once(path: Path, content: bytes) -> None:
    """Atomically create immutable evidence without overwriting a prior receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        _ = target.write(content)
        target.flush()
        os.fsync(target.fileno())
    _ = path.chmod(0o400)


def write_canonical_report(root: Path, value: Mapping[str, JsonValue]) -> Path:
    encoded = canonical_json(value)
    digest = hashlib.sha256(encoded).hexdigest()
    destination = root / f"gate-report.{digest}.json"
    write_once(destination, encoded)
    return destination


def write_receipt_manifest(root: Path, files: tuple[Path, ...]) -> Path:
    entries: dict[str, JsonValue] = {}
    for path in sorted(files):
        content = path.read_bytes()
        entries[str(path.relative_to(root))] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    manifest: dict[str, JsonValue] = {"schema_version": 1, "files": entries}
    destination = root / "receipt-manifest.json"
    write_once(destination, canonical_json(manifest))
    return destination
