"""Verified immutable snapshots for the released legacy database adapter."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_ALLOWED_WORKER_SCHEMAS: Final = {6, 7, 8, 9, 10}


class LegacyImportError(ValueError):
    """A released legacy source or resumable receipt is inconsistent."""


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    name: str
    path: Path
    backup: Path
    schema: str
    sha256: str
    size_bytes: int


def snapshot_source(name: str, path: Path, state_directory: Path) -> SourceSnapshot:
    source = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
        timeout=5.0,
        isolation_level=None,
    )
    try:
        if source.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise sqlite3.DatabaseError(f"{name} integrity check failed")
        schema = _source_schema(name, source)
        backup_directory = state_directory / "legacy-backups"
        backup_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, raw = tempfile.mkstemp(prefix=f".{name}-", suffix=".tmp", dir=backup_directory)
        os.close(descriptor)
        temporary = Path(raw)
        try:
            destination = sqlite3.connect(temporary)
            try:
                source.backup(destination)
                if destination.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise sqlite3.DatabaseError(f"{name} backup integrity check failed")
            finally:
                destination.close()
            encoded = temporary.read_bytes()
            digest = hashlib.sha256(encoded).hexdigest()
            backup = backup_directory / f"{name}-schema-{schema}-{digest}.sqlite3"
            if backup.exists():
                _validate_existing_backup(backup, digest, name)
                temporary.unlink()
            else:
                os.replace(temporary, backup)
                backup.chmod(0o600)
            return SourceSnapshot(name, path, backup, schema, digest, backup.stat().st_size)
        except (OSError, sqlite3.Error, LegacyImportError):
            temporary.unlink(missing_ok=True)
            raise
    finally:
        source.close()


def _validate_existing_backup(backup: Path, digest: str, source_name: str) -> None:
    if hashlib.sha256(backup.read_bytes()).hexdigest() != digest:
        raise sqlite3.DatabaseError(f"{source_name} backup digest collision")


def _source_schema(name: str, connection: sqlite3.Connection) -> str:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if name == "catalog":
        if version != 3:
            raise LegacyImportError(f"unsupported catalog schema {version}; expected 3")
        return str(version)
    if name == "worker":
        if version not in _ALLOWED_WORKER_SCHEMAS:
            raise LegacyImportError(
                f"unsupported worker outbox schema {version}; expected 6, 7, 8, 9, or 10"
            )
        return str(version)
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(connection_settings)")}
    required = {
        "facility_code",
        "client_installation_ref",
        "edge_installation_id",
        "enrollment_generation",
    }
    if not required <= columns:
        raise LegacyImportError(
            "unsupported connection schema; run released connection migration first"
        )
    return "connection-v2"


__all__ = ["LegacyImportError", "SourceSnapshot", "snapshot_source"]
