"""Atomic local storage for rendered alert snapshots."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import final

from worker.pipeline.output.evidence.clip_config import configured_store_dir
from worker.pipeline.output.evidence.snapshot_files import SnapshotFiles


@dataclass(frozen=True, slots=True)
class StoredSnapshot:
    snapshot_id: str
    path: str
    sha256: str
    size_bytes: int
    mime_type: str
    captured_at: str
    camera_id: str
    edge_event_id: str | None


@dataclass(slots=True)
class SnapshotConflictError(Exception):
    snapshot_id: str

    def __str__(self) -> str:
        return f"snapshot {self.snapshot_id} conflicts with existing bytes or metadata"


@final
class SnapshotStore(SnapshotFiles):
    def __init__(self, store_dir: Path | None = None) -> None:
        super().__init__(configured_store_dir() if store_dir is None else store_dir)

    def store(
        self,
        jpeg: bytes,
        *,
        snapshot_id: str,
        captured_at: str,
        camera_id: str,
        edge_event_id: str | None,
    ) -> StoredSnapshot:
        """Atomically compare-or-insert one immutable snapshot identity."""
        relative_path = self._relative_path(camera_id, captured_at, snapshot_id)
        destination = self.store_dir / relative_path
        identity_key = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
        lock_name = f"{identity_key[:2]}.lock"
        lock_directory = self._open_directory(Path(".snapshot-locks"), create=True)
        try:
            lock_descriptor = self._open_file(
                lock_directory,
                lock_name,
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            with os.fdopen(lock_descriptor, "a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                return self._store_locked(
                    jpeg,
                    snapshot_id=snapshot_id,
                    captured_at=captured_at,
                    camera_id=camera_id,
                    edge_event_id=edge_event_id,
                    relative_path=relative_path,
                    destination=destination,
                    identity_key=identity_key,
                )
        finally:
            os.close(lock_directory)

    def _store_locked(
        self,
        jpeg: bytes,
        *,
        snapshot_id: str,
        captured_at: str,
        camera_id: str,
        edge_event_id: str | None,
        relative_path: Path,
        destination: Path,
        identity_key: str,
    ) -> StoredSnapshot:
        expected = StoredSnapshot(
            snapshot_id=snapshot_id,
            path=relative_path.as_posix(),
            sha256=hashlib.sha256(jpeg).hexdigest(),
            size_bytes=len(jpeg),
            mime_type="image/jpeg",
            captured_at=captured_at,
            camera_id=camera_id,
            edge_event_id=edge_event_id,
        )
        metadata_relative = Path(".snapshot-identities") / f"{identity_key}.json"
        metadata_content = self._read_file(metadata_relative)
        if metadata_content is not None:
            try:
                actual = StoredSnapshot(**json.loads(metadata_content.decode("utf-8")))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SnapshotConflictError(snapshot_id) from exc
            if actual != expected:
                raise SnapshotConflictError(snapshot_id)
        else:
            self._write_atomic(
                self.store_dir / metadata_relative,
                json.dumps(
                    asdict(expected),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )

        existing = self._read_file(relative_path)
        if existing is not None:
            if existing != jpeg:
                raise SnapshotConflictError(snapshot_id)
            directory = self._open_directory(relative_path.parent, create=False)
            try:
                self._fsync_directory(directory)
            finally:
                os.close(directory)
        else:
            self._write_atomic(destination, jpeg)
        return expected

    @staticmethod
    def _relative_path(camera_id: str, captured_at: str, snapshot_id: str) -> Path:
        date = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).date().isoformat()
        camera_key = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:16]
        snapshot_key = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
        return Path("snapshots") / camera_key / date / f"{snapshot_key}.jpg"


__all__ = ["SnapshotConflictError", "SnapshotStore", "StoredSnapshot"]
