"""Crash-recoverable bounded local storage for rendered alert snapshots."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, final

from worker.pipeline.output.evidence.clip_config import configured_store_dir
from worker.pipeline.output.evidence.snapshot_files import SnapshotFiles

LOGGER: Final = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class SnapshotLimits:
    max_pending_global: int = 1024
    max_pending_per_camera: int = 256
    max_files_global: int = 10_000
    max_files_per_camera: int = 2_000
    max_bytes_global: int = 2 * 1024 * 1024 * 1024
    max_bytes_per_camera: int = 512 * 1024 * 1024
    max_age: timedelta = timedelta(days=60)
    max_pending_age: timedelta = timedelta(days=1)

    def __post_init__(self) -> None:
        numeric = (
            self.max_pending_global,
            self.max_pending_per_camera,
            self.max_files_global,
            self.max_files_per_camera,
            self.max_bytes_global,
            self.max_bytes_per_camera,
        )
        if any(value <= 0 for value in numeric):
            raise ValueError("snapshot limits must be positive")
        if self.max_age <= timedelta(0) or self.max_pending_age <= timedelta(0):
            raise ValueError("snapshot age limits must be positive")


@dataclass(slots=True)
class SnapshotStats:
    staged: int = 0
    published: int = 0
    committed: int = 0
    dropped_capacity: int = 0
    discarded_unreferenced: int = 0
    quarantined_corrupt: int = 0
    purged: int = 0
    pending_files: int = 0
    pending_bytes: int = 0


@dataclass(frozen=True, slots=True)
class SnapshotDiscardReport:
    discarded: int = 0
    corrupt: int = 0


@dataclass(slots=True)
class SnapshotConflictError(Exception):
    snapshot_id: str

    def __str__(self) -> str:
        return f"snapshot {self.snapshot_id} conflicts with existing bytes or metadata"


@dataclass(slots=True)
class SnapshotCapacityError(Exception):
    reason: str

    def __str__(self) -> str:
        return f"snapshot capacity exceeded: {self.reason}"


@final
class SnapshotStore(SnapshotFiles):
    """Own the filesystem half of snapshot two-phase publication.

    ``stage`` durably writes immutable metadata and bytes below
    ``.snapshot-staging``. Only after the event transaction commits may
    ``publish`` atomically move bytes to their final path. ``commit`` moves the
    staging metadata into the retained identity directory after the central
    snapshot relation commits. A crash at every boundary therefore leaves a
    named transition that startup reconciliation can resume.
    """

    def __init__(
        self,
        store_dir: Path | None = None,
        *,
        limits: SnapshotLimits | None = None,
    ) -> None:
        super().__init__(configured_store_dir() if store_dir is None else store_dir)
        self.limits = SnapshotLimits() if limits is None else limits
        self.stats = SnapshotStats()

    def stage(
        self,
        jpeg: bytes,
        *,
        snapshot_id: str,
        captured_at: str,
        camera_id: str,
        edge_event_id: str | None,
    ) -> StoredSnapshot:
        """Durably reserve immutable identity and bytes without publishing a final."""
        expected = self._expected(
            jpeg,
            snapshot_id=snapshot_id,
            captured_at=captured_at,
            camera_id=camera_id,
            edge_event_id=edge_event_id,
        )
        identity_key = self._identity_key(snapshot_id)
        with self._lock("admission.lock"):
            with self._lock(f"{identity_key[:2]}.lock"):
                existing = self._load_identity(identity_key)
                if existing is not None:
                    self._require_same(existing, expected)
                    self._require_bytes(Path(existing.path), existing)
                    self._refresh_pending_stats()
                    return existing
                staged = self._load_staged(identity_key)
                if staged is not None:
                    self._require_same(staged, expected)
                    blob = self._read_file(self._staged_blob(identity_key))
                    if blob is None:
                        final = self._read_file(Path(expected.path))
                        if final is None:
                            self._write_atomic(
                                self.store_dir / self._staged_blob(identity_key), jpeg
                            )
                        else:
                            self._require_content(final, expected)
                    elif hashlib.sha256(blob).hexdigest() != expected.sha256 or len(blob) != len(
                        jpeg
                    ):
                        raise SnapshotConflictError(snapshot_id)
                    self._refresh_pending_stats()
                    return staged
                orphan_blob = self._read_file(self._staged_blob(identity_key))
                if orphan_blob is not None and orphan_blob != jpeg:
                    raise SnapshotConflictError(snapshot_id)
                self._enforce_capacity(expected)
                self._write_atomic(
                    self.store_dir / self._staged_metadata(identity_key),
                    self._encode(expected),
                )
                if orphan_blob is None:
                    self._write_atomic(self.store_dir / self._staged_blob(identity_key), jpeg)
                self.stats.staged += 1
                self._refresh_pending_stats()
                return expected

    def publish(self, snapshot: StoredSnapshot) -> None:
        """Atomically publish validated staged bytes while retaining transition metadata."""
        identity_key = self._identity_key(snapshot.snapshot_id)
        with self._lock(f"{identity_key[:2]}.lock"):
            identity = self._load_identity(identity_key)
            if identity is not None:
                self._require_same(identity, snapshot)
                self._require_bytes(Path(snapshot.path), snapshot)
                return
            staged = self._load_staged(identity_key)
            if staged is not None:
                self._require_same(staged, snapshot)
            final = self._read_file(Path(snapshot.path))
            if final is not None:
                self._require_content(final, snapshot)
                return
            if staged is None:
                raise SnapshotConflictError(snapshot.snapshot_id)
            staged_blob = self._read_file(self._staged_blob(identity_key))
            if staged_blob is None:
                raise SnapshotConflictError(snapshot.snapshot_id)
            self._require_content(staged_blob, snapshot)
            self._replace_atomic(
                self.store_dir / self._staged_blob(identity_key),
                self.store_dir / snapshot.path,
            )
            self.stats.published += 1
            self._refresh_pending_stats()

    def commit(self, snapshot: StoredSnapshot) -> None:
        """Complete the filesystem transition after the DB relation commits."""
        identity_key = self._identity_key(snapshot.snapshot_id)
        with self._lock(f"{identity_key[:2]}.lock"):
            existing = self._load_identity(identity_key)
            if existing is not None:
                self._require_same(existing, snapshot)
            else:
                staged = self._load_staged(identity_key)
                self._require_bytes(Path(snapshot.path), snapshot)
                if staged is None:
                    self._write_atomic(
                        self.store_dir / self._identity_metadata(identity_key),
                        self._encode(snapshot),
                    )
                else:
                    self._require_same(staged, snapshot)
                    self._replace_atomic(
                        self.store_dir / self._staged_metadata(identity_key),
                        self.store_dir / self._identity_metadata(identity_key),
                    )
            self._unlink_file(self._staged_blob(identity_key))
            self.stats.committed += 1
            self._refresh_pending_stats()

    def store(
        self,
        jpeg: bytes,
        *,
        snapshot_id: str,
        captured_at: str,
        camera_id: str,
        edge_event_id: str | None,
    ) -> StoredSnapshot:
        """Compatibility helper for callers that need a local-only committed snapshot."""
        snapshot = self.stage(
            jpeg,
            snapshot_id=snapshot_id,
            captured_at=captured_at,
            camera_id=camera_id,
            edge_event_id=edge_event_id,
        )
        self.publish(snapshot)
        self.commit(snapshot)
        return snapshot

    def staged_records(self) -> tuple[StoredSnapshot, ...]:
        root = self.store_dir / ".snapshot-staging"
        if not root.exists():
            return ()
        records: list[StoredSnapshot] = []
        for metadata in sorted(root.glob("*.json")):
            record = self._decode_file(metadata)
            if record is not None:
                records.append(record)
        return tuple(records)

    def identity_records(self) -> tuple[StoredSnapshot, ...]:
        return self._records_in(Path(".snapshot-identities"))

    def retention_records(self) -> tuple[StoredSnapshot, ...]:
        return self._records_in(Path(".snapshot-retention"))

    def stage_retention(self, snapshot: StoredSnapshot) -> None:
        """Persist deletion intent before the central retention transaction."""
        identity_key = self._identity_key(snapshot.snapshot_id)
        with self._lock(f"{identity_key[:2]}.lock"):
            existing = self._load_identity(identity_key)
            if existing is not None:
                self._require_same(existing, snapshot)
            marker = self._decode(self._read_file(self._retention_metadata(identity_key)))
            if marker is not None:
                self._require_same(marker, snapshot)
                return
            self._write_atomic(
                self.store_dir / self._retention_metadata(identity_key),
                self._encode(snapshot),
            )

    def commit_retention(self, snapshot: StoredSnapshot) -> None:
        identity_key = self._identity_key(snapshot.snapshot_id)
        with self._lock(f"{identity_key[:2]}.lock"):
            self._unlink_file(self._retention_metadata(identity_key))

    def cancel_retention(self, snapshot: StoredSnapshot) -> None:
        self.commit_retention(snapshot)

    def discard_unreferenced_staging(
        self,
        referenced_snapshot_ids: set[str],
        *,
        now: datetime,
    ) -> SnapshotDiscardReport:
        """Delete staging absent from the DB, reporting every disposition."""
        del now
        root = self.store_dir / ".snapshot-staging"
        if not root.exists():
            return SnapshotDiscardReport()
        discarded = 0
        corrupt = 0
        for metadata in sorted(root.glob("*.json")):
            identity_key = metadata.stem
            record = self._decode_file(metadata)
            if record is None:
                self._unlink_file(self._staged_metadata(identity_key))
                self._unlink_file(self._staged_blob(identity_key))
                corrupt += 1
                continue
            if record.snapshot_id in referenced_snapshot_ids:
                continue
            self._unlink_file(self._staged_metadata(identity_key))
            self._unlink_file(self._staged_blob(identity_key))
            final = self._read_file(Path(record.path))
            if final is not None:
                self._unlink_file(Path(record.path))
            discarded += 1
        known = {path.stem for path in root.glob("*.json")}
        for blob in sorted(root.glob("*.jpg")):
            if blob.stem not in known:
                self._unlink_file(Path(".snapshot-staging") / blob.name)
                corrupt += 1
        self.stats.discarded_unreferenced += discarded
        self.stats.quarantined_corrupt += corrupt
        self._refresh_pending_stats()
        if discarded or corrupt:
            LOGGER.warning(
                "snapshot staging reconciliation removed artifacts: discarded=%d corrupt=%d",
                discarded,
                corrupt,
                extra={"discarded": discarded, "corrupt": corrupt},
            )
        return SnapshotDiscardReport(discarded=discarded, corrupt=corrupt)

    def matches_committed(self, snapshot: StoredSnapshot) -> bool:
        identity_key = self._identity_key(snapshot.snapshot_id)
        identity = self._load_identity(identity_key)
        if identity != snapshot:
            return False
        content = self._read_file(Path(snapshot.path))
        if content is None:
            return False
        try:
            self._require_content(content, snapshot)
        except SnapshotConflictError:
            return False
        return True

    def remove_committed(self, snapshot: StoredSnapshot) -> None:
        """Remove final bytes and identity while durable retention intent remains."""
        identity_key = self._identity_key(snapshot.snapshot_id)
        with self._lock(f"{identity_key[:2]}.lock"):
            identity = self._load_identity(identity_key)
            if identity is not None:
                self._require_same(identity, snapshot)
            marker = self._decode(self._read_file(self._retention_metadata(identity_key)))
            if marker is None:
                raise SnapshotConflictError(snapshot.snapshot_id)
            self._require_same(marker, snapshot)
            self._unlink_file(Path(snapshot.path))
            self._unlink_file(self._identity_metadata(identity_key))
            self.stats.purged += 1

    def _records_in(self, relative: Path) -> tuple[StoredSnapshot, ...]:
        root = self.store_dir / relative
        if not root.exists():
            return ()
        records: list[StoredSnapshot] = []
        for metadata in sorted(root.glob("*.json")):
            record = self._decode_file(metadata)
            if record is not None:
                records.append(record)
        return tuple(records)

    def _expected(
        self,
        jpeg: bytes,
        *,
        snapshot_id: str,
        captured_at: str,
        camera_id: str,
        edge_event_id: str | None,
    ) -> StoredSnapshot:
        if not snapshot_id or not camera_id:
            raise ValueError("snapshot identity and camera must be set")
        _ = _parse_utc_required(captured_at)
        relative_path = self._relative_path(camera_id, captured_at, snapshot_id)
        return StoredSnapshot(
            snapshot_id=snapshot_id,
            path=relative_path.as_posix(),
            sha256=hashlib.sha256(jpeg).hexdigest(),
            size_bytes=len(jpeg),
            mime_type="image/jpeg",
            captured_at=captured_at,
            camera_id=camera_id,
            edge_event_id=edge_event_id,
        )

    def _enforce_capacity(self, incoming: StoredSnapshot) -> None:
        pending = self.staged_records()
        identities = self.identity_records()
        all_records = (*pending, *identities)
        pending_camera = [record for record in pending if record.camera_id == incoming.camera_id]
        camera = [record for record in all_records if record.camera_id == incoming.camera_id]
        checks = (
            (len(pending) + 1 > self.limits.max_pending_global, "global pending files"),
            (
                len(pending_camera) + 1 > self.limits.max_pending_per_camera,
                "per-camera pending files",
            ),
            (len(all_records) + 1 > self.limits.max_files_global, "global files"),
            (len(camera) + 1 > self.limits.max_files_per_camera, "per-camera files"),
            (
                sum(record.size_bytes for record in all_records) + incoming.size_bytes
                > self.limits.max_bytes_global,
                "global bytes",
            ),
            (
                sum(record.size_bytes for record in camera) + incoming.size_bytes
                > self.limits.max_bytes_per_camera,
                "per-camera bytes",
            ),
        )
        for exceeded, reason in checks:
            if exceeded:
                self.stats.dropped_capacity += 1
                self._refresh_pending_stats(pending)
                LOGGER.warning(
                    "snapshot dropped by bounded admission: camera_id=%s reason=%s",
                    incoming.camera_id,
                    reason,
                    extra={"reason": reason, "camera_id": incoming.camera_id},
                )
                raise SnapshotCapacityError(reason)

    def _refresh_pending_stats(self, records: Iterable[StoredSnapshot] | None = None) -> None:
        pending = tuple(self.staged_records() if records is None else records)
        self.stats.pending_files = len(pending)
        self.stats.pending_bytes = sum(record.size_bytes for record in pending)

    def _load_staged(self, identity_key: str) -> StoredSnapshot | None:
        return self._decode(self._read_file(self._staged_metadata(identity_key)))

    def _load_identity(self, identity_key: str) -> StoredSnapshot | None:
        return self._decode(self._read_file(self._identity_metadata(identity_key)))

    def _require_bytes(self, relative: Path, expected: StoredSnapshot) -> None:
        content = self._read_file(relative)
        if content is None:
            raise SnapshotConflictError(expected.snapshot_id)
        self._require_content(content, expected)

    @staticmethod
    def _require_content(content: bytes, expected: StoredSnapshot) -> None:
        if (
            len(content) != expected.size_bytes
            or hashlib.sha256(content).hexdigest() != expected.sha256
        ):
            raise SnapshotConflictError(expected.snapshot_id)

    @staticmethod
    def _require_same(actual: StoredSnapshot, expected: StoredSnapshot) -> None:
        if actual != expected:
            raise SnapshotConflictError(expected.snapshot_id)

    @contextmanager
    def _lock(self, name: str) -> Generator[None, None, None]:
        for attempt in range(2):
            directory = self._open_directory(Path(".snapshot-locks"), create=True)
            try:
                descriptor = self._open_file(
                    directory,
                    name,
                    os.O_RDWR | os.O_CREAT,
                    0o600,
                )
            except FileNotFoundError:
                os.close(directory)
                if attempt == 1:
                    raise
                continue
            try:
                with os.fdopen(descriptor, "a+b") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    yield
            finally:
                os.close(directory)
            return
        raise AssertionError("snapshot lock retry exhausted")

    @staticmethod
    def _encode(snapshot: StoredSnapshot) -> bytes:
        return json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _decode(content: bytes | None) -> StoredSnapshot | None:
        if content is None:
            return None
        try:
            decoded: object = json.loads(content.decode("utf-8"))
            if not isinstance(decoded, dict):
                return None
            expected_keys = {
                "snapshot_id",
                "path",
                "sha256",
                "size_bytes",
                "mime_type",
                "captured_at",
                "camera_id",
                "edge_event_id",
            }
            if set(decoded) != expected_keys:
                return None
            snapshot_id = decoded["snapshot_id"]
            path = decoded["path"]
            sha256 = decoded["sha256"]
            size_bytes = decoded["size_bytes"]
            mime_type = decoded["mime_type"]
            captured_at = decoded["captured_at"]
            camera_id = decoded["camera_id"]
            edge_event_id = decoded["edge_event_id"]
            if not all(
                isinstance(value, str)
                for value in (snapshot_id, path, sha256, mime_type, captured_at, camera_id)
            ):
                return None
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
                return None
            if edge_event_id is not None and not isinstance(edge_event_id, str):
                return None
            assert isinstance(snapshot_id, str)
            assert isinstance(path, str)
            assert isinstance(sha256, str)
            assert isinstance(mime_type, str)
            assert isinstance(captured_at, str)
            assert isinstance(camera_id, str)
            snapshot = StoredSnapshot(
                snapshot_id=snapshot_id,
                path=path,
                sha256=sha256,
                size_bytes=size_bytes,
                mime_type=mime_type,
                captured_at=captured_at,
                camera_id=camera_id,
                edge_event_id=edge_event_id,
            )
            if snapshot.mime_type != "image/jpeg" or snapshot.size_bytes <= 0:
                return None
            if len(snapshot.sha256) != 64 or any(
                character not in "0123456789abcdef" for character in snapshot.sha256
            ):
                return None
            if (
                snapshot.path
                != SnapshotStore._relative_path(
                    snapshot.camera_id, snapshot.captured_at, snapshot.snapshot_id
                ).as_posix()
            ):
                return None
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        else:
            return snapshot

    def _decode_file(self, path: Path) -> StoredSnapshot | None:
        try:
            return self._decode(path.read_bytes())
        except OSError:
            return None

    @staticmethod
    def _identity_key(snapshot_id: str) -> str:
        return hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()

    @staticmethod
    def _staged_metadata(identity_key: str) -> Path:
        return Path(".snapshot-staging") / f"{identity_key}.json"

    @staticmethod
    def _staged_blob(identity_key: str) -> Path:
        return Path(".snapshot-staging") / f"{identity_key}.jpg"

    @staticmethod
    def _identity_metadata(identity_key: str) -> Path:
        return Path(".snapshot-identities") / f"{identity_key}.json"

    @staticmethod
    def _retention_metadata(identity_key: str) -> Path:
        return Path(".snapshot-retention") / f"{identity_key}.json"

    @staticmethod
    def _relative_path(camera_id: str, captured_at: str, snapshot_id: str) -> Path:
        date = _parse_utc_required(captured_at).date().isoformat()
        camera_key = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:16]
        snapshot_key = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
        return Path("snapshots") / camera_key / date / f"{snapshot_key}.jpg"


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _parse_utc_required(value: str) -> datetime:
    parsed = _parse_utc(value)
    if parsed is None:
        raise ValueError("snapshot captured_at must be timezone-aware ISO-8601")
    return parsed


__all__ = [
    "SnapshotCapacityError",
    "SnapshotConflictError",
    "SnapshotDiscardReport",
    "SnapshotLimits",
    "SnapshotStats",
    "SnapshotStore",
    "StoredSnapshot",
]
