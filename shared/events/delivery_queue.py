"""Filesystem-backed, publish-once delivery queue for edge relay envelopes."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, TypeAlias

from shared.events import envelope_limits as limits

MAX_ACCEPTED_ENTRIES: Final = 4096
MAX_ACCEPTED_BYTES: Final = 256 * 1024 * 1024

#: Bound on evidence retained after refusal. Unbounded retention is a different
#: failure with the same cause: the runtime slot filling a disk it shares with
#: the clip store and the backend database. Sized to the live queue so retention
#: can never exceed what the queue itself was budgeted for.
MAX_DEAD_LETTERED_ENTRIES: Final = MAX_ACCEPTED_ENTRIES
MAX_DEAD_LETTERED_BYTES: Final = MAX_ACCEPTED_BYTES
_ENTRY_SUFFIX: Final = ".json"
_SAFE_TEXT = re.compile(r"^[ -~]*$")


class EntryKind(StrEnum):
    EVENT = "EVENT"
    SNAPSHOT_ATTACHMENT = "SNAPSHOT_ATTACHMENT"
    SNAPSHOT_DISPOSITION = "SNAPSHOT_DISPOSITION"


class AdmissionFault(StrEnum):
    ENTRY_CAPACITY = "entry_capacity"
    BYTE_CAPACITY = "byte_capacity"
    CONFLICT = "conflict"
    LOCK_UNAVAILABLE = "lock_unavailable"


@dataclass(frozen=True, slots=True)
class EventEntry:
    edge_event_id: str
    event_type: str
    detected_at: str
    camera_id: str
    facility_id: str
    decision_trace: bytes
    values: bytes
    shed_detail_keys: tuple[str, ...] = ()
    entry_id: str = ""

    @property
    def kind(self) -> EntryKind:
        return EntryKind.EVENT

    def __post_init__(self) -> None:
        if not self.entry_id:
            object.__setattr__(self, "entry_id", f"event-{self.edge_event_id}")
        _validate_text(self.entry_id, limits.ENTRY_ID_MAX_CHARS, "entry_id")
        _validate_text(self.edge_event_id, limits.EDGE_EVENT_ID_MAX_CHARS, "edge_event_id")
        _validate_text(self.event_type, limits.EVENT_TYPE_MAX_CHARS, "event_type")
        _validate_text(self.detected_at, limits.DETECTED_AT_MAX_CHARS, "detected_at")
        _validate_text(self.camera_id, limits.CAMERA_ID_MAX_CHARS, "camera_id")
        _validate_text(self.facility_id, limits.FACILITY_ID_MAX_CHARS, "facility_id")
        _validate_bytes(self.decision_trace, limits.DECISION_TRACE_BYTES_MAX, "decision_trace")
        _validate_bytes(self.values, limits.VALUES_BYTES_MAX, "values")
        if (
            not isinstance(self.shed_detail_keys, tuple)
            or any(not isinstance(key, str) or not key for key in self.shed_detail_keys)
        ):
            raise ValueError("shed_detail_keys must be a tuple of non-empty strings")


@dataclass(frozen=True, slots=True)
class SnapshotAttachmentEntry:
    edge_event_id: str
    snapshot_id: str
    sha256: str
    media_reference: str
    size_bytes: int
    mime_type: str
    entry_id: str = ""

    @property
    def kind(self) -> EntryKind:
        return EntryKind.SNAPSHOT_ATTACHMENT

    def __post_init__(self) -> None:
        if not self.entry_id:
            object.__setattr__(
                self,
                "entry_id",
                _keyed_id("attachment", self.edge_event_id, self.snapshot_id, self.sha256),
            )
        _validate_text(self.entry_id, limits.ENTRY_ID_MAX_CHARS, "entry_id")
        _validate_text(self.edge_event_id, limits.EDGE_EVENT_ID_MAX_CHARS, "edge_event_id")
        _validate_text(self.snapshot_id, limits.SNAPSHOT_ID_MAX_CHARS, "snapshot_id")
        _validate_text(self.sha256, limits.SHA256_MAX_CHARS, "sha256")
        _validate_text(self.media_reference, limits.MEDIA_REFERENCE_MAX_CHARS, "media_reference")
        _validate_text(self.mime_type, limits.MIME_TYPE_MAX_CHARS, "mime_type")
        if not 0 <= self.size_bytes <= limits.SNAPSHOT_SIZE_BYTES_MAX:
            raise ValueError("size_bytes is outside its finite envelope limit")


@dataclass(frozen=True, slots=True)
class SnapshotDispositionEntry:
    edge_event_id: str
    snapshot_id: str
    disposition: str
    reason: str
    entry_id: str = ""

    @property
    def kind(self) -> EntryKind:
        return EntryKind.SNAPSHOT_DISPOSITION

    def __post_init__(self) -> None:
        if not self.entry_id:
            object.__setattr__(
                self,
                "entry_id",
                _keyed_id("disposition", self.edge_event_id, self.snapshot_id, self.disposition),
            )
        _validate_text(self.entry_id, limits.ENTRY_ID_MAX_CHARS, "entry_id")
        _validate_text(self.edge_event_id, limits.EDGE_EVENT_ID_MAX_CHARS, "edge_event_id")
        _validate_text(self.snapshot_id, limits.SNAPSHOT_ID_MAX_CHARS, "snapshot_id")
        _validate_text(self.disposition, limits.DISPOSITION_MAX_CHARS, "disposition")
        _validate_text(self.reason, limits.DISPOSITION_REASON_MAX_CHARS, "reason")


DeliveryEntry: TypeAlias = EventEntry | SnapshotAttachmentEntry | SnapshotDispositionEntry


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    accepted: bool
    fault: AdmissionFault | None = None
    already_admitted: bool = False


@dataclass(frozen=True, slots=True)
class DeliveryQueueCapacitySnapshot:
    accepted_count: int
    accepted_bytes: int
    max_accepted_entries: int
    max_accepted_bytes: int
    by_kind: dict[EntryKind, int]
    #: Entries the backend refused or that exhausted delivery. They are
    #: retained on disk, not delivered, and need operator action; a
    #: deployment cannot act on what it cannot see.
    dead_lettered_count: int = 0
    dead_lettered_bytes: int = 0


class DeliveryQueue:
    """A bounded queue whose published entry files are its only durable state."""

    def __init__(self, directory: Path, *, recover: bool = True) -> None:
        self._directory = directory
        created = not directory.exists()
        self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if created:
            # The directory entry must reach disk before anything is admitted
            # into it. Without this a power loss can lose the queue directory
            # name itself, taking every admitted entry with it -- the whole
            # point of the queue is that those survive.
            _fsync_directory(directory.parent)
        self._thread_lock = threading.Lock()
        self._lock_path = self._directory / ".delivery-queue.lock"
        self._lock_path.touch(mode=0o600, exist_ok=True)
        if recover:
            with self._locked():
                self._remove_orphan_temps()
                self._count, self._bytes = self._scan_totals()
        else:
            # Capacity is re-derived under the queue lock by admission. Fatal
            # callers skip blocking recovery so they can use the zero-wait path.
            self._count = 0
            self._bytes = 0

    @property
    def accepted_count(self) -> int:
        with self._locked():
            self._count, self._bytes = self._scan_totals()
            return self._count

    @property
    def accepted_bytes(self) -> int:
        with self._locked():
            self._count, self._bytes = self._scan_totals()
            return self._bytes

    @property
    def capacity_snapshot(self) -> DeliveryQueueCapacitySnapshot:
        """Return one locked, filesystem-derived view of queue capacity."""
        with self._locked():
            paths = tuple(self._published_paths())
            by_kind = dict.fromkeys(EntryKind, 0)
            for path in paths:
                entry = json.loads(path.read_bytes())
                kind = EntryKind(entry["kind"])
                by_kind[kind] += 1
            accepted_count = len(paths)
            accepted_bytes = sum(path.stat().st_size for path in paths)
            self._count, self._bytes = accepted_count, accepted_bytes
            dead_directory = self.dead_letter_directory
            dead_count = 0
            dead_bytes = 0
            if dead_directory.is_dir():
                for retained in dead_directory.iterdir():
                    if retained.is_file():
                        dead_count += 1
                        dead_bytes += retained.stat().st_size
            return DeliveryQueueCapacitySnapshot(
                accepted_count=accepted_count,
                accepted_bytes=accepted_bytes,
                max_accepted_entries=MAX_ACCEPTED_ENTRIES,
                max_accepted_bytes=MAX_ACCEPTED_BYTES,
                by_kind=by_kind,
                dead_lettered_count=dead_count,
                dead_lettered_bytes=dead_bytes,
            )

    def try_admit(self, entry: DeliveryEntry) -> AdmissionResult:
        payload = _serialize(entry)
        target = self._entry_path(entry.entry_id)
        with self._locked():
            return self._admit_unlocked(payload, target)

    def try_admit_nonblocking(self, entry: DeliveryEntry) -> AdmissionResult:
        """Attempt one durable admission without waiting for either queue lock."""
        payload = _serialize(entry)
        target = self._entry_path(entry.entry_id)
        with self._try_locked() as acquired:
            if not acquired:
                return AdmissionResult(False, AdmissionFault.LOCK_UNAVAILABLE)
            return self._admit_unlocked(payload, target)

    def _admit_unlocked(self, payload: bytes, target: Path) -> AdmissionResult:
        self._remove_orphan_temps()
        self._count, self._bytes = self._scan_totals()
        if target.exists():
            if target.read_bytes() == payload:
                return AdmissionResult(accepted=True, already_admitted=True)
            return AdmissionResult(False, AdmissionFault.CONFLICT)
        if self._count >= MAX_ACCEPTED_ENTRIES:
            return AdmissionResult(False, AdmissionFault.ENTRY_CAPACITY)
        if self._bytes + len(payload) > MAX_ACCEPTED_BYTES:
            return AdmissionResult(False, AdmissionFault.BYTE_CAPACITY)
        temporary = self._directory / f".{uuid.uuid4().hex}.tmp"
        try:
            _write_durable(temporary, payload)
            os.replace(temporary, target)
            _fsync_directory(self._directory)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._count += 1
        self._bytes += len(payload)
        return AdmissionResult(True)

    def acknowledge(self, entry_id: str) -> bool:
        """Delete exactly one committed entry; no event cascade is possible."""
        _validate_text(entry_id, limits.ENTRY_ID_MAX_CHARS, "entry_id")
        target = self._entry_path(entry_id)
        with self._locked():
            if not target.exists():
                return False
            target.unlink()
            _fsync_directory(self._directory)
            self._count, self._bytes = self._scan_totals()
            return True

    @property
    def dead_letter_directory(self) -> Path:
        """Where evidence the backend refused is retained for an operator."""
        return self._directory.parent / f"{self._directory.name}-dead-letter"

    def dead_letter(self, entry_id: str, status_code: int) -> bool:
        """Retain a rejected entry outside the live queue instead of deleting it.

        A 422 means the backend refused this payload. Deleting it destroys the
        evidence and reports success, which is exactly how 41 real bed-exit
        events were lost in this deployment: the cause was an undeclared field,
        but the mechanism was this deletion. Fixing one undeclared field does
        not make the next one safe.

        The entry leaves the live queue so admission bounds still hold, and lands
        in a sibling directory that survives restart and is visible to an
        operator. It is never reported as acknowledged.
        """
        _validate_text(entry_id, limits.ENTRY_ID_MAX_CHARS, "entry_id")
        source = self._entry_path(entry_id)
        with self._locked():
            if not source.exists():
                return False
            destination_directory = self.dead_letter_directory
            created = not destination_directory.exists()
            destination_directory.mkdir(parents=True, exist_ok=True)
            if created:
                # The directory entry itself must survive a crash, or the
                # retained file has nowhere durable to live.
                _fsync_directory(destination_directory.parent)
            # No-clobber. `os.replace` silently overwrites, so re-admitting the
            # same entry id after an earlier refusal would destroy the first
            # retained copy -- reintroducing exactly the evidence loss this
            # directory exists to prevent. A monotonic suffix keeps every
            # distinct refusal.
            retained = sorted(
                path for path in destination_directory.iterdir() if path.is_file()
            )
            retained_bytes = sum(path.stat().st_size for path in retained)
            if (
                len(retained) >= MAX_DEAD_LETTERED_ENTRIES
                or retained_bytes + source.stat().st_size > MAX_DEAD_LETTERED_BYTES
            ):
                # Full. Refuse to retain rather than evicting: silently dropping
                # the oldest refused evidence to make room for the newest is the
                # deletion this directory exists to prevent, just slower. The
                # entry stays in the live queue where it is still counted and
                # still visible, and the operator must drain the retention area.
                return False
            # Name as "<status>.<ordinal>.<original>" so the original entry
            # filename is always recoverable by dropping exactly two leading
            # components. Appending the disambiguator to the END produced names
            # like "...json.1", and requeueing one of those wrote a file the
            # queue's own *.json scan cannot see -- evidence present on disk and
            # invisible to delivery, which is silent loss wearing a fix's
            # clothes.
            ordinal = 0
            destination = destination_directory / f"{status_code}.{ordinal}.{source.name}"
            while destination.exists():
                ordinal += 1
                destination = destination_directory / f"{status_code}.{ordinal}.{source.name}"
            # Order matters under power loss. The link must be durable BEFORE
            # the live copy is removed: unlinking first leaves a window where a
            # crash loses both names and the evidence is gone for good. Linking
            # first can at worst leave a duplicate, which the requeue path
            # already treats as byte-identical and idempotent.
            os.link(source, destination)
            _fsync_directory(destination_directory)
            source.unlink()
            _fsync_directory(self._directory)
            self._count, self._bytes = self._scan_totals()
            return True

    def requeue_dead_lettered(self, retained: Path) -> bool:
        """Return one retained entry to the live queue under the queue's own lock.

        Writing the file back directly would bypass every property this class
        provides: the exclusive lock, the capacity bounds, atomic publication,
        and byte-identical duplicate detection. An operator command repairing an
        evidence problem must not introduce a worse one.

        Returns False when the live queue cannot accept it, leaving the retained
        copy untouched so the operation is resumable and nothing is lost.
        """
        payload = retained.read_bytes()
        # "<status>.<ordinal>.<original>": drop exactly the two leading
        # components so the recovered name is the identity the queue admitted
        # the entry under, and is therefore visible to its own scan.
        _, _, remainder = retained.name.partition(".")
        _, _, original = remainder.partition(".")
        if not original or not original.endswith(_ENTRY_SUFFIX):
            raise ValueError(f"retained entry has no recoverable identity: {retained.name}")
        target = self._directory / original
        with self._locked():
            self._count, self._bytes = self._scan_totals()
            if target.exists():
                if target.read_bytes() == payload:
                    # The removal happens in the RETENTION directory, so that is
                    # what must be made durable. Fsyncing the live queue instead
                    # left the unlink unpersisted: a power loss here resurrects a
                    # refusal the operator had already cleared, and it blocks the
                    # cutover gate again.
                    retained.unlink()
                    _fsync_directory(retained.parent)
                    return True
                return False
            if self._count >= MAX_ACCEPTED_ENTRIES:
                return False
            if self._bytes + len(payload) > MAX_ACCEPTED_BYTES:
                return False
            temporary = self._directory / f".{uuid.uuid4().hex}.tmp"
            try:
                _write_durable(temporary, payload)
                os.replace(temporary, target)
                _fsync_directory(self._directory)
            finally:
                if temporary.exists():
                    temporary.unlink()
            retained.unlink()
            _fsync_directory(retained.parent)
            self._count += 1
            self._bytes += len(payload)
            return True

    def acknowledge_backend(self, entry_id: str, status_code: int) -> bool:
        """Delete entries the backend genuinely holds.

        409 means the backend already has this entry, so removing our copy is
        correct. 422 is a refusal and is handled by :meth:`dead_letter`; it must
        never reach here, because deleting refused evidence and calling it
        acknowledged is indistinguishable from delivering it.
        """
        if status_code not in {200, 201, 202, 204, 409}:
            return False
        return self.acknowledge(entry_id)

    def entries(self) -> Iterator[dict[str, object]]:
        with self._locked():
            paths = tuple(sorted(self._published_paths()))
        for path in paths:
            yield json.loads(path.read_bytes())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock, self._lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _try_locked(self) -> Iterator[bool]:
        if not self._thread_lock.acquire(blocking=False):
            yield False
            return
        try:
            with self._lock_path.open("a+b") as lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    yield False
                    return
                try:
                    yield True
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._thread_lock.release()

    def _published_paths(self) -> Iterator[Path]:
        yield from self._directory.glob(f"*{_ENTRY_SUFFIX}")

    def _scan_totals(self) -> tuple[int, int]:
        paths = tuple(self._published_paths())
        return len(paths), sum(path.stat().st_size for path in paths)

    def _remove_orphan_temps(self) -> None:
        orphans = tuple(self._directory.glob(".*.tmp"))
        for path in orphans:
            path.unlink()
        if orphans:
            _fsync_directory(self._directory)

    def _entry_path(self, entry_id: str) -> Path:
        return self._directory / f"{entry_id}{_ENTRY_SUFFIX}"


def _serialize(entry: DeliveryEntry) -> bytes:
    data = asdict(entry)
    if isinstance(entry, EventEntry):
        data["decision_trace_b64"] = base64.b64encode(data.pop("decision_trace")).decode("ascii")
        data["values_b64"] = base64.b64encode(data.pop("values")).decode("ascii")
    data["kind"] = entry.kind.value
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _keyed_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("ascii")).hexdigest()
    return f"{prefix}-{digest}"


def _validate_text(value: str, maximum: int, name: str) -> None:
    valid = (
        isinstance(value, str)
        and bool(value)
        and len(value) <= maximum
        and _SAFE_TEXT.fullmatch(value) is not None
    )
    if not valid:
        raise ValueError(f"{name} must be non-empty printable ASCII within its envelope limit")
    if name == "entry_id" and ("/" in value or "\\" in value):
        raise ValueError(f"{name} cannot contain a path separator")


def _validate_bytes(value: bytes, maximum: int, name: str) -> None:
    if not isinstance(value, bytes) or len(value) > maximum:
        raise ValueError(f"{name} exceeds its envelope limit")


def _write_durable(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "MAX_ACCEPTED_BYTES",
    "MAX_ACCEPTED_ENTRIES",
    "AdmissionFault",
    "AdmissionResult",
    "DeliveryEntry",
    "DeliveryQueue",
    "DeliveryQueueCapacitySnapshot",
    "EntryKind",
    "EventEntry",
    "SnapshotAttachmentEntry",
    "SnapshotDispositionEntry",
]
