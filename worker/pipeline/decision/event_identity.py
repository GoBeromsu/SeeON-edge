from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Protocol, TypeAlias, final, override
from uuid import uuid4

from pydantic import UUID4, BaseModel, ConfigDict, ValidationError

RETENTION_SEC: Final = 90 * 24 * 3600
MAX_JOURNAL_BYTES: Final = 16 * 1024 * 1024
Clock: TypeAlias = Callable[[], float]


@dataclass(frozen=True, slots=True)
class EventIdentityStoreError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"event identity journal {self.path}: {self.detail}"


class _PersistedIdentity(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source_key: str
    edge_event_id: UUID4
    recorded_at: float


class _LegacyIdentity(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source_key: str
    edge_event_id: UUID4


@final
class EventIdentityStore:
    def __init__(
        self,
        path: Path | None = None,
        *,
        clock: Clock | None = None,
        retention_sec: float = RETENTION_SEC,
        max_bytes: int = MAX_JOURNAL_BYTES,
    ) -> None:
        self._path = path
        self._clock: Clock = clock if clock is not None else _wall_clock
        self._retention_sec = retention_sec
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        now = self._clock()
        loaded = self._load(path, now)
        self._by_source_key = self._select_retained(loaded, now)
        if path is not None and path.exists() and loaded:
            self._rewrite(self._by_source_key)

    def resolve(self, source_key: str) -> str:
        with self._lock:
            persisted = self._by_source_key.get(source_key)
            if persisted is None:
                self._refresh_locked()
                persisted = self._by_source_key.get(source_key)
            if persisted is None:
                persisted = _PersistedIdentity(
                    source_key=source_key,
                    edge_event_id=uuid4(),
                    recorded_at=self._clock(),
                )
                updated = dict(self._by_source_key)
                updated[source_key] = persisted
                retained = self._select_retained(updated, self._clock())
                self._rewrite(retained)
                persisted = retained.get(source_key, persisted)
                self._by_source_key = retained
        return str(persisted.edge_event_id)

    def _refresh_locked(self) -> None:
        loaded = self._load(self._path, self._clock())
        for source_key, record in loaded.items():
            current = self._by_source_key.get(source_key)
            if current is None:
                self._by_source_key[source_key] = record

    def _select_retained(
        self, records: dict[str, _PersistedIdentity], now: float
    ) -> dict[str, _PersistedIdentity]:
        cutoff = now - self._retention_sec
        eligible = [
            record
            for record in records.values()
            if cutoff <= record.recorded_at <= now
        ]
        eligible.sort(key=lambda record: (record.recorded_at, record.source_key), reverse=True)
        retained: dict[str, _PersistedIdentity] = {}
        size = 0
        for record in eligible:
            encoded = _encoded_line(record)
            if size + len(encoded) > self._max_bytes:
                break
            retained[record.source_key] = record
            size += len(encoded)
        return retained

    def _rewrite(self, records: dict[str, _PersistedIdentity]) -> None:
        path = self._path
        if path is None:
            return
        ordered = sorted(
            records.values(), key=lambda record: (record.recorded_at, record.source_key)
        )
        payload = "".join(_encoded_line(record).decode("utf-8") for record in ordered)
        _atomic_replace(path, payload)

    @staticmethod
    def _load(path: Path | None, now: float) -> dict[str, _PersistedIdentity]:
        if path is None or not path.exists():
            return {}
        loaded: dict[str, _PersistedIdentity] = {}
        try:
            with path.open(encoding="utf-8") as journal:
                for line_number, line in enumerate(journal, start=1):
                    if line.strip() == "":
                        continue
                    parsed = _parse_line(line, now)
                    if parsed is None:
                        detail = f"malformed line {line_number}"
                        raise EventIdentityStoreError(path, detail)
                    previous = loaded.get(parsed.source_key)
                    if previous is not None and previous.edge_event_id != parsed.edge_event_id:
                        detail = f"conflicting source key at line {line_number}"
                        raise EventIdentityStoreError(path, detail)
                    loaded[parsed.source_key] = parsed
        except OSError as error:
            raise EventIdentityStoreError(path, str(error)) from error
        return loaded


def event_identity_path(camera_id: str, state_dir: Path) -> Path:
    camera_digest = hashlib.sha256(camera_id.encode()).hexdigest()
    return state_dir / "event-identities" / f"{camera_digest}.jsonl"


def _parse_line(line: str, now: float) -> _PersistedIdentity | None:
    stripped = line.strip()
    if stripped == "":
        return None
    try:
        record = _PersistedIdentity.model_validate_json(stripped)
    except ValidationError:
        try:
            legacy = _LegacyIdentity.model_validate_json(stripped)
        except ValidationError:
            return None
        return _PersistedIdentity(
            source_key=legacy.source_key,
            edge_event_id=legacy.edge_event_id,
            recorded_at=now,
        )
    if record.recorded_at > now:
        return None
    return record


def _encoded_line(record: _PersistedIdentity) -> bytes:
    return (record.model_dump_json() + "\n").encode("utf-8")


def _atomic_replace(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            _write_text(handle, payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        if temporary.exists():
            temporary.unlink()
        raise EventIdentityStoreError(path, str(error)) from error
    finally:
        if temporary.exists():
            temporary.unlink()


class _TextWriter(Protocol):
    def write(self, s: str, /) -> int: ...


def _write_text(handle: _TextWriter, payload: str) -> None:
    _ = handle.write(payload)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _wall_clock() -> float:
    from time import time

    return time()


__all__ = [
    "MAX_JOURNAL_BYTES",
    "RETENTION_SEC",
    "EventIdentityStore",
    "EventIdentityStoreError",
    "event_identity_path",
]
