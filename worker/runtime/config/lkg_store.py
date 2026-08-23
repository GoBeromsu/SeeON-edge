from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from pydantic import JsonValue

from worker.runtime.config.restart import RestartDirective
from worker.runtime.state_dir import resolve_state_dir

CONFIG_HISTORY_RETENTION_COUNT: Final = 50
WORKER_STATE_DB_FILENAME: Final = "config-lkg"
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class StoredConfigPayload:
    payload: JsonObject
    directive: RestartDirective
    registry_version: int


class WorkerConfigLkgStore:
    """A bounded, verified filesystem cache of the last accepted relay config."""

    def __init__(self, database_path: Path | None = None, *, state_dir: Path | None = None) -> None:
        root = resolve_state_dir() if state_dir is None else state_dir
        self.directory = (
            root / "config-lkg"
            if database_path is None
            else (
                database_path
                if database_path.name.endswith("config-lkg")
                else database_path.parent / f"{database_path.name}.config-lkg"
            )
        )
        self.database_path = self.directory
        self._lock = threading.Lock()

    def save(self, payload: JsonObject, directive: RestartDirective) -> bool:
        record = {
            "generation": directive.generation,
            "config_version": directive.version,
            "registry_version": _registry_version(payload),
            "payload": payload,
        }
        try:
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as error:
            print(f"worker config LKG payload unavailable: {error}", file=sys.stderr)
            return False
        with self._lock:
            try:
                with self._locked():
                    current = self._load_current()
                    candidate = (directive, record["registry_version"])
                    if current is not None and candidate < (
                        current.directive,
                        current.registry_version,
                    ):
                        return False
                    revision = self._revision_path(directive, record["registry_version"])
                    _write_atomic(revision, encoded)
                    _write_atomic(self._current_path, encoded)
                    self._prune_revisions()
            except (OSError, ValueError, json.JSONDecodeError) as error:
                _report_unavailable(self.directory, error)
                return False
        return True

    def load(self) -> StoredConfigPayload | None:
        with self._lock:
            if not self.directory.is_dir():
                if _blocked_parent(self.directory):
                    _report_unavailable(self.directory, NotADirectoryError(self.directory))
                return None
            try:
                with self._locked():
                    return self._load_current()
            except (OSError, ValueError, json.JSONDecodeError) as error:
                _report_unavailable(self.directory, error)
                return None

    def clear(self) -> bool:
        with self._lock:
            if not self.directory.is_dir():
                if _blocked_parent(self.directory):
                    _report_unavailable(self.directory, NotADirectoryError(self.directory))
                    return False
                return True
            try:
                with self._locked():
                    try:
                        self._current_path.unlink()
                    except FileNotFoundError:
                        return True
                    _fsync_directory(self.directory)
            except OSError as error:
                _report_unavailable(self.directory, error)
                return False
        return True

    @property
    def _current_path(self) -> Path:
        return self.directory / "current.json"

    @property
    def _revision_directory(self) -> Path:
        return self.directory / "revisions"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.directory / ".lock"
        lock_path.touch(mode=0o600, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_current(self) -> StoredConfigPayload | None:
        try:
            contents = self._current_path.read_bytes()
        except FileNotFoundError:
            return None
        return _decode_record(contents)

    def _revision_path(self, directive: RestartDirective, registry_version: int) -> Path:
        return self._revision_directory / (
            f"{directive.generation:020d}-{directive.version:020d}-{registry_version:020d}.json"
        )

    def _prune_revisions(self) -> None:
        paths = sorted(self._revision_directory.glob("*.json"), reverse=True)
        for path in paths[CONFIG_HISTORY_RETENTION_COUNT:]:
            path.unlink()
        _fsync_directory(self._revision_directory)


def _decode_record(contents: bytes) -> StoredConfigPayload:
    value = json.loads(contents)
    if not isinstance(value, dict):
        raise TypeError("cached config record is not an object")
    payload = value.get("payload")
    generation = value.get("generation")
    version = value.get("config_version")
    registry_version = value.get("registry_version")
    if (
        not isinstance(payload, dict)
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or not isinstance(version, int)
        or isinstance(version, bool)
        or not isinstance(registry_version, int)
        or isinstance(registry_version, bool)
    ):
        raise TypeError("cached config record has invalid fields")
    return StoredConfigPayload(
        payload=payload,
        directive=RestartDirective(generation=generation, version=version),
        registry_version=registry_version,
    )


def _write_atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _report_unavailable(directory: Path, error: Exception) -> None:
    print(f"worker config LKG store unavailable at {directory}: {error}", file=sys.stderr)


def _blocked_parent(path: Path) -> bool:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.exists() and not candidate.is_dir()


def _registry_version(payload: JsonObject) -> int:
    value = payload.get("registry_version", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = [
    "CONFIG_HISTORY_RETENTION_COUNT",
    "WORKER_STATE_DB_FILENAME",
    "JsonObject",
    "JsonValue",
    "StoredConfigPayload",
    "WorkerConfigLkgStore",
]
