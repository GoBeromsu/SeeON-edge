from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, final, override
from uuid import uuid4

from pydantic import UUID4, BaseModel, ConfigDict, ValidationError


@dataclass(slots=True)
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


@final
class EventIdentityStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._by_source_key = self._load(path)

    def resolve(self, source_key: str) -> str:
        with self._lock:
            persisted = self._by_source_key.get(source_key)
            if persisted is None:
                self._refresh()
                persisted = self._by_source_key.get(source_key)
            if persisted is None:
                persisted = _PersistedIdentity(
                    source_key=source_key,
                    edge_event_id=uuid4(),
                )
                self._append(persisted)
                self._by_source_key[source_key] = persisted
        return str(persisted.edge_event_id)

    def _refresh(self) -> None:
        path = self._path
        if path is not None and path.exists():
            self._by_source_key.update(self._load(path))

    @staticmethod
    def _load(path: Path | None) -> dict[str, _PersistedIdentity]:
        if path is None or not path.exists():
            return {}
        loaded: dict[str, _PersistedIdentity] = {}
        try:
            with path.open(encoding="utf-8") as journal:
                for line_number, line in enumerate(journal, start=1):
                    if line.strip() == "":
                        continue
                    record = _PersistedIdentity.model_validate_json(line)
                    previous = loaded.get(record.source_key)
                    if previous is not None and previous != record:
                        detail = f"conflicting source key at line {line_number}"
                        raise EventIdentityStoreError(path, detail)
                    loaded[record.source_key] = record
        except (OSError, ValidationError) as error:
            raise EventIdentityStoreError(path, str(error)) from error
        return loaded

    def _append(self, record: _PersistedIdentity) -> None:
        path = self._path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as journal:
                _ = journal.write(record.model_dump_json())
                _ = journal.write("\n")
                journal.flush()
                os.fsync(journal.fileno())
        except OSError as error:
            raise EventIdentityStoreError(path, str(error)) from error


def event_identity_path(camera_id: str, state_dir: Path) -> Path:
    camera_digest = hashlib.sha256(camera_id.encode()).hexdigest()
    return state_dir / "event-identities" / f"{camera_digest}.jsonl"


__all__ = [
    "EventIdentityStore",
    "EventIdentityStoreError",
    "event_identity_path",
]
