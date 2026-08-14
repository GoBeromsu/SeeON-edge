from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar, final
from uuid import uuid4

from pydantic import (
    UUID4,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
)
from pydantic_core import PydanticCustomError
from typing_extensions import override

from worker.pipeline.output.evidence.event_payload import (
    MutableWorkerEventPayload,
    WorkerEventPayload,
)


@dataclass(slots=True)
class EventIdentityStoreError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"event identity store {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class EventIdentity:
    edge_event_id: str
    detected_at: str


class _PersistedIdentity(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source_key: str | None
    edge_event_id: UUID4
    detected_at: AwareDatetime

    @field_validator("detected_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise PydanticCustomError("utc_timestamp", "detected_at must be UTC")
        return value.astimezone(UTC)

    def as_identity(self) -> EventIdentity:
        return EventIdentity(
            edge_event_id=str(self.edge_event_id),
            detected_at=_rfc3339_z(self.detected_at),
        )


@final
class EventIdentityStore:
    """Append journal assigning one durable identity per upstream incident."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._by_source_key = self._load(path)

    def enrich(
        self,
        event: WorkerEventPayload,
        facility_id: str,
        camera_id: str,
    ) -> MutableWorkerEventPayload:
        source_key = _source_key(event, facility_id, camera_id)
        with self._lock:
            persisted = None if source_key is None else self._by_source_key.get(source_key)
            if persisted is None:
                persisted = _PersistedIdentity(
                    source_key=source_key,
                    edge_event_id=uuid4(),
                    detected_at=datetime.now(UTC),
                )
                self._append(persisted)
                if source_key is not None:
                    self._by_source_key[source_key] = persisted
        identity = persisted.as_identity()
        enriched: MutableWorkerEventPayload = {}
        if "event_type" in event:
            enriched["event_type"] = event["event_type"]
        if "probability" in event:
            enriched["probability"] = event["probability"]
        if "confidence" in event:
            enriched["confidence"] = event["confidence"]
        if "camera_id" in event:
            enriched["camera_id"] = event["camera_id"]
        if "facility_id" in event:
            enriched["facility_id"] = event["facility_id"]
        if "domain" in event:
            enriched["domain"] = event["domain"]
        if "identity" in event:
            enriched["identity"] = event["identity"]
        if "time_sec" in event:
            enriched["time_sec"] = event["time_sec"]
        if "person_id" in event:
            enriched["person_id"] = event["person_id"]
        if "bed_id" in event:
            enriched["bed_id"] = event["bed_id"]
        if "idempotency_key" in event:
            enriched["idempotency_key"] = event["idempotency_key"]
        if "event_id" in event:
            enriched["event_id"] = event["event_id"]
        if "clip_id" in event:
            enriched["clip_id"] = event["clip_id"]
        if "evidence" in event:
            enriched["evidence"] = dict(event["evidence"])
        if "audit" in event:
            enriched["audit"] = dict(event["audit"])
        if "snapshot_jpeg" in event:
            enriched["snapshot_jpeg"] = event["snapshot_jpeg"]
        if "snapshot" in event:
            enriched["snapshot"] = dict(event["snapshot"])
        enriched["edge_event_id"] = identity.edge_event_id
        enriched["detected_at"] = identity.detected_at
        return enriched

    def _load(self, path: Path | None) -> dict[str, _PersistedIdentity]:
        if path is None or not path.exists():
            return {}
        loaded: dict[str, _PersistedIdentity] = {}
        try:
            with path.open(encoding="utf-8") as journal:
                for line_number, line in enumerate(journal, start=1):
                    if line.strip() == "":
                        continue
                    record = _PersistedIdentity.model_validate_json(line)
                    if record.source_key is None:
                        continue
                    previous = loaded.get(record.source_key)
                    if previous is not None and previous != record:
                        raise EventIdentityStoreError(
                            path,
                            f"conflicting source key at line {line_number}",
                        )
                    loaded[record.source_key] = record
        except (OSError, ValidationError) as exc:
            raise EventIdentityStoreError(path, str(exc)) from exc
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
        except OSError as exc:
            raise EventIdentityStoreError(path, str(exc)) from exc


def event_identity_path(camera_id: str, state_dir: Path) -> Path:
    camera_digest = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()
    return state_dir / "event-identities" / f"{camera_digest}.jsonl"


def _source_key(event: WorkerEventPayload, facility_id: str, camera_id: str) -> str | None:
    source_name = "idempotency_key"
    source_value = event.get(source_name)
    if source_value is None or str(source_value) == "":
        source_name = "event_id"
        source_value = event.get(source_name)
    if source_value is None or str(source_value) == "":
        return None
    return json.dumps(
        [
            facility_id,
            camera_id,
            str(event.get("event_type", "")),
            source_name,
            str(source_value),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _rfc3339_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "EventIdentity",
    "EventIdentityStore",
    "EventIdentityStoreError",
    "event_identity_path",
]
