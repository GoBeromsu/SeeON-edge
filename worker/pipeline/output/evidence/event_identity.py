"""Deterministic in-envelope event identity without an on-disk journal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from worker.pipeline.output.evidence.event_payload import (
    MutableWorkerEventPayload,
    WorkerEventPayload,
)


@dataclass(frozen=True, slots=True)
class EventIdentity:
    edge_event_id: str
    detected_at: str


class EventIdentityStore:
    """Compatibility name for the stateless identity derivation seam.

    Identity is carried by the delivery envelope.  The optional constructor path
    is intentionally ignored: no replay journal is created or read.
    """

    def __init__(self, path: Path | None = None) -> None:
        del path

    def enrich(
        self,
        event: WorkerEventPayload,
        facility_id: str,
        camera_id: str,
    ) -> MutableWorkerEventPayload:
        detected_at = _detected_at(event, facility_id, camera_id)
        edge_event_id = _event_id(event, facility_id, camera_id, detected_at)
        enriched: MutableWorkerEventPayload = {
            key: dict(value)
            if key in {"audit", "evidence", "snapshot"} and isinstance(value, dict)
            else value
            for key, value in event.items()
        }
        enriched["edge_event_id"] = edge_event_id
        enriched["detected_at"] = detected_at
        return enriched


def event_identity_path(camera_id: str, state_dir: Path) -> Path:
    """Removed journal path retained only for callers during composition cutover."""
    del camera_id, state_dir
    raise RuntimeError("event identity journals are no longer supported")


def _event_id(event: WorkerEventPayload, facility_id: str, camera_id: str, detected_at: str) -> str:
    supplied = event.get("edge_event_id")
    if isinstance(supplied, str):
        try:
            parsed = UUID(supplied)
        except ValueError:
            pass
        else:
            if parsed.version == 4 and parsed.variant == "specified in RFC 4122":
                return str(parsed)
    source = json.dumps(
        [
            facility_id,
            camera_id,
            event.get("event_type", ""),
            event.get("idempotency_key", event.get("event_id", "")),
            detected_at,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    if source:
        raw = bytearray(hashlib.sha256(source).digest()[:16])
        raw[6] = (raw[6] & 0x0F) | 0x40
        raw[8] = (raw[8] & 0x3F) | 0x80
        return str(UUID(bytes=bytes(raw)))
    return str(uuid4())


def _detected_at(event: WorkerEventPayload, facility_id: str, camera_id: str) -> str:
    supplied = event.get("detected_at")
    if isinstance(supplied, str) and supplied:
        try:
            parsed = datetime.fromisoformat(supplied.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            if parsed.tzinfo is not None:
                return (
                    parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                )
    time_sec = event.get("time_sec")
    if isinstance(time_sec, int | float):
        return (
            datetime.fromtimestamp(float(time_sec), UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    material = json.dumps(
        [facility_id, camera_id, event.get("event_type", ""), event.get("event_id", "")],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    seconds = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return (
        datetime.fromtimestamp(seconds, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


__all__ = ["EventIdentity", "EventIdentityStore", "event_identity_path"]
