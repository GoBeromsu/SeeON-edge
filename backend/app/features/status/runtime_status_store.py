"""API-owned worker runtime-status snapshots.

The worker publishes telemetry through the relay HTTP boundary. This store owns
only the API-local, latest snapshot for each facility; it never reads worker
runtime state directly.

Latency history persists in the ``runtime_latency`` table (one row per
facility) of the shared ``catalog.sqlite3`` database (see
``backend/app/features/clips/catalog.py``). This module bootstraps its own
table independently via an idempotent ``CREATE TABLE IF NOT EXISTS``
(identical statement text to ``catalog.py``'s ``_V3_TABLE_STATEMENTS``)
rather than depending on ``CatalogStore``, and never touches
``PRAGMA user_version`` (that stays exclusively ``CatalogStore``'s
responsibility).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import TypeAlias

from pydantic import TypeAdapter, ValidationError

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.status.detection_health import (
    DetectionHealth,
    accept_detection_sample,
    detection_health_fields,
    parse_detection_counters,
)
from backend.app.shared.sqlite_bootstrap import connect_catalog_store

DEFAULT_RUNTIME_STATUS_STALE_AFTER_SEC: float = 15.0
logger = logging.getLogger(__name__)

_CREATE_RUNTIME_LATENCY_TABLE = (
    "CREATE TABLE IF NOT EXISTS runtime_latency (facility_id TEXT PRIMARY KEY, "
    "payload_json TEXT NOT NULL) STRICT"
)

JsonObject: TypeAlias = dict[str, object]
LatencyRow: TypeAlias = tuple[str, str]
_LATENCY_ROW: TypeAdapter[LatencyRow] = TypeAdapter(LatencyRow)


@dataclass(frozen=True, slots=True)
class RuntimeStatusRecordResult:
    accepted: bool
    generation: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStatusSnapshot:
    facility_id: str
    generation: int
    seq: int
    received_at: float
    cameras: tuple[JsonObject, ...]
    clip_recorder: JsonObject
    clip_export: JsonObject | None
    gpu: JsonObject | None
    worker: JsonObject | None
    delivery_queue: JsonObject | None


@dataclass(slots=True)
class RuntimeStatusStore:
    """Keep one ordered runtime-status snapshot per facility."""

    stale_after_sec: float = DEFAULT_RUNTIME_STATUS_STALE_AFTER_SEC
    _snapshots: dict[str, RuntimeStatusSnapshot] = field(default_factory=dict)
    _latest_generation: dict[str, int] = field(default_factory=dict)
    _latency_by_facility: dict[str, JsonObject] = field(default_factory=dict)
    _detection_health: dict[tuple[str, str], DetectionHealth] = field(default_factory=dict)
    latency_state_path: Path = field(default_factory=lambda: EDGE_DATABASE_PATH)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _latency_loaded: bool = field(default=False, init=False, repr=False)
    _latency_load_error: str | None = field(default=None, init=False, repr=False)
    _connection: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def record(
        self,
        payload: Mapping[str, object],
        *,
        received_at: float | None = None,
    ) -> RuntimeStatusRecordResult:
        """Record a payload after auth and facility validation.

        A missing generation always starts a new generation. Within a generation
        sequence numbers may repeat for retry delivery, but may not go backwards.
        """
        facility_id = str(payload["facility_id"])
        seq = _require_int(payload["seq"], field="seq")
        cameras_raw = payload["cameras"]
        clip_recorder_raw = payload["clip_recorder"]
        clip_export_raw = payload.get("clip_export")
        gpu_raw = payload.get("gpu")
        worker_raw = payload.get("worker")
        delivery_queue_raw = payload.get("delivery_queue")
        if not isinstance(cameras_raw, list) or not isinstance(clip_recorder_raw, dict):
            raise TypeError("runtime status payload has invalid telemetry fields")
        if clip_export_raw is not None and not isinstance(clip_export_raw, dict):
            raise TypeError("runtime status payload has invalid clip_export")
        if gpu_raw is not None and not isinstance(gpu_raw, dict):
            raise TypeError("runtime status payload has invalid gpu")
        if worker_raw is not None and not isinstance(worker_raw, dict):
            raise TypeError("runtime status payload has invalid worker")
        if delivery_queue_raw is not None and not isinstance(delivery_queue_raw, dict):
            raise TypeError("runtime status payload has invalid delivery_queue")
        cameras = _object_dicts(cameras_raw)
        clip_recorder = _object_dict(clip_recorder_raw)
        clip_export = None if clip_export_raw is None else _object_dict(clip_export_raw)
        gpu = None if gpu_raw is None else _object_dict(gpu_raw)
        worker = None if worker_raw is None else _object_dict(worker_raw)
        delivery_queue = (
            None if delivery_queue_raw is None else _object_dict(delivery_queue_raw)
        )
        requested_generation = _optional_int(payload.get("generation"), field="generation")
        stamped = time() if received_at is None else received_at
        with self._lock:
            previous = self._snapshots.get(facility_id)
            latest_generation = self._latest_generation.get(facility_id, 0)

            if requested_generation is None:
                generation = latest_generation + 1
            else:
                generation = requested_generation
                if generation < latest_generation:
                    return RuntimeStatusRecordResult(False, latest_generation, "old_generation")

            if previous is not None and generation == previous.generation and seq < previous.seq:
                return RuntimeStatusRecordResult(False, previous.generation, "old_seq")

            self._snapshots[facility_id] = RuntimeStatusSnapshot(
                facility_id=facility_id,
                generation=generation,
                seq=seq,
                received_at=stamped,
                cameras=tuple(deepcopy(cameras)),
                clip_recorder=deepcopy(clip_recorder),
                clip_export=deepcopy(clip_export),
                gpu=deepcopy(gpu),
                worker=deepcopy(worker),
                delivery_queue=deepcopy(delivery_queue),
            )
            self._latest_generation[facility_id] = max(latest_generation, generation)
            self._accept_detection_samples(facility_id, cameras, accepted_at=stamped)
            return RuntimeStatusRecordResult(True, generation)

    def snapshot(self, *, now: float | None = None) -> dict[str, object]:
        """Return API-stamped, facility-keyed telemetry with derived staleness."""
        current = time() if now is None else now
        with self._lock:
            self._prune_detection_health()
            facilities: dict[str, JsonObject] = {}
            for facility_id, status in sorted(self._snapshots.items()):
                stale = current - status.received_at > self.stale_after_sec
                facilities[facility_id] = {
                    "facility_id": status.facility_id,
                    "generation": status.generation,
                    "seq": status.seq,
                    "received_at": status.received_at,
                    "stale": stale,
                    "cameras": self._cameras_with_detection_health(
                        facility_id,
                        status.cameras,
                        now=current,
                        stale=stale,
                    ),
                    "clip_recorder": deepcopy(status.clip_recorder),
                    "clip_export": deepcopy(status.clip_export),
                    "gpu": deepcopy(status.gpu),
                    "worker": deepcopy(status.worker),
                    "delivery_queue": deepcopy(status.delivery_queue),
                    "latency": deepcopy(self._latency_for_facility(facility_id)),
                }
        return {
            "facilities": facilities,
            "stale_after_sec": self.stale_after_sec,
        }

    def _accept_detection_samples(
        self,
        facility_id: str,
        cameras: list[JsonObject],
        *,
        accepted_at: float,
    ) -> None:
        active: set[tuple[str, str]] = set()
        for camera in cameras:
            camera_id = camera.get("camera_id")
            if not isinstance(camera_id, str) or not camera_id:
                continue
            key = (facility_id, camera_id)
            counters = parse_detection_counters(camera.get("detection"))
            if counters is None:
                _ = self._detection_health.pop(key, None)
                continue
            active.add(key)
            self._detection_health[key] = accept_detection_sample(
                self._detection_health.get(key),
                counters,
                accepted_at=accepted_at,
            )
        for key in tuple(self._detection_health):
            if key[0] == facility_id and key not in active:
                del self._detection_health[key]

    def _prune_detection_health(self) -> None:
        active = {
            (facility_id, camera_id)
            for facility_id, status in self._snapshots.items()
            for camera in status.cameras
            if isinstance((camera_id := camera.get("camera_id")), str)
            and parse_detection_counters(camera.get("detection")) is not None
        }
        for key in tuple(self._detection_health):
            if key not in active:
                del self._detection_health[key]

    def _cameras_with_detection_health(
        self,
        facility_id: str,
        cameras: tuple[JsonObject, ...],
        *,
        now: float,
        stale: bool,
    ) -> list[JsonObject]:
        projected: list[JsonObject] = []
        for camera in cameras:
            row = deepcopy(camera)
            camera_id = row.get("camera_id")
            raw_detection = row.get("detection")
            missing = parse_detection_counters(raw_detection) is None
            health = (
                self._detection_health.get((facility_id, camera_id))
                if isinstance(camera_id, str)
                else None
            )
            derived = detection_health_fields(
                health,
                now=now,
                stale=stale,
                missing=missing,
            )
            if isinstance(raw_detection, dict):
                row["detection"] = {**raw_detection, **derived}
            else:
                row["detection"] = derived
            projected.append(row)
        return projected

    def record_latency(
        self, facility_id: str, detected_at: str, *, received_at: float | None = None
    ) -> None:
        try:
            detected = datetime.fromisoformat(detected_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return
        stamped = time() if received_at is None else received_at
        elapsed = stamped - detected
        with self._lock:
            if not self._load_latency():
                return
            previous = self._latency_by_facility.get(facility_id, {})
            if elapsed < 0 or not math.isfinite(elapsed):
                self._latency_by_facility[facility_id] = {
                    **previous,
                    "last_error": "invalid_elapsed",
                }
                self._persist_latency()
                return
            samples = _as_int(previous.get("first_attempt_samples"), default=0) + 1
            max_sec = max(_as_float(previous.get("max_sec"), default=elapsed), elapsed)
            self._latency_by_facility[facility_id] = {
                **previous,
                "first_attempt_samples": samples,
                "max_sec": max_sec,
                "since_sec": _as_float(previous.get("since_sec"), default=stamped),
            }
            self._persist_latency()

    def _latency_for_facility(self, facility_id: str) -> JsonObject | None:
        if not self._load_latency():
            return {"state": "unknown", "reason": self._latency_load_error}
        latency = self._latency_by_facility.get(facility_id)
        return None if latency is None else deepcopy(latency)

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = connect_catalog_store(
                self.latency_state_path, (_CREATE_RUNTIME_LATENCY_TABLE,)
            )
        return self._connection

    def _load_latency(self) -> bool:
        if self._latency_loaded:
            return True
        try:
            connection = self._connect()
            rows = connection.execute(
                "SELECT facility_id, payload_json FROM runtime_latency"
            ).fetchall()
        except (OSError, sqlite3.Error) as exc:
            self._latency_load_error = exc.__class__.__name__
            return False
        latency_by_facility: dict[str, JsonObject] = {}
        for row in rows:
            parsed = _parse_latency_row(row)
            if parsed is None:
                continue
            facility_id, item = parsed
            latency_by_facility[facility_id] = item
        self._latency_by_facility = latency_by_facility
        self._latency_loaded = True
        return True

    def _persist_latency(self) -> None:
        """Best-effort persist; an unwritable state dir must not crash the caller.

        Mirrors ``CatalogStore.get_catalog_store``'s graceful-degradation
        pattern (``backend/app/features/clips/catalog.py``): a store that
        cannot durably persist still functions in-memory for the process
        lifetime, it just loses latency history across restarts.
        """
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("DELETE FROM runtime_latency")
                for facility_id, item in self._latency_by_facility.items():
                    encoded = json.dumps(
                        item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    connection.execute(
                        "INSERT INTO runtime_latency (facility_id, payload_json) VALUES (?, ?)",
                        (facility_id, encoded),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        except (OSError, sqlite3.Error) as exc:
            message = f"runtime latency store unavailable at {self.latency_state_path}: {exc}"
            logger.warning(message)


def get_runtime_status_store(app: object) -> RuntimeStatusStore:
    """Return the app-owned runtime store, creating it for no-lifespan tests."""
    state = getattr(app, "state", None)
    if state is None:
        raise TypeError("app has no state")
    store = getattr(state, "runtime_status_store", None)
    if not isinstance(store, RuntimeStatusStore):
        store = RuntimeStatusStore()
        _assign_state_attr(state, "runtime_status_store", store)
    return store


def _assign_state_attr(state: object, name: str, value: object) -> None:
    """Write a dynamic app.state attribute without untyped attribute access."""
    inner = getattr(state, "_state", None)
    if isinstance(inner, dict):
        inner[name] = value
        return
    namespace = getattr(state, "__dict__", None)
    if isinstance(namespace, dict):
        namespace[name] = value
        return
    raise TypeError(f"app state cannot store {name!r}")


def _parse_latency_row(row: object) -> tuple[str, JsonObject] | None:
    try:
        facility_id, encoded = _LATENCY_ROW.validate_python(row)
    except ValidationError:
        return None
    try:
        item = json.loads(encoded)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(item, dict):
        return None
    return facility_id, {str(key): value for key, value in item.items()}


def _require_int(value: object, *, field: str) -> int:
    parsed = _optional_int(value, field=field)
    if parsed is None:
        raise TypeError(f"runtime status payload has invalid {field}")
    return parsed


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"runtime status payload has invalid {field}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise TypeError(f"runtime status payload has invalid {field}") from exc
    raise TypeError(f"runtime status payload has invalid {field}")


def _object_dicts(values: list[object]) -> list[JsonObject]:
    cameras: list[JsonObject] = []
    for item in values:
        if not isinstance(item, dict):
            raise TypeError("runtime status payload has invalid telemetry fields")
        cameras.append(_object_dict(item))
    return cameras


def _object_dict(value: Mapping[object, object]) -> JsonObject:
    return {str(field_key): field_value for field_key, field_value in value.items()}


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_float(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


__all__ = [
    "DEFAULT_RUNTIME_STATUS_STALE_AFTER_SEC",
    "RuntimeStatusRecordResult",
    "RuntimeStatusSnapshot",
    "RuntimeStatusStore",
    "get_runtime_status_store",
]
