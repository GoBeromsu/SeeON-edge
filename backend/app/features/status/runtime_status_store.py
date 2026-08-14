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
from typing import Any

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from backend.app.shared.state_dir import resolve_state_dir

DEFAULT_RUNTIME_STATUS_STALE_AFTER_SEC: float = 15.0
logger = logging.getLogger(__name__)

_CREATE_RUNTIME_LATENCY_TABLE = (
    "CREATE TABLE IF NOT EXISTS runtime_latency (facility_id TEXT PRIMARY KEY, "
    "payload_json TEXT NOT NULL) STRICT"
)


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
    cameras: tuple[dict[str, Any], ...]
    clip_recorder: dict[str, Any]
    clip_export: dict[str, Any] | None
    gpu: dict[str, Any] | None
    worker: dict[str, Any] | None


@dataclass(slots=True)
class RuntimeStatusStore:
    """Keep one ordered runtime-status snapshot per facility."""

    stale_after_sec: float = DEFAULT_RUNTIME_STATUS_STALE_AFTER_SEC
    _snapshots: dict[str, RuntimeStatusSnapshot] = field(default_factory=dict)
    _latest_generation: dict[str, int] = field(default_factory=dict)
    _latency_by_facility: dict[str, dict[str, Any]] = field(default_factory=dict)
    latency_state_path: Path = field(
        default_factory=lambda: resolve_state_dir("ml-api") / "catalog.sqlite3"
    )
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
        requested_generation = payload.get("generation")
        seq = int(payload["seq"])
        cameras = payload["cameras"]
        clip_recorder = payload["clip_recorder"]
        clip_export = payload.get("clip_export")
        gpu = payload.get("gpu")
        worker = payload.get("worker")
        if not isinstance(cameras, list) or not isinstance(clip_recorder, dict):
            raise TypeError("runtime status payload has invalid telemetry fields")
        if clip_export is not None and not isinstance(clip_export, dict):
            raise TypeError("runtime status payload has invalid clip_export")
        if gpu is not None and not isinstance(gpu, dict):
            raise TypeError("runtime status payload has invalid gpu")
        if worker is not None and not isinstance(worker, dict):
            raise TypeError("runtime status payload has invalid worker")
        stamped = time() if received_at is None else received_at
        with self._lock:
            previous = self._snapshots.get(facility_id)
            latest_generation = self._latest_generation.get(facility_id, 0)

            if requested_generation is None:
                generation = latest_generation + 1
            else:
                generation = int(requested_generation)
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
            )
            self._latest_generation[facility_id] = max(latest_generation, generation)
            return RuntimeStatusRecordResult(True, generation)

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        """Return API-stamped, facility-keyed telemetry with derived staleness."""
        current = time() if now is None else now
        with self._lock:
            facilities = {
                facility_id: {
                    "facility_id": status.facility_id,
                    "generation": status.generation,
                    "seq": status.seq,
                    "received_at": status.received_at,
                    "stale": current - status.received_at > self.stale_after_sec,
                    "cameras": deepcopy(list(status.cameras)),
                    "clip_recorder": deepcopy(status.clip_recorder),
                    "clip_export": deepcopy(status.clip_export),
                    "gpu": deepcopy(status.gpu),
                    "worker": deepcopy(status.worker),
                    "latency": deepcopy(self._latency_for_facility(facility_id)),
                }
                for facility_id, status in sorted(self._snapshots.items())
            }
        return {
            "facilities": facilities,
            "stale_after_sec": self.stale_after_sec,
        }

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
            samples = int(previous.get("first_attempt_samples", 0)) + 1
            max_sec = max(float(previous.get("max_sec", elapsed)), elapsed)
            self._latency_by_facility[facility_id] = {
                **previous,
                "first_attempt_samples": samples,
                "max_sec": max_sec,
                "since_sec": float(previous.get("since_sec", stamped)),
            }
            self._persist_latency()

    def _latency_for_facility(self, facility_id: str) -> dict[str, Any] | None:
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
        latency_by_facility: dict[str, dict[str, Any]] = {}
        for facility_id, encoded in rows:
            try:
                item = json.loads(encoded)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(facility_id, str) and isinstance(item, dict):
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
    state = app.state  # type: ignore[attr-defined]
    store = getattr(state, "runtime_status_store", None)
    if not isinstance(store, RuntimeStatusStore):
        store = RuntimeStatusStore()
        state.runtime_status_store = store
    return store


__all__ = [
    "DEFAULT_RUNTIME_STATUS_STALE_AFTER_SEC",
    "RuntimeStatusRecordResult",
    "RuntimeStatusSnapshot",
    "RuntimeStatusStore",
    "get_runtime_status_store",
]
