"""API-owned worker runtime-status snapshots.

The worker publishes telemetry through the relay HTTP boundary. This store owns
only the API-local, latest snapshot for each facility; it never reads worker
runtime state directly. Latency and status are latest-only memory. Restart
forgets every observation. Missing is explicit, never a fabricated zero.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import TypeAlias

from backend.app.features.status.detection_health import (
    DetectionHealth,
    parse_detection_counters,
)
from backend.app.features.status.runtime_status_support import (
    JsonObject,
    LatestLatency,
    accept_camera_samples,
    object_dict,
    object_dicts,
    optional_int,
    project_cameras,
    prune_health,
    require_int,
)

DEFAULT_RUNTIME_STATUS_STALE_AFTER_SEC: float = 15.0
Clock: TypeAlias = Callable[[], float]


def _system_clock() -> float:
    return time()


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
    clock: Clock = field(default=_system_clock)
    _snapshots: dict[str, RuntimeStatusSnapshot] = field(default_factory=dict)
    _latest_generation: dict[str, int] = field(default_factory=dict)
    _latency: LatestLatency = field(default_factory=LatestLatency)
    _detection_health: dict[tuple[str, str], DetectionHealth] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

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
        seq = require_int(payload["seq"], field="seq")
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
        cameras = object_dicts(cameras_raw)
        clip_recorder = object_dict(clip_recorder_raw)
        clip_export = None if clip_export_raw is None else object_dict(clip_export_raw)
        gpu = None if gpu_raw is None else object_dict(gpu_raw)
        worker = None if worker_raw is None else object_dict(worker_raw)
        delivery_queue = (
            None if delivery_queue_raw is None else object_dict(delivery_queue_raw)
        )
        requested_generation = optional_int(payload.get("generation"), field="generation")
        with self._lock:
            now = self.clock()
            stamped = now if received_at is None else received_at
            if stamped > now:
                return RuntimeStatusRecordResult(False, 0, "future_timestamp")
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
            accept_camera_samples(
                self._detection_health, facility_id, cameras, accepted_at=stamped
            )
            return RuntimeStatusRecordResult(True, generation)

    def snapshot(self, *, now: float | None = None) -> JsonObject:
        """Return API-stamped, facility-keyed telemetry with derived staleness."""
        with self._lock:
            current = self.clock() if now is None else now
            prune_health(self._detection_health, _active_health_keys(self._snapshots))
            facilities: dict[str, JsonObject] = {}
            for facility_id, status in sorted(self._snapshots.items()):
                stale = current - status.received_at > self.stale_after_sec
                facilities[facility_id] = {
                    "facility_id": status.facility_id,
                    "generation": status.generation,
                    "seq": status.seq,
                    "received_at": status.received_at,
                    "stale": stale,
                    "cameras": project_cameras(
                        self._detection_health,
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
                    "latency": self._latency.get(facility_id),
                }
        return {
            "facilities": facilities,
            "stale_after_sec": self.stale_after_sec,
        }

    def record_latency(
        self, facility_id: str, detected_at: str, *, received_at: float | None = None
    ) -> None:
        with self._lock:
            stamped = self.clock() if received_at is None else received_at
            self._latency.record(facility_id, detected_at, stamped)

    def _latency_for_facility(self, facility_id: str) -> JsonObject | None:
        return self._latency.get(facility_id)


def _active_health_keys(
    snapshots: dict[str, RuntimeStatusSnapshot],
) -> set[tuple[str, str]]:
    return {
        (facility_id, camera_id)
        for facility_id, status in snapshots.items()
        for camera in status.cameras
        if isinstance((camera_id := camera.get("camera_id")), str)
        and parse_detection_counters(camera.get("detection")) is not None
    }


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


__all__ = [
    "DEFAULT_RUNTIME_STATUS_STALE_AFTER_SEC",
    "RuntimeStatusRecordResult",
    "RuntimeStatusSnapshot",
    "RuntimeStatusStore",
    "get_runtime_status_store",
]
