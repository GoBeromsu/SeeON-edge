"""Runtime status route.

``/status`` merges two API-owned relay snapshots: heartbeat-derived camera
liveness and worker-published runtime diagnostics. It never reads worker runtime
state directly (no cross-process shared state).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Request

from backend.app.features.cameras.store import CameraRegistryStore, registry_expected_cameras
from backend.app.features.runtime_settings.store import get_runtime_settings_store
from backend.app.features.status.heartbeat_store import get_heartbeat_store
from backend.app.features.status.runtime_status_store import get_runtime_status_store

router = APIRouter(tags=["status"])


@router.get("/status")
def status(request: Request) -> dict[str, object]:
    heartbeat_store = get_heartbeat_store(request.app)
    registry = getattr(request.app.state, "camera_registry", None)
    registry_store = registry if isinstance(registry, CameraRegistryStore) else None
    expected = registry_expected_cameras(registry_store)
    response = heartbeat_store.snapshot(expected)
    runtime = get_runtime_status_store(request.app).snapshot()
    facilities = runtime["facilities"]
    primary = _primary_facility(facilities)
    runtime["cameras"] = _flatten_runtime_cameras(
        facilities,
        alias_to_local=_local_runtime_id_resolver(registry_store),
    )
    runtime["worker"] = primary.get("worker") if primary else None
    runtime["device"] = _to_device_diagnostics(primary.get("gpu") if primary else None)
    runtime["clip_recorder"] = primary.get("clip_recorder") if primary else None
    runtime["clip_export_applied"] = _clip_export_applied(primary)
    response["runtime"] = runtime
    response["runtime_settings"] = get_runtime_settings_store(request.app).get().as_dict()
    return response


def _local_runtime_id_resolver(
    registry: CameraRegistryStore | None,
) -> dict[str, str] | None:
    """Read-only map of registry-local id and backend_camera_id -> local id.

    Dashboard ``runtime.cameras`` is keyed by the registry-local camera id.
    Workers may publish either alias; this resolver never writes the registry
    and never changes Hub egress identity. ``None`` means no registry is
    mounted, so reported ids stay as-is without an unresolved marker.
    """
    if registry is None:
        return None
    snapshot = registry.snapshot()
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list):
        return {}
    alias_to_local: dict[str, str] = {}
    for record in cameras:
        if not isinstance(record, dict):
            continue
        local_id = record.get("id")
        if not isinstance(local_id, str) or not local_id.strip():
            continue
        alias_to_local[local_id] = local_id
        backend_id = record.get("backend_camera_id")
        if isinstance(backend_id, str) and backend_id.strip():
            alias_to_local[backend_id] = local_id
    return alias_to_local


def _flatten_runtime_cameras(
    facilities: dict[str, Any],
    *,
    alias_to_local: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Merge per-facility camera diagnostics into a single camera_id-keyed dict.

    The front-end status contract (``front/src/shared/api/statusNormalizer.ts``)
    expects ``runtime.cameras`` as a flat dict, independent of facility
    grouping. Single-tenant deployments have exactly one facility today, so a
    later facility's entry for the same camera_id simply wins on collision.
    When both a local id and its mapped ``backend_camera_id`` appear, they
    collapse to one local row and the newest accepted snapshot wins.

    Staleness lives on the facility (``facility["stale"]``, derived from
    ``received_at`` vs ``stale_after_sec``) but the front-end only reads this
    flat dict (issue #160) — a dead worker's last-known ``measured_fps`` would
    otherwise keep rendering as if it were live. So each camera's staleness is
    propagated here rather than dropped, without erasing the last measured
    value (the front-end needs both to tell "never measured" apart from
    "measurement stopped").
    """
    aliases = alias_to_local
    cameras: dict[str, Any] = {}
    newest_received_at: dict[str, float] = {}
    ordered_facilities = sorted(
        facilities.values(),
        key=lambda facility: facility.get("received_at", 0.0),
    )
    for facility in ordered_facilities:
        stale = bool(facility.get("stale"))
        received_at = facility.get("received_at", 0.0)
        received_at_sec = received_at if isinstance(received_at, (int, float)) else 0.0
        for camera in facility.get("cameras", []):
            camera_id = camera.get("camera_id")
            if not camera_id:
                continue
            local_id = (
                aliases.get(camera_id)
                if aliases is not None and isinstance(camera_id, str)
                else None
            )
            if local_id is None:
                row = {**camera, "stale": stale}
                if aliases is not None:
                    row["unresolved"] = True
                cameras[camera_id] = row
                newest_received_at[camera_id] = float(received_at_sec)
                continue
            previous = newest_received_at.get(local_id)
            if previous is not None and float(received_at_sec) < previous:
                continue
            cameras[local_id] = {**camera, "camera_id": local_id, "stale": stale}
            newest_received_at[local_id] = float(received_at_sec)
    return cameras


def _clip_export_applied(facility: dict[str, Any] | None) -> dict[str, object]:
    if facility is None:
        return {"enabled": None, "version": None, "freshness": "unknown"}
    applied = facility.get("clip_export")
    if not isinstance(applied, dict):
        return {"enabled": None, "version": None, "freshness": "unknown"}
    worker = facility.get("worker")
    if isinstance(worker, dict) and worker.get("alive") is False:
        freshness = "offline"
    elif facility.get("stale") is True:
        freshness = "stale"
    else:
        freshness = "fresh"
    return {
        "enabled": applied.get("enabled"),
        "version": applied.get("version"),
        "freshness": freshness,
    }


def _primary_facility(facilities: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the facility whose worker/device/clip_recorder diagnostics are
    exposed at the flat ``runtime.*`` level the front-end contract expects.

    Deployments are single-tenant in practice (one facility), so this is
    almost always unambiguous. When more than one facility has reported
    (e.g. a stale test fixture), the most recently received one wins.
    """
    if not facilities:
        return None
    return max(facilities.values(), key=lambda facility: facility.get("received_at", 0.0))


def _clip_export_applied(facility: dict[str, Any] | None) -> dict[str, object]:
    if facility is None:
        return {"enabled": None, "version": None, "freshness": "unknown"}
    applied = facility.get("clip_export")
    if not isinstance(applied, dict):
        return {"enabled": None, "version": None, "freshness": "unknown"}
    worker = facility.get("worker")
    if isinstance(worker, dict) and worker.get("alive") is False:
        freshness = "offline"
    elif facility.get("stale") is True:
        freshness = "stale"
    else:
        freshness = "fresh"
    return {
        "enabled": applied.get("enabled"),
        "version": applied.get("version"),
        "freshness": freshness,
    }


def _to_device_diagnostics(gpu: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map the GPU telemetry payload onto the front-end's device-adaptive
    ``RuntimeDeviceDiagnostics`` shape (``front/src/shared/api/types.ts``).

    Only ``device_name``/``captured_at_sec`` carry over directly; ``backend``
    has no global-scope source today (decode backend is tracked per-camera,
    see ``runtime.cameras[*].decode``) so it is left null.
    """
    if not gpu:
        return None
    return {
        "backend": None,
        "available": gpu.get("cuda_context_ok"),
        "device_name": gpu.get("device_name"),
        "captured_at_sec": gpu.get("captured_at_sec"),
    }


__all__ = ["router"]
