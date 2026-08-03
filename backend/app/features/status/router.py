"""Runtime status route.

``/status`` merges two API-owned relay snapshots: heartbeat-derived camera
liveness and worker-published runtime diagnostics. It never reads worker runtime
state directly (no cross-process shared state).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from backend.app.features.status.heartbeat_store import get_heartbeat_store
from backend.app.features.status.runtime_status_store import get_runtime_status_store

router = APIRouter(tags=["status"])


@router.get("/status")
def status(request: Request) -> dict[str, object]:
    heartbeat_store = get_heartbeat_store(request.app)
    inventory = getattr(request.app.state, "camera_inventory", {})
    response = heartbeat_store.snapshot(inventory)
    runtime = get_runtime_status_store(request.app).snapshot()
    facilities = runtime["facilities"]
    primary = _primary_facility(facilities)
    runtime["cameras"] = _flatten_runtime_cameras(facilities)
    runtime["worker"] = primary.get("worker") if primary else None
    runtime["device"] = _to_device_diagnostics(primary.get("gpu") if primary else None)
    runtime["clip_recorder"] = primary.get("clip_recorder") if primary else None
    response["runtime"] = runtime
    return response


def _flatten_runtime_cameras(facilities: dict[str, Any]) -> dict[str, Any]:
    """Merge per-facility camera diagnostics into a single camera_id-keyed dict.

    The front-end status contract (``front/src/shared/api/statusNormalizer.ts``)
    expects ``runtime.cameras`` as a flat dict, independent of facility
    grouping. Single-tenant deployments have exactly one facility today, so a
    later facility's entry for the same camera_id simply wins on collision.
    """
    cameras: dict[str, Any] = {}
    for facility in facilities.values():
        for camera in facility.get("cameras", []):
            camera_id = camera.get("camera_id")
            if camera_id:
                cameras[camera_id] = camera
    return cameras


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
