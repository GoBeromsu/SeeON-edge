"""Health routes for api."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, status

from shared.release_identity import (
    EDGE_DATABASE_FORMAT_IDENTITY,
    EDGE_DATABASE_SCHEMA_VERSION,
)

probe_router = APIRouter(tags=["health"])
router = APIRouter(tags=["health"])


@probe_router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@probe_router.get("/health/release-identity")
def release_identity() -> dict[str, object]:
    return {
        "format": EDGE_DATABASE_FORMAT_IDENTITY,
        "edge_database_schema_version": EDGE_DATABASE_SCHEMA_VERSION,
    }


@probe_router.get("/health/ready")
def ready(request: Request, response: Response) -> dict[str, Any]:
    readiness = getattr(request.app.state, "readiness", {"ready": False, "reason": "booting"})
    if not readiness.get("ready", False):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return readiness


@router.get("/health")
def legacy_health(request: Request) -> dict[str, Any]:
    from backend.app.features.cameras.store import CameraRegistryStore, registry_expected_cameras

    registry = getattr(request.app.state, "camera_registry", None)
    expected = registry_expected_cameras(
        registry if isinstance(registry, CameraRegistryStore) else None
    )
    return {
        "status": "ok",
        "gateway": (
            "ready"
            if getattr(request.app.state, "readiness", {}).get("ready")
            else "booting"
        ),
        "relay": {
            "token_configured": bool(getattr(request.app.state, "edge_relay_token", None)),
            "backend_configured": hasattr(request.app.state, "backend_ingest_client"),
            "camera_count": len(expected),
        },
    }
