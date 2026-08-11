"""Gateway metadata route."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["models"])


@router.get("/models")
def models(request: Request) -> dict[str, object]:
    from backend.app.features.cameras.store import CameraRegistryStore, registry_expected_cameras

    registry = getattr(request.app.state, "camera_registry", None)
    expected = registry_expected_cameras(
        registry if isinstance(registry, CameraRegistryStore) else None
    )
    return {
        "service": "ml-api",
        "role": "gateway",
        "ml": "external-worker",
        "relay": {
            "backend_configured": hasattr(request.app.state, "backend_ingest_client"),
            "camera_count": len(expected),
        },
    }
