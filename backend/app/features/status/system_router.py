"""System status route."""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from backend.app.features.clips.store import CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR
from backend.app.shared.backend_client_bundle import backend_client_bundle

router = APIRouter(tags=["system"])


class BackendStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configured: bool
    reachable: bool | None
    last_ok_at: str | None

class ImageDigestsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ml_api: str | None
    ml_worker: str | None


class ClipStoreStorageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_bytes: int | None
    used_bytes: int | None
    used_pct: float | None


class StorageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_store: ClipStoreStorageResponse


class SystemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: BackendStatusResponse
    version: str
    image_digests: ImageDigestsResponse
    storage: StorageResponse
    updated_at: str


@router.get("/system", response_model=SystemResponse)
def system(request: Request) -> dict[str, object]:
    enrolled = backend_client_bundle(request.app) is not None
    reachable = getattr(request.app.state, "backend_reachable", None)
    last_ok_at = getattr(request.app.state, "backend_last_ok_at", None)
    return {
        "version": os.environ.get("ML_EDGE_VERSION", "dev"),
        "image_digests": {
            "ml_api": _clean_env("ML_API_IMAGE_DIGEST"),
            "ml_worker": _clean_env("ML_WORKER_IMAGE_DIGEST"),
        },
        "backend": {
            "configured": enrolled,
            "reachable": reachable,
            "last_ok_at": last_ok_at,
        },
        "storage": {"clip_store": _clip_store_usage()},
        "updated_at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }


def _clip_store_usage() -> dict[str, object]:
    path = os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR)
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return {"total_bytes": None, "used_bytes": None, "used_pct": None}
    used = usage.total - usage.free
    used_pct = (used / usage.total * 100.0) if usage.total else None
    return {"total_bytes": usage.total, "used_bytes": used, "used_pct": used_pct}


def _clean_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


__all__ = ["router"]
