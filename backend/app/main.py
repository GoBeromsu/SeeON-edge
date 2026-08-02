"""FastAPI backend app factory."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import get_settings
from backend.app.features.auth.router import router as auth_router
from backend.app.features.cameras.router import router as cameras_router
from backend.app.features.cameras.streams_router import router as streams_router
from backend.app.features.clips.router import router as clips_router
from backend.app.features.connection.router import router as connection_router
from backend.app.features.evidence.router import router as evidence_router
from backend.app.features.relay.router import router as relay_router
from backend.app.features.status.router import router as status_router
from backend.app.features.status.system_router import router as system_router
from backend.app.lifespan import lifespan as serving_lifespan
from backend.app.routes import health as health_routes
from backend.app.routes.models import router as models_router

LifespanFactory = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def create_app(*, lifespan: LifespanFactory | None = serving_lifespan) -> FastAPI:
    """Create the backend FastAPI app with feature-slice routers registered."""
    settings = get_settings()
    prefix = settings.api_v1_prefix
    app = FastAPI(
        title="fall-detector api",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(health_routes.probe_router)

    api_router = APIRouter()
    api_router.include_router(health_routes.router)
    api_router.include_router(auth_router)
    api_router.include_router(status_router)
    api_router.include_router(models_router)
    api_router.include_router(relay_router)
    api_router.include_router(evidence_router)
    api_router.include_router(cameras_router)
    api_router.include_router(connection_router)
    api_router.include_router(clips_router)
    api_router.include_router(streams_router)
    api_router.include_router(system_router)
    app.include_router(api_router, prefix=prefix)
    _mount_front_dist(app)
    return app


def _mount_front_dist(app: FastAPI) -> None:
    front_dist = Path(os.environ.get("API_FRONT_DIST", "/app/front"))
    if front_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(front_dist), html=True), name="front")


@asynccontextmanager
async def no_lifespan(app: FastAPI) -> AsyncGenerator[None]:
    del app
    yield


app = create_app()

__all__ = [
    "app",
    "create_app",
    "no_lifespan",
]
