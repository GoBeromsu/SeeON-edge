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
from backend.app.features.cameras.bed_zone_router import router as bed_zone_router
from backend.app.features.cameras.router import router as cameras_router
from backend.app.features.cameras.streams_router import router as streams_router
from backend.app.features.clips.router import router as clips_router
from backend.app.features.clips.storage_router import router as clip_storage_router
from backend.app.features.connection.router import router as connection_router
from backend.app.features.connection.topology_confirmation_router import (
    router as topology_confirmation_router,
)
from backend.app.features.detection_settings.router import router as detection_settings_router
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
    # relay 토큰의 유일한 출처는 `app.state.edge_relay_token`이다. 인증
    # 호출부는 이 값만 읽는다(env를 다시 읽지 않는다). 여기서 채워 두면
    # lifespan이 도는 서빙 경로와 lifespan 없이 만드는 테스트 경로가 같은
    # 한 곳을 보게 된다. lifespan 쪽 시딩은 `hasattr` 가드가 있어 이 값을
    # 덮지 않는다.
    app.state.edge_relay_token = os.environ.get("API_EDGE_RELAY_TOKEN")
    app.include_router(health_routes.probe_router)

    api_router = APIRouter()
    api_router.include_router(health_routes.router)
    api_router.include_router(auth_router)
    api_router.include_router(status_router)
    api_router.include_router(models_router)
    api_router.include_router(relay_router)
    api_router.include_router(evidence_router)
    api_router.include_router(cameras_router)
    api_router.include_router(bed_zone_router)
    api_router.include_router(connection_router)
    api_router.include_router(topology_confirmation_router)
    api_router.include_router(clips_router)
    api_router.include_router(clip_storage_router)
    api_router.include_router(detection_settings_router)
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
