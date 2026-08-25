from __future__ import annotations

from typing import Final

from fastapi.routing import APIRoute

from backend.app.core.config import get_settings
from backend.app.main import create_app, no_lifespan

UNVERSIONED_ALLOWLIST: Final = {
    "/health/live",
    "/health/ready",
    "/health/release-identity",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}


def test_routes_are_versioned_except_explicit_unversioned_allowlist() -> None:
    app = create_app(lifespan=no_lifespan)
    prefix = get_settings().api_v1_prefix
    paths = {route.path for route in app.routes}
    api_route_paths = {route.path for route in app.routes if isinstance(route, APIRoute)}

    escaped = sorted(
        path
        for path in api_route_paths
        if path not in UNVERSIONED_ALLOWLIST and not path.startswith(prefix)
    )

    assert not escaped
    assert UNVERSIONED_ALLOWLIST.issubset(paths)
    assert f"{prefix}/relay/alerts" in api_route_paths
    assert f"{prefix}/status" in api_route_paths
    assert "/status" not in paths
