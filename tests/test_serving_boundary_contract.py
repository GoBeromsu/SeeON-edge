from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Final

from fastapi.routing import APIRoute

from backend.app.main import create_app, no_lifespan

ML_ROOT: Final = Path(__file__).resolve().parents[1]
SERVING_ROOT: Final = ML_ROOT / "backend" / "app"
ALLOWED_PATHS: Final = {
    "/api/v1/auth/session",
    "/api/v1/health",
    "/api/v1/models",
    "/api/v1/relay/alerts",
    "/api/v1/relay/capabilities",
    "/api/v1/relay/clips/{clip_id}",
    "/api/v1/relay/config",
    "/api/v1/relay/heartbeat",
    "/api/v1/relay/restart",
    "/api/v1/relay/runtime-status",
    "/api/v1/audit",
    "/api/v1/cameras",
    "/api/v1/cameras/worker-config",
    "/api/v1/cameras/{camera_id}",
    "/api/v1/cameras/{camera_id}/test",
    "/api/v1/clips",
    "/api/v1/connection",
    "/api/v1/connection/sync-cameras",
    "/api/v1/connection/test",
    "/api/v1/clips/{clip_id}/label",
    "/api/v1/clips/{clip_id}/video",
    "/api/v1/status",
    "/api/v1/streams/{camera_id}",
    "/api/v1/streams/{camera_id}/snapshot",
    "/api/v1/system",
    "/docs",
    "/docs/oauth2-redirect",
    "/health/live",
    "/health/ready",
    "/openapi.json",
    "/redoc",
}
PRODUCTION_ROUTE_TERMS: Final = (
    "rtsp",
    "frame relay",
    "frame_relay",
    "camera stream",
)
FORBIDDEN_IMPORTS: Final = (
    "worker",
    "shared.events.outbox",
    "shared.events.schemas",
    "runtime.edge_worker",
    "runners",
    "sources",
    "perception",
    "domains",
)


def _serving_python_files() -> list[Path]:
    return sorted(path for path in SERVING_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _route_descriptor(route: APIRoute) -> str:
    tags = " ".join(str(tag) for tag in route.tags)
    return f"{route.name} {route.path} {tags}".lower()


def _import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return node.names[0].name
    if isinstance(node, ast.ImportFrom):
        return node.module
    return None


def _is_forbidden_import(import_name: str, forbidden: str) -> bool:
    return import_name == forbidden or import_name.startswith(f"{forbidden}.")


def test_serving_app_exposes_only_documented_boundary_routes() -> None:
    app = create_app(lifespan=no_lifespan)
    api_routes = [route for route in app.routes if isinstance(route, APIRoute)]

    exposed_paths = {route.path for route in app.routes}
    production_routes = [
        route.path
        for route in api_routes
        if any(term in _route_descriptor(route) for term in PRODUCTION_ROUTE_TERMS)
    ]

    assert exposed_paths == ALLOWED_PATHS
    assert not production_routes, (
        "Serving must not expose production edge/runtime surfaces: " + ", ".join(production_routes)
    )


def test_serving_files_do_not_import_ml_runtime_or_event_schema_modules() -> None:
    violations: list[str] = []

    for path in _serving_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            import_name = _import_name(node)
            if import_name is None:
                continue
            for forbidden in FORBIDDEN_IMPORTS:
                if _is_forbidden_import(import_name, forbidden):
                    violations.append(
                        f"{path.relative_to(ML_ROOT)}:{node.lineno}: imports {import_name}"
                    )

    assert not violations, "\n".join(violations)


def test_serving_import_allows_api_owned_backend_ingest_client_but_not_ml_runtime() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import backend.app.main; "
                "forbidden = {'shared.events.outbox', 'shared.events.schemas', "
                "'worker', 'edge.runtime.edge_worker', "
                "'edge.runners.registry', 'edge.sources.registry'}; "
                "loaded = sorted(forbidden.intersection(sys.modules)); "
                "print(loaded); raise SystemExit(1 if loaded else 0)"
            ),
        ],
        cwd=ML_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert probe.returncode == 0, probe.stdout
