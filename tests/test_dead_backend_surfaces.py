"""Reachability, route, package, and image contracts for retired backend surfaces."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan

ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = ROOT / "backend" / "app"
DEAD_RELATIVE_PATHS = (
    "backend/app/features/clips/listing_index.py",
    "backend/app/features/clips/listing_repository.py",
    "backend/app/features/clips/listing_queries.py",
    "backend/app/features/clips/listing_schema.py",
    "backend/app/features/clips/listing_schema_migration.py",
    "backend/app/features/clips/_listing_rows.py",
    "backend/app/features/clips/listing_generation.py",
    "backend/app/features/clips/audit_log.py",
    "backend/app/features/cameras/topology_schema.py",
    "backend/app/features/audit/owners.py",
)
DEAD_MODULES = (
    "backend.app.features.clips.listing_index",
    "backend.app.features.clips.listing_repository",
    "backend.app.features.clips.listing_queries",
    "backend.app.features.clips.listing_schema",
    "backend.app.features.clips.listing_schema_migration",
    "backend.app.features.clips._listing_rows",
    "backend.app.features.clips.listing_generation",
    "backend.app.features.clips.audit_log",
    "backend.app.features.cameras.topology_schema",
    "backend.app.features.audit.owners",
)
DEAD_SYMBOLS = frozenset(
    {
        "ClipListingIndex",
        "ListingRepository",
        "AuditLogStore",
        "post_backend_backup",
        "API_BACKEND_CLIP_EVENTS_URL_ENV",
        "push_camera",
        "put_roster",
        "BackendCameraMapper",
        "production_action_owners",
        "TOPOLOGY_SCHEMA",
        "API_FACILITY_ID_ENV",
    }
)
ACTIVE_CLIP_PATHS = frozenset(
    {
        "/api/v1/clips",
        "/api/v1/clips/{clip_id}/artifacts",
        "/api/v1/clips/{clip_id}/metadata",
        "/api/v1/clips/{clip_id}/thumbnail",
        "/api/v1/clips/{clip_id}/video",
        "/api/v1/clips/{clip_id}",
        "/api/v1/audit",
    }
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def _defined_or_imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_dead_backend_modules_are_absent_from_the_package() -> None:
    for relative in DEAD_RELATIVE_PATHS:
        assert not (ROOT / relative).exists(), relative
    for module_name in DEAD_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_create_app_import_graph_does_not_reach_dead_backend_modules() -> None:
    roots = (
        ROOT / "backend" / "app" / "main.py",
        ROOT / "backend" / "app" / "lifespan.py",
    )
    offenders: list[str] = []
    for path in roots:
        imported = _imported_modules(path)
        for module_name in DEAD_MODULES:
            if module_name in imported or module_name.rsplit(".", 1)[-1] in {
                item.rsplit(".", 1)[-1] for item in imported
            }:
                offenders.append(f"{path.relative_to(ROOT)}:{module_name}")
    assert offenders == []


def test_backend_production_tree_has_no_dead_backend_symbols() -> None:
    offenders: list[str] = []
    for path in sorted(BACKEND_APP.rglob("*.py")):
        found = _defined_or_imported_names(path) & DEAD_SYMBOLS
        if found:
            offenders.append(f"{path.relative_to(ROOT)}:{sorted(found)}")
    assert offenders == []


def test_active_clip_and_audit_routes_remain_registered() -> None:
    app = create_app(lifespan=no_lifespan)
    registered = {route.path for route in app.routes if isinstance(route, APIRoute)}
    assert registered >= ACTIVE_CLIP_PATHS
    assert "/api/v1/clips/{clip_id}/analysis" not in registered
    assert "/api/v1/relay/analysis-traces" not in registered


def test_backend_image_copy_tree_cannot_ship_dead_backend_modules() -> None:
    dockerfile = (ROOT / "Dockerfile.backend").read_text(encoding="utf-8")
    assert "COPY backend ./backend" in dockerfile
    present = [relative for relative in DEAD_RELATIVE_PATHS if (ROOT / relative).exists()]
    assert present == []


def test_lifespan_has_no_declaration_only_facility_or_config_constants() -> None:
    text = (ROOT / "backend" / "app" / "lifespan.py").read_text(encoding="utf-8")
    assert "API_FACILITY_ID_ENV" not in text
    assert "API_BACKEND_CONFIG_URL_ENV" not in text
    tree = ast.parse(text, filename="lifespan.py")
    assigned = {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assigned.update(
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    )
    assert "API_FACILITY_ID_ENV" not in assigned
    assert "API_BACKEND_CONFIG_URL_ENV" not in assigned


def test_backend_client_bundle_has_no_camera_mapper_slot() -> None:
    from backend.app.shared.backend_client_bundle import BackendClientBundle

    assert "camera_mapper" not in BackendClientBundle.__dataclass_fields__
    app = create_app(lifespan=no_lifespan)
    with TestClient(app):
        assert not hasattr(app.state, "backend_camera_mapper")


def test_catalog_backfill_has_no_jsonl_audit_side_channel() -> None:
    from backend.app.features.clips.catalog import CatalogStore

    assert "audit_log" not in CatalogStore.backfill.__code__.co_varnames
