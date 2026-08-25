from __future__ import annotations

import ast
import importlib
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.compact_schema import COMPACT_APPLICATION_TABLES
from backend.app.edge_db.migrator import migrate_database
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[1]
RETIRED_TABLES = frozenset(
    {
        "control_heartbeats",
        "runtime_latency",
        "qa_replay_runs",
        "qa_replay_comparisons",
        "qa_label_revisions",
        "qa_label_state",
        "clip_listing_generation",
        "clip_listing_rows",
        "clip_listing_summary",
        "clip_listing_thumbnails",
        "runtime_analysis_traces",
        "runtime_analysis_components",
        "runtime_analysis_persons",
        "runtime_analysis_beds",
        "runtime_analysis_keypoints",
        "runtime_analysis_bed_points",
        "runtime_manifest_contents",
        "runtime_manifest_boots",
        "runtime_manifest_cameras",
        "faults",
    }
)
_STATUS_ROOT = ROOT / "backend/app/features/status"
_LISTING_RUNTIME = ROOT / "backend/app/features/clips/listing_runtime.py"
_QA_ROOT = ROOT / "backend/app/features/qa"


def _module_sql_kinds(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sqlite3" or alias.name.startswith("sqlite3."):
                    kinds.add("import:sqlite3")
        if isinstance(node, ast.ImportFrom) and node.module == "sqlite3":
            kinds.add("from-import:sqlite3")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.upper()
            if "CREATE TABLE" in text:
                kinds.add("ddl:create-table")
            if "INSERT INTO" in text and any(
                table.upper() in text for table in RETIRED_TABLES
            ):
                kinds.add("dml:retired-upsert")
    return frozenset(kinds)


def test_qa_package_is_absent() -> None:
    assert not _QA_ROOT.exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.store")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.runtime_trace_store")


def test_listing_runtime_startup_module_is_absent() -> None:
    assert not _LISTING_RUNTIME.exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_runtime")


def test_status_modules_have_no_sqlite_or_retired_ddl() -> None:
    offenders: list[str] = []
    for path in sorted(_STATUS_ROOT.glob("*.py")):
        kinds = _module_sql_kinds(path)
        if kinds:
            offenders.append(f"{path.relative_to(ROOT)}:{sorted(kinds)}")
    assert offenders == []


def test_reintroducing_status_sqlite_fails_this_named_boundary() -> None:
    kinds = _module_sql_kinds(_STATUS_ROOT / "heartbeat_store.py")
    injected = ast.parse("import sqlite3\nsqlite3.connect('x')\nCREATE = 'CREATE TABLE x (id INT)'")
    visitor_kinds = _module_sql_kinds
    del visitor_kinds
    found = set(kinds)
    for node in ast.walk(injected):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sqlite3":
                    found.add("import:sqlite3")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "CREATE TABLE" in node.value.upper():
                found.add("ddl:create-table")
    assert "import:sqlite3" in found
    assert "ddl:create-table" in found
    assert kinds == frozenset()


def test_schema18_runtime_has_no_telemetry_qa_or_listing_tables(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert tables == COMPACT_APPLICATION_TABLES
    assert tables.isdisjoint(RETIRED_TABLES)


def test_lifespan_boot_does_not_create_retired_tables_or_listing_index() -> None:
    app = create_app()
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        status = client.get("/api/v1/status")
    assert live.status_code == 200
    assert ready.status_code in {200, 503}
    assert status.status_code == 200
    assert not hasattr(app.state, "clip_listing_index")
    body = status.json()
    assert body["cameras"] == {}
    assert body["runtime"]["facilities"] == {}


def test_authorizer_denies_telemetry_ddl(tmp_path: Path) -> None:
    from backend.app.edge_db.connection import RuntimeActor, open_runtime_database

    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    connection = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(
                "CREATE TABLE control_heartbeats (camera_id TEXT PRIMARY KEY, "
                "facility_id TEXT NOT NULL, received_at REAL NOT NULL, "
                "config_version INTEGER) STRICT"
            )
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(
                "CREATE TABLE runtime_latency (facility_id TEXT PRIMARY KEY, "
                "payload_json TEXT NOT NULL) STRICT"
            )
    finally:
        connection.close()
