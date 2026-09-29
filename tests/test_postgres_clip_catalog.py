"""The PostgreSQL clip catalogue across the API lifecycle: startup, restart and reads."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.clips.catalog_indexer import (
    API_CLIP_CATALOG_INTERVAL_SEC_ENV,
    InvalidClipCatalogIntervalError,
    ReconcileOutcome,
)
from backend.app.features.clips.store import ClipStore
from backend.app.main import create_app
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_app_env import inject_sandbox_root
from tests_support.postgres_clip_app import index_clips
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_NOW = "2026-08-13T00:00:00Z"
_THUMBNAIL = b"\xff\xd8thumbnail\xff\xd9"
_ROW_VERSIONS_SQL = "SELECT clip_id, local_state, revision, xmin::text FROM clips ORDER BY clip_id"
# Row count plus every row version: any INSERT, UPDATE or DELETE changes it.
_TABLE_VERSIONS_SQL = {
    "clips": "SELECT count(*), coalesce(string_agg(xmin::text, ',' ORDER BY clip_id), '') "
    "FROM clips",
    "incidents": "SELECT count(*), coalesce(string_agg(xmin::text, ',' ORDER BY incident_id), '') "
    "FROM incidents",
    "artifacts": "SELECT count(*), coalesce(string_agg(xmin::text, ',' ORDER BY artifact_id), '') "
    "FROM artifacts",
}


def _write_clip(root: Path, index: int, *, thumbnail: bytes | None = None) -> str:
    clip_id = f"clip-{index:05d}"
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    _ = (clip_dir / "clip.mp4").write_bytes(bytes([index % 256]) * (512 + index))
    if thumbnail is not None:
        _ = (clip_dir / "thumbnail.jpg").write_bytes(thumbnail)
    payload = {
        "clip_id": clip_id,
        "camera_id": "camera-a",
        "event_ref": f"event-{index}",
        "event_type": "fall",
        "started_at": f"2026-08-01T00:00:{index:02d}Z",
        "duration_s": 1.0,
        "codec": "h264",
        "path": f"clips/{clip_id}/clip.mp4",
        "video_available": True,
        "finalized": True,
    }
    _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return clip_id


def _login(client: TestClient) -> None:
    login = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert login.status_code == 204


def _listed(client: TestClient) -> tuple[int, list[str]]:
    response = client.get("/api/v1/clips", params={"limit": 10})
    assert response.status_code == 200
    body = response.json()
    return body["pagination"]["total"], [clip["clip_id"] for clip in body["clips"]]


def _file_versions(directory: Path) -> dict[str, tuple[str, int]]:
    return {
        str(path.relative_to(directory)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def make_app(
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> Callable[[Path], FastAPI]:
    """An app without the lifespan; tests reconcile explicitly."""

    def make(root: Path) -> FastAPI:
        monkeypatch.setenv("CLIP_STORE_DIR", str(root))
        app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
        app.state.clip_store = ClipStore(root)
        return app

    return make


@pytest.fixture
def lifespan_app(
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> Callable[[Path], FastAPI]:
    """The serving app with its real lifespan on the sandbox schema.

    The interval is the inclusive maximum, so after the startup reconcile the
    indexer loop does not run again within a test. The store root is injected
    as app state: the serving lifespan refuses the retired ``CLIP_STORE_DIR``.
    """
    monkeypatch.setenv(API_CLIP_CATALOG_INTERVAL_SEC_ENV, "300")

    def make(root: Path) -> FastAPI:
        app = create_app()
        app.state.clip_store = ClipStore(root)
        inject_sandbox_root(app, postgres_product_sandbox, postgres_audit_runtime)
        return app

    return make


def test_catalog_is_queryable_after_an_api_restart_without_being_rewritten(
    tmp_path: Path,
    lifespan_app: Callable[[Path], FastAPI],
    postgres_product_sandbox: ProductSandbox,
) -> None:
    root = tmp_path / "clip-store"
    for index in range(3):
        _write_clip(root, index)
    admin = postgres_product_sandbox.admin

    with TestClient(lifespan_app(root)) as client:
        _login(client)
        assert _listed(client) == (3, ["clip-00002", "clip-00001", "clip-00000"])
    rows_before = admin.execute(_ROW_VERSIONS_SQL).fetchall()

    with TestClient(lifespan_app(root)) as client:
        _login(client)
        assert _listed(client) == (3, ["clip-00002", "clip-00001", "clip-00000"])
    assert admin.execute(_ROW_VERSIONS_SQL).fetchall() == rows_before, (
        "the restarted API serves the persisted rows; its startup reconcile rewrites none"
    )


def test_listing_reads_the_postgres_catalog_and_never_touches_edge_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifespan_app: Callable[[Path], FastAPI],
    postgres_product_sandbox: ProductSandbox,
) -> None:
    root = tmp_path / "clip-store"
    for index in range(3):
        _write_clip(root, index)
    admin = postgres_product_sandbox.admin
    sentinel = tmp_path / ".central-fixture" / "edge.sqlite3"

    with TestClient(lifespan_app(root)) as client:
        _login(client)
        # Disk and catalogue now disagree both ways: clip-00001 is on disk but
        # not catalogued, clip-00003 is on disk and not yet indexed.
        _ = admin.execute("DELETE FROM clips WHERE clip_id = %s", ("clip-00001",))
        unindexed = _write_clip(root, 3)
        catalogued = {row[0] for row in admin.execute("SELECT clip_id FROM clips").fetchall()}
        files_before = _file_versions(tmp_path)
        assert str(sentinel.relative_to(tmp_path)) in files_before
        connects: list[object] = []
        real_connect = sqlite3.connect

        def recording_connect(database: object, *args: object, **kwargs: object) -> object:
            connects.append(database)
            return real_connect(database, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(sqlite3, "connect", recording_connect)
        total, listed = _listed(client)
        files_after = _file_versions(tmp_path)

    assert catalogued == {"clip-00000", "clip-00002"}
    assert unindexed == "clip-00003"
    assert (total, listed) == (2, ["clip-00002", "clip-00000"])
    assert connects == []
    assert files_after == files_before, "GET /clips opens or writes no file, edge.sqlite3 included"


def test_clip_reads_write_no_catalog_incident_or_artifact_row(
    tmp_path: Path,
    make_app: Callable[[Path], FastAPI],
    postgres_product_sandbox: ProductSandbox,
) -> None:
    root = tmp_path / "clip-store"
    clip_id = _write_clip(root, 0, thumbnail=_THUMBNAIL)
    app = make_app(root)
    admin = postgres_product_sandbox.admin
    index_clips(app)
    media = (root / "clips" / clip_id / "clip.mp4").read_bytes()
    with admin.transaction():
        _ = admin.execute(
            "INSERT INTO incidents ("
            "incident_id, edge_event_id, facility_id, camera_id, event_type, "
            "probability, detected_at, lifecycle_state, provenance_state, "
            "provenance_missing_reason, review_version, revision, created_at, updated_at"
            ") VALUES ('incident-a','event-0','facility-1','camera-a','fall',0.8,%s,'OPEN',"
            "'MISSING','NOT_RECORDED',0,1,%s,%s)",
            (_NOW, _NOW, _NOW),
        )
        _ = admin.execute(
            "INSERT INTO artifacts ("
            "incident_id, kind, artifact_id, clip_id, state, contained_relpath, "
            "content_sha256, size_bytes, mime_type, codec, revision, created_at, updated_at"
            ") VALUES ('incident-a','PRIMARY_CLIP','primary-a',%s,'AVAILABLE',%s,%s,%s,"
            "'video/mp4','h264',1,%s,%s)",
            (
                clip_id,
                f"clips/{clip_id}/clip.mp4",
                hashlib.sha256(media).hexdigest(),
                len(media),
                _NOW,
                _NOW,
            ),
        )
    _write_clip(root, 1)

    def versions() -> dict[str, object]:
        return {table: admin.execute(sql).fetchone() for table, sql in _TABLE_VERSIONS_SQL.items()}

    before = versions()
    with TestClient(app) as client:
        _login(client)
        assert _listed(client) == (1, [clip_id])
        for path in ("metadata", "artifacts", "thumbnail", "video"):
            response = client.get(f"/api/v1/clips/{clip_id}/{path}")
            assert response.status_code == 200, path
    assert versions() == before


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "301", "abc"])
def test_invalid_catalog_interval_refuses_startup(
    interval: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifespan_app: Callable[[Path], FastAPI],
) -> None:
    root = tmp_path / "clip-store"
    _write_clip(root, 0)
    app = lifespan_app(root)
    monkeypatch.setenv(API_CLIP_CATALOG_INTERVAL_SEC_ENV, interval)
    with (
        pytest.raises(InvalidClipCatalogIntervalError, match=API_CLIP_CATALOG_INTERVAL_SEC_ENV),
        TestClient(app),
    ):
        pass


def test_failed_initial_reconcile_refuses_startup(
    tmp_path: Path,
    lifespan_app: Callable[[Path], FastAPI],
    postgres_product_sandbox: ProductSandbox,
) -> None:
    root = tmp_path / "clip-store"
    _write_clip(root, 0)
    app = lifespan_app(root)
    _ = postgres_product_sandbox.admin.execute("ALTER TABLE clips RENAME TO clips_away")
    with pytest.raises(psycopg.errors.UndefinedTable), TestClient(app):
        pass


def test_missing_store_root_is_not_treated_as_deleted_clips(
    tmp_path: Path,
    make_app: Callable[[Path], FastAPI],
    postgres_product_sandbox: ProductSandbox,
) -> None:
    root = tmp_path / "clip-store"
    for index in range(3):
        _write_clip(root, index)
    app = make_app(root)
    admin = postgres_product_sandbox.admin
    index_clips(app)
    rows_before = admin.execute(_ROW_VERSIONS_SQL).fetchall()
    assert [row[1] for row in rows_before] == ["AVAILABLE"] * 3

    unmounted = root.rename(tmp_path / "clip-store-unmounted")
    assert app.state.clip_catalog_indexer.reconcile(ClipStore(root)) == ReconcileOutcome(0, 0, 0)
    assert admin.execute(_ROW_VERSIONS_SQL).fetchall() == rows_before

    _ = unmounted.rename(root)
    with TestClient(app) as client:
        _login(client)
        assert _listed(client) == (3, ["clip-00002", "clip-00001", "clip-00000"])
