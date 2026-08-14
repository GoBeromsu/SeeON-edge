"""API-level tests for the clip storage location slice (see
backend/app/features/clips/storage_router.py):

``GET /api/v1/clips/storage`` -- usage snapshot + selected subdirectory.
``GET /api/v1/clips/storage/browse`` -- directories-only listing, traversal-safe.
``PUT /api/v1/clips/storage/location`` -- validate + persist a new selection.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan

DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


@pytest.fixture(autouse=True)
def clip_store_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "clip-store"
    root.mkdir()
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    return root


def test_get_storage_reports_usage_and_defaults_selected_path_to_empty(clip_store_env) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/storage")

    assert response.status_code == 200
    body = response.json()
    assert body["mount_label"] == "clip-store"
    assert "root" not in body
    assert str(clip_store_env) not in response.text
    assert body["selected_path"] == ""
    assert isinstance(body["total_bytes"], int)
    assert isinstance(body["used_bytes"], int)
    assert isinstance(body["used_pct"], float)
    assert 0.0 <= body["used_pct"] <= 100.0


def test_get_storage_degrades_to_null_usage_when_the_root_does_not_exist(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "does-not-exist"))
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/storage")

    assert response.status_code == 200
    body = response.json()
    assert body["total_bytes"] is None
    assert body["used_bytes"] is None
    assert body["used_pct"] is None


def test_browse_lists_root_level_directories_only(clip_store_env) -> None:
    (clip_store_env / "external-drive").mkdir()
    (clip_store_env / "backup").mkdir()
    (clip_store_env / "manifest.json").write_text("{}", encoding="utf-8")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/storage/browse")

    assert response.status_code == 200
    body = response.json()
    assert body["path"] == ""
    assert body["parent"] is None
    assert sorted(entry["name"] for entry in body["directories"]) == ["backup", "external-drive"]
    assert {"name": "backup", "path": "backup"} in body["directories"]


def test_browse_lists_a_nested_subdirectory_and_reports_its_parent(clip_store_env) -> None:
    (clip_store_env / "backup" / "clips").mkdir(parents=True)
    (clip_store_env / "backup" / "sibling").mkdir()

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/storage/browse", params={"path": "backup"})

    assert response.status_code == 200
    body = response.json()
    assert body["path"] == "backup"
    assert body["parent"] == ""
    assert sorted(entry["name"] for entry in body["directories"]) == ["clips", "sibling"]
    assert {"name": "clips", "path": "backup/clips"} in body["directories"]


@pytest.mark.parametrize(
    "path",
    ["/etc", "../escape", "sub/../../escape", "sub\x00null"],
)
def test_browse_rejects_traversal_and_absolute_paths_with_400(clip_store_env, path: str) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/storage/browse", params={"path": path})

    assert response.status_code == 400


def test_browse_returns_404_for_a_nonexistent_path(clip_store_env) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/storage/browse", params={"path": "nope"})

    assert response.status_code == 404


def test_browse_rejects_a_symlink_escape(clip_store_env, tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").mkdir()
    os.symlink(outside, clip_store_env / "escape-link", target_is_directory=True)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        root_listing = client.get("/api/v1/clips/storage/browse")
        walk_into_link = client.get("/api/v1/clips/storage/browse", params={"path": "escape-link"})

    # A symlink is filtered out of the directory listing (not reported as a
    # real directory)...
    assert root_listing.status_code == 200
    assert "escape-link" not in [d["name"] for d in root_listing.json()["directories"]]
    # ...and walking into it directly is rejected as not-a-real-directory
    # rather than silently followed outside the store root.
    assert walk_into_link.status_code == 404


def test_put_location_persists_the_selection_and_returns_the_storage_shape(
    clip_store_env,
) -> None:
    (clip_store_env / "backup").mkdir()

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.put("/api/v1/clips/storage/location", json={"path": "backup"})
        get_response = client.get("/api/v1/clips/storage")

    assert response.status_code == 200
    assert response.json()["selected_path"] == "backup"
    assert response.json()["mount_label"] == "clip-store"
    assert "root" not in response.json()
    assert str(clip_store_env) not in response.text
    assert get_response.json()["selected_path"] == "backup"


def test_put_location_can_reset_the_selection_to_the_empty_root(clip_store_env) -> None:
    (clip_store_env / "backup").mkdir()

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        client.put("/api/v1/clips/storage/location", json={"path": "backup"})
        reset_response = client.put("/api/v1/clips/storage/location", json={"path": ""})

    assert reset_response.status_code == 200
    assert reset_response.json()["selected_path"] == ""


@pytest.mark.parametrize("path", ["/etc", "../escape"])
def test_put_location_rejects_traversal_and_absolute_paths_with_400(
    clip_store_env, path: str
) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.put("/api/v1/clips/storage/location", json={"path": path})

    assert response.status_code == 400


def test_put_location_returns_404_for_a_nonexistent_path(clip_store_env) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.put("/api/v1/clips/storage/location", json={"path": "nope"})

    assert response.status_code == 404


def test_clip_storage_routes_require_a_dashboard_session(clip_store_env) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        get_storage = client.get("/api/v1/clips/storage")
        browse = client.get("/api/v1/clips/storage/browse")
        put_location = client.put("/api/v1/clips/storage/location", json={"path": ""})

    assert get_storage.status_code == 401
    assert browse.status_code == 401
    assert put_location.status_code == 401
