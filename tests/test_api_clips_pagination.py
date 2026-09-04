from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan


def _write_manifest(
    store_root: Path,
    clip_id: str,
    *,
    camera_id: str,
    event_type: str | None,
    event_ref: str,
    started_at: str = "2026-08-09T00:00:00Z",
    layout_prefix: str = "",
) -> None:
    layout_root = store_root / layout_prefix
    clip_dir = layout_root / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    video = f"video:{clip_id}".encode()
    _ = (clip_dir / "clip.mp4").write_bytes(video)
    relative_path = "/".join(part for part in (layout_prefix, "clips", clip_id) if part)
    payload: dict[str, str | float | bool] = {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "event_ref": event_ref,
        "started_at": started_at,
        "duration_s": 30.0,
        "codec": "h264",
        "path": relative_path,
        "video_available": True,
        "finalized": True,
    }
    if event_type is not None:
        payload["event_type"] = event_type
    _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


def _app() -> object:
    return create_app(lifespan=no_lifespan)


@pytest.fixture(autouse=True)
def clip_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "label-store"))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.delenv("API_AUDIT_LOG", raising=False)
    return tmp_path


def test_first_page_returns_compact_cursor(
    clip_env: Path,
) -> None:
    store_root = clip_env / "clip-store"
    for index in range(60):
        camera_id = "camera-a" if index < 50 else "camera-b"
        if index < 25:
            event_type, event_ref = "fall", f"event-{index}"
        elif index < 40:
            event_type, event_ref = "bed-exit", f"event-{index}"
        elif index < 50:
            event_type, event_ref = None, "legacy"
        else:
            event_type, event_ref = "fall", f"event-{index}"
        _write_manifest(
            store_root,
            f"clip-{index:03d}",
            camera_id=camera_id,
            event_type=event_type,
            event_ref=event_ref,
        )

    with TestClient(_app()) as client:
        _login(client)
        response = client.get(
            "/api/v1/clips",
            params={"limit": 48, "offset": 0},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["pagination"] == {
        "limit": 48,
        "offset": 0,
        "total": 60,
        "has_more": True,
        "next_cursor": body["pagination"]["next_cursor"],
    }
    assert isinstance(body["pagination"]["next_cursor"], str)
    assert body["event_type_counts"] == {"bed-exit": 15, "fall": 35, "other": 10}
    assert [clip["clip_id"] for clip in body["clips"]] == [
        f"clip-{index:03d}" for index in range(59, 11, -1)
    ]


@pytest.mark.parametrize(
    ("event_type", "expected_ids", "expected_total"),
    [
        ("fall", ["clip-004", "clip-003"], 5),
        ("other", ["clip-011", "clip-010"], 2),
    ],
)
def test_event_filter_uses_effective_category_and_keeps_camera_scoped_facets(
    clip_env: Path,
    event_type: str,
    expected_ids: list[str],
    expected_total: int,
) -> None:
    store_root = clip_env / "clip-store"
    for index in range(15):
        camera_id = "camera-a" if index < 12 else "camera-b"
        if index < 5:
            category, event_ref = "fall", f"event-{index}"
        elif index < 10:
            category, event_ref = "bed-exit", f"event-{index}"
        elif index < 12:
            category, event_ref = None, "legacy"
        else:
            category, event_ref = "fall", f"event-{index}"
        _write_manifest(
            store_root,
            f"clip-{index:03d}",
            camera_id=camera_id,
            event_type=category,
            event_ref=event_ref,
        )

    with TestClient(_app()) as client:
        _login(client)
        response = client.get(
            "/api/v1/clips",
            params={
                "camera_id": "camera-a",
                "event_type": event_type,
                "limit": 2,
                "offset": 0,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert [clip["clip_id"] for clip in body["clips"]] == expected_ids
    assert body["pagination"]["total"] == expected_total
    assert body["pagination"]["has_more"] is (expected_total > len(expected_ids))
    assert body["event_type_counts"] == {"bed-exit": 5, "fall": 5, "other": 2}


def test_paged_list_rebuilds_without_a_listing_generation() -> None:
    # Given: an application with no legacy listing-generation index.
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)

        # When: a bounded listing is requested.
        response = client.get("/api/v1/clips", params={"limit": 48})

    # Then: schema-18 clips is rebuilt directly and returns an empty keyset page.
    assert response.status_code == 200
    assert response.json()["pagination"] == {
        "limit": 48,
        "offset": 0,
        "total": 0,
        "has_more": False,
        "next_cursor": None,
    }


def test_unpaged_list_preserves_all_clips_and_reports_unbounded_pagination(clip_env: Path) -> None:
    store_root = clip_env / "clip-store"
    _write_manifest(
        store_root,
        "clip-a",
        camera_id="camera-a",
        event_type="fall",
        event_ref="event-a",
    )
    _write_manifest(
        store_root,
        "clip-b",
        camera_id="camera-b",
        event_type="bed-exit",
        event_ref="event-b",
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    body = response.json()
    assert [clip["clip_id"] for clip in body["clips"]] == ["clip-b", "clip-a"]
    assert body["pagination"] == {
        "limit": None,
        "offset": 0,
        "total": 2,
        "has_more": False,
        "next_cursor": None,
    }
    assert body["event_type_counts"] == {"bed-exit": 1, "fall": 1}


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"offset": 1},
    ],
)
def test_list_rejects_invalid_pagination(params: dict[str, int]) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips", params=params)

    assert response.status_code == 422


@pytest.mark.parametrize("layout_prefix", ["", "archive", "external/drive-1"])
def test_single_clip_metadata_resolves_every_historical_layout(
    layout_prefix: str,
    clip_env: Path,
) -> None:
    store_root = clip_env / "clip-store"
    clip_id = "clip-history"
    _write_manifest(
        store_root,
        clip_id,
        camera_id="camera-a",
        event_type="fall",
        event_ref="event-history",
        layout_prefix=layout_prefix,
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{clip_id}/metadata")

    assert response.status_code == 200
    assert response.json() == {
        "clip_id": clip_id,
        "camera_id": "camera-a",
        "event_ref": "event-history",
        "event_type": "fall",
        "started_at": "2026-08-09T00:00:00Z",
        "duration_s": 30.0,
        "codec": "h264",
        "path": "/".join(part for part in (layout_prefix, "clips", clip_id) if part),
        "video_available": True,
        "video_error": None,
        "finalized": True,
        "size_bytes": len(f"video:{clip_id}".encode()),
        "thumbnail_available": False,
        "detected_at": None,
        "truncation_reasons": [],
        "extension": None,
    }
