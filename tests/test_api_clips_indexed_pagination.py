from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.clips import listing_generation as listing_generation_module
from backend.app.features.clips.listing_index import ClipListingIndex
from backend.app.features.clips.store import ClipStore
from backend.app.main import create_app, no_lifespan

_CLIP_COUNT = 9_313
_PAGE_SIZE = 48


def _write_fixture(root: Path) -> None:
    for index in range(_CLIP_COUNT):
        clip_id = f"clip-{index:05d}"
        clip_dir = root / "clips" / clip_id
        clip_dir.mkdir(parents=True)
        payload: dict[str, str | float | bool] = {
            "clip_id": clip_id,
            "camera_id": "camera-a",
            "event_ref": f"event-{index}",
            "started_at": (
                f"2026-08-09T{index // 3600:02d}:"
                f"{index // 60 % 60:02d}:{index % 60:02d}Z"
            ),
            "duration_s": 0.0,
            "codec": "",
            "video_available": False,
            "finalized": True,
        }
        facet_case = index % 4
        if facet_case == 0:
            payload["event_type"] = "fall"
        elif facet_case == 1:
            payload["event_type"] = "bed-exit"
        elif facet_case == 2:
            payload["event_type"] = f"unknown-{index}"
        _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_large_indexed_listing_has_bounded_work_and_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: 9,313 manifests projected into the ml-api catalog database.
    root = tmp_path / "clip-store"
    _write_fixture(root)
    store = ClipStore(root)
    index = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    initial = index.reconcile(store)
    app = create_app(lifespan=no_lifespan)
    app.state.clip_store = store
    app.state.clip_listing_index = index
    manifest_reads = 0
    original_reader = listing_generation_module.read_manifest_file

    def instrumented_reader(path: Path):
        nonlocal manifest_reads
        manifest_reads += 1
        return original_reader(path)

    monkeypatch.setattr(listing_generation_module, "read_manifest_file", instrumented_reader)

    # When: a client reads every bounded page through the HTTP route.
    traversed_ids: list[str] = []
    first_response_bytes = 0
    first_facets: dict[str, int] = {}
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "admin", "password": "admin"},
        )
        assert login.status_code == 204
        for offset in range(0, _CLIP_COUNT, _PAGE_SIZE):
            response = client.get(
                "/api/v1/clips",
                params={"limit": _PAGE_SIZE, "offset": offset},
            )
            assert response.status_code == 200
            body = response.json()
            if offset == 0:
                first_response_bytes = len(response.content)
                first_facets = body["event_type_counts"]
                assert len(body["clips"]) == _PAGE_SIZE
                assert body["pagination"]["total"] == _CLIP_COUNT
            traversed_ids.extend(clip["clip_id"] for clip in body["clips"])
    index.close()

    # Then: requests read no manifests, facets and bytes stay bounded, and IDs appear once.
    assert initial.read == _CLIP_COUNT
    assert manifest_reads == 0
    assert set(first_facets) == {"bed-exit", "fall", "other"}
    assert first_response_bytes < 50_000
    assert len(traversed_ids) == _CLIP_COUNT
    assert len(set(traversed_ids)) == _CLIP_COUNT
