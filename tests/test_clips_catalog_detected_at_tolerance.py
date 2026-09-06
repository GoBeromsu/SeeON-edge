"""P0-AC7 reader tolerance: the catalogue reads manifests with and without ``detected_at``.

This lands ahead of the worker writer so that reverting the writer never
leaves already-written manifests unreadable, and this reader never has to be
reverted while such manifests exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.clips.store import ClipStore

_EVENT_ID = "00000000-0000-4000-8000-000000000001"


def _ready_manifest(clip_id: str = "clip-1") -> dict[str, object]:
    return {
        "manifest_schema_version": 2,
        "clip_id": clip_id,
        "camera_id": "cam-1",
        "event_ref": _EVENT_ID,
        "event_refs": [_EVENT_ID],
        "clip_start_at": "2026-07-01T00:00:00Z",
        "clip_end_at": "2026-07-01T00:00:01Z",
        "finalized_at": "2026-07-01T00:00:02Z",
        "started_at": "2026-07-01T00:00:00Z",
        "duration_s": 1.0,
        "codec": "h264",
        "path": f"clips/{clip_id}/clip.mp4",
        "finalized": True,
        "video_available": True,
        "duration_ms": 1000,
        "sha256": "a" * 64,
        "size_bytes": 1,
        "mime_type": "video/mp4",
        "state": "READY",
        "state_version": 2,
    }


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    path = root / "clips" / str(payload["clip_id"]) / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("detected_at", [None, "2026-09-02T16:43:24.147354Z"])
def test_backfill_accepts_both_manifest_generations(
    tmp_path: Path, detected_at: str | None
) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    if detected_at is not None:
        payload["detected_at"] = detected_at
    _write_manifest(root, payload)
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        store.backfill(ClipStore(root))
        assert [record["clip_id"] for record in store.records("clips")] == ["clip-1"]
    finally:
        store.close()


def test_backfill_rejects_non_utc_detected_at(tmp_path: Path) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    payload["detected_at"] = "2026-09-02T16:43:24+09:00"
    _write_manifest(root, payload)
    store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    try:
        with pytest.raises((TypeError, ValueError)):
            store.backfill(ClipStore(root))
        assert store.records("clips") == []
    finally:
        store.close()
