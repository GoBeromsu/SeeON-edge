"""P0-AC7 reader tolerance: the catalogue reads manifests with and without ``detected_at``.

This lands ahead of the worker writer so that reverting the writer never
leaves already-written manifests unreadable, and this reader never has to be
reverted while such manifests exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.features.clips.catalog_indexer import (
    ClipCatalogIndexer,
    ClipCatalogQuery,
    PostgresClipCatalog,
)
from backend.app.features.clips.store import ClipStore
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

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


def _listed(sandbox: ProductSandbox, root: Path) -> list[str]:
    store = ClipStore(root)
    outcome = ClipCatalogIndexer(sandbox.database, sandbox.authority).reconcile(store)
    assert (outcome.remaining, outcome.isolated) == (0, 0)
    page = PostgresClipCatalog(sandbox.database).page(
        store, ClipCatalogQuery(camera_id=None, event_type=None, limit=50, cursor=None)
    )
    return [clip.manifest.clip_id for clip in page.clips]


@pytest.mark.parametrize("detected_at", [None, "2026-09-02T16:43:24.147354Z"])
def test_catalogue_lists_both_manifest_generations(
    tmp_path: Path, postgres_product_sandbox: ProductSandbox, detected_at: str | None
) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    if detected_at is not None:
        payload["detected_at"] = detected_at
    _write_manifest(root, payload)

    assert _listed(postgres_product_sandbox, root) == ["clip-1"]


def test_catalogue_does_not_list_non_utc_detected_at(
    tmp_path: Path, postgres_product_sandbox: ProductSandbox
) -> None:
    root = tmp_path / "clip-store"
    payload = _ready_manifest()
    payload["detected_at"] = "2026-09-02T16:43:24+09:00"
    _write_manifest(root, payload)

    assert _listed(postgres_product_sandbox, root) == []
