from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from backend.app.features.clips.listing import ClipPage
from backend.app.features.clips.listing_generation import prepare_generation
from backend.app.features.clips.listing_repository import ListingRepository
from backend.app.features.clips.listing_schema import SELECT_ACTIVE_GENERATION
from backend.app.features.clips.schemas import ClipListQuery
from backend.app.features.clips.store import ClipStore

_GENERATION_ROWS = TypeAdapter(list[tuple[int]])


def _write_manifests(root: Path, clip_ids: tuple[str, ...], event_type: str) -> None:
    for clip_id in clip_ids:
        clip_dir = root / "clips" / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, str | float | bool] = {
            "clip_id": clip_id,
            "camera_id": "camera-a",
            "event_ref": "event-ref",
            "event_type": event_type,
            "started_at": f"2026-08-09T00:00:{clip_id[-2:]}Z",
            "duration_s": 1.0,
            "codec": "h264",
            "video_available": False,
            "finalized": True,
        }
        _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_page_keeps_one_generation_snapshot_during_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: generation N is active and a page reader pauses after selecting it.
    old_root = tmp_path / "old-clips"
    _write_manifests(old_root, ("clip-01", "clip-02"), "fall")
    new_root = tmp_path / "new-clips"
    _write_manifests(new_root, ("clip-11", "clip-12", "clip-13"), "bed-exit")
    path = tmp_path / "catalog.sqlite3"
    repository = ListingRepository.open(path)
    repository.publish(prepare_generation(ClipStore(old_root), {}))
    replacement = prepare_generation(ClipStore(new_root), {})
    generation_selected = threading.Event()
    resume_reader = threading.Event()
    selected_generations: list[int] = []
    pages: list[ClipPage] = []

    def blocking_active_generation(connection: sqlite3.Connection) -> int:
        generation = _GENERATION_ROWS.validate_python(
            connection.execute(SELECT_ACTIVE_GENERATION).fetchall()
        )[0][0]
        selected_generations.append(generation)
        generation_selected.set()
        assert resume_reader.wait(timeout=2)
        return generation

    monkeypatch.setattr(
        ListingRepository,
        "_active_generation",
        staticmethod(blocking_active_generation),
    )
    reader = threading.Thread(
        target=lambda: pages.append(repository.page(ClipListQuery(limit=48)))
    )
    reader.start()
    assert generation_selected.wait(timeout=1)

    # When: generation N+1 is activated and generation N is cleaned before the read resumes.
    repository.publish(replacement)
    with sqlite3.connect(path) as connection:
        retained_generations = connection.execute(
            "SELECT DISTINCT generation FROM clip_listing_rows ORDER BY generation"
        ).fetchall()
    resume_reader.set()
    reader.join(timeout=2)
    repository.close()

    # Then: the page is one complete old-or-new generation, never an empty or mixed result.
    assert not reader.is_alive()
    assert selected_generations == [1]
    assert retained_generations == [(2,)]
    observed = (
        tuple(manifest.clip_id for manifest in pages[0].manifests),
        pages[0].total,
        tuple(sorted(pages[0].event_type_counts.items())),
    )
    assert observed in {
        (("clip-02", "clip-01"), 2, (("fall", 2),)),
        (("clip-13", "clip-12", "clip-11"), 3, (("bed-exit", 3),)),
    }
