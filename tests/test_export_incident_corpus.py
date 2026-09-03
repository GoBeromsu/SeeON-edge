import json
import sqlite3
from pathlib import Path

import pytest

from scripts.qa.export_incident_corpus import CorpusValidationError, export


def _snapshot(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE incidents (incident_id TEXT, edge_event_id TEXT, camera_id TEXT, "
        "event_type TEXT, detected_at TEXT)"
    )
    connection.execute(
        "INSERT INTO incidents VALUES ('i1', 'event-1', 'camera-1', 'fall', "
        "'2026-01-01T00:00:00+00:00')"
    )
    connection.commit()
    connection.close()


def _clip(store: Path, clip_id: str, refs: list[str], *, media: bool = True) -> Path:
    directory = store / "clips" / clip_id
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"clip_id": clip_id, "event_refs": refs, "duration_s": 12}), encoding="utf-8"
    )
    if media:
        (directory / "clip.mp4").write_bytes(b"video")
    return directory


def test_export_reads_only_claimed_canonical_clip_manifest(tmp_path: Path) -> None:
    snapshot = tmp_path / "edge.sqlite3"
    store = tmp_path / "store"
    output = tmp_path / "corpus.jsonl"
    _snapshot(snapshot)
    _clip(store, "clip-1", ["event-1"])
    _clip(store / "ignored", "clip-2", ["event-1"])

    assert export(snapshot, store, output)[:2] == (1, 1)
    record = json.loads(output.read_text())
    assert record["clip_id"] == "clip-1"
    assert record["clip_path"].endswith("clips/clip-1/clip.mp4")


def test_export_fails_closed_for_every_malformed_claimed_clip(tmp_path: Path) -> None:
    snapshot = tmp_path / "edge.sqlite3"
    store = tmp_path / "store"
    _snapshot(snapshot)
    _clip(store, "bad-one", ["event-1"])
    (store / "clips" / "bad-one" / "manifest.json").write_text("{", encoding="utf-8")
    _clip(store, "bad-two", [])

    with pytest.raises(CorpusValidationError, match="bad-one: malformed manifest") as exc:
        export(snapshot, store, tmp_path / "out.jsonl")
    assert "bad-two: invalid event_refs" in str(exc.value)


def test_export_rejects_duplicate_event_ref_claims(tmp_path: Path) -> None:
    snapshot = tmp_path / "edge.sqlite3"
    store = tmp_path / "store"
    _snapshot(snapshot)
    _clip(store, "clip-one", ["event-1"])
    _clip(store, "clip-two", ["event-1"])

    with pytest.raises(CorpusValidationError) as exc:
        export(snapshot, store, tmp_path / "out.jsonl")
    assert "event-1: duplicate event_ref claimed by clip-one and clip-two" in str(exc.value)


def test_export_requires_media_for_claimed_clip(tmp_path: Path) -> None:
    snapshot = tmp_path / "edge.sqlite3"
    store = tmp_path / "store"
    _snapshot(snapshot)
    _clip(store, "clip-one", ["event-1"], media=False)

    with pytest.raises(CorpusValidationError, match="clip-one: missing clip.mp4"):
        export(snapshot, store, tmp_path / "out.jsonl")
