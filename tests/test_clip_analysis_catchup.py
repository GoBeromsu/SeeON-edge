from __future__ import annotations

import os
from pathlib import Path

from worker.runtime import clip_analysis_catchup


class _Supervisor:
    def __init__(self, results: list[bool]) -> None:
        self._results = iter(results)
        self.clip_ids: list[str] = []

    def enqueue(self, clip_id: str, *_args: object, **_kwargs: object) -> bool:
        self.clip_ids.append(clip_id)
        return next(self._results)


def test_catchup_enqueues_newest_first_and_stops_at_full_queue(
    tmp_path: Path, monkeypatch: object
) -> None:
    oldest = tmp_path / "oldest" / "clip.mp4"
    newest = tmp_path / "newest" / "clip.mp4"
    ignored = tmp_path / "ignored" / "clip.mp4"
    for path, timestamp in ((oldest, 1), (newest, 3), (ignored, 2)):
        path.parent.mkdir()
        path.write_bytes(b"clip")
        manifest = path.with_name("manifest.json")
        manifest.write_text("{}")
        os.utime(manifest, (timestamp, timestamp))
    monkeypatch.setattr(
        clip_analysis_catchup, "_ready_clips", lambda _store: [oldest, newest, ignored]
    )
    monkeypatch.setattr(
        clip_analysis_catchup,
        "manifest_facts",
        lambda path: None if path == ignored else ("a" * 64, 1, 1, 1, 1),
    )
    supervisor = _Supervisor([True, False])

    clip_analysis_catchup.catch_up_clip_analysis(tmp_path, supervisor)

    assert supervisor.clip_ids == ["newest", "oldest"]
