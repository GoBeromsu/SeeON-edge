from __future__ import annotations

import os
import threading
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


def test_catchup_stops_on_event_and_respects_candidate_bound(
    tmp_path: Path, monkeypatch: object
) -> None:
    clips: list[Path] = []
    for number in range(257):
        clip = tmp_path / f"clip-{number}" / "clip.mp4"
        clip.parent.mkdir()
        clip.write_bytes(b"clip")
        clip.with_name("manifest.json").write_text("{}")
        clips.append(clip)
    monkeypatch.setattr(clip_analysis_catchup, "_ready_clips", lambda _store: clips)
    monkeypatch.setattr(
        clip_analysis_catchup, "manifest_facts", lambda _path: ("a" * 64, 1, 1, 1, 1)
    )
    bounded = _Supervisor([True] * 256)
    clip_analysis_catchup.catch_up_clip_analysis(tmp_path, bounded)
    assert len(bounded.clip_ids) == 256

    stop = threading.Event()

    class _StoppingSupervisor(_Supervisor):
        def enqueue(self, *args: object, **kwargs: object) -> bool:
            result = super().enqueue(*args, **kwargs)
            stop.set()
            return result

    stopped = _StoppingSupervisor([True] * 256)
    clip_analysis_catchup.catch_up_clip_analysis(tmp_path, stopped, stop)
    assert len(stopped.clip_ids) == 1
