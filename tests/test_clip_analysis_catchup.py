from __future__ import annotations

import json
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
        clip_analysis_catchup,
        "_ready_candidates",
        lambda *_args: [
            clip_analysis_catchup._Candidate(
                path.parent.name,
                path,
                "a" * 64,
                1,
                1,
                path.with_name("manifest.json").stat().st_mtime,
            )
            for path in (oldest, newest)
        ],
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
    monkeypatch.setattr(
        clip_analysis_catchup,
        "_ready_candidates",
        lambda *_args: [
            clip_analysis_catchup._Candidate(path.parent.name, path, "a" * 64, 1, 1, 0)
            for path in clips[:256]
        ],
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


def test_ready_candidate_accepts_flow_published_manifest_without_source_media(
    tmp_path: Path,
) -> None:
    clip_dir = tmp_path / "clips" / "camera-20260908T000000000000Z-0123456789ab"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"clip")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 2,
                "state": "READY",
                "clip_id": clip_dir.name,
                "camera_id": "camera",
                "event_refs": ["123e4567-e89b-42d3-a456-426614174000"],
                "clip_start_at": "2026-09-08T00:00:00Z",
                "clip_end_at": "2026-09-08T00:00:01Z",
                "finalized_at": "2026-09-08T00:00:01Z",
                "sha256": "a" * 64,
                "size_bytes": 4,
                "duration_ms": 1000,
                "state_version": 2,
                "encoder": "deepstream-smart-record",
            }
        )
    )

    candidates = clip_analysis_catchup._ready_candidates(tmp_path, threading.Event(), float("inf"))

    assert [(item.clip_id, item.size_bytes, item.duration_ms) for item in candidates] == [
        (clip_dir.name, 4, 1000)
    ]
