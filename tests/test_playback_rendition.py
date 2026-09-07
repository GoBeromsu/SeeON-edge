from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from worker.pipeline.output.evidence.playback_rendition import (
    PLAYBACK_DIGEST_NAME,
    PLAYBACK_NAME,
    PLAYBACK_TIMING_NAME,
    PlaybackRenditionError,
    VideoTiming,
    write_playback_rendition,
)
from worker.tools import clip_playback_backfill


def _runner(codec: str):
    def run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "ffprobe":
            return subprocess.CompletedProcess(arguments, 0, stdout=f"{codec}\n", stderr="")
        output = Path(arguments[-1])
        output.write_bytes(b"browser-safe-rendition")
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    return run


def _timing(_path: Path) -> VideoTiming:
    return VideoTiming(1, 12_000, (0, 400, 800))


def test_h264_original_does_not_create_playback_rendition(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"h264-original")

    result = write_playback_rendition(clip, run=_runner("h264"), read_timing=_timing)

    assert result is None
    assert not (tmp_path / PLAYBACK_NAME).exists()
    assert not (tmp_path / PLAYBACK_DIGEST_NAME).exists()


def test_hevc_original_creates_rendition_and_matching_digest(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"hevc-original")

    result = write_playback_rendition(clip, run=_runner("hevc"), read_timing=_timing)

    assert result == tmp_path / PLAYBACK_NAME
    assert result.read_bytes() == b"browser-safe-rendition"
    assert (tmp_path / PLAYBACK_DIGEST_NAME).read_text(encoding="ascii") == (
        hashlib.sha256(result.read_bytes()).hexdigest() + "\n"
    )
    assert json.loads((tmp_path / PLAYBACK_TIMING_NAME).read_text(encoding="ascii")) == {
        "frames": 3,
        "pts_identical": True,
        "rendition_sha256": hashlib.sha256(b"browser-safe-rendition").hexdigest(),
        "source_sha256": hashlib.sha256(b"hevc-original").hexdigest(),
        "source_frames": 3,
        "time_base": "1/12000",
    }


def test_hevc_rendition_records_non_identical_timing(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"hevc-original")

    def non_identical(path: Path) -> VideoTiming:
        return (
            VideoTiming(1, 12_000, (0, 400, 800))
            if path.name == "clip.mp4"
            else VideoTiming(1, 12_000, (0, 401, 800))
        )

    write_playback_rendition(clip, run=_runner("hevc"), read_timing=non_identical)

    assert (
        json.loads((tmp_path / PLAYBACK_TIMING_NAME).read_text(encoding="ascii"))["pts_identical"]
        is False
    )


def test_ffmpeg_failure_leaves_no_partial_file_or_sidecar(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"hevc-original")

    def failing_run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "ffprobe":
            return subprocess.CompletedProcess(arguments, 0, stdout="hevc\n", stderr="")
        Path(arguments[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(arguments, 1, stdout="", stderr="failed")

    with pytest.raises(PlaybackRenditionError):
        write_playback_rendition(clip, run=failing_run, read_timing=_timing)

    assert not (tmp_path / PLAYBACK_NAME).exists()
    assert not (tmp_path / PLAYBACK_DIGEST_NAME).exists()
    assert not list(tmp_path.glob(".*.mp4"))
    assert not caplog.records


def test_ffmpeg_timeout_leaves_no_partial_file_or_sidecar(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"hevc-original")

    def timeout_run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "ffprobe":
            return subprocess.CompletedProcess(arguments, 0, stdout="hevc\n", stderr="")
        Path(arguments[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(arguments, 120)

    with pytest.raises(PlaybackRenditionError):
        write_playback_rendition(clip, run=timeout_run, read_timing=_timing)

    assert not (tmp_path / PLAYBACK_NAME).exists()
    assert not (tmp_path / PLAYBACK_DIGEST_NAME).exists()
    assert not list(tmp_path.glob(".*.mp4"))


def test_digest_write_failure_removes_new_rendition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"hevc-original")
    old = tmp_path / PLAYBACK_NAME
    old.write_bytes(b"old-rendition")

    def fail_digest(_sidecar: Path, _digest: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "worker.pipeline.output.evidence.playback_rendition._write_digest_sidecar",
        fail_digest,
    )

    with pytest.raises(PlaybackRenditionError):
        write_playback_rendition(clip, run=_runner("hevc"), read_timing=_timing)

    assert not old.exists()
    timing = json.loads((tmp_path / PLAYBACK_TIMING_NAME).read_text(encoding="ascii"))
    assert timing["rendition_sha256"] == hashlib.sha256(b"browser-safe-rendition").hexdigest()


def test_backfill_dry_run_reports_missing_timing_sidecars_as_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = tmp_path / "clips"
    existing = clips / "existing"
    existing.mkdir(parents=True)
    (existing / "clip.mp4").write_bytes(b"hevc")
    (existing / PLAYBACK_NAME).write_bytes(b"already-rendered")
    (existing / PLAYBACK_TIMING_NAME).write_text(
        json.dumps(
            {
                "pts_identical": True,
                "source_sha256": hashlib.sha256(b"hevc").hexdigest(),
                "rendition_sha256": hashlib.sha256(b"already-rendered").hexdigest(),
            }
        ),
        encoding="ascii",
    )
    pending = clips / "pending"
    pending.mkdir()
    (pending / "clip.mp4").write_bytes(b"hevc")
    calls: list[Path] = []
    monkeypatch.setattr(clip_playback_backfill, "probe_video_codec", lambda _path: "hevc")
    monkeypatch.setattr(
        clip_playback_backfill,
        "write_playback_rendition",
        lambda path: calls.append(path) or path.with_name(PLAYBACK_NAME),
    )

    summary = clip_playback_backfill.backfill(tmp_path, dry_run=True)

    assert summary == {
        "scanned": 2,
        "h264": 0,
        "created": 0,
        "skipped": 1,
        "pending": 1,
        "failed": 0,
        "dry_run": True,
    }
    assert calls == []
    assert not (pending / PLAYBACK_NAME).exists()


def test_backfill_regenerates_non_identical_existing_rendition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "clips" / "existing" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"hevc")
    (clip.parent / PLAYBACK_NAME).write_bytes(b"already-rendered")
    (clip.parent / PLAYBACK_TIMING_NAME).write_text('{"pts_identical":false}', encoding="ascii")
    calls: list[Path] = []
    monkeypatch.setattr(
        clip_playback_backfill,
        "probe_video_codec",
        lambda _path: "hevc",
    )
    monkeypatch.setattr(
        clip_playback_backfill,
        "write_playback_rendition",
        lambda path: calls.append(path) or path.with_name(PLAYBACK_NAME),
    )

    summary = clip_playback_backfill.backfill(tmp_path)

    assert summary["skipped"] == 0
    assert summary["created"] == 1
    assert calls == [clip]


def test_thumbnail_backfill_uses_manifest_duration_and_skips_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = tmp_path / "clips" / "created"
    created.mkdir(parents=True)
    (created / "clip.mp4").write_bytes(b"hevc")
    (created / "manifest.json").write_text('{"duration_s": 22.5}', encoding="utf-8")
    existing = tmp_path / "clips" / "existing"
    existing.mkdir()
    (existing / "clip.mp4").write_bytes(b"hevc")
    (existing / "manifest.json").write_text('{"duration_s": 10}', encoding="utf-8")
    (existing / "thumbnail.jpg").write_bytes(b"already-rendered")
    calls: list[tuple[Path, Path, float]] = []

    class ThumbnailGenerator:
        def generate(self, video: Path, thumbnail: Path, duration_s: float) -> Path:
            calls.append((video, thumbnail, duration_s))
            thumbnail.write_bytes(b"thumbnail")
            return thumbnail

    monkeypatch.setattr(clip_playback_backfill, "FfmpegThumbnailGenerator", ThumbnailGenerator)

    summary = clip_playback_backfill.backfill(tmp_path, thumbnails=True)

    assert summary == {
        "scanned": 2,
        "created": 1,
        "skipped": 1,
        "pending": 0,
        "failed": 0,
        "dry_run": False,
    }
    assert calls == [(created / "clip.mp4", created / "thumbnail.jpg", 22.5)]
    assert (created / "thumbnail.jpg").read_bytes() == b"thumbnail"


def test_backfill_scans_bounded_historical_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "old" / "archive" / "clips" / "camera-1" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"hevc")
    monkeypatch.setattr(clip_playback_backfill, "probe_video_codec", lambda _path: "hevc")
    monkeypatch.setattr(
        clip_playback_backfill,
        "write_playback_rendition",
        lambda path: path.with_name(PLAYBACK_NAME),
    )

    summary = clip_playback_backfill.backfill(tmp_path)

    assert summary["scanned"] == 1
    assert summary["created"] == 1
