from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from worker.pipeline.output.evidence.playback_rendition import (
    PLAYBACK_DIGEST_NAME,
    PLAYBACK_NAME,
    PlaybackRenditionError,
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


def test_h264_original_does_not_create_playback_rendition(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"h264-original")

    result = write_playback_rendition(clip, run=_runner("h264"))

    assert result is None
    assert not (tmp_path / PLAYBACK_NAME).exists()
    assert not (tmp_path / PLAYBACK_DIGEST_NAME).exists()


def test_hevc_original_creates_rendition_and_matching_digest(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"hevc-original")

    result = write_playback_rendition(clip, run=_runner("hevc"))

    assert result == tmp_path / PLAYBACK_NAME
    assert result.read_bytes() == b"browser-safe-rendition"
    assert (tmp_path / PLAYBACK_DIGEST_NAME).read_text(encoding="ascii") == (
        hashlib.sha256(result.read_bytes()).hexdigest() + "\n"
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
        write_playback_rendition(clip, run=failing_run)

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
        write_playback_rendition(clip, run=timeout_run)

    assert not (tmp_path / PLAYBACK_NAME).exists()
    assert not (tmp_path / PLAYBACK_DIGEST_NAME).exists()
    assert not list(tmp_path.glob(".*.mp4"))


def test_backfill_skips_existing_renditions_and_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = tmp_path / "clips"
    existing = clips / "existing"
    existing.mkdir(parents=True)
    (existing / "clip.mp4").write_bytes(b"hevc")
    (existing / PLAYBACK_NAME).write_bytes(b"already-rendered")
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
        "skipped": 2,
        "failed": 0,
        "dry_run": True,
    }
    assert calls == []
    assert not (pending / PLAYBACK_NAME).exists()


def test_backfill_skips_existing_rendition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = tmp_path / "clips" / "existing" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"hevc")
    (clip.parent / PLAYBACK_NAME).write_bytes(b"already-rendered")
    monkeypatch.setattr(
        clip_playback_backfill,
        "probe_video_codec",
        lambda _path: pytest.fail("existing rendition should not be probed"),
    )

    summary = clip_playback_backfill.backfill(tmp_path)

    assert summary["skipped"] == 1
    assert summary["created"] == 0
