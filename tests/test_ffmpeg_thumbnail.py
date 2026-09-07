from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from worker.adapters.media.ffmpeg_thumbnail import (
    FfmpegThumbnailGenerator,
    ThumbnailUnavailable,
)


def _successful_run(calls: list[list[str]]):
    def run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        Path(arguments[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    return run


def test_generates_atomically_published_thumbnail_at_detection_offset(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls: list[list[str]] = []

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        assert kwargs["timeout"] == 30.0
        Path(arguments[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    thumbnail = FfmpegThumbnailGenerator(run=run).generate(
        video,
        tmp_path / "thumbnail.jpg",
        30.0,
    )

    assert thumbnail == tmp_path / "thumbnail.jpg"
    assert thumbnail.read_bytes() == b"jpeg"
    assert calls[0][0:6] == ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss"]
    assert calls[0][calls[0].index("-ss") + 1] == "15.0"
    assert not list(tmp_path.glob(".thumbnail.*.jpg"))


@pytest.mark.parametrize(
    ("duration_s", "expected_offset"),
    ((0.0, "0.0"), (0.25, "0.0"), (0.5, "0.0"), (1.0, "0.5")),
)
def test_clamps_thumbnail_offset_for_short_clips(
    tmp_path: Path,
    duration_s: float,
    expected_offset: str,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls: list[list[str]] = []

    FfmpegThumbnailGenerator(run=_successful_run(calls)).generate(
        video,
        tmp_path / "thumbnail.jpg",
        duration_s,
    )

    assert calls[0][calls[0].index("-ss") + 1] == expected_offset


def test_raises_when_ffmpeg_returns_nonzero_and_removes_partial_thumbnail(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    def run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(arguments[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(arguments, 1, stdout="", stderr="failure")

    with pytest.raises(ThumbnailUnavailable, match="ffmpeg did not create"):
        FfmpegThumbnailGenerator(run=run).generate(video, tmp_path / "thumbnail.jpg", 30.0)

    assert not (tmp_path / "thumbnail.jpg").exists()
    assert not list(tmp_path.glob(".thumbnail.*.jpg"))


def test_raises_when_ffmpeg_times_out_and_removes_partial_thumbnail(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    def run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(arguments[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(arguments, 30)

    with pytest.raises(ThumbnailUnavailable, match="timed out"):
        FfmpegThumbnailGenerator(run=run).generate(video, tmp_path / "thumbnail.jpg", 30.0)

    assert not (tmp_path / "thumbnail.jpg").exists()
    assert not list(tmp_path.glob(".thumbnail.*.jpg"))


def test_raises_when_ffmpeg_creates_an_empty_thumbnail(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    def run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    with pytest.raises(ThumbnailUnavailable, match="empty thumbnail"):
        FfmpegThumbnailGenerator(run=run).generate(video, tmp_path / "thumbnail.jpg", 30.0)

    assert not (tmp_path / "thumbnail.jpg").exists()
    assert not list(tmp_path.glob(".thumbnail.*.jpg"))
