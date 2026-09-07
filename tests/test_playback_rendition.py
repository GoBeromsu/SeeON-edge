from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from worker.pipeline.output.evidence import playback_rendition_publish
from worker.pipeline.output.evidence.playback_rendition import (
    PLAYBACK_MANIFEST_NAME,
    PLAYBACK_RENDITION_PREFIX,
    PlaybackRenditionError,
    VideoTiming,
    write_playback_rendition,
)
from worker.tools import clip_playback_backfill


def _runner(codec: str, payload: bytes = b"browser-safe-rendition"):
    def run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "ffprobe":
            return subprocess.CompletedProcess(arguments, 0, stdout=f"{codec}\n", stderr="")
        Path(arguments[-1]).write_bytes(payload)
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    return run


def _timing(_path: Path) -> VideoTiming:
    return VideoTiming(1, 12_000, (0, 400, 800))


def _clip(tmp_path: Path, contents: bytes = b"hevc-original") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(contents)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": hashlib.sha256(contents).hexdigest()}), encoding="utf-8"
    )
    return clip


def test_h264_original_does_not_create_playback_rendition(tmp_path: Path) -> None:
    clip = _clip(tmp_path)

    assert write_playback_rendition(clip, run=_runner("h264"), read_timing=_timing) is None
    assert not (tmp_path / PLAYBACK_MANIFEST_NAME).exists()
    assert not list(tmp_path.glob(f"{PLAYBACK_RENDITION_PREFIX}*.mp4"))


def test_hevc_original_publishes_attested_immutable_bundle(tmp_path: Path) -> None:
    clip = _clip(tmp_path)

    result = write_playback_rendition(clip, run=_runner("hevc"), read_timing=_timing)

    digest = hashlib.sha256(b"browser-safe-rendition").hexdigest()
    assert result == tmp_path / f"{PLAYBACK_RENDITION_PREFIX}{digest[:16]}.mp4"
    assert result.read_bytes() == b"browser-safe-rendition"
    assert json.loads((tmp_path / PLAYBACK_MANIFEST_NAME).read_text(encoding="ascii")) == {
        "frames": 3,
        "pts_identical": True,
        "rendition": result.name,
        "rendition_sha256": digest,
        "source_sha256": hashlib.sha256(b"hevc-original").hexdigest(),
        "source_frames": 3,
        "time_base": "1/12000",
    }


def test_manifest_source_digest_mismatch_does_not_publish_rendition(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": "a" * 64}), encoding="utf-8")

    with pytest.raises(PlaybackRenditionError, match="does not match"):
        write_playback_rendition(clip, run=_runner("hevc"), read_timing=_timing)

    assert not (tmp_path / PLAYBACK_MANIFEST_NAME).exists()
    assert not list(tmp_path.glob(f"{PLAYBACK_RENDITION_PREFIX}*.mp4"))


def test_failure_before_rendition_rename_leaves_old_bundle_untouched(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    old_digest = hashlib.sha256(b"old").hexdigest()
    old = tmp_path / f"{PLAYBACK_RENDITION_PREFIX}{old_digest[:16]}.mp4"
    old.write_bytes(b"old")
    old_manifest = {"rendition": old.name, "rendition_sha256": old_digest}
    (tmp_path / PLAYBACK_MANIFEST_NAME).write_text(json.dumps(old_manifest), encoding="ascii")

    def failed_run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "ffprobe":
            return subprocess.CompletedProcess(arguments, 0, stdout="hevc\n", stderr="")
        Path(arguments[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(arguments, 1, stdout="", stderr="failed")

    with pytest.raises(PlaybackRenditionError):
        write_playback_rendition(clip, run=failed_run, read_timing=_timing)

    assert old.read_bytes() == b"old"
    assert (
        json.loads((tmp_path / PLAYBACK_MANIFEST_NAME).read_text(encoding="ascii")) == old_manifest
    )
    assert not list(tmp_path.glob(".playback-rendition.*"))


def test_manifest_rename_failure_preserves_old_bundle_and_removes_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(tmp_path)
    old_digest = hashlib.sha256(b"old").hexdigest()
    old = tmp_path / f"{PLAYBACK_RENDITION_PREFIX}{old_digest[:16]}.mp4"
    old.write_bytes(b"old")
    old_manifest = {"rendition": old.name, "rendition_sha256": old_digest}
    (tmp_path / PLAYBACK_MANIFEST_NAME).write_text(json.dumps(old_manifest), encoding="ascii")

    def fail_manifest(_path: Path, _payload: dict[str, bool | int | str]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "worker.pipeline.output.evidence.playback_rendition_publish._write_manifest",
        fail_manifest,
    )
    with pytest.raises(PlaybackRenditionError):
        write_playback_rendition(clip, run=_runner("hevc"), read_timing=_timing)

    assert old.read_bytes() == b"old"
    assert (
        json.loads((tmp_path / PLAYBACK_MANIFEST_NAME).read_text(encoding="ascii")) == old_manifest
    )
    assert list(tmp_path.glob(f"{PLAYBACK_RENDITION_PREFIX}*.mp4")) == [old]


def test_manifest_directory_fsync_failure_retains_pointer_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(tmp_path)
    calls = 0
    original_fsync_directory = playback_rendition_publish.fsync_directory

    def fail_manifest_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory sync failed")
        original_fsync_directory(path)

    monkeypatch.setattr(playback_rendition_publish, "fsync_directory", fail_manifest_fsync)
    with pytest.raises(PlaybackRenditionError, match="durability is uncertain"):
        write_playback_rendition(clip, run=_runner("hevc"), read_timing=_timing)

    manifest = json.loads((tmp_path / PLAYBACK_MANIFEST_NAME).read_text(encoding="ascii"))
    assert (tmp_path / manifest["rendition"]).is_file()


def test_successful_switch_removes_older_versioned_renditions(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    old = tmp_path / f"{PLAYBACK_RENDITION_PREFIX}{'a' * 16}.mp4"
    old.write_bytes(b"old")

    result = write_playback_rendition(clip, run=_runner("hevc"), read_timing=_timing)

    assert result is not None
    assert list(tmp_path.glob(f"{PLAYBACK_RENDITION_PREFIX}*.mp4")) == [result]


def test_backfill_regenerates_missing_invalid_or_nonidentical_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(tmp_path / "clips" / "clip")
    calls: list[Path] = []
    monkeypatch.setattr(clip_playback_backfill, "probe_video_codec", lambda _path: "hevc")
    monkeypatch.setattr(
        clip_playback_backfill,
        "write_playback_rendition",
        lambda path: calls.append(path) or path,
    )

    summary = clip_playback_backfill.backfill(tmp_path)

    assert summary["created"] == 1
    assert calls == [clip]


def test_backfill_is_idempotent_for_identical_attested_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _clip(tmp_path / "clips" / "clip")
    rendition = clip.parent / f"{PLAYBACK_RENDITION_PREFIX}{'b' * 16}.mp4"
    rendition.write_bytes(b"rendered")
    digest = hashlib.sha256(b"rendered").hexdigest()
    rendition.rename(clip.parent / f"{PLAYBACK_RENDITION_PREFIX}{digest[:16]}.mp4")
    rendition = clip.parent / f"{PLAYBACK_RENDITION_PREFIX}{digest[:16]}.mp4"
    (clip.parent / PLAYBACK_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "rendition": rendition.name,
                "rendition_sha256": digest,
                "source_sha256": hashlib.sha256(b"hevc-original").hexdigest(),
                "pts_identical": True,
                "time_base": "1/12000",
                "frames": 3,
                "source_frames": 3,
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(clip_playback_backfill, "probe_video_codec", lambda _path: "hevc")

    summary = clip_playback_backfill.backfill(tmp_path)

    assert summary["skipped"] == 1
    assert summary["created"] == 0
