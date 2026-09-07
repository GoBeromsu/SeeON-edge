"""Atomic publication of browser playback rendition bundles."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from worker.pipeline.output.evidence.durability import fsync_directory
from worker.pipeline.output.evidence.playback_rendition import (
    PLAYBACK_MANIFEST_NAME,
    PLAYBACK_RENDITION_PREFIX,
    PlaybackRenditionError,
    VideoTiming,
    probe_video_codec,
    read_video_timing,
)

LOGGER = logging.getLogger(__name__)


class _ManifestPublicationUncertain(OSError):
    """The manifest replacement may already be visible on durable storage."""


def write_playback_rendition(
    clip_path: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_s: float = 120.0,
    read_timing: Callable[[Path], VideoTiming] | None = None,
) -> Path | None:
    """Publish a digest-addressed rendition and its single attestation pointer."""
    if probe_video_codec(clip_path, run) in {"avc1", "h264"}:
        return None
    timing_reader = read_video_timing if read_timing is None else read_timing
    source_timing = timing_reader(clip_path)
    source_digest = _verified_source_sha256(clip_path)
    temporary = _temporary_path(clip_path.parent)
    rendition: Path | None = None
    published_new = False
    try:
        _transcode(clip_path, temporary, source_timing, run, timeout_s)
        rendition_timing = timing_reader(temporary)
        rendition_digest = _sha256(temporary)
        rendition = clip_path.with_name(f"{PLAYBACK_RENDITION_PREFIX}{rendition_digest[:16]}.mp4")
        _fsync_file(temporary)
        if rendition.exists():
            temporary.unlink()
        else:
            os.replace(temporary, rendition)
            published_new = True
            fsync_directory(rendition.parent)
        _write_manifest(
            clip_path.with_name(PLAYBACK_MANIFEST_NAME),
            _attestation(
                source_timing, rendition_timing, source_digest, rendition_digest, rendition.name
            ),
        )
    except _ManifestPublicationUncertain as exc:
        temporary.unlink(missing_ok=True)
        raise PlaybackRenditionError("playback rendition manifest durability is uncertain") from exc
    except PlaybackRenditionError:
        temporary.unlink(missing_ok=True)
        _remove_unpublished_rendition(rendition if published_new else None)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        _remove_unpublished_rendition(rendition if published_new else None)
        raise PlaybackRenditionError("could not publish playback rendition") from exc
    _remove_old_renditions(clip_path.parent, rendition)
    return rendition


def _transcode(
    clip_path: Path,
    temporary: Path,
    source_timing: VideoTiming,
    run: Callable[..., subprocess.CompletedProcess[str]],
    timeout_s: float,
) -> None:
    try:
        result = run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(clip_path),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "passthrough",
                "-enc_time_base",
                "-1",
                "-video_track_timescale",
                str(source_timing.time_base_denominator),
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PlaybackRenditionError("ffmpeg failed") from exc
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        raise PlaybackRenditionError("ffmpeg did not create a rendition")


def _verified_source_sha256(clip_path: Path) -> str:
    try:
        payload = json.loads(clip_path.with_name("manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlaybackRenditionError("clip manifest is unreadable") from exc
    digest = payload.get("sha256") if isinstance(payload, dict) else None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise PlaybackRenditionError("clip manifest has no valid source digest")
    try:
        actual_digest = _sha256(clip_path)
    except OSError as exc:
        raise PlaybackRenditionError("could not hash clip source") from exc
    if actual_digest != digest:
        raise PlaybackRenditionError("clip source digest does not match manifest")
    return actual_digest


def _attestation(
    source: VideoTiming,
    rendition: VideoTiming,
    source_digest: str,
    rendition_digest: str,
    rendition_name: str,
) -> dict[str, bool | int | str]:
    return {
        "rendition": rendition_name,
        "rendition_sha256": rendition_digest,
        "source_sha256": source_digest,
        "pts_identical": source == rendition,
        "time_base": f"{source.time_base_numerator}/{source.time_base_denominator}",
        "frames": len(rendition.pts),
        "source_frames": len(source.pts),
    }


def _write_manifest(path: Path, payload: dict[str, bool | int | str]) -> None:
    temporary = _temporary_path(path.parent)
    try:
        with temporary.open("x", encoding="ascii") as output:
            _ = output.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        try:
            fsync_directory(path.parent)
        except OSError as exc:
            raise _ManifestPublicationUncertain from exc
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(parent: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".playback-rendition.", suffix=".mp4", dir=parent)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _remove_unpublished_rendition(rendition: Path | None) -> None:
    if rendition is not None:
        rendition.unlink(missing_ok=True)


def _remove_old_renditions(parent: Path, current: Path) -> None:
    for candidate in parent.glob(f"{PLAYBACK_RENDITION_PREFIX}*.mp4"):
        if candidate == current:
            continue
        try:
            candidate.unlink()
        except OSError as exc:
            LOGGER.warning(
                "playback rendition cleanup failed path=%s exception_class=%s",
                candidate,
                type(exc).__name__,
            )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
