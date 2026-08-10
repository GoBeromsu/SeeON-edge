"""Filesystem discovery and compatibility parsing for clip manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pydantic import JsonValue, TypeAdapter, ValidationError

_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MEDIA_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
_MANIFEST_PAYLOAD = TypeAdapter(dict[str, JsonValue])


class ClipManifestPayload(TypedDict):
    clip_id: str
    camera_id: str
    event_ref: str
    event_type: str | None
    started_at: str
    duration_s: float
    codec: str
    path: str | None
    video_available: bool
    video_error: str | None
    finalized: bool
    size_bytes: int | None
    thumbnail_available: bool


@dataclass(frozen=True, slots=True)
class ClipManifest:
    clip_id: str
    camera_id: str
    event_ref: str
    event_type: str | None
    started_at: str
    duration_s: float
    codec: str
    path: str | None
    video_available: bool
    video_error: str | None
    finalized: bool
    size_bytes: int | None = None
    thumbnail_available: bool = False

    def as_response(self) -> ClipManifestPayload:
        return {
            "clip_id": self.clip_id,
            "camera_id": self.camera_id,
            "event_ref": self.event_ref,
            "event_type": self.event_type,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "codec": self.codec,
            "path": self.path,
            "video_available": self.video_available,
            "video_error": self.video_error,
            "finalized": self.finalized,
            "size_bytes": self.size_bytes,
            "thumbnail_available": self.thumbnail_available,
        }


def discover_manifest_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths = list(root.glob("clips/*/manifest.json"))
    paths.extend(root.glob("*/clips/*/manifest.json"))
    paths.extend(root.glob("*/*/clips/*/manifest.json"))
    return paths


def read_manifest_file(path: Path) -> ClipManifest | None:
    try:
        parsed = _MANIFEST_PAYLOAD.validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None
    return _manifest_from_mapping(parsed)


def is_valid_clip_id(value: str) -> bool:
    return bool(_CLIP_ID_RE.fullmatch(value))


def video_file_from_dir(directory: Path, clip_id: str) -> Path:
    preferred = [
        directory / f"{clip_id}.mp4",
        directory / "clip.mp4",
        directory / "video.mp4",
        directory / "final.mp4",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    media_files = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in _MEDIA_SUFFIXES
    )
    if len(media_files) == 1:
        return media_files[0]
    raise FileNotFoundError(str(directory))


def _manifest_from_mapping(data: Mapping[str, JsonValue]) -> ClipManifest | None:
    clip_id = _text(data.get("clip_id"))
    camera_id = _text(data.get("camera_id"))
    event_ref = _text(data.get("event_ref"))
    event_type = _text(data.get("event_type")) or None
    started_at = _text(data.get("started_at"))
    codec = _text(data.get("codec"))
    path = _text(data.get("path")) or None
    video_error = _text(data.get("reason_code")) or _text(data.get("video_error")) or None
    video_available_raw = data.get("video_available")
    finalized = data.get("finalized")
    duration_s_raw = data.get("duration_s")
    if not all((clip_id, camera_id, event_ref, started_at)) or not is_valid_clip_id(clip_id):
        return None
    if not isinstance(finalized, bool):
        return None
    if isinstance(duration_s_raw, bool) or not isinstance(duration_s_raw, int | float):
        duration_s = 0.0
    else:
        duration_s = float(duration_s_raw)
    if isinstance(video_available_raw, bool):
        video_available = video_available_raw
    else:
        video_available = path is not None
    return ClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        event_ref=event_ref,
        event_type=event_type,
        started_at=started_at,
        duration_s=duration_s,
        codec=codec,
        path=path,
        video_available=video_available,
        video_error=video_error,
        finalized=finalized,
    )


def _text(value: JsonValue | None) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


__all__ = [
    "ClipManifest",
    "discover_manifest_paths",
    "is_valid_clip_id",
    "read_manifest_file",
    "video_file_from_dir",
]
