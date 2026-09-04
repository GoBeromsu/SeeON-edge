"""Filesystem discovery and compatibility parsing for clip manifests."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from pydantic import JsonValue, TypeAdapter, ValidationError

_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9:_-]{1,128}$")
_MEDIA_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
_MANIFEST_PAYLOAD = TypeAdapter(dict[str, JsonValue])
_EXTENSION_BOUNDARIES = {"none", "extension_bounded", "extension_raced"}


class ExtensionContributorPayload(TypedDict):
    event_ref: str
    detected_at: str


class ClipExtensionPayload(TypedDict):
    contributors: list[ExtensionContributorPayload]
    duration_s: float
    boundary: str


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
    detected_at: str | None
    truncation_reasons: list[str]
    extension: ClipExtensionPayload | None


@dataclass(frozen=True, slots=True)
class ExtensionContributor:
    event_ref: str
    detected_at: str


@dataclass(frozen=True, slots=True)
class ClipExtension:
    contributors: tuple[ExtensionContributor, ...]
    duration_s: float
    boundary: str


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
    detected_at: str | None = None
    truncation_reasons: tuple[str, ...] = ()
    event_refs: tuple[str, ...] = ()
    extension: ClipExtension | None = None

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
            "detected_at": self.detected_at,
            "truncation_reasons": list(self.truncation_reasons),
            "extension": (
                {
                    "contributors": [
                        {
                            "event_ref": contributor.event_ref,
                            "detected_at": contributor.detected_at,
                        }
                        for contributor in self.extension.contributors
                    ],
                    "duration_s": self.extension.duration_s,
                    "boundary": self.extension.boundary,
                }
                if self.extension is not None
                else None
            ),
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
    detected_at = _optional_rfc3339_z(data.get("detected_at"))
    codec = _text(data.get("codec"))
    path = _text(data.get("path")) or None
    video_error = _text(data.get("reason_code")) or _text(data.get("video_error")) or None
    video_available_raw = data.get("video_available")
    finalized = data.get("finalized")
    duration_s_raw = data.get("duration_s")
    if (
        not all((clip_id, camera_id, event_ref, started_at))
        or not is_valid_clip_id(clip_id)
        or (data.get("detected_at") is not None and detected_at is None)
    ):
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
    truncation_reasons = _truncation_reasons(data.get("truncation_reasons"))
    event_refs = _event_refs(data.get("event_refs"), event_ref)
    extension = _extension(data.get("extension"))
    if data.get("extension") is not None and extension is None:
        return None
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
        detected_at=detected_at,
        truncation_reasons=truncation_reasons,
        event_refs=event_refs,
        extension=extension,
    )


def _text(value: JsonValue | None) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _optional_rfc3339_z(value: JsonValue | None) -> str | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


def _truncation_reasons(value: JsonValue | None) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(reason, str) or not reason for reason in value
    ):
        return ()
    return tuple(value)


def _event_refs(value: JsonValue | None, event_ref: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(reference, str) or not reference.strip() for reference in value
    ):
        return (event_ref,)
    references = tuple(dict.fromkeys(reference.strip() for reference in value))
    return references or (event_ref,)


def _extension(value: JsonValue | None) -> ClipExtension | None:
    if not isinstance(value, dict) or set(value) != {
        "contributors",
        "duration_s",
        "boundary",
    }:
        return None
    boundary = value["boundary"]
    duration_s = value["duration_s"]
    contributors = value["contributors"]
    if (
        not isinstance(boundary, str)
        or boundary not in _EXTENSION_BOUNDARIES
        or isinstance(duration_s, bool)
        or not isinstance(duration_s, int | float)
        or not math.isfinite(duration_s)
        or duration_s < 0
        or not isinstance(contributors, list)
        or not contributors
    ):
        return None
    parsed_contributors: list[ExtensionContributor] = []
    for contributor in contributors:
        if not isinstance(contributor, dict) or set(contributor) != {"event_ref", "detected_at"}:
            return None
        event_ref = contributor["event_ref"]
        detected_at = contributor["detected_at"]
        if not isinstance(event_ref, str) or not event_ref.strip():
            return None
        parsed_detected_at = _optional_rfc3339_z(detected_at)
        if parsed_detected_at is None:
            return None
        parsed_contributors.append(
            ExtensionContributor(event_ref=event_ref, detected_at=parsed_detected_at)
        )
    return ClipExtension(
        contributors=tuple(parsed_contributors),
        duration_s=float(duration_s),
        boundary=boundary,
    )


__all__ = [
    "ClipExtension",
    "ClipManifest",
    "ExtensionContributor",
    "discover_manifest_paths",
    "is_valid_clip_id",
    "read_manifest_file",
    "video_file_from_dir",
]
