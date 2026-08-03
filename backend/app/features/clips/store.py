"""Read-only manifest access and API-owned clip labels."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

CLIP_STORE_DIR_ENV = "CLIP_STORE_DIR"
API_LABEL_STORE_ENV = "API_LABEL_STORE"
DEFAULT_CLIP_STORE_DIR = "/var/lib/clip-store"
DEFAULT_LABEL_STORE_DIR = "/var/lib/ml-api-labels"

LabelValue = Literal["TRUE_POSITIVE", "FALSE_POSITIVE"] | None

_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MEDIA_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


@dataclass(frozen=True)
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
    # Not read from manifest.json (the recorder never writes it there); the
    # router stats the resolved video file at response time and overrides
    # this, so it stays None whenever that stat is unavailable/skipped.
    size_bytes: int | None = None

    def as_response(self) -> dict[str, object]:
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
        }


@dataclass(frozen=True)
class LabelRecord:
    clip_id: str
    label: LabelValue
    reviewer: str
    reviewed_at: str

    def as_response(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "label": self.label,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
        }


class ClipStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def from_env(cls) -> ClipStore:
        return cls(os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))

    def list_manifests(self, *, camera_id: str | None = None) -> list[ClipManifest]:
        manifests: list[ClipManifest] = []
        for manifest_path in self._manifest_paths():
            manifest = self._read_manifest_file(manifest_path)
            if (
                manifest is None
                or not manifest.finalized
                or manifest.clip_id != manifest_path.parent.name
            ):
                continue
            if camera_id is not None and manifest.camera_id != camera_id:
                continue
            manifests.append(manifest)
        return sorted(manifests, key=lambda manifest: manifest.started_at, reverse=True)

    def _manifest_paths(self) -> list[Path]:
        """Manifests can live directly under the store root
        (``root/clips/*/manifest.json`` -- the layout before any storage
        location was ever selected) or nested one or two levels down under a
        chosen ``clip_store_subdir`` (``root/<sub>/clips/*/manifest.json``,
        ``root/<sub2>/<sub1>/clips/*/manifest.json`` -- ``store_subdir`` may
        itself be a multi-segment relative path, see
        ``ClipRecordingConfig.store_subdir``). Listing must keep finding
        clips recorded under any past selection, not just the current one, so
        all three layouts are always checked -- bounded to two subdir levels
        rather than an unbounded recursive walk, since a clip store can
        accumulate many unrelated directories over time.
        """
        if not self.root.is_dir():
            return []
        paths: list[Path] = []
        paths.extend(self.root.glob("clips/*/manifest.json"))
        paths.extend(self.root.glob("*/clips/*/manifest.json"))
        paths.extend(self.root.glob("*/*/clips/*/manifest.json"))
        return paths

    def get_manifest(self, clip_id: str) -> ClipManifest | None:
        if not is_valid_clip_id(clip_id):
            raise ValueError("invalid clip_id")
        manifest = self._read_manifest_file(self.root / "clips" / clip_id / "manifest.json")
        if manifest is None or not manifest.finalized or manifest.clip_id != clip_id:
            return None
        return manifest

    def resolve_video_path(self, manifest: ClipManifest) -> Path:
        if manifest.path is None:
            raise FileNotFoundError(str(self.root))
        raw_path = Path(manifest.path)
        candidate = raw_path if raw_path.is_absolute() else self.root / raw_path
        resolved = candidate.resolve(strict=False)
        root = self.root.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise ValueError("manifest path escapes clip store")
        if resolved.is_dir():
            resolved = _video_file_from_dir(resolved, manifest.clip_id)
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        return resolved

    def _read_manifest_file(self, path: Path) -> ClipManifest | None:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(parsed, dict):
            return None
        return _manifest_from_mapping(parsed)


class LabelStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def from_env(cls) -> LabelStore:
        return cls(os.environ.get(API_LABEL_STORE_ENV, DEFAULT_LABEL_STORE_DIR))

    def save(self, record: LabelRecord) -> bool:
        """Best-effort persist; an unwritable state dir must not crash the caller.

        Mirrors ``RuntimeStatusStore._persist_latency``'s graceful-degradation
        pattern (``backend/app/features/status/runtime_status_store.py``): a
        label write that cannot land durably is dropped (with a warning)
        rather than crashing the labeling request. Returns ``True`` only when
        the record actually landed durably, so the caller (``label_clip`` in
        ``router.py``) can surface a degraded save to the client instead of
        silently reporting success.
        """
        if not is_valid_clip_id(record.clip_id):
            raise ValueError("invalid clip_id")
        labels_dir = self.root / "labels"
        try:
            labels_dir.mkdir(parents=True, exist_ok=True)
            target = labels_dir / f"{record.clip_id}.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(record.as_response(), ensure_ascii=False, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, target)
        except OSError as exc:
            logger.warning("label store unavailable at %s: %s", labels_dir, exc)
            return False
        return True

    def get(self, clip_id: str) -> LabelRecord | None:
        if not is_valid_clip_id(clip_id):
            raise ValueError("invalid clip_id")
        target = self.root / "labels" / f"{clip_id}.json"
        try:
            parsed = json.loads(target.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(parsed, dict):
            return None
        return LabelRecord(
            clip_id=str(parsed.get("clip_id", "")),
            label=_label_value(parsed.get("label")),
            reviewer=str(parsed.get("reviewer", "")),
            reviewed_at=str(parsed.get("reviewed_at", "")),
        )


def is_valid_clip_id(value: str) -> bool:
    return bool(_CLIP_ID_RE.fullmatch(value))


def _manifest_from_mapping(data: dict[str, object]) -> ClipManifest | None:
    clip_id = _text(data.get("clip_id"))
    camera_id = _text(data.get("camera_id"))
    event_ref = _text(data.get("event_ref"))
    event_type = _text(data.get("event_type")) or None
    started_at = _text(data.get("started_at"))
    codec = _text(data.get("codec"))
    path = _text(data.get("path")) or None
    video_error = _text(data.get("video_error")) or None
    video_available_raw = data.get("video_available")
    finalized = data.get("finalized")
    duration_s_raw = data.get("duration_s")
    # Path-less manifests are kept as diagnostic rows so the event->clip
    # correlation survives even when the encoder produced no playable video.
    if not all((clip_id, camera_id, event_ref, started_at)):
        return None
    if not is_valid_clip_id(clip_id):
        return None
    if not isinstance(finalized, bool):
        return None
    # Missing/invalid duration_s defaults to 0.0 rather than dropping the
    # manifest: the wire contract (ClipManifestResponse.duration_s) requires
    # a real, non-negative number, so a manifest can never be represented
    # with duration_s absent -- but it must still be listed (diagnostic value)
    # rather than silently disappearing from GET /clips.
    if isinstance(duration_s_raw, bool) or not isinstance(duration_s_raw, int | float):
        duration_s = 0.0
    else:
        duration_s = float(duration_s_raw)
    if isinstance(video_available_raw, bool):
        video_available = video_available_raw
    else:
        # Legacy manifests predate the field: assume playable when a path exists.
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


def _label_value(value: object) -> LabelValue:
    if value is None or value in ("TRUE_POSITIVE", "FALSE_POSITIVE"):
        return value
    raise ValueError("invalid label")


def _text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _video_file_from_dir(directory: Path, clip_id: str) -> Path:
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


__all__ = [
    "API_LABEL_STORE_ENV",
    "CLIP_STORE_DIR_ENV",
    "ClipManifest",
    "ClipStore",
    "LabelRecord",
    "LabelStore",
    "LabelValue",
    "is_valid_clip_id",
]
