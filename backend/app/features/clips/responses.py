"""Computed response fields for returned clip items."""

from __future__ import annotations

from typing import assert_never

from backend.app.features.clips.schemas import ClipManifestResponse
from backend.app.features.clips.store import ClipManifest, ClipStore, LocatedClip


def clip_response(
    manifest: ClipManifest,
    size_bytes: int | None,
    thumbnail_available: bool,
    scene_available: bool = False,
) -> ClipManifestResponse:
    return ClipManifestResponse.model_validate(
        {
            **manifest.as_response(),
            "size_bytes": size_bytes,
            "thumbnail_available": thumbnail_available,
            "scene_available": scene_available,
        }
    )


def resolved_video_size(
    store: ClipStore,
    clip: ClipManifest | LocatedClip,
) -> int | None:
    try:
        match clip:
            case LocatedClip(manifest=manifest):
                if not manifest.video_available:
                    return None
                video_path = store.resolve_located_video_path(clip)
            case ClipManifest() as manifest:
                if not manifest.video_available:
                    return None
                video_path = store.resolve_video_path(clip)
            case unreachable:
                assert_never(unreachable)
        return video_path.stat().st_size
    except (ValueError, FileNotFoundError, OSError):
        return None


__all__ = ["clip_response", "resolved_video_size"]
