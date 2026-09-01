"""Descriptor-pinned access to claimed clip scene-index sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from backend.app.features.clips.descriptor_files import (
    OpenedRegularFile,
    open_contained_regular_file,
)
from backend.app.features.clips.manifest import SceneIndexClaim

MAX_SCENE_INDEX_BYTES: Final = 8 * 1024 * 1024


def resolve_scene_index(
    root: Path,
    manifest_path: Path,
    claim: SceneIndexClaim | None,
) -> OpenedRegularFile | None:
    """Open a claimed sidecar only when its descriptor still matches its size."""
    if claim is None:
        return None
    try:
        opened = open_contained_regular_file(root, manifest_path.parent / claim.path)
    except FileNotFoundError:
        return None
    if opened.size_bytes != claim.size_bytes:
        opened.handle.close()
        return None
    return opened


__all__ = ["MAX_SCENE_INDEX_BYTES", "resolve_scene_index"]
