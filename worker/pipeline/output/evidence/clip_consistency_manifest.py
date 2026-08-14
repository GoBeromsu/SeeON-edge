"""Validated authoritative final manifests and same-ID staging discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_io import (
    validate_directory,
    validate_regular,
)
from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError
from worker.pipeline.output.evidence.evidence_manifest import (
    ClipEvidenceError,
    ReadyClipManifest,
    UnavailableClipManifest,
    parse_manifest,
    verify_ready_manifest,
)


@dataclass(frozen=True, slots=True)
class ManifestAuthority:
    desired: dict[str, tuple[str, ...]]
    ready_count: int
    unavailable_count: int
    staging: tuple[Path, ...]


def scan_manifest_authority(
    clip_store: Path,
    *,
    expected_uid: int,
    ffprobe_bin: str,
) -> ManifestAuthority:
    validate_directory(
        clip_store,
        expected_uid=expected_uid,
        owner_controlled=False,
        label="clip store",
    )
    clips_root = clip_store / "clips"
    staging_root = clips_root / ".staging"
    for path, label in ((clips_root, "clips root"), (staging_root, "staging root")):
        validate_directory(
            path,
            expected_uid=expected_uid,
            owner_controlled=False,
            label=label,
        )
    for entry in staging_root.iterdir():
        if entry.is_symlink():
            raise ClipConsistencyError("unsafe_path", "staging contains a symlink")
    desired: dict[str, tuple[str, ...]] = {}
    ready = unavailable = 0
    for clip_dir in sorted(clips_root.iterdir(), key=lambda candidate: candidate.name):
        if clip_dir.name == ".staging":
            continue
        validate_directory(
            clip_dir,
            expected_uid=expected_uid,
            owner_controlled=False,
            label="final clip",
        )
        for child in clip_dir.iterdir():
            if child.is_symlink():
                raise ClipConsistencyError("unsafe_path", "final clip contains a symlink")
        manifest_path = clip_dir / "manifest.json"
        validate_regular(
            manifest_path,
            expected_uid=expected_uid,
            exact_mode=None,
            label="final manifest",
        )
        try:
            manifest = parse_manifest(manifest_path)
            if manifest.clip_id != clip_dir.name:
                raise ClipConsistencyError("final_invalid", "final identity mismatch")
            match manifest:
                case ReadyClipManifest():
                    media = clip_dir / "clip.mp4"
                    validate_regular(
                        media,
                        expected_uid=expected_uid,
                        exact_mode=None,
                        label="final media",
                    )
                    verify_ready_manifest(manifest, media, ffprobe_bin=ffprobe_bin)
                    ready += 1
                case UnavailableClipManifest():
                    if (clip_dir / "clip.mp4").exists():
                        raise ClipConsistencyError(
                            "final_invalid", "unavailable final contains media"
                        )
                    unavailable += 1
            desired[clip_dir.name] = tuple(manifest.event_refs)
        except ClipEvidenceError as exc:
            raise ClipConsistencyError("final_invalid", "final authority invalid") from exc
    overlaps: list[Path] = []
    for clip_id in desired:
        candidate = staging_root / clip_id
        if candidate.exists() or candidate.is_symlink():
            validate_directory(
                candidate,
                expected_uid=expected_uid,
                owner_controlled=False,
                label="same-ID staging",
            )
            overlaps.append(candidate)
    return ManifestAuthority(desired, ready, unavailable, tuple(overlaps))


__all__ = ["ManifestAuthority", "scan_manifest_authority"]
