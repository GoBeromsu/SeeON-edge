#!/usr/bin/env python3
"""Independently verify catalog records, manifests, and clip media."""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from backend.app.features.clips.catalog import (
    CatalogRecord,
    CatalogStore,
    sanitized_camera_payload,
    strict_camera_snapshot,
    strict_manifest_records,
)
from backend.app.features.clips.store import ClipStore
from backend.app.shared.state_dir import resolve_state_dir

parser = argparse.ArgumentParser()
parser.add_argument(
    "--catalog",
    type=Path,
    default=resolve_state_dir("ml-api") / "catalog.sqlite3",
)
parser.add_argument("--clip-store", type=Path, required=True)
parser.add_argument(
    "--cameras",
    type=Path,
    default=resolve_state_dir("ml-api") / "cameras.json",
)
args = parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as media:
        for chunk in iter(lambda: media.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_media_file(clip_directory: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
        for path in clip_directory.iterdir()
    )


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _safe_snapshot_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("invalid snapshot path")
    relative = Path(value)
    if relative.parts[:1] != ("snapshots",) or ".." in relative.parts:
        raise ValueError("invalid snapshot path")
    candidate = root / relative
    resolved = candidate.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if resolved_root not in resolved.parents:
        raise ValueError("snapshot path escapes clip store")
    return candidate


def _snapshot_file_paths(root: Path) -> set[str]:
    snapshots_dir = root / "snapshots"
    if not snapshots_dir.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in snapshots_dir.rglob("*")
        if (path.is_file() and not path.is_symlink()) or path.is_symlink()
    }


def _canonical_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _canonical_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return False
    canonical = parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value == canonical


def _validate_snapshot_payload(snapshot: dict[str, Any]) -> None:
    required = {
        "snapshot_id",
        "path",
        "sha256",
        "size_bytes",
        "mime_type",
        "captured_at",
        "camera_id",
        "edge_event_id",
    }
    if set(snapshot) != required:
        raise ValueError("snapshot fields do not match the relay contract")
    snapshot_id = snapshot["snapshot_id"]
    camera_id = snapshot["camera_id"]
    captured_at = snapshot["captured_at"]
    edge_event_id = snapshot["edge_event_id"]
    if (
        not _canonical_uuid(snapshot_id)
        or edge_event_id != snapshot_id
        or not _canonical_uuid(edge_event_id)
    ):
        raise ValueError("snapshot identity is invalid")
    if not isinstance(camera_id, str) or not camera_id:
        raise ValueError("snapshot camera_id is invalid")
    if not _canonical_utc_timestamp(captured_at):
        raise ValueError("snapshot captured_at is invalid")
    if snapshot["mime_type"] != "image/jpeg":
        raise ValueError("snapshot mime_type is invalid")
    if (
        not isinstance(snapshot["sha256"], str)
        or not _SHA256_RE.fullmatch(snapshot["sha256"])
        or isinstance(snapshot["size_bytes"], bool)
        or not isinstance(snapshot["size_bytes"], int)
        or snapshot["size_bytes"] <= 0
    ):
        raise ValueError("snapshot media declaration is invalid")
    camera_key = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:16]
    snapshot_key = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
    date = datetime.fromisoformat(captured_at.replace("Z", "+00:00")).date().isoformat()
    expected_path = f"snapshots/{camera_key}/{date}/{snapshot_key}.jpg"
    if snapshot["path"] != expected_path:
        raise ValueError("snapshot path does not match its identity")


def _snapshot_metadata_errors(
    root: Path, snapshots: list[CatalogRecord]
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    malformed: list[str] = []
    missing: list[str] = []
    size_mismatches: list[str] = []
    sha256_mismatches: list[str] = []
    declared_paths: set[str] = set()
    for record in snapshots:
        snapshot = record.payload
        identifier = record.key
        try:
            _validate_snapshot_payload(snapshot)
            path = _safe_snapshot_path(root, snapshot["path"])
        except (TypeError, ValueError):
            malformed.append(identifier)
            continue
        relative = path.relative_to(root).as_posix()
        if relative in declared_paths:
            malformed.append(identifier)
            continue
        declared_paths.add(relative)
        try:
            info = path.lstat()
        except OSError:
            missing.append(identifier)
            continue
        if not stat.S_ISREG(info.st_mode):
            malformed.append(identifier)
            continue
        if info.st_size != snapshot["size_bytes"]:
            size_mismatches.append(identifier)
        try:
            actual_sha256 = _sha256(path)
        except OSError:
            missing.append(identifier)
            continue
        if actual_sha256 != snapshot["sha256"]:
            sha256_mismatches.append(identifier)
    orphan_files = sorted(_snapshot_file_paths(root) - declared_paths)
    return (
        sorted(malformed),
        sorted(missing),
        sorted(size_mismatches),
        sorted(sha256_mismatches),
        orphan_files,
    )


def _promoted_payload_mismatches(records: list[CatalogRecord], key_column: str) -> list[str]:
    mismatches: list[str] = []
    for record in records:
        if record.payload.get(key_column) != record.key or any(
            record.columns[column] != record.payload.get(column) for column in record.columns
        ):
            mismatches.append(record.key)
    return sorted(mismatches)


catalog = CatalogStore.open(args.catalog)
try:
    clip_store = ClipStore(args.clip_store)
    indexed = {record["clip_id"]: record for record in catalog.list_clips()}
    manifests: dict[str, dict[str, Any]] = {}
    media: dict[str, tuple[int, str]] = {}
    manifest_errors: list[str] = []
    media_errors: list[str] = []
    media_absent_declared: list[str] = []
    unavailable_media_present: list[str] = []
    state_counts: Counter[str] = Counter()
    try:
        strict_records = strict_manifest_records(clip_store)
    except (TypeError, ValueError) as exc:
        manifest_errors.append(str(exc))
        strict_records = []
    for record in strict_records:
        item = record.manifest
        manifest_path = record.path
        payload = record.payload
        manifests[item.clip_id] = payload
        state = payload.get("state")
        state_counts[str(state) if state is not None else "MISSING"] += 1
        if state == "UNAVAILABLE":
            media_absent_declared.append(item.clip_id)
            if _has_media_file(manifest_path.parent):
                unavailable_media_present.append(item.clip_id)
            continue
        try:
            media_path = clip_store.resolve_video_path(item)
            media[item.clip_id] = (media_path.stat().st_size, _sha256(media_path))
        except (OSError, ValueError):
            media_errors.append(item.clip_id)

    orphan_records = sorted(set(indexed) - set(manifests))
    unindexed_files = sorted(set(manifests) - set(indexed))
    record_mismatches = sorted(
        clip_id
        for clip_id in set(indexed) & set(manifests)
        if indexed[clip_id] != manifests[clip_id]
    )
    promoted_mismatches = _promoted_payload_mismatches(
        catalog.records_with_columns("clips"), "clip_id"
    )
    indexed_cameras = {record["id"]: record for record in catalog.records("cameras")}
    camera_snapshot = strict_camera_snapshot(args.cameras)
    camera_source = str(args.cameras) if camera_snapshot is not None else "absent"
    cameras = {
        camera["id"]: sanitized_camera_payload(camera)
        for camera in (camera_snapshot or {"cameras": []})["cameras"]
        if isinstance(camera.get("id"), str) and camera["id"]
    }
    orphan_camera_records = sorted(set(indexed_cameras) - set(cameras))
    unindexed_cameras = sorted(set(cameras) - set(indexed_cameras))
    camera_mismatches = sorted(
        camera_id
        for camera_id in set(indexed_cameras) & set(cameras)
        if indexed_cameras[camera_id] != cameras[camera_id]
    )
    camera_promoted_mismatches = _promoted_payload_mismatches(
        catalog.records_with_columns("cameras"), "id"
    )
    snapshot_records = catalog.records_with_columns("snapshots")
    snapshot_promoted_mismatches = _promoted_payload_mismatches(snapshot_records, "snapshot_id")
    (
        malformed_snapshot_paths,
        missing_snapshots,
        snapshot_size_mismatches,
        snapshot_sha256_mismatches,
        orphan_snapshot_files,
    ) = _snapshot_metadata_errors(clip_store.root, snapshot_records)
    media_sha256_mismatches = sorted(
        clip_id
        for clip_id, (_, actual_sha256) in media.items()
        if manifests[clip_id].get("sha256") != actual_sha256
        or indexed.get(clip_id, {}).get("sha256") != actual_sha256
    )
    media_size_mismatches = sorted(
        clip_id
        for clip_id, (actual_size, _) in media.items()
        if manifests[clip_id].get("size_bytes") != actual_size
        or indexed.get(clip_id, {}).get("size_bytes") != actual_size
    )
    mismatch_count = sum(
        map(
            len,
            (
                orphan_records,
                unindexed_files,
                manifest_errors,
                media_errors,
                record_mismatches,
                promoted_mismatches,
                media_sha256_mismatches,
                media_size_mismatches,
                unavailable_media_present,
                snapshot_promoted_mismatches,
                malformed_snapshot_paths,
                missing_snapshots,
                snapshot_size_mismatches,
                snapshot_sha256_mismatches,
                orphan_snapshot_files,
                orphan_camera_records,
                unindexed_cameras,
                camera_mismatches,
                camera_promoted_mismatches,
            ),
        )
    )
    state_summary = (
        ",".join(f"{state}:{count}" for state, count in sorted(state_counts.items())) or "none"
    )
    print(
        f"catalog records={len(indexed)} files={len(manifests)} "
        f"states={state_summary} media_present={len(media)} "
        f"media_absent_declared={len(media_absent_declared)} "
        f"orphan_records={len(orphan_records)} unindexed_files={len(unindexed_files)} "
        f"manifest_errors={len(manifest_errors)} media_errors={len(media_errors)} "
        f"record_mismatches={len(record_mismatches)} "
        f"promoted_mismatches={len(promoted_mismatches)} "
        f"media_sha256_mismatches={len(media_sha256_mismatches)} "
        f"media_size_mismatches={len(media_size_mismatches)} "
        f"snapshot_promoted_mismatches={len(snapshot_promoted_mismatches)} "
        f"unavailable_media_present={len(unavailable_media_present)} "
        f"malformed_snapshot_paths={len(malformed_snapshot_paths)} "
        f"missing_snapshots={len(missing_snapshots)} "
        f"snapshot_sha256_mismatches={len(snapshot_sha256_mismatches)} "
        f"snapshot_size_mismatches={len(snapshot_size_mismatches)} "
        f"orphan_snapshot_files={len(orphan_snapshot_files)} "
        f"camera_source={camera_source} cameras={len(cameras)} "
        f"orphan_camera_records={len(orphan_camera_records)} "
        f"unindexed_cameras={len(unindexed_cameras)} camera_mismatches={len(camera_mismatches)} "
        f"camera_promoted_mismatches={len(camera_promoted_mismatches)}"
    )
    for category, values in (
        ("orphan record", orphan_records),
        ("unindexed file", unindexed_files),
        ("manifest error", manifest_errors),
        ("media error", media_errors),
        ("record mismatch", record_mismatches),
        ("promoted mismatch", promoted_mismatches),
        ("media sha256 mismatch", media_sha256_mismatches),
        ("media size mismatch", media_size_mismatches),
        ("snapshot promoted mismatch", snapshot_promoted_mismatches),
        ("snapshot malformed path", malformed_snapshot_paths),
        ("missing snapshot", missing_snapshots),
        ("snapshot sha256 mismatch", snapshot_sha256_mismatches),
        ("snapshot size mismatch", snapshot_size_mismatches),
        ("orphan snapshot file", orphan_snapshot_files),
        ("orphan camera record", orphan_camera_records),
        ("unindexed camera", unindexed_cameras),
        ("camera mismatch", camera_mismatches),
        ("camera promoted mismatch", camera_promoted_mismatches),
        ("unavailable media present", unavailable_media_present),
    ):
        for clip_id in values:
            print(f"{category}: {clip_id}")
    raise SystemExit(1 if mismatch_count else 0)
finally:
    catalog.close()
