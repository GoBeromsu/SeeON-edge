"""Project verified v17 and filesystem facts into a schema-18 candidate."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Final

from backend.app.edge_db.compact_artifact_projection import project_artifacts
from backend.app.edge_db.compact_configuration_projection import project_configuration
from backend.app.edge_db.compact_policy_projection import project_policies
from backend.app.features.clips.manifest import (
    discover_manifest_paths,
    read_manifest_file,
    video_file_from_dir,
)

_ZERO_HASH: Final = "0" * 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> None:
    stat = path.lstat()
    if path.is_symlink() or not path.is_file() or stat.st_nlink < 1:
        raise sqlite3.DatabaseError(f"unsafe cutover file: {path}")


def filesystem_inventory_sha256(clip_store: Path, worker_state: Path) -> str:
    """Hash names, sizes, and contents for manifests, media, queue, and dead-letter."""
    digest = hashlib.sha256()
    roots = (
        clip_store,
        worker_state / "delivery-queue",
        worker_state / "delivery-queue-dead-letter",
    )
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise sqlite3.DatabaseError(f"unsafe cutover inventory root: {root}")
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            _regular(path)
            relative = f"{root.name}/{path.relative_to(root).as_posix()}".encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def require_filesystem_drain(clip_store: Path, worker_state: Path) -> None:
    """Refuse queued, dead-lettered, staged clip, or staged snapshot evidence."""
    pending: list[Path] = []
    for directory in (
        worker_state / "delivery-queue",
        worker_state / "delivery-queue-dead-letter",
    ):
        if directory.exists():
            pending.extend(path for path in directory.rglob("*") if path.is_file())
    for directory in (clip_store / "clips" / ".staging", clip_store / ".snapshot-staging"):
        if directory.exists():
            pending.extend(path for path in directory.rglob("*") if path.is_file() or path.is_dir())
    if pending:
        raise sqlite3.DatabaseError("EDGE_DB_FILESYSTEM_DRAIN_INCOMPLETE")


def _copy_credentials(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    row = source.execute(
        "SELECT id,username,algorithm,salt,password_hash,updated_at FROM credentials WHERE id=1"
    ).fetchone()
    if row is not None:
        target.execute("INSERT INTO credentials VALUES (?,?,?,?,?,?)", row)


def _review(
    source: sqlite3.Connection, incident_id: str
) -> tuple[int, str | None, str | None, str | None, str | None]:
    row = source.execute(
        "SELECT revision.review_version,revision.disposition,revision.actor_id,"
        "revision.reviewed_at,revision.notes FROM control_evidence_review_state AS state "
        "JOIN control_evidence_review_revisions AS revision "
        "ON revision.incident_id=state.incident_id "
        "AND revision.review_version=state.current_version WHERE state.incident_id=?",
        (incident_id,),
    ).fetchone()
    if row is None:
        return (0, None, None, None, None)
    disposition = {"TRUE_POSITIVE": "TP", "FALSE_POSITIVE": "FP"}.get(str(row[1]))
    if disposition is None:
        return (0, None, None, None, None)
    return (int(row[0]), disposition, str(row[2]), str(row[3]), row[4])


def _copy_incidents(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    facility_row = source.execute(
        "SELECT facility_id FROM connection_settings WHERE id=1"
    ).fetchone()
    facility_id = (
        "unknown" if facility_row is None or facility_row[0] is None else str(facility_row[0])
    )
    events = {
        str(row[0]): row
        for row in source.execute(
            "SELECT edge_event_id,backend_event_id FROM evidence_events ORDER BY edge_event_id"
        )
    }
    for row in source.execute(
        "SELECT incident_id,edge_event_id,camera_id,event_type,detected_at,"
        "runtime_manifest_sha256,module_qualified_id,policy_qualified_id,provenance_state,"
        "provenance_missing_reason,lifecycle_state,failure_reason,revision,created_at,updated_at "
        "FROM evidence_incidents ORDER BY incident_id"
    ):
        incident_id, edge_event_id = str(row[0]), str(row[1])
        event = events.get(edge_event_id)
        backend_event_id = None if event is None else event[1]
        qualified = (
            row[8] == "QUALIFIED"
            and backend_event_id is not None
            and row[5] is not None
            and row[6] is not None
            and row[7] is not None
        )
        provenance = "QUALIFIED" if qualified else "MISSING"
        missing_reason = None if qualified else str(row[9] or "NOT_RECORDED")[:64]
        review = _review(source, incident_id)
        lifecycle = row[10] if row[10] in {"COMPLETE", "FAILED"} else "OPEN"
        failure_reason = row[11] if lifecycle == "FAILED" else None
        target.execute(
            "INSERT INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                incident_id,
                edge_event_id,
                facility_id,
                row[2],
                row[3],
                None,
                row[4],
                lifecycle,
                failure_reason,
                backend_event_id if qualified else None,
                row[5] if qualified else None,
                row[6] if qualified else None,
                row[7] if qualified else None,
                provenance,
                missing_reason,
                *review,
                max(int(row[12]), int(review[0]) + 1),
                row[13],
                row[14],
            ),
        )


def _copy_manifests(clip_store: Path, target: sqlite3.Connection) -> None:
    seen: set[str] = set()
    for manifest_path in sorted(discover_manifest_paths(clip_store)):
        _regular(manifest_path)
        manifest = read_manifest_file(manifest_path)
        if manifest is None:
            raise sqlite3.DatabaseError(f"malformed clip manifest: {manifest_path}")
        if manifest.clip_id in seen:
            raise sqlite3.DatabaseError(f"duplicate clip manifest: {manifest.clip_id}")
        seen.add(manifest.clip_id)
        media = video_file_from_dir(manifest_path.parent, manifest.clip_id)
        _regular(media)
        thumbnail = manifest_path.parent / "thumbnail.jpg"
        if thumbnail.exists():
            _regular(thumbnail)
        facet = manifest.event_type if manifest.event_type in {"fall", "bed-exit"} else "other"
        relative_manifest = manifest_path.relative_to(clip_store).as_posix()
        relative_media = media.relative_to(clip_store).as_posix()
        relative_thumbnail = (
            thumbnail.relative_to(clip_store).as_posix() if thumbnail.exists() else None
        )
        target.execute(
            "INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                manifest.clip_id,
                manifest.camera_id,
                facet,
                manifest.started_at,
                manifest.started_at if manifest.finalized else None,
                max(0, round(manifest.duration_s * 1000)),
                manifest.codec or None,
                "video/mp4",
                relative_manifest,
                relative_media,
                relative_thumbnail,
                _sha256(manifest_path),
                _sha256(media),
                _sha256(thumbnail) if thumbnail.exists() else None,
                manifest_path.stat().st_size,
                media.stat().st_size,
                thumbnail.stat().st_size if thumbnail.exists() else None,
                "AVAILABLE",
                None,
                "WAITING",
                None,
                None,
                "RETAINED",
                None,
                None,
                None,
                1,
                manifest.started_at,
                manifest.started_at,
            ),
        )


def verify_manifest_projection(clip_store: Path, candidate: Path) -> None:
    """Prove every verified manifest/media pair has exact candidate hashes."""
    connection = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)
    try:
        manifests = sorted(discover_manifest_paths(clip_store))
        for manifest_path in manifests:
            manifest = read_manifest_file(manifest_path)
            if manifest is None:
                raise sqlite3.DatabaseError(f"malformed clip manifest: {manifest_path}")
            media = video_file_from_dir(manifest_path.parent, manifest.clip_id)
            expected = (_sha256(manifest_path), _sha256(media), media.stat().st_size)
            actual = connection.execute(
                "SELECT manifest_sha256,media_sha256,media_size_bytes FROM clips WHERE clip_id=?",
                (manifest.clip_id,),
            ).fetchone()
            if actual != expected:
                raise sqlite3.DatabaseError(f"clip manifest projection differs: {manifest.clip_id}")
        if connection.execute("SELECT count(*) FROM clips").fetchone() != (len(manifests),):
            raise sqlite3.DatabaseError("candidate contains a clip absent from inventory")
    finally:
        connection.close()


def project_compact_data(source_path: Path, candidate: Path, clip_store: Path) -> None:
    """Insert locked current facts after schema 18 has created empty target authorities."""
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    target = sqlite3.connect(candidate)
    try:
        target.execute("PRAGMA foreign_keys=ON")
        target.execute("BEGIN IMMEDIATE")
        _copy_credentials(source, target)
        project_configuration(source, target)
        project_policies(source, target)
        _copy_incidents(source, target)
        _copy_manifests(clip_store, target)
        project_artifacts(source, target)
        target.commit()
    finally:
        target.close()
        source.close()


__all__ = [
    "filesystem_inventory_sha256",
    "project_compact_data",
    "require_filesystem_drain",
    "verify_manifest_projection",
]
