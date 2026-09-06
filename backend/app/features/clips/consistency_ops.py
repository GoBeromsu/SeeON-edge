"""Backend-owned, one-shot repair of evidence clip/event relations.

The final manifest is authoritative.  This module deliberately uses the API
runtime database connection: it must never open a second SQLite path.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.edge_db.ownership import Writer, writer_for_table

_UNAVAILABLE_REASONS = {
    "ENCODER_FAILED",
    "NO_FRAMES",
    "FINALIZE_FAILED",
    "INTERRUPTED_FINALIZE",
    "MISSING",
    "CORRUPT",
}


class ClipConsistencyError(Exception):
    """A safety refusal suitable for showing to an operator."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class RepairCounters:
    relations_before: int
    relations_after: int
    mismatch_clips: int
    mismatch_tuples: int
    sql_relations_deleted: int
    sql_relations_inserted: int

    @property
    def changes(self) -> int:
        return self.mismatch_tuples


@dataclass(frozen=True)
class RepairReceipt:
    format_version: int
    mode: Literal["dry-run", "apply"]
    state: Literal["DRY_RUN", "DONE"]
    schema_version: int
    counters: RepairCounters
    receipt_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "changes": self.counters.changes}


@dataclass(frozen=True)
class RepairRequest:
    clip_store: Path
    maintenance_root: Path
    quiescence_receipt: Path
    apply: bool = False
    expected_owner_uid: int = os.getuid()
    database_path: Path = EDGE_DATABASE_PATH


@dataclass(frozen=True)
class FinalizedClipEvidence:
    """A finalized clip-store outcome verified from its manifest and media."""

    local_state: str
    unavailable_reason: str | None
    manifest_path: Path
    media_relpath: str | None
    sha256: str | None
    size_bytes: int | None
    mime_type: str | None
    codec: str | None
    duration_ms: int | None
    finalized_at: str | None


def repair_clip_consistency(request: RepairRequest) -> RepairReceipt:
    """Plan or apply a manifest-authoritative relation repair.

    Apply is guarded by a short-lived, exact quiescence assertion and stores a
    content-addressed receipt.  The same plan returns the existing receipt,
    making retries idempotent.
    """
    _validate_roots(request)
    if request.apply:
        _validate_quiescence(request)
    connection = open_runtime_database(
        request.database_path, actor=RuntimeActor.API, check_same_thread=False
    )
    try:
        connection.execute("BEGIN IMMEDIATE" if request.apply else "BEGIN")
        schema_version = _validate_database(connection)
        if request.apply and writer_for_table("clip_events") is not Writer.API:
            raise ClipConsistencyError(
                "ownership_migration_required",
                "clip_events is still Worker-owned; run after schema 17 is registered",
            )
        desired = _scan_authority(request.clip_store, request.expected_owner_uid)
        plan, counters = _plan(connection, desired)
        if not request.apply:
            connection.rollback()
            return RepairReceipt(1, "dry-run", "DRY_RUN", schema_version, counters, None)
        identity = _request_identity(request, desired)
        receipt_path = request.maintenance_root / f"clip-consistency-{identity}.json"
        existing = _read_receipt(receipt_path, request.expected_owner_uid)
        if existing is not None:
            if existing.get("relations_after_sha256") != _relations_sha256(connection):
                raise ClipConsistencyError(
                    "receipt_conflict", "database relation preimage differs from receipt"
                )
            connection.rollback()
            return _receipt_from_payload(existing, receipt_path)
        _apply(connection, plan)
        connection.commit()
        receipt = RepairReceipt(1, "apply", "DONE", schema_version, counters, str(receipt_path))
        _write_receipt(receipt_path, request, plan, _relations_sha256(connection), receipt)
        return receipt  # noqa: TRY300
    except sqlite3.Error as exc:
        if connection.in_transaction:
            connection.rollback()
        raise ClipConsistencyError("database_error", "SQLite operation failed") from exc
    finally:
        connection.close()


def _validate_roots(request: RepairRequest) -> None:
    for path, label, mode in (
        (request.clip_store, "clip store", None),
        (request.maintenance_root, "maintenance root", 0o700),
    ):
        try:
            info = path.stat()
        except OSError as exc:
            raise ClipConsistencyError("unsafe_path", f"{label} is unavailable") from exc
        if not path.is_dir() or path.is_symlink() or info.st_uid != request.expected_owner_uid:
            raise ClipConsistencyError("unsafe_path", f"{label} is not a trusted directory")
        if mode is not None and info.st_mode & 0o777 != mode:
            raise ClipConsistencyError("unsafe_path", f"{label} permissions differ")
    if not (request.clip_store / "clips" / ".staging").is_dir():
        raise ClipConsistencyError("unsafe_path", "clip store layout is incomplete")


def _validate_quiescence(request: RepairRequest) -> None:
    path = request.quiescence_receipt
    if path.parent != request.maintenance_root or path.is_symlink():
        raise ClipConsistencyError("quiescence_invalid", "receipt is outside maintenance root")
    try:
        info = path.stat()
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClipConsistencyError("quiescence_invalid", "receipt cannot be read") from exc
    required = {
        "format_version",
        "state_db",
        "clip_store",
        "stopped_service",
        "stopped_db_writers",
        "operator_uid",
        "issued_at",
        "expires_at",
    }
    now = time.time()
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or info.st_uid != request.expected_owner_uid
        or info.st_mode & 0o777 != 0o600
        or payload.get("format_version") != 1
        or payload.get("state_db") != str(request.database_path.resolve())
        or payload.get("clip_store") != str(request.clip_store.resolve())
        or payload.get("stopped_service") != "ml-worker"
        or payload.get("stopped_db_writers") != ["event", "config", "fault"]
        or payload.get("operator_uid") != request.expected_owner_uid
        or not all(
            isinstance(payload.get(key), int) and not isinstance(payload.get(key), bool)
            for key in ("issued_at", "expires_at")
        )
        or not payload["issued_at"] <= now <= payload["expires_at"]
        or not 0 < payload["expires_at"] - payload["issued_at"] <= 3600
    ):
        raise ClipConsistencyError("quiescence_invalid", "receipt assertion differs")


def _validate_database(connection: sqlite3.Connection) -> int:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version < 1 or connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ClipConsistencyError("schema_drift", "database integrity or version differs")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ClipConsistencyError("foreign_key_drift", "database foreign keys are invalid")
    for table in ("evidence_events", "evidence_clips", "clip_events"):
        if (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            is None
        ):
            raise ClipConsistencyError("schema_drift", "evidence relation schema differs")
    active = connection.execute(
        "SELECT 1 FROM evidence_events WHERE state='IN_FLIGHT' AND lease_expires_at > ? LIMIT 1",
        (time.time(),),
    ).fetchone()
    if active is not None:
        raise ClipConsistencyError("active_lease", "active evidence lease exists")
    return version


def _scan_authority(root: Path, uid: int) -> dict[str, tuple[str, ...]]:
    clips = root / "clips"
    desired: dict[str, tuple[str, ...]] = {}
    try:
        directories = sorted(clips.iterdir())
    except OSError as exc:
        raise ClipConsistencyError("final_read_error", "final authority cannot be listed") from exc
    for directory in directories:
        if directory.name == ".staging":
            continue
        try:
            directory_info = directory.lstat()
        except OSError as exc:
            raise ClipConsistencyError(
                "final_read_error", "final authority cannot be read"
            ) from exc
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or stat.S_ISLNK(directory_info.st_mode)
            or directory_info.st_uid != uid
        ):
            raise ClipConsistencyError("final_invalid", "final authority is unsafe")
        evidence, refs = _inspect_finalized_directory(directory, uid)
        del evidence
        desired[directory.name] = refs
    if len([ref for refs in desired.values() for ref in refs]) != len(
        {ref for refs in desired.values() for ref in refs}
    ):
        raise ClipConsistencyError("manifest_conflict", "event has multiple authorities")
    return desired


def inspect_finalized_clip(
    clip_store: Path, clip_id: str, *, clips_dir_fd: int | None = None
) -> FinalizedClipEvidence:
    """Verify one finalized clip using the consistency repair's authority rules."""
    # Guarded on both paths: the pathname variant would join an absolute or
    # ``../`` clip id straight onto the store and escape it just as readily.
    _require_bounded_clip_id(clip_id)
    if clips_dir_fd is not None:
        return _inspect_finalized_clip_at(clip_store, clip_id, clips_dir_fd)
    directory = clip_store / "clips" / clip_id
    try:
        directory_info = directory.lstat()
    except FileNotFoundError:
        raise ClipConsistencyError("missing", "final clip directory is missing") from None
    except OSError as exc:
        raise ClipConsistencyError(
            "final_read_error", "final clip directory cannot be read"
        ) from exc
    if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(directory_info.st_mode):
        raise ClipConsistencyError("final_invalid", "final clip directory is unsafe")
    evidence, _ = _inspect_finalized_directory(directory, os.getuid())
    return evidence


def _require_bounded_clip_id(clip_id: str) -> None:
    """Reject a clip id that could name anything outside the pinned store.

    ``evidence_clips.clip_id`` is unconstrained TEXT, and it is passed straight
    to ``os.stat``/``os.open`` with ``dir_fd``. An absolute value ignores the
    descriptor entirely and a ``../`` value walks out of it, so a matching
    manifest and media planted anywhere on the filesystem could be recorded
    ``VERIFIED`` for a clip in the database, and its publication terminalized on
    that basis. The identifier must therefore name exactly one ordinary path
    component before any filesystem call is made with it.
    """
    if (
        not clip_id
        or "/" in clip_id
        or "\\" in clip_id
        or clip_id in {".", ".."}
        or clip_id != PurePosixPath(clip_id).name
    ):
        raise ClipConsistencyError(
            "clip_id_unsafe",
            "clip id is not a single bounded path component",
        )


def _inspect_finalized_clip_at(
    clip_store: Path, clip_id: str, clips_dir_fd: int
) -> FinalizedClipEvidence:
    """Inspect a clip through the descriptor pinned by legacy recovery."""
    try:
        directory_info = os.stat(clip_id, dir_fd=clips_dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise ClipConsistencyError("missing", "final clip directory is missing") from None
    except OSError as exc:
        raise ClipConsistencyError(
            "final_read_error", "final clip directory cannot be read"
        ) from exc
    if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(directory_info.st_mode):
        raise ClipConsistencyError("final_invalid", "final clip directory is unsafe")
    try:
        directory_fd = os.open(
            clip_id,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=clips_dir_fd,
        )
    except OSError as exc:
        raise ClipConsistencyError(
            "final_read_error", "final clip directory cannot be opened"
        ) from exc
    try:
        evidence, _ = _inspect_finalized_directory_at(
            clip_store / "clips" / clip_id, directory_fd, os.getuid()
        )
    finally:
        os.close(directory_fd)
    return evidence


def _inspect_finalized_directory(
    directory: Path, uid: int
) -> tuple[FinalizedClipEvidence, tuple[str, ...]]:
    manifest_path = directory / "manifest.json"
    try:
        manifest_info = manifest_path.lstat()
        if stat.S_ISLNK(manifest_info.st_mode) or not stat.S_ISREG(manifest_info.st_mode):
            raise ValueError  # noqa: TRY301
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        refs = payload["event_refs"]
        if (
            payload.get("manifest_schema_version") != 2
            or payload.get("clip_id") != directory.name
            or not isinstance(refs, list)
            or not all(isinstance(ref, str) and ref for ref in refs)
            or len(refs) != len(set(refs))
        ):
            raise ValueError  # noqa: TRY301
        if payload.get("state") == "READY":
            media = directory / "clip.mp4"
            media_info = media.lstat()
            if (
                stat.S_ISLNK(media_info.st_mode)
                or not stat.S_ISREG(media_info.st_mode)
                or media_info.st_uid != uid
                or payload.get("path") != f"clips/{directory.name}/clip.mp4"
            ):
                raise ValueError  # noqa: TRY301
            if payload.get("size_bytes") != media_info.st_size or payload.get("sha256") != _sha256(
                media
            ):
                raise ValueError  # noqa: TRY301
            evidence = FinalizedClipEvidence(
                "VERIFIED",
                None,
                manifest_path,
                f"clips/{directory.name}/clip.mp4",
                # Subscripted, not .get(): a missing field must refuse the clip
                # rather than record the literal string "None" as if it were
                # verified metadata. The descriptor-relative variant below is
                # written the same way, and tests pin the two to agree -- a
                # divergence in strictness would classify identical clips
                # differently depending on which path inspected them.
                str(payload["sha256"]),
                int(payload["size_bytes"]),
                str(payload["mime_type"]),
                str(payload["codec"]),
                int(payload["duration_ms"]),
                str(payload["finalized_at"]) if payload.get("finalized_at") is not None else None,
            )
        elif payload.get("state") == "UNAVAILABLE":
            media = directory / "clip.mp4"
            try:
                media.lstat()
            except FileNotFoundError:
                pass
            else:
                raise ValueError  # noqa: TRY301
            reason = payload.get("reason_code")
            evidence = FinalizedClipEvidence(
                "UNAVAILABLE",
                reason if reason in _UNAVAILABLE_REASONS else "MISSING",
                manifest_path,
                None,
                None,
                None,
                None,
                None,
                None,
                str(payload["finalized_at"]) if payload.get("finalized_at") is not None else None,
            )
        else:
            raise ValueError  # noqa: TRY301
    except OSError as exc:
        raise ClipConsistencyError("final_read_error", "final authority cannot be read") from exc
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ClipConsistencyError("final_invalid", "final authority invalid") from exc
    return evidence, tuple(refs)


def _inspect_finalized_directory_at(
    directory: Path, directory_fd: int, uid: int
) -> tuple[FinalizedClipEvidence, tuple[str, ...]]:
    """Read and hash a finalized clip without resolving the store pathname."""
    try:
        manifest_info = os.stat("manifest.json", dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(manifest_info.st_mode) or not stat.S_ISREG(manifest_info.st_mode):
            raise ValueError  # noqa: TRY301
        manifest_fd = os.open("manifest.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        with os.fdopen(manifest_fd, encoding="utf-8") as manifest:
            payload = json.load(manifest)
        refs = payload["event_refs"]
        if (
            payload.get("manifest_schema_version") != 2
            or payload.get("clip_id") != directory.name
            or not isinstance(refs, list)
            or not all(isinstance(ref, str) and ref for ref in refs)
            or len(refs) != len(set(refs))
        ):
            raise ValueError  # noqa: TRY301
        if payload.get("state") == "READY":
            media_info = os.stat("clip.mp4", dir_fd=directory_fd, follow_symlinks=False)
            if (
                stat.S_ISLNK(media_info.st_mode)
                or not stat.S_ISREG(media_info.st_mode)
                or media_info.st_uid != uid
                or payload.get("path") != f"clips/{directory.name}/clip.mp4"
            ):
                raise ValueError  # noqa: TRY301
            media_fd = os.open("clip.mp4", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            with os.fdopen(media_fd, "rb") as media:
                digest = hashlib.file_digest(media, "sha256").hexdigest()
            if payload.get("size_bytes") != media_info.st_size or payload.get("sha256") != digest:
                raise ValueError  # noqa: TRY301
            evidence = FinalizedClipEvidence(
                "VERIFIED",
                None,
                directory / "manifest.json",
                f"clips/{directory.name}/clip.mp4",
                str(payload["sha256"]),
                int(payload["size_bytes"]),
                str(payload["mime_type"]),
                str(payload["codec"]),
                int(payload["duration_ms"]),
                str(payload["finalized_at"]) if payload.get("finalized_at") is not None else None,
            )
        elif payload.get("state") == "UNAVAILABLE":
            try:
                os.stat("clip.mp4", dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError  # noqa: TRY301
            reason = payload.get("reason_code")
            evidence = FinalizedClipEvidence(
                "UNAVAILABLE",
                reason if reason in _UNAVAILABLE_REASONS else "MISSING",
                directory / "manifest.json",
                None,
                None,
                None,
                None,
                None,
                None,
                str(payload["finalized_at"]) if payload.get("finalized_at") is not None else None,
            )
        else:
            raise ValueError  # noqa: TRY301
    except OSError as exc:
        raise ClipConsistencyError("final_read_error", "final authority cannot be read") from exc
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ClipConsistencyError("final_invalid", "final authority invalid") from exc
    return evidence, tuple(refs)


def _plan(
    connection: sqlite3.Connection, desired: dict[str, tuple[str, ...]]
) -> tuple[dict[str, object], RepairCounters]:
    clip_ids = {row[0] for row in connection.execute("SELECT clip_id FROM evidence_clips")}
    refs = [ref for values in desired.values() for ref in values]
    event_ids = {row[0] for row in connection.execute("SELECT edge_event_id FROM evidence_events")}
    if not set(desired) <= clip_ids or not set(refs) <= event_ids:
        raise ClipConsistencyError("database_drift", "manifest authority is absent from database")
    before = tuple(
        connection.execute(
            "SELECT clip_id, edge_event_id, ordinal FROM clip_events "
            "ORDER BY clip_id, ordinal, edge_event_id"
        )
    )
    desired_rows = tuple(
        (clip, ref, ordinal)
        for clip in sorted(desired)
        for ordinal, ref in enumerate(desired[clip])
    )
    scope = {row for row in before if row[0] in desired or row[1] in set(refs)}
    changed = {
        clip
        for clip in desired
        if tuple(row[1:] for row in before if row[0] == clip)
        != tuple(row[1:] for row in desired_rows if row[0] == clip)
    }
    owners = {
        row[0]: row[1]
        for row in connection.execute("SELECT edge_event_id, clip_id FROM clip_events")
        if row[0] in refs
    }
    delete_ids = sorted(
        {row[1] for row in before if row[0] in changed}
        | {ref for ref, owner in owners.items() if owner not in desired}
    )
    inserts = tuple(row for row in desired_rows if row[0] in changed)
    after = tuple(sorted([row for row in before if row[1] not in set(delete_ids)] + list(inserts)))
    counters = RepairCounters(
        len(before),
        len(after),
        len(changed),
        len(scope.symmetric_difference(set(desired_rows))),
        len(delete_ids),
        len(inserts),
    )
    return {"delete_event_ids": delete_ids, "insert_rows": [list(row) for row in inserts]}, counters


def _apply(connection: sqlite3.Connection, plan: dict[str, object]) -> None:
    # The plan is a heterogeneous dict, so narrow before touching the database:
    # an unchecked cast here would let a malformed plan reach DML on evidence.
    delete_ids = plan["delete_event_ids"]
    if not isinstance(delete_ids, Sequence):
        raise ClipConsistencyError("plan_invalid", "delete_event_ids is not a sequence")
    insert_rows = plan["insert_rows"]
    if not isinstance(insert_rows, Sequence):
        raise ClipConsistencyError("plan_invalid", "insert_rows is not a sequence")

    if delete_ids:
        connection.executemany(
            "DELETE FROM clip_events WHERE edge_event_id = ?", ((value,) for value in delete_ids)
        )
    connection.executemany(
        "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, ?)",
        insert_rows,
    )


def _request_identity(request: RepairRequest, desired: dict[str, tuple[str, ...]]) -> str:
    payload = {
        "database": str(request.database_path.resolve()),
        "clip_store": str(request.clip_store.resolve()),
        "manifest_authority": desired,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as media:
        return hashlib.file_digest(media, "sha256").hexdigest()


def _relations_sha256(connection: sqlite3.Connection) -> str:
    rows = tuple(
        connection.execute(
            "SELECT clip_id, edge_event_id, ordinal FROM clip_events "
            "ORDER BY clip_id, ordinal, edge_event_id"
        )
    )
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _read_receipt(path: Path, uid: int) -> dict[str, object] | None:
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_uid != uid or path.stat().st_mode & 0o777 != 0o600:
        raise ClipConsistencyError("receipt_invalid", "existing receipt is unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClipConsistencyError("receipt_invalid", "existing receipt is invalid") from exc
    if not isinstance(payload, dict) or payload.get("state") != "DONE":
        raise ClipConsistencyError("receipt_invalid", "existing receipt differs")
    return payload


def _receipt_from_payload(payload: dict[str, object], path: Path) -> RepairReceipt:
    counters = payload.get("counters")
    if not isinstance(counters, dict):
        raise ClipConsistencyError("receipt_invalid", "existing counters differ")
    schema_version = payload["schema_version"]
    if not isinstance(schema_version, int):
        raise ClipConsistencyError("receipt_invalid", "schema_version is not an integer")
    return RepairReceipt(1, "apply", "DONE", schema_version, RepairCounters(**counters), str(path))


def _write_receipt(
    path: Path,
    request: RepairRequest,
    plan: dict[str, object],
    relations_after_sha256: str,
    receipt: RepairReceipt,
) -> None:
    payload = receipt.to_dict() | {
        "database": str(request.database_path.resolve()),
        "clip_store": str(request.clip_store.resolve()),
        "plan": plan,
        "relations_after_sha256": relations_after_sha256,
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "ClipConsistencyError",
    "FinalizedClipEvidence",
    "RepairCounters",
    "RepairReceipt",
    "RepairRequest",
    "inspect_finalized_clip",
    "repair_clip_consistency",
]
