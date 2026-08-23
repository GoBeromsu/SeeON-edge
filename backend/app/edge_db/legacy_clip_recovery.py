"""Backend-owned terminal recovery for schema-16 evidence clips."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from backend.app.features.clips.consistency_ops import (
    ClipConsistencyError,
    FinalizedClipEvidence,
    inspect_finalized_clip,
)

_TERMINAL_PUBLICATION_REASON = "LEGACY_CLIP_PUBLICATION_UNSUPPORTED"
_TERMINAL_IN_FLIGHT_PUBLICATION_REASON = "LEGACY_CLIP_PUBLISHER_RETIRED"


@dataclass(frozen=True, slots=True)
class LegacyClipRecoveryResult:
    verified: int
    unavailable: int
    corrupt: int
    publication_terminalized: int
    unresolved: int


class LegacyClipStoreUnavailableError(RuntimeError):
    """The clip store root is missing, unreadable, or not a clip store."""


class LegacyClipRecovery:
    """Finalize schema-16 clips from the authoritative finalized clip store.

    Schema-16 clips have no backend publisher.  Waiting publication outcomes are
    therefore explicitly recorded as permanent rather than silently stranded.
    """

    def __init__(self, database: Path, clip_store: Path) -> None:
        self._database = database
        self._clip_store = clip_store

    def _require_mounted_store(self) -> None:
        """Refuse before touching the database unless the store is really there.

        Absent this check, a mistyped or unmounted ``--clip-store`` is
        indistinguishable from a store whose media is genuinely gone: every
        ``AWAITING_FINALIZE`` clip is classified ``UNAVAILABLE``, its publication
        is terminalized, the command exits 0, and the schema-17 gate opens. One
        wrong path would therefore write off all 1053 live clips and wave the
        forward-only migration through behind them.

        ``<root>/clips/.staging`` is the structural marker the existing clip
        consistency ops already treat as authoritative, so a mounted-but-empty
        store still proceeds while a wrong root stops here.
        """
        marker = self._clip_store / "clips" / ".staging"
        if not marker.is_dir():
            raise LegacyClipStoreUnavailableError(
                f"clip store at {self._clip_store} is not a mounted finalized clip "
                f"store: expected the directory {marker}. Refusing to classify any "
                f"clip, because a wrong or unmounted path would mark every clip "
                f"UNAVAILABLE and open the schema-17 migration gate."
            )

    def _open_store_handle(self) -> int:
        """Open the store directory and keep the descriptor for the whole scan.

        Comparing ``(st_dev, st_ino)`` before and after is not enough: if the
        mount disappears while clips are being classified and the *same* mount
        returns before the second check, the identity matches and every
        `missing` verdict gathered during the gap would commit. Holding an open
        descriptor and verifying it still resolves to the live path proves the
        directory was continuously present, not merely present at two instants.
        """
        try:
            return os.open(self._clip_store / "clips", os.O_RDONLY | os.O_DIRECTORY)
        except OSError as error:
            raise LegacyClipStoreUnavailableError(
                f"clip store at {self._clip_store} could not be opened: {error}"
            ) from error

    def _require_same_store(self, handle: int) -> None:
        """Confirm the held descriptor still names the live store directory."""
        try:
            held = os.fstat(handle)
            live = (self._clip_store / "clips").stat()
        except OSError as error:
            raise LegacyClipStoreUnavailableError(
                f"clip store at {self._clip_store} became unreadable during the scan: "
                f"{error}"
            ) from error
        if (held.st_dev, held.st_ino) != (live.st_dev, live.st_ino):
            raise LegacyClipStoreUnavailableError(
                f"clip store at {self._clip_store} was replaced during the scan; "
                f"refusing to record any classification, because clips inspected "
                f"afterwards were read from a different mount."
            )
        marker = self._clip_store / "clips" / ".staging"
        if not marker.is_dir():
            raise LegacyClipStoreUnavailableError(
                f"clip store marker {marker} disappeared during the scan; refusing "
                f"to record classifications gathered against an absent mount."
            )

    def run(self) -> LegacyClipRecoveryResult:
        self._require_mounted_store()
        # Pinned across the scan. A mount that disappears midway makes later
        # lstat calls raise FileNotFoundError, which inspect_finalized_clip
        # legitimately reads as 'missing' -- so every remaining clip would be
        # committed UNAVAILABLE and the migration gate opened, on evidence that
        # was never actually examined. Comparing identity before the write
        # transaction separates a vanished mount from genuinely absent media.
        store_handle = self._open_store_handle()
        verified = unavailable = corrupt = 0
        with sqlite3.connect(self._database) as connection:
            clip_ids = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT clip_id FROM evidence_clips
                    WHERE local_state = 'AWAITING_FINALIZE'
                    ORDER BY clip_id
                    """
                )
            ]

        classifications = []
        for clip_id in clip_ids:
            outcome, unavailable_reason, evidence = self._classify(clip_id, store_handle)
            manifest_sha256 = manifest_size_bytes = None
            if evidence is not None:
                try:
                    manifest = self._read_manifest_bytes(store_handle, clip_id)
                except OSError as error:
                    raise LegacyClipStoreUnavailableError(
                        f"clip store at {self._clip_store} cannot read the manifest for "
                        f"{clip_id}; refusing to mutate any clip"
                    ) from error
                manifest_sha256 = hashlib.sha256(manifest).hexdigest()
                manifest_size_bytes = len(manifest)
            classifications.append(
                (
                    clip_id,
                    outcome,
                    unavailable_reason,
                    evidence,
                    manifest_sha256,
                    manifest_size_bytes,
                )
            )
            if outcome == "VERIFIED":
                verified += 1
            elif outcome == "UNAVAILABLE":
                unavailable += 1
            else:
                corrupt += 1

        try:
            self._require_same_store(store_handle)
        finally:
            os.close(store_handle)

        with sqlite3.connect(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for (
                clip_id,
                local_state,
                unavailable_reason,
                evidence,
                manifest_sha256,
                manifest_size_bytes,
            ) in classifications:
                self._record(
                    connection,
                    clip_id,
                    local_state,
                    unavailable_reason,
                    evidence,
                )
                self._reconcile_projection(
                    connection,
                    clip_id,
                    local_state,
                    unavailable_reason,
                    evidence,
                    manifest_sha256,
                    manifest_size_bytes,
                )
            cursor = connection.execute(
                """
                UPDATE evidence_clips
                SET publish_state = 'PERMANENT',
                    last_error_code = CASE publish_state
                        WHEN 'WAITING' THEN ?
                        WHEN 'IN_FLIGHT' THEN ?
                    END,
                    publish_attempt_count = publish_attempt_count + 1,
                    publish_lease_owner = NULL,
                    publish_lease_expires_at = NULL
                WHERE publish_state IN ('WAITING', 'IN_FLIGHT')
                  AND local_state IN ('VERIFIED', 'UNAVAILABLE', 'CORRUPT')
                """,
                (
                    _TERMINAL_PUBLICATION_REASON,
                    _TERMINAL_IN_FLIGHT_PUBLICATION_REASON,
                ),
            )
            publication_terminalized = cursor.rowcount
            unresolved = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM evidence_clips
                    WHERE local_state = 'AWAITING_FINALIZE'
                       OR publish_state = 'IN_FLIGHT'
                    """
                ).fetchone()[0]
            )

        return LegacyClipRecoveryResult(
            verified=verified,
            unavailable=unavailable,
            corrupt=corrupt,
            publication_terminalized=publication_terminalized,
            unresolved=unresolved,
        )

    def _classify(
        self, clip_id: str, store_handle: int
    ) -> tuple[str, str | None, FinalizedClipEvidence | None]:
        try:
            evidence = inspect_finalized_clip(
                self._clip_store, clip_id, clips_dir_fd=store_handle
            )
        except ClipConsistencyError as exc:
            if exc.code == "final_read_error":
                raise LegacyClipStoreUnavailableError(
                    f"clip store at {self._clip_store} cannot be read while inspecting "
                    f"{clip_id}; refusing to classify or mutate any clip"
                ) from exc
            if exc.code == "missing":
                return "UNAVAILABLE", "MISSING", None
            return "CORRUPT", "CORRUPT", None
        return evidence.local_state, evidence.unavailable_reason, evidence

    @staticmethod
    def _read_manifest_bytes(store_handle: int, clip_id: str) -> bytes:
        """Read the projection manifest through the same pinned store descriptor."""
        directory_handle = os.open(
            clip_id,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=store_handle,
        )
        try:
            manifest_handle = os.open(
                "manifest.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_handle
            )
            with os.fdopen(manifest_handle, "rb") as manifest:
                return manifest.read()
        finally:
            os.close(directory_handle)

    def _record(
        self,
        connection: sqlite3.Connection,
        clip_id: str,
        local_state: str,
        unavailable_reason: str | None,
        evidence: FinalizedClipEvidence | None,
    ) -> None:
        connection.execute(
            """
            UPDATE evidence_clips
            SET local_state = ?, unavailable_reason = ?,
                manifest_path = ?, media_relpath = ?, sha256 = ?, size_bytes = ?,
                mime_type = ?, codec = ?, duration_ms = ?, finalized_at = ?,
                state_version = state_version + 1
            WHERE clip_id = ? AND local_state = 'AWAITING_FINALIZE'
            """,
            (
                local_state,
                unavailable_reason,
                str(evidence.manifest_path) if evidence is not None else None,
                evidence.media_relpath if evidence is not None else None,
                evidence.sha256 if evidence is not None else None,
                evidence.size_bytes if evidence is not None else None,
                evidence.mime_type if evidence is not None else None,
                evidence.codec if evidence is not None else None,
                evidence.duration_ms if evidence is not None else None,
                evidence.finalized_at if evidence is not None else None,
                clip_id,
            ),
        )

    def _reconcile_projection(
        self,
        connection: sqlite3.Connection,
        clip_id: str,
        local_state: str,
        unavailable_reason: str | None,
        evidence: FinalizedClipEvidence | None,
        manifest_sha256: str | None,
        manifest_size_bytes: int | None,
    ) -> None:
        """Resolve the schema-16 pending primary slot in the clip transaction."""
        rows = connection.execute(
            """
            SELECT incident.incident_id, incident.updated_at
            FROM evidence_incidents AS incident
            JOIN clip_events AS relation USING (edge_event_id)
            JOIN evidence_artifact_slots AS slot
              ON slot.incident_id = incident.incident_id
             AND slot.slot_name = 'PRIMARY_CLIP'
             AND slot.state = 'PENDING'
            WHERE relation.clip_id = ?
            """,
            (clip_id,),
        ).fetchall()
        for incident_id, updated_at in rows:
            timestamp = evidence.finalized_at if evidence is not None else str(updated_at)
            if local_state == "VERIFIED":
                assert evidence is not None
                assert manifest_sha256 is not None and manifest_size_bytes is not None
                assert evidence.media_relpath is not None
                media_id = "legacy-primary:" + hashlib.sha256(
                    f"{evidence.sha256}\0{evidence.media_relpath}".encode()
                ).hexdigest()
                connection.execute(
                    """
                    INSERT INTO evidence_media_objects (
                        media_id, content_sha256, size_bytes, mime_type, contained_relpath,
                        basename, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(media_id) DO NOTHING
                    """,
                    (
                        media_id,
                        evidence.sha256,
                        evidence.size_bytes,
                        evidence.mime_type,
                        evidence.media_relpath,
                        evidence.media_relpath.rsplit("/", 1)[-1],
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO evidence_primary_clips (
                        incident_id, clip_id, manifest_relpath, manifest_sha256,
                        manifest_size_bytes, media_id, codec, duration_ms, finalized_at,
                        source_packet_preserved, source_missing_reason, truncation_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0,
                              'LEGACY_SOURCE_FACTS_NOT_RECORDED', '[]', ?)
                    ON CONFLICT(incident_id) DO NOTHING
                    """,
                    (
                        incident_id, clip_id, f"clips/{clip_id}/manifest.json", manifest_sha256,
                        manifest_size_bytes, media_id, evidence.codec, evidence.duration_ms,
                        evidence.finalized_at, timestamp,
                    ),
                )
                connection.execute(
                    """
                    UPDATE evidence_artifact_slots
                    SET state = 'AVAILABLE', media_id = ?, updated_at = ?, revision = revision + 1
                    WHERE incident_id = ? AND slot_name = 'PRIMARY_CLIP' AND state = 'PENDING'
                    """,
                    (media_id, timestamp, incident_id),
                )
                connection.execute(
                    """
                    UPDATE evidence_incidents
                    SET primary_clip_id = ?, lifecycle_state = 'MEDIA_READY',
                        updated_at = ?, revision = revision + 1
                    WHERE incident_id = ? AND lifecycle_state = 'STAGING'
                    """,
                    (clip_id, timestamp, incident_id),
                )
            else:
                reason = unavailable_reason or local_state
                connection.execute(
                    """
                    INSERT INTO evidence_primary_clips (
                        incident_id, clip_id, source_packet_preserved, source_missing_reason,
                        truncation_json, unavailable_reason, created_at
                    ) VALUES (?, ?, 0, 'LEGACY_SOURCE_FACTS_NOT_RECORDED', '[]', ?, ?)
                    ON CONFLICT(incident_id) DO NOTHING
                    """,
                    (incident_id, clip_id, reason, timestamp),
                )
                connection.execute(
                    """
                    UPDATE evidence_artifact_slots
                    SET state = ?, reason = ?, updated_at = ?, revision = revision + 1
                    WHERE incident_id = ? AND slot_name = 'PRIMARY_CLIP' AND state = 'PENDING'
                    """,
                    (local_state, reason, timestamp, incident_id),
                )
                connection.execute(
                    """
                    UPDATE evidence_incidents
                    SET lifecycle_state = 'FAILED', failure_reason = ?,
                        updated_at = ?, revision = revision + 1
                    WHERE incident_id = ? AND lifecycle_state = 'STAGING'
                    """,
                    (reason, timestamp, incident_id),
                )


__all__ = [
    "LegacyClipRecovery",
    "LegacyClipRecoveryResult",
    "LegacyClipStoreUnavailableError",
]
