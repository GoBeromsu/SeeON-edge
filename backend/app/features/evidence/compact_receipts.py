"""Transaction-guarded schema-18 clip receipt persistence."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from backend.app.edge_db.configuration import utc_now
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.compact_receipt_sql import (
    commit_clip,
    commit_primary_artifact,
    commit_unavailable_primary,
)
from backend.app.features.evidence.receipt_files import (
    ReceiptFiles,
    ReceiptHooks,
    ReceiptManifest,
    open_receipt_media,
)
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptVerificationError,
    ReceiptMissingIncidentError,
    VerifiedArtifact,
    verified_artifact,
)


class CompactArtifactReceiptStore:
    """Bind compact publication facts to one live verified media descriptor."""

    def __init__(
        self,
        database_path: Path,
        clip_root: Path,
        hooks: ReceiptHooks | None = None,
    ) -> None:
        self._database_path = database_path
        self._clip_store = ClipStore(clip_root)
        self._hooks = hooks or ReceiptHooks()

    def commit(
        self,
        receipt: ArtifactReceipt,
        *,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> ArtifactReceipt:
        located = self._clip_store.locate_manifest(receipt.artifact_id)
        if located is None:
            raise ArtifactReceiptVerificationError("clip manifest is missing")
        opened = open_receipt_media(
            self._clip_store.root, located.manifest_path.parent / "clip.mp4"
        )
        try:
            return self.commit_verified(
                receipt, verified_artifact(opened.handle), after_write=after_write
            )
        finally:
            opened.handle.close()

    def commit_verified(
        self,
        receipt: ArtifactReceipt,
        route_verified: VerifiedArtifact,
        *,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> ArtifactReceipt:
        files = ReceiptFiles.capture(self._clip_store, receipt, route_verified)
        _run_hook(self._hooks.after_preflight)
        connection = open_runtime_database(self._database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                receipt_timestamp = utc_now()
                projection = files.verify()
                commit_clip(connection, projection)
                for incident_id, edge_event_id in _manifest_incidents(
                    connection, projection.manifest.event_refs
                ):
                    commit_primary_artifact(
                        connection,
                        incident_id,
                        edge_event_id,
                        projection,
                        timestamp=receipt_timestamp,
                    )
                _run_hook(self._hooks.before_final_check)
                final_verified = files.verify().verified
                if after_write is not None:
                    after_write(connection)
        finally:
            connection.close()
        return ArtifactReceipt(
            receipt.artifact_id,
            final_verified.sha256,
            final_verified.size_bytes,
        )

    def commit_unavailable(self, clip_id: str, reason: str) -> None:
        manifest = ReceiptManifest.capture(self._clip_store, clip_id)
        timestamp = utc_now()
        connection = open_runtime_database(self._database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                manifest.verify()
                incidents = _manifest_incidents(connection, manifest.manifest.event_refs)
                for incident_id, _edge_event_id in incidents:
                    commit_unavailable_primary(connection, incident_id, reason, timestamp)
                manifest.verify()
        finally:
            connection.close()

    def get(self, artifact_id: str) -> ArtifactReceipt | None:
        connection = open_runtime_database(self._database_path, actor=RuntimeActor.API)
        try:
            row = connection.execute(
                "SELECT media_sha256, media_size_bytes FROM clips "
                "WHERE clip_id = ? AND publish_state = 'PUBLISHED'",
                (artifact_id,),
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else ArtifactReceipt(artifact_id, str(row[0]), int(row[1]))


def _manifest_incidents(
    connection: sqlite3.Connection, event_refs: tuple[str, ...]
) -> list[tuple[str, str]]:
    incidents: list[tuple[str, str]] = []
    for event_ref in event_refs:
        incident = connection.execute(
            "SELECT incident_id, edge_event_id FROM incidents WHERE edge_event_id = ?",
            (event_ref,),
        ).fetchone()
        if incident is None:
            raise ReceiptMissingIncidentError(f"manifest incident is missing: {event_ref}")
        incidents.append((str(incident[0]), str(incident[1])))
    return incidents


def _run_hook(hook: Callable[[], None] | None) -> None:
    if hook is not None:
        hook()


__all__ = ["CompactArtifactReceiptStore"]
