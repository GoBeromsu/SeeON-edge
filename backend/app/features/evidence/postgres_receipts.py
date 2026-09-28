"""Descriptor-verified native receipt transactions, never HTTP acknowledgement.

The injected database owns authority admission, transaction exit and pool return.
An optional borrowed audit callback is tentative until this owner returns. It
runs once on every successful ready operation, including an identical receipt.
Unavailable receipts retain the existing zero-callback contract.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.configuration import utc_now
from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.postgres_receipt_sql import (
    commit_clip,
    commit_primary_artifact,
    commit_unavailable_primary,
    lock_clip,
)
from backend.app.features.evidence.receipt_files import (
    ReceiptFiles,
    ReceiptHooks,
    ReceiptManifest,
    open_receipt_media,
)
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptPersistenceError,
    ReceiptMissingIncidentError,
    VerifiedArtifact,
    verified_artifact,
)


class PostgresArtifactReceiptStore:
    def __init__(
        self,
        database: PostgresDatabase,
        authority: AuthorityToken,
        clip_root: Path,
        hooks: ReceiptHooks | None = None,
    ) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("native receipts require a PostgreSQL owner")
        if not isinstance(authority, AuthorityToken):
            raise TypeError("native receipts require deployment authority")
        self.database, self.authority = database, authority
        self._clip_store = ClipStore(clip_root)
        self._hooks = hooks or ReceiptHooks()

    def commit(
        self,
        receipt: ArtifactReceipt,
        *,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> ArtifactReceipt:
        manifest = ReceiptManifest.capture(self._clip_store, receipt.artifact_id)
        opened = open_receipt_media(self._clip_store.root, manifest.media_path)
        with opened.handle:
            return self.commit_verified(
                receipt, verified_artifact(opened.handle), after_write=after_write
            )

    def commit_verified(
        self,
        receipt: ArtifactReceipt,
        route_verified: VerifiedArtifact,
        *,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> ArtifactReceipt:
        files = ReceiptFiles.capture(self._clip_store, receipt, route_verified)
        if self._hooks.after_preflight is not None:
            self._hooks.after_preflight()

        def write(connection: psycopg.Connection) -> ArtifactReceipt:
            require_authority(connection, self.authority)
            # Serialize absent clips too, before any incident lock is taken.
            lock_clip(connection, receipt.artifact_id)
            incidents = _manifest_incidents(connection, files.manifest.manifest.event_refs)
            projection = files.verify()
            timestamp = utc_now()
            commit_clip(connection, projection)
            for incident_id, edge_event_id in incidents:
                commit_primary_artifact(
                    connection, incident_id, edge_event_id, projection, timestamp=timestamp
                )
            if self._hooks.before_final_check is not None:
                self._hooks.before_final_check()
            final = files.verify().verified
            if after_write is not None:
                after_write(connection)
            return ArtifactReceipt(receipt.artifact_id, final.sha256, final.size_bytes)

        return self.database.transact(write)

    def commit_unavailable(self, clip_id: str, reason: str) -> None:
        manifest = ReceiptManifest.capture(self._clip_store, clip_id)

        def write(connection: psycopg.Connection) -> None:
            require_authority(connection, self.authority)
            lock_clip(connection, clip_id)
            incidents = _manifest_incidents(connection, manifest.manifest.event_refs)
            manifest.verify()
            timestamp = utc_now()
            for incident_id, _ in incidents:
                commit_unavailable_primary(connection, incident_id, reason, timestamp)
            manifest.verify()

        self.database.transact(write)

    def get(self, artifact_id: str) -> ArtifactReceipt | None:
        def read(connection: psycopg.Connection) -> ArtifactReceipt | None:
            row = connection.execute(
                "SELECT media_sha256,media_size_bytes FROM clips "
                "WHERE clip_id=%s AND publish_state='PUBLISHED'",
                (artifact_id,),
            ).fetchone()
            if row is None:
                return None
            if not isinstance(row[0], str) or type(row[1]) is not int or row[1] <= 0:
                raise ArtifactReceiptPersistenceError("stored receipt identity is unreadable")
            try:
                return ArtifactReceipt(artifact_id, row[0], row[1])
            except ValueError:
                raise ArtifactReceiptPersistenceError(
                    "stored receipt identity is unreadable"
                ) from None

        return self.database.read(read)


def _manifest_incidents(
    connection: psycopg.Connection, event_refs: tuple[str, ...]
) -> list[tuple[str, str]]:
    # Event identities are immutable. All receipt writers take this same order,
    # independent of the ordering of references in the immutable manifest bytes.
    incidents: list[tuple[str, str]] = []
    for event_ref in sorted(set(event_refs)):
        row = connection.execute(
            "SELECT incident_id,edge_event_id FROM incidents WHERE edge_event_id=%s FOR UPDATE",
            (event_ref,),
        ).fetchone()
        if row is None:
            raise ReceiptMissingIncidentError(f"manifest incident is missing: {event_ref}")
        incidents.append((str(row[0]), str(row[1])))
    return incidents


__all__ = ["PostgresArtifactReceiptStore"]
