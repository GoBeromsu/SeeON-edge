"""Transaction-guarded schema-18 clip receipt persistence."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from backend.app.features.clips.descriptor_files import (
    OpenedRegularFile,
    open_contained_regular_file,
)
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.compact_receipt_sql import (
    ClipProjection,
    commit_clip,
    commit_primary_artifact,
)
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptVerificationError,
    VerifiedArtifact,
    verified_artifact,
)


@dataclass(frozen=True, slots=True)
class CompactReceiptHooks:
    """Deterministic race hooks; production leaves both callbacks absent."""

    after_preflight: Callable[[], None] | None = None
    before_final_check: Callable[[], None] | None = None


class CompactArtifactReceiptStore:
    """Bind compact publication facts to one live verified media descriptor."""

    def __init__(
        self,
        database_path: Path,
        clip_root: Path,
        hooks: CompactReceiptHooks | None = None,
    ) -> None:
        self._database_path = database_path
        self._clip_store = ClipStore(clip_root)
        self._hooks = hooks or CompactReceiptHooks()

    def commit(
        self,
        receipt: ArtifactReceipt,
        *,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> ArtifactReceipt:
        located = self._clip_store.locate_manifest(receipt.artifact_id)
        if located is None:
            raise ArtifactReceiptVerificationError("clip manifest is missing")
        opened = self._open_media(located.manifest_path.parent / "clip.mp4")
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
        located = self._clip_store.locate_manifest(receipt.artifact_id)
        if located is None:
            raise ArtifactReceiptVerificationError("clip manifest is missing")
        media_path = located.manifest_path.parent / "clip.mp4"
        preflight = self._verify_descriptor(route_verified, receipt)
        self._verify_current_path(media_path, preflight.identity)
        _run_hook(self._hooks.after_preflight)
        projection_facts = self._manifest_projection_facts(located.manifest_path)
        connection = open_runtime_database(self._database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                transaction_verified = self._verify_descriptor(route_verified, receipt)
                self._verify_current_path(media_path, transaction_verified.identity)
                projection = ClipProjection(
                    receipt=receipt,
                    verified=transaction_verified,
                    manifest=located.manifest,
                    manifest_relpath=projection_facts[0],
                    media_relpath=media_path.relative_to(self._clip_store.root).as_posix(),
                    manifest_hash=projection_facts[1],
                    manifest_size=projection_facts[2],
                )
                commit_clip(connection, projection)
                for event_ref in located.manifest.event_refs:
                    incident = connection.execute(
                        "SELECT incident_id, edge_event_id FROM incidents WHERE edge_event_id = ?",
                        (event_ref,),
                    ).fetchone()
                    if incident is not None:
                        commit_primary_artifact(
                            connection, str(incident[0]), str(incident[1]), projection
                        )
                _run_hook(self._hooks.before_final_check)
                final_verified = self._verify_descriptor(route_verified, receipt)
                if final_verified.identity != transaction_verified.identity:
                    raise ArtifactReceiptVerificationError(
                        "verified descriptor changed during transaction"
                    )
                self._verify_current_path(media_path, final_verified.identity)
                if after_write is not None:
                    after_write(connection)
        finally:
            connection.close()
        return ArtifactReceipt(
            receipt.artifact_id,
            final_verified.sha256,
            final_verified.size_bytes,
        )

    def _verify_descriptor(
        self,
        route_verified: VerifiedArtifact,
        receipt: ArtifactReceipt,
    ) -> VerifiedArtifact:
        verified = verified_artifact(route_verified.handle)
        if verified.identity != route_verified.identity:
            raise ArtifactReceiptVerificationError("verified descriptor identity changed")
        if (verified.sha256, verified.size_bytes) != (receipt.sha256, receipt.size_bytes):
            raise ArtifactReceiptVerificationError("declared receipt differs from media bytes")
        return verified

    def _verify_current_path(self, path: Path, identity: tuple[int, int]) -> None:
        current = self._open_media(path)
        try:
            current_stat = os.fstat(current.handle.fileno())
            if (current_stat.st_dev, current_stat.st_ino) != identity:
                raise ArtifactReceiptVerificationError("clip media inode changed")
        finally:
            current.handle.close()

    def _open_media(self, path: Path) -> OpenedRegularFile:
        try:
            path_stat = os.lstat(path)
            if not stat.S_ISREG(path_stat.st_mode):
                raise ArtifactReceiptVerificationError("clip media pathname is not regular")
            opened = open_contained_regular_file(self._clip_store.root, path)
            opened_stat = os.fstat(opened.handle.fileno())
        except (FileNotFoundError, OSError, ValueError) as error:
            raise ArtifactReceiptVerificationError("clip media is missing") from error
        if (path_stat.st_dev, path_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
            opened.handle.close()
            raise ArtifactReceiptVerificationError("clip media pathname changed")
        return opened

    def _manifest_projection_facts(self, path: Path) -> tuple[str, str, int]:
        opened = open_contained_regular_file(self._clip_store.root, path)
        try:
            content = opened.handle.read()
        finally:
            opened.handle.close()
        return (
            path.relative_to(self._clip_store.root).as_posix(),
            hashlib.sha256(content).hexdigest(),
            len(content),
        )

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


def _run_hook(hook: Callable[[], None] | None) -> None:
    if hook is not None:
        hook()


__all__ = ["CompactArtifactReceiptStore", "CompactReceiptHooks"]
