"""Terminal recovery for central intents stranded by the schema-17 ownership cutover."""

from __future__ import annotations

import errno
import os
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from backend.app.edge_db.legacy_clip_recovery import (
    LegacyClipRecovery,
    LegacyClipStoreUnavailableError,
)

_DERIVATIVE_CANCEL_REASON = "LEGACY_DERIVATIVE_EXECUTOR_RETIRED"
_RETENTION_PRESENT_REASON = "LEGACY_RETENTION_MEDIA_STILL_PRESENT"


@dataclass(frozen=True, slots=True)
class LegacyTransitionalRecoveryResult:
    derivatives_cancelled: int
    slots_terminalized: int
    retention_purged: int
    retention_failed: int
    unresolved: int


class LegacyTransitionalRecovery:
    """Resolve only facts that can be established after the worker is retired."""

    def __init__(self, database: Path, clip_store: Path) -> None:
        self._database = database
        self._clip_store = clip_store

    def run(self) -> LegacyTransitionalRecoveryResult:
        # Retention must not turn a missing mount into a false PURGED fact. Keep
        # the same continuously-pinned store proof used by clip recovery and do
        # all filesystem inspection before opening the write transaction.
        store = LegacyClipRecovery(self._database, self._clip_store)
        store._require_mounted_store()
        handle = store._open_store_handle()
        try:
            with sqlite3.connect(self._database) as connection:
                retention = list(
                    connection.execute(
                        """
                        -- LEFT JOIN deliberately: a retention row whose clip
                        -- record is already gone must still be resolvable, or
                        -- it blocks the migration gate forever with no actor
                        -- able to clear it.
                        SELECT retention.clip_id, clip.media_relpath
                        FROM evidence_retention_states AS retention
                        LEFT JOIN evidence_clips AS clip USING (clip_id)
                        WHERE retention.state = 'PENDING'
                        ORDER BY retention.clip_id
                        """
                    )
                )
            retention_outcomes = [
                (
                    str(clip_id),
                    self._retention_outcome(str(clip_id), media_relpath, handle),
                )
                for clip_id, media_relpath in retention
            ]
            store._require_same_store(handle)
        finally:
            os.close(handle)

        with sqlite3.connect(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            derivatives_cancelled = connection.execute(
                """
                UPDATE derivative_jobs
                SET state = 'CANCELLED', reason = ?, cancel_requested = 1,
                    updated_at = updated_at, revision = revision + 1
                WHERE state IN ('PENDING', 'RUNNING')
                """,
                (_DERIVATIVE_CANCEL_REASON,),
            ).rowcount
            # A PENDING artifact slot has no executor after the ownership
            # change, exactly like a PENDING job. It is recorded UNAVAILABLE
            # with a reason rather than left to block the gate with nothing
            # able to resolve it. UNAVAILABLE requires a reason per the CHECK.
            slots_terminalized = connection.execute(
                """
                UPDATE derivative_evidence_slots
                SET state = 'UNAVAILABLE', reason = ?, media_id = NULL,
                    updated_at = updated_at, revision = revision + 1
                WHERE state = 'PENDING'
                """,
                (_DERIVATIVE_CANCEL_REASON,),
            ).rowcount
            retention_purged = retention_failed = 0
            for clip_id, (state, reason) in retention_outcomes:
                cursor = connection.execute(
                    """
                    UPDATE evidence_retention_states
                    SET state = ?, reason = ?, updated_at = updated_at, revision = revision + 1
                    WHERE clip_id = ? AND state = 'PENDING'
                    """,
                    (state, reason, clip_id),
                )
                if cursor.rowcount:
                    if state == "PURGED":
                        retention_purged += 1
                    else:
                        retention_failed += 1
            unresolved = int(
                connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM derivative_jobs
                         WHERE state IN ('PENDING', 'RUNNING'))
                      + (SELECT COUNT(*) FROM derivative_evidence_slots
                         WHERE state = 'PENDING')
                      + (SELECT COUNT(*) FROM evidence_retention_states
                         WHERE state = 'PENDING')
                    """
                ).fetchone()[0]
            )
        return LegacyTransitionalRecoveryResult(
            derivatives_cancelled=derivatives_cancelled,
            slots_terminalized=slots_terminalized,
            retention_purged=retention_purged,
            retention_failed=retention_failed,
            unresolved=unresolved,
        )

    def _retention_outcome(
        self, clip_id: str, media_relpath: object, clips_dir_fd: int
    ) -> tuple[str, str | None]:
        if not isinstance(media_relpath, str) or not media_relpath:
            return ("FAILED", "LEGACY_RETENTION_MEDIA_PATH_UNVERIFIABLE")
        candidate = PurePosixPath(media_relpath)
        if (
            candidate.is_absolute()
            or candidate.as_posix() != media_relpath
            or len(candidate.parts) < 2
            or candidate.parts[0] != "clips"
            or any(part in {".", ".."} for part in candidate.parts)
        ):
            return ("FAILED", "LEGACY_RETENTION_MEDIA_PATH_UNVERIFIABLE")
        # Descend one component at a time under the pinned descriptor. A single
        # os.stat on the whole relative path with follow_symlinks=False protects
        # only the final component, so an intermediate symlink would resolve
        # outside the store and let an external object decide PURGED or FAILED.
        try:
            exists = self._exists_within(candidate.parts[1:], clips_dir_fd)
        except _EscapedStoreError:
            return ("FAILED", "LEGACY_RETENTION_MEDIA_PATH_UNVERIFIABLE")
        except OSError as error:
            raise LegacyClipStoreUnavailableError(
                f"clip store at {self._clip_store} cannot inspect retention media for {clip_id}; "
                "refusing to record a purge verdict"
            ) from error
        if exists:
            return ("FAILED", _RETENTION_PRESENT_REASON)
        return ("PURGED", None)

    @staticmethod
    def _exists_within(parts: tuple[str, ...], clips_dir_fd: int) -> bool:
        """Resolve ``parts`` strictly beneath ``clips_dir_fd``.

        Every intermediate component is opened with ``O_NOFOLLOW``, so a symlink
        anywhere along the path raises rather than silently escaping. Returns
        whether the final component exists; raises :class:`_EscapedStoreError`
        when the path leaves the pinned store.
        """
        current = os.dup(clips_dir_fd)
        try:
            for component in parts[:-1]:
                try:
                    nxt = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=current,
                    )
                except FileNotFoundError:
                    return False
                except OSError as error:
                    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise _EscapedStoreError from error
                    raise
                os.close(current)
                current = nxt
            try:
                final = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                return False
            if stat.S_ISLNK(final.st_mode):
                # The final component is itself a symlink. ``follow_symlinks=False``
                # means we observed the link inode, never its target, so we cannot
                # say whether the clip's media exists. Reporting it as present
                # would tell an operator the file is sitting there when what we
                # actually found is a pointer we deliberately refused to follow.
                raise _EscapedStoreError
            return True
        finally:
            os.close(current)


class _EscapedStoreError(Exception):
    """A retention path left the pinned clip store via a symlink."""


__all__ = ["LegacyTransitionalRecovery", "LegacyTransitionalRecoveryResult"]
