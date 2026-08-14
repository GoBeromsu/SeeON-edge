"""API-owned internal ML QA store: replay runs, comparisons, and labels.

Persists deterministic replay evidence produced by ``worker.replay`` (a
separate, worker-owned, import-independent package -- this store never
imports it) plus operator disposition labels with full audit history. Mirrors
``backend/app/features/evidence/record_store.py``'s CAS/immutable-revision
conventions for the ``qa_`` table family.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from secrets import token_hex

from shared.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction


class QaLabelDisposition(StrEnum):
    TRUE_POSITIVE = "TP"
    FALSE_POSITIVE = "FP"
    FALSE_NEGATIVE = "FN"
    TRUE_NEGATIVE = "TN"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class QaReplayRun:
    run_id: str
    camera_id: str
    module_qualified_id: str
    policy_qualified_id: str
    effective_policy_id: str
    frame_count: int
    event_count: int
    source_kind: str
    source_run_id: str | None
    requested_by: str
    requested_at: str
    result_sha256: str
    result: dict[str, object]


@dataclass(frozen=True, slots=True)
class QaReplayComparison:
    comparison_id: str
    baseline_run_id: str
    candidate_run_id: str
    identical: bool
    mismatch_count: int
    created_at: str
    comparison_sha256: str
    comparison: dict[str, object]


@dataclass(frozen=True, slots=True)
class QaLabel:
    label_id: str
    comparison_id: str
    version: int
    actor_id: str
    labeled_at: str
    disposition: QaLabelDisposition
    notes: str | None


@dataclass(slots=True)
class QaConflictError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return f"internal ML QA conflict: {self.detail}"


class QaStore:
    """DDL-free API-owned writer/reader for the ``qa_`` table family."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def record_run(
        self,
        *,
        camera_id: str,
        module_qualified_id: str,
        policy_qualified_id: str,
        effective_policy_id: str,
        frame_count: int,
        event_count: int,
        source_kind: str,
        source_run_id: str | None,
        requested_by: str,
        requested_at: str,
        result: dict[str, object],
    ) -> QaReplayRun:
        if source_kind not in ("captured", "replay"):
            raise ValueError("source_kind must be 'captured' or 'replay'")
        if (source_kind == "captured") != (source_run_id is None):
            raise ValueError("source_run_id is required iff source_kind is 'replay'")
        result_json = _canonical_json(result)
        result_sha256 = hashlib.sha256(result_json.encode()).hexdigest()
        run_id = result_sha256
        connection = _connect(self.database_path)
        try:
            with write_transaction(connection):
                existing = connection.execute(
                    "SELECT result_json FROM qa_replay_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if existing is not None:
                    if existing[0] != result_json:
                        raise QaConflictError("run identity resolves to contradictory content")
                else:
                    connection.execute(
                        """
                        INSERT INTO qa_replay_runs (
                            run_id, camera_id, module_qualified_id, policy_qualified_id,
                            effective_policy_id, frame_count, event_count, source_kind,
                            source_run_id, requested_by, requested_at, result_sha256,
                            result_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            camera_id,
                            module_qualified_id,
                            policy_qualified_id,
                            effective_policy_id,
                            frame_count,
                            event_count,
                            source_kind,
                            source_run_id,
                            requested_by,
                            requested_at,
                            result_sha256,
                            result_json,
                        ),
                    )
        finally:
            connection.close()
        return QaReplayRun(
            run_id=run_id,
            camera_id=camera_id,
            module_qualified_id=module_qualified_id,
            policy_qualified_id=policy_qualified_id,
            effective_policy_id=effective_policy_id,
            frame_count=frame_count,
            event_count=event_count,
            source_kind=source_kind,
            source_run_id=source_run_id,
            requested_by=requested_by,
            requested_at=requested_at,
            result_sha256=result_sha256,
            result=result,
        )

    def record_comparison(
        self,
        *,
        baseline_run_id: str,
        candidate_run_id: str,
        created_at: str,
        comparison: dict[str, object],
    ) -> QaReplayComparison:
        if baseline_run_id == candidate_run_id:
            raise ValueError("baseline and candidate runs must differ")
        identical = bool(comparison.get("identical"))
        mismatches = comparison.get("mismatches")
        mismatch_count = len(mismatches) if isinstance(mismatches, list) else 0
        if identical != (mismatch_count == 0):
            raise ValueError("comparison identical flag contradicts mismatch count")
        comparison_json = _canonical_json(comparison)
        comparison_sha256 = hashlib.sha256(comparison_json.encode()).hexdigest()
        comparison_id = hashlib.sha256(
            f"{baseline_run_id}:{candidate_run_id}:{comparison_sha256}".encode()
        ).hexdigest()
        connection = _connect(self.database_path)
        try:
            with write_transaction(connection):
                for run_id in (baseline_run_id, candidate_run_id):
                    if (
                        connection.execute(
                            "SELECT 1 FROM qa_replay_runs WHERE run_id = ?", (run_id,)
                        ).fetchone()
                        is None
                    ):
                        raise QaConflictError(f"unknown replay run {run_id!r}")
                existing = connection.execute(
                    "SELECT comparison_json FROM qa_replay_comparisons WHERE comparison_id = ?",
                    (comparison_id,),
                ).fetchone()
                if existing is not None:
                    if existing[0] != comparison_json:
                        raise QaConflictError(
                            "comparison identity resolves to contradictory content"
                        )
                else:
                    connection.execute(
                        """
                        INSERT INTO qa_replay_comparisons (
                            comparison_id, baseline_run_id, candidate_run_id, identical,
                            mismatch_count, created_at, comparison_sha256, comparison_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            comparison_id,
                            baseline_run_id,
                            candidate_run_id,
                            int(identical),
                            mismatch_count,
                            created_at,
                            comparison_sha256,
                            comparison_json,
                        ),
                    )
        finally:
            connection.close()
        return QaReplayComparison(
            comparison_id=comparison_id,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            identical=identical,
            mismatch_count=mismatch_count,
            created_at=created_at,
            comparison_sha256=comparison_sha256,
            comparison=comparison,
        )

    def label(
        self,
        *,
        comparison_id: str,
        expected_version: int,
        actor_id: str,
        labeled_at: str,
        disposition: QaLabelDisposition,
        notes: str | None,
    ) -> QaLabel:
        if not actor_id.strip():
            raise ValueError("actor_id must be non-empty")
        label_id = f"qa-label:{token_hex(16)}"
        next_version = expected_version + 1
        connection = _connect(self.database_path)
        try:
            try:
                with write_transaction(connection):
                    relation = connection.execute(
                        "SELECT 1 FROM qa_replay_comparisons WHERE comparison_id = ?",
                        (comparison_id,),
                    ).fetchone()
                    if relation is None:
                        raise QaConflictError("label must reference an existing comparison")
                    current = connection.execute(
                        "SELECT current_version FROM qa_label_state WHERE comparison_id = ?",
                        (comparison_id,),
                    ).fetchone()
                    current_version = 0 if current is None else int(current[0])
                    if current_version != expected_version:
                        raise QaConflictError(
                            f"expected version {expected_version} changed to {current_version}"
                        )
                    connection.execute(
                        """
                        INSERT INTO qa_label_revisions (
                            label_id, comparison_id, label_version, actor_id,
                            labeled_at, disposition, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            label_id,
                            comparison_id,
                            next_version,
                            actor_id,
                            labeled_at,
                            disposition.value,
                            notes,
                        ),
                    )
                    if expected_version == 0:
                        connection.execute(
                            "INSERT INTO qa_label_state (comparison_id, current_version) "
                            "VALUES (?, 1)",
                            (comparison_id,),
                        )
                    else:
                        changed = connection.execute(
                            "UPDATE qa_label_state SET current_version = ? "
                            "WHERE comparison_id = ? AND current_version = ?",
                            (next_version, comparison_id, expected_version),
                        ).rowcount
                        if changed != 1:
                            raise QaConflictError(f"expected version {expected_version} changed")
            except sqlite3.IntegrityError as error:
                raise QaConflictError(str(error)) from error
        finally:
            connection.close()
        return QaLabel(
            label_id=label_id,
            comparison_id=comparison_id,
            version=next_version,
            actor_id=actor_id,
            labeled_at=labeled_at,
            disposition=disposition,
            notes=notes,
        )

    def get_run(self, run_id: str) -> QaReplayRun | None:
        connection = _connect(self.database_path)
        try:
            row = connection.execute(
                "SELECT run_id, camera_id, module_qualified_id, policy_qualified_id, "
                "effective_policy_id, frame_count, event_count, source_kind, "
                "source_run_id, requested_by, requested_at, result_sha256, result_json "
                "FROM qa_replay_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return QaReplayRun(
            run_id=str(row[0]),
            camera_id=str(row[1]),
            module_qualified_id=str(row[2]),
            policy_qualified_id=str(row[3]),
            effective_policy_id=str(row[4]),
            frame_count=int(row[5]),
            event_count=int(row[6]),
            source_kind=str(row[7]),
            source_run_id=None if row[8] is None else str(row[8]),
            requested_by=str(row[9]),
            requested_at=str(row[10]),
            result_sha256=str(row[11]),
            result=json.loads(str(row[12])),
        )

    def get_comparison(self, comparison_id: str) -> QaReplayComparison | None:
        connection = _connect(self.database_path)
        try:
            row = connection.execute(
                "SELECT comparison_id, baseline_run_id, candidate_run_id, identical, "
                "mismatch_count, created_at, comparison_sha256, comparison_json "
                "FROM qa_replay_comparisons WHERE comparison_id = ?",
                (comparison_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return QaReplayComparison(
            comparison_id=str(row[0]),
            baseline_run_id=str(row[1]),
            candidate_run_id=str(row[2]),
            identical=bool(row[3]),
            mismatch_count=int(row[4]),
            created_at=str(row[5]),
            comparison_sha256=str(row[6]),
            comparison=json.loads(str(row[7])),
        )

    def current_label(self, comparison_id: str) -> QaLabel | None:
        connection = _connect(self.database_path)
        try:
            row = connection.execute(
                """
                SELECT revision.label_id, revision.comparison_id, revision.label_version,
                       revision.actor_id, revision.labeled_at, revision.disposition,
                       revision.notes
                FROM qa_label_state AS state
                JOIN qa_label_revisions AS revision
                  ON revision.comparison_id = state.comparison_id
                 AND revision.label_version = state.current_version
                WHERE state.comparison_id = ?
                """,
                (comparison_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return QaLabel(
            label_id=str(row[0]),
            comparison_id=str(row[1]),
            version=int(row[2]),
            actor_id=str(row[3]),
            labeled_at=str(row[4]),
            disposition=QaLabelDisposition(str(row[5])),
            notes=None if row[6] is None else str(row[6]),
        )

    def label_history(self, comparison_id: str) -> tuple[QaLabel, ...]:
        connection = _connect(self.database_path)
        try:
            rows = connection.execute(
                "SELECT label_id, comparison_id, label_version, actor_id, labeled_at, "
                "disposition, notes FROM qa_label_revisions WHERE comparison_id = ? "
                "ORDER BY label_version",
                (comparison_id,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            QaLabel(
                label_id=str(row[0]),
                comparison_id=str(row[1]),
                version=int(row[2]),
                actor_id=str(row[3]),
                labeled_at=str(row[4]),
                disposition=QaLabelDisposition(str(row[5])),
                notes=None if row[6] is None else str(row[6]),
            )
            for row in rows
        )


def _canonical_json(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _connect(database_path: Path) -> sqlite3.Connection:
    return open_runtime_database(database_path, actor=RuntimeActor.API)


__all__ = [
    "QaConflictError",
    "QaLabel",
    "QaLabelDisposition",
    "QaReplayComparison",
    "QaReplayRun",
    "QaStore",
]
