from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from worker.pipeline.output.annotated_derivative import AnnotatedDerivativeJob, DerivativeKind


class DerivativeJobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    CORRUPT = "CORRUPT"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class DerivativeJobRecord:
    incident_id: str
    derivative_kind: DerivativeKind
    request_id: str
    state: DerivativeJobState
    media_id: str | None
    reason: str | None
    attempt_count: int
    cancel_requested: bool
    revision: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class DerivativeRequestResult:
    record: DerivativeJobRecord
    scheduled: bool


class DerivativeJobConflictError(RuntimeError):
    pass


class DerivativeJobStore:
    """Worker-owned DDL-free request slots for bounded derivative execution."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def request(self, job: AnnotatedDerivativeJob, *, updated_at: str) -> DerivativeRequestResult:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                row = connection.execute(
                    "SELECT incident_id,derivative_kind,request_id,state,media_id,reason,"
                    "attempt_count,cancel_requested,revision,created_at,updated_at "
                    "FROM derivative_jobs WHERE incident_id=? AND derivative_kind=?",
                    (job.incident_id, job.derivative_kind.value),
                ).fetchone()
                if row is None:
                    relation = connection.execute(
                        "SELECT primary_clip_id,lifecycle_state FROM evidence_incidents "
                        "WHERE incident_id=?",
                        (job.incident_id,),
                    ).fetchone()
                    if relation is None or str(relation[0]) != job.primary_clip_id:
                        raise DerivativeJobConflictError("derivative incident relation is absent")
                    if str(relation[1]) not in {"PUBLISHED", "DERIVATIVE_PENDING", "COMPLETE"}:
                        raise DerivativeJobConflictError("primary evidence is not published")
                    retention = connection.execute(
                        "SELECT state FROM evidence_retention_states WHERE clip_id=?",
                        (job.primary_clip_id,),
                    ).fetchone()
                    if retention is not None and str(retention[0]) in {"PENDING", "PURGED"}:
                        raise DerivativeJobConflictError("primary evidence is retained")
                    connection.execute(
                        "INSERT INTO derivative_jobs "
                        "(incident_id,derivative_kind,request_id,state,created_at,updated_at) "
                        "VALUES (?,?,?,'PENDING',?,?)",
                        (
                            job.incident_id,
                            job.derivative_kind.value,
                            job.identity,
                            updated_at,
                            updated_at,
                        ),
                    )
                    record = self._get_with(connection, job.incident_id, job.derivative_kind)
                    assert record is not None
                    return DerivativeRequestResult(record, True)
                record = _record(row)
                if record.request_id != job.identity:
                    raise DerivativeJobConflictError("immutable derivative request differs")
                return DerivativeRequestResult(
                    record,
                    record.state is DerivativeJobState.PENDING and not record.cancel_requested,
                )
        finally:
            connection.close()

    def get(self, incident_id: str, kind: DerivativeKind) -> DerivativeJobRecord | None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            return self._get_with(connection, incident_id, kind)
        finally:
            connection.close()

    def recoverable(self) -> tuple[DerivativeJobRecord, ...]:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            rows = connection.execute(
                "SELECT incident_id,derivative_kind,request_id,state,media_id,reason,"
                "attempt_count,cancel_requested,revision,created_at,updated_at "
                "FROM derivative_jobs WHERE state IN ('PENDING','RUNNING') "
                "ORDER BY created_at,request_id"
            ).fetchall()
            return tuple(_record(row) for row in rows)
        finally:
            connection.close()

    def mark_running(self, job: AnnotatedDerivativeJob, *, updated_at: str) -> bool:
        return self._transition(
            job.incident_id,
            job.derivative_kind,
            expected="PENDING",
            target="RUNNING",
            updated_at=updated_at,
            extra="attempt_count=attempt_count+1",
            require_not_cancelled=True,
        )

    def reset_running(self, *, updated_at: str) -> int:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                return connection.execute(
                    "UPDATE derivative_jobs SET state='PENDING',revision=revision+1,updated_at=? "
                    "WHERE state='RUNNING' AND cancel_requested=0",
                    (updated_at,),
                ).rowcount
        finally:
            connection.close()

    def request_cancel(self, incident_id: str, kind: DerivativeKind, *, updated_at: str) -> bool:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                row = connection.execute(
                    "SELECT state,revision,cancel_requested FROM derivative_jobs "
                    "WHERE incident_id=? AND derivative_kind=?",
                    (incident_id, kind.value),
                ).fetchone()
                if row is None:
                    return False
                if str(row[0]) not in {"PENDING", "RUNNING"}:
                    return str(row[0]) == "CANCELLED"
                if bool(row[2]):
                    return True
                connection.execute(
                    "UPDATE derivative_jobs SET cancel_requested=1,revision=revision+1,"
                    "updated_at=? WHERE incident_id=? AND derivative_kind=? AND revision=?",
                    (updated_at, incident_id, kind.value, int(row[1])),
                )
                return True
        finally:
            connection.close()

    def mark_cancelled(self, job: AnnotatedDerivativeJob, *, updated_at: str) -> bool:
        return self._terminal(job, "CANCELLED", "CANCELLED", updated_at)

    def mark_unavailable(
        self, job: AnnotatedDerivativeJob, reason: str, *, updated_at: str
    ) -> bool:
        return self._terminal(job, "UNAVAILABLE", reason, updated_at)

    def mark_unavailable_record(
        self, record: DerivativeJobRecord, reason: str, *, updated_at: str
    ) -> bool:
        if not reason or len(reason) > 128:
            raise ValueError("derivative terminal reason is invalid")
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                return (
                    connection.execute(
                        "UPDATE derivative_jobs SET state='UNAVAILABLE',reason=?,"
                        "revision=revision+1,updated_at=? WHERE incident_id=? "
                        "AND derivative_kind=? AND state IN ('PENDING','RUNNING')",
                        (reason, updated_at, record.incident_id, record.derivative_kind.value),
                    ).rowcount
                    == 1
                )
        finally:
            connection.close()

    def mark_interrupted(self, job: AnnotatedDerivativeJob, *, updated_at: str) -> bool:
        """Return RUNNING work to PENDING only when operator cancel did not win.

        Leaves ``cancel_requested`` untouched. When the durable cancel flag is
        already set, the update is refused so callers can mark CANCELLED instead
        of clearing operator intent across a runtime stop.
        """
        return self._transition(
            job.incident_id,
            job.derivative_kind,
            expected="RUNNING",
            target="PENDING",
            updated_at=updated_at,
            require_not_cancelled=True,
        )

    def _terminal(
        self, job: AnnotatedDerivativeJob, target: str, reason: str, updated_at: str
    ) -> bool:
        if not reason or len(reason) > 128:
            raise ValueError("derivative terminal reason is invalid")
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                return (
                    connection.execute(
                        "UPDATE derivative_jobs SET state=?,reason=?,revision=revision+1,"
                        "updated_at=? WHERE incident_id=? AND derivative_kind=? "
                        "AND state IN ('PENDING','RUNNING')",
                        (target, reason, updated_at, job.incident_id, job.derivative_kind.value),
                    ).rowcount
                    == 1
                )
        finally:
            connection.close()

    def _transition(
        self,
        incident_id: str,
        kind: DerivativeKind,
        *,
        expected: str,
        target: str,
        updated_at: str,
        extra: str | None = None,
        require_not_cancelled: bool,
    ) -> bool:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                cancel = " AND cancel_requested=0" if require_not_cancelled else ""
                assignments = "state=?"
                if extra is not None:
                    assignments = f"{assignments},{extra}"
                assignments = f"{assignments},revision=revision+1,updated_at=?"
                return (
                    connection.execute(
                        f"UPDATE derivative_jobs SET {assignments} WHERE incident_id=? "
                        "AND derivative_kind=? AND state=?" + cancel,
                        (target, updated_at, incident_id, kind.value, expected),
                    ).rowcount
                    == 1
                )
        finally:
            connection.close()

    @staticmethod
    def _get_with(
        connection: sqlite3.Connection,
        incident_id: str,
        kind: DerivativeKind,
    ) -> DerivativeJobRecord | None:
        row = connection.execute(
            "SELECT incident_id,derivative_kind,request_id,state,media_id,reason,"
            "attempt_count,cancel_requested,revision,created_at,updated_at "
            "FROM derivative_jobs WHERE incident_id=? AND derivative_kind=?",
            (incident_id, kind.value),
        ).fetchone()
        return None if row is None else _record(row)


def _required_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"derivative job {field_name} is not an integer")
    return value


def _record(row: tuple[object, ...]) -> DerivativeJobRecord:
    return DerivativeJobRecord(
        str(row[0]),
        DerivativeKind(str(row[1])),
        str(row[2]),
        DerivativeJobState(str(row[3])),
        None if row[4] is None else str(row[4]),
        None if row[5] is None else str(row[5]),
        _required_integer(row[6], "attempt_count"),
        bool(row[7]),
        _required_integer(row[8], "revision"),
        str(row[9]),
        str(row[10]),
    )


__all__ = [
    "DerivativeJobConflictError",
    "DerivativeJobRecord",
    "DerivativeJobState",
    "DerivativeJobStore",
    "DerivativeRequestResult",
]
