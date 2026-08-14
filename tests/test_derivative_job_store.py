from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    DerivativeKind,
)
from worker.pipeline.output.evidence.derivative_job_store import DerivativeJobStore


def _job() -> AnnotatedDerivativeJob:
    # Store transitions only read incident_id/kind; avoid full scene construction.
    return cast(
        AnnotatedDerivativeJob,
        cast(
            object,
            SimpleNamespace(
                incident_id="incident-a",
                derivative_kind=DerivativeKind.STILL,
            ),
        ),
    )


def _insert_running(database: Path, *, cancel_requested: int = 0) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO derivative_jobs "
            "(incident_id,derivative_kind,request_id,state,attempt_count,cancel_requested,"
            "revision,created_at,updated_at) VALUES "
            "('incident-a','STILL',?,'RUNNING',1,?,1,'now','now')",
            ("a" * 64, cancel_requested),
        )
        connection.commit()


@pytest.mark.parametrize(
    ("attempt_count", "revision", "field_name"),
    (
        ("malformed", 1, "attempt_count"),
        (0, "malformed", "revision"),
    ),
)
def test_job_store_rejects_malformed_integer_rows(
    tmp_path: Path,
    attempt_count: object,
    revision: object,
    field_name: str,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA writable_schema = ON")
        row = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'derivative_jobs'"
        ).fetchone()
        assert row is not None
        schema = str(row[0])
        assert ") STRICT" in schema
        connection.execute(
            "UPDATE sqlite_schema SET sql = ? WHERE name = 'derivative_jobs'",
            (schema.replace(") STRICT", ")"),),
        )
        connection.execute("PRAGMA writable_schema = OFF")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO derivative_jobs "
            "(incident_id,derivative_kind,request_id,state,attempt_count,cancel_requested,"
            "revision,created_at,updated_at) VALUES "
            "('incident-a','STILL',?,'PENDING',?,0,?,'now','now')",
            ("a" * 64, attempt_count, revision),
        )

    with pytest.raises(TypeError, match=field_name):
        DerivativeJobStore(database).get("incident-a", DerivativeKind.STILL)


def test_mark_interrupted_returns_pending_without_clearing_uncancelled_flag(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _insert_running(database, cancel_requested=0)
    store = DerivativeJobStore(database)

    assert store.mark_interrupted(_job(), updated_at="later") is True
    record = store.get("incident-a", DerivativeKind.STILL)
    assert record is not None
    assert record.state.value == "PENDING"
    assert not record.cancel_requested


def test_mark_interrupted_refuses_when_cancel_requested(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _insert_running(database, cancel_requested=1)
    store = DerivativeJobStore(database)

    assert store.mark_interrupted(_job(), updated_at="later") is False
    record = store.get("incident-a", DerivativeKind.STILL)
    assert record is not None
    assert record.state.value == "RUNNING"
    assert record.cancel_requested
    assert store.mark_cancelled(_job(), updated_at="later") is True
    cancelled = store.get("incident-a", DerivativeKind.STILL)
    assert cancelled is not None
    assert cancelled.state.value == "CANCELLED"
    assert cancelled.cancel_requested
