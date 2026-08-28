"""The runbook's precondition query must agree with the migrator, by execution.

The runbook tells an operator to run a query and proceed when it reports that
nothing blocks. If that query disagrees with ``_require_schema17_drain``, the
operator gets a green light during a maintenance window and the migrator then
refuses with ``EDGE_DB_DRAIN_INCOMPLETE``.

The first version of this file checked for substrings and counted ``EXISTS``.
That proved almost nothing: flipping ``OR`` to ``AND``, inserting a ``NOT``, or
moving the tokens into prose would all have passed. So this version extracts the
SQL the runbook actually tells the operator to run, executes it against seeded
databases, and compares its verdict with what ``migrate_database`` really does.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.app.edge_db.migrator import MIGRATIONS, migrate_database
from backend.app.edge_db.schema import SchemaV17MigrationError

_ROOT = Path(__file__).resolve().parents[1]
_RUNBOOK = _ROOT / "docs/runbooks/backend-only-sqlite-cutover.md"
_SCHEMA_16 = 16


def _runbook_gate_sql() -> str:
    """Extract the SELECT the runbook embeds for the operator's gate check."""
    text = _RUNBOOK.read_text(encoding="utf-8")
    match = re.search(
        r"(SELECT EXISTS\(SELECT 1 FROM evidence_events.*?)\\\"\\\"\\\"", text, re.DOTALL
    )
    assert match, (
        "could not find the gate predicate in the runbook; the operator has no "
        "executable way to establish the precondition they are told to verify"
    )
    # The runbook embeds it inside a shell-quoted python heredoc.
    return match.group(1).replace('\\"', '"').strip()


def _seeded(tmp_path: Path, seed: Callable[[sqlite3.Connection], None], name: str) -> Path:
    database = tmp_path / f"{name}.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:_SCHEMA_16])
    connection = sqlite3.connect(database)
    try:
        seed(connection)
        connection.commit()
    finally:
        connection.close()
    return database


def _runbook_says_blocked(database: Path) -> bool:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        return bool(connection.execute(_runbook_gate_sql()).fetchone()[0])
    finally:
        connection.close()


def _migrator_refuses(database: Path) -> bool:
    try:
        migrate_database(database)
    except SchemaV17MigrationError:
        return True
    return False


def _clip(local_state: str = "VERIFIED", publish_state: str = "WAITING") -> Callable[..., None]:
    def seed(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, local_state, publish_state, state_version) "
            "VALUES ('clip:probe', ?, ?, 1)",
            (local_state, publish_state),
        )

    return seed


def _event(state: str) -> Callable[..., None]:
    # The schema enforces that IN_FLIGHT rows carry a lease and others do not.
    leased = state == "IN_FLIGHT"

    def seed(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, "
            " next_attempt_at, attempt_count, lease_owner, lease_expires_at, "
            " delivery_state) "
            "VALUES ('e1', '2026-08-22T00:00:00Z', '{}', ?, 1787000000.0, "
            "        1787000000.0, 0, ?, ?, 'PENDING')",
            (state, "sender-1" if leased else None, 1787003600.0 if leased else None),
        )

    return seed


def _slot_fields(state: str) -> tuple[str | None, str | None]:
    """The schema ties media_id/reason to the state; honour that.

    ``derivative_jobs`` additionally requires a reason for ``CANCELLED``.
    """
    if state == "AVAILABLE":
        return ("media:1", None)
    if state in {"UNAVAILABLE", "CORRUPT", "CANCELLED"}:
        return (None, "probe")
    return (None, None)


def _derivative_job(state: str) -> Callable[..., None]:
    def seed(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO derivative_jobs "
            "(incident_id, derivative_kind, request_id, state, media_id, reason, "
            " attempt_count, cancel_requested, revision, created_at, updated_at) "
            "VALUES ('inc:1', 'VIDEO', '" + ('r' * 64) + "', ?, ?, ?, 0, 0, 1, "
            "        '2026-08-22T00:00:00Z', '2026-08-22T00:00:00Z')",
            (state, *_slot_fields(state)),
        )

    return seed


def _derivative_slot(state: str) -> Callable[..., None]:
    def seed(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO derivative_evidence_slots "
            "(incident_id, derivative_kind, state, media_id, reason, revision, "
            " created_at, updated_at) "
            "VALUES ('inc:1', 'ANNOTATED_CLIP', ?, ?, ?, 1, "
            "        '2026-08-22T00:00:00Z', '2026-08-22T00:00:00Z')",
            (state, *_slot_fields(state)),
        )

    return seed


def _retention(state: str) -> Callable[..., None]:
    def seed(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO evidence_retention_states "
            "(clip_id, state, reason, revision, requested_at, updated_at) "
            "VALUES ('clip:r', ?, ?, 1, '2026-08-22T00:00:00Z', '2026-08-22T00:00:00Z')",
            (state, "disk" if state == "FAILED" else None),
        )

    return seed


#: One case per blocking predicate, plus the clean and deliberately-permitted ones.
_CASES: tuple[tuple[str, Callable[..., None]], ...] = (
    ("clean", lambda _: None),
    ("derivative_job_pending", _derivative_job("PENDING")),
    ("derivative_job_running", _derivative_job("RUNNING")),
    ("derivative_job_available", _derivative_job("AVAILABLE")),
    ("derivative_job_cancelled", _derivative_job("CANCELLED")),
    ("derivative_job_unavailable", _derivative_job("UNAVAILABLE")),
    ("derivative_job_corrupt", _derivative_job("CORRUPT")),
    ("derivative_slot_pending", _derivative_slot("PENDING")),
    ("derivative_slot_available", _derivative_slot("AVAILABLE")),
    ("derivative_slot_unavailable", _derivative_slot("UNAVAILABLE")),
    ("derivative_slot_corrupt", _derivative_slot("CORRUPT")),
    ("retention_pending", _retention("PENDING")),
    ("retention_purged", _retention("PURGED")),
    ("retention_failed", _retention("FAILED")),
    ("event_staged", _event("STAGED")),
    ("event_ready", _event("READY")),
    ("event_in_flight", _event("IN_FLIGHT")),
    ("event_acked", _event("ACKED")),
    ("clip_awaiting_finalize", _clip(local_state="AWAITING_FINALIZE")),
    ("clip_publish_in_flight", _clip(publish_state="IN_FLIGHT")),
    ("clip_publish_waiting", _clip(publish_state="WAITING")),
    ("clip_publish_published", _clip(publish_state="PUBLISHED")),
    ("clip_publish_permanent", _clip(publish_state="PERMANENT")),
    ("clip_publish_compatibility", _clip(publish_state="COMPATIBILITY")),
    ("clip_unavailable", _clip(local_state="UNAVAILABLE")),
    ("clip_corrupt", _clip(local_state="CORRUPT")),
)


@pytest.mark.parametrize(("name", "seed"), _CASES, ids=[case[0] for case in _CASES])
def test_the_runbook_query_matches_the_migrator(
    tmp_path: Path, name: str, seed: Callable[..., None]
) -> None:
    """Executed equivalence: the operator's answer must be the migrator's answer."""
    for_query = _seeded(tmp_path, seed, f"{name}-query")
    for_migrate = _seeded(tmp_path, seed, f"{name}-migrate")

    runbook_blocked = _runbook_says_blocked(for_query)
    migrator_refused = _migrator_refuses(for_migrate)

    assert runbook_blocked == migrator_refused, (
        f"case {name!r}: the runbook query reports "
        f"{'blocked' if runbook_blocked else 'clear'} while the migrator "
        f"{'refuses' if migrator_refused else 'proceeds'}. An operator following "
        f"the runbook would be misled during a maintenance window."
    )


def test_the_extracted_query_is_actually_executable_sql() -> None:
    """Guard the guard: a malformed extraction would make every case vacuous."""
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE evidence_events (state TEXT)")
    connection.execute("CREATE TABLE evidence_clips (local_state TEXT, publish_state TEXT)")
    connection.execute("CREATE TABLE derivative_jobs (state TEXT)")
    connection.execute("CREATE TABLE derivative_evidence_slots (state TEXT)")
    connection.execute("CREATE TABLE evidence_retention_states (state TEXT)")

    result = connection.execute(_runbook_gate_sql()).fetchone()[0]

    assert result in (0, 1), "the extracted runbook query did not evaluate to a boolean"
