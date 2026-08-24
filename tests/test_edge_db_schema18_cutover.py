from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.app.edge_db.cutover_authorization import (
    CompactCutoverAuthorization,
    CompactCutoverRequiredError,
    CompactCutoverSourceError,
    issue_compact_cutover_authorization,
)
from backend.app.edge_db.migrator import DeploymentLock, deployment_lock, migrate_database
from backend.app.edge_db.schema import MIGRATIONS, SchemaV18MigrationError

RECONCILIATION = b"schema18-reconciliation-receipt"


def _v17(database: Path) -> None:
    migrate_database(database, migrations=MIGRATIONS[:17])


def _issue(lock: DeploymentLock, candidate: Path, source: Path | None = None) -> object:
    return issue_compact_cutover_authorization(
        lock,
        source=candidate if source is None else source,
        candidate=candidate,
        reconciliation=RECONCILIATION,
    )


def test_default_migrate_refuses_drained_v17_before_mutation(tmp_path: Path) -> None:
    database = tmp_path / "live.sqlite3"
    _v17(database)
    before = database.read_bytes()

    with pytest.raises(CompactCutoverRequiredError, match="EDGE_DB_CUTOVER_UNAUTHORIZED"):
        migrate_database(database)

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert "runtime_settings" in tables
        assert "edge_site" not in tables


def test_forged_authorization_refuses_v17_upgrade(tmp_path: Path) -> None:
    database = tmp_path / "forged.sqlite3"
    _v17(database)
    before = database.read_bytes()
    forged = CompactCutoverAuthorization(_ticket="forged")

    with deployment_lock(database.parent) as lock:
        with pytest.raises(CompactCutoverRequiredError, match="FORGED_LOCK"):
            migrate_database(database, lock=lock, cutover=forged)

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_forged_lock_cannot_issue_or_redeem(tmp_path: Path) -> None:
    database = tmp_path / "live.sqlite3"
    _v17(database)
    fake = DeploymentLock(database.parent.resolve(), 0, True)
    with pytest.raises(CompactCutoverRequiredError, match="FORGED_LOCK"):
        _issue(fake, database)


def test_arbitrary_hash_is_not_trusted(tmp_path: Path) -> None:
    database = tmp_path / "cand.sqlite3"
    _v17(database)
    before = database.read_bytes()
    with deployment_lock(database.parent) as lock:
        with pytest.raises(TypeError):
            issue_compact_cutover_authorization(
                lock,
                source_schema_version=17,
                source_db_sha256="ab" * 32,
                reconciliation_sha256="cd" * 32,
            )
        authorization = _issue(lock, database)
        result = migrate_database(database, lock=lock, cutover=authorization)
    assert result.current_version == 18
    expected_source = hashlib.sha256(before).hexdigest()
    expected_recon = hashlib.sha256(RECONCILIATION).hexdigest()
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT source_schema_version, source_db_sha256, reconciliation_sha256 "
            "FROM schema_migrations WHERE version = 18"
        ).fetchone()
        assert row == (17, expected_source, expected_recon)
        assert row[1] != "ab" * 32


def test_source_candidate_preimage_must_match(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    _v17(source)
    _v17(candidate)
    with sqlite3.connect(candidate) as connection:
        connection.execute("CREATE TABLE extra_preimage (id INTEGER PRIMARY KEY)")
        connection.commit()
    with deployment_lock(candidate.parent) as lock:
        with pytest.raises(CompactCutoverSourceError):
            issue_compact_cutover_authorization(
                lock,
                source=source,
                candidate=candidate,
                reconciliation=RECONCILIATION,
            )


def test_authorized_candidate_upgrade_populates_row18_provenance(tmp_path: Path) -> None:
    source = tmp_path / "archive.sqlite3"
    candidate = tmp_path / "state" / "candidate.sqlite3"
    candidate.parent.mkdir()
    _v17(source)
    candidate.write_bytes(source.read_bytes())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    recon = hashlib.sha256(RECONCILIATION).hexdigest()
    assert digest != recon

    with deployment_lock(candidate.parent) as lock:
        authorization = issue_compact_cutover_authorization(
            lock,
            source=source,
            candidate=candidate,
            reconciliation=RECONCILIATION,
        )
        result = migrate_database(candidate, lock=lock, cutover=authorization)

    assert result.previous_version == 17
    assert result.current_version == 18
    with sqlite3.connect(candidate) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        row = connection.execute(
            "SELECT source_schema_version, source_db_sha256, reconciliation_sha256 "
            "FROM schema_migrations WHERE version = 18"
        ).fetchone()
        assert row == (17, digest, recon)
        historical = connection.execute(
            "SELECT source_schema_version, source_db_sha256, reconciliation_sha256 "
            "FROM schema_migrations WHERE version < 18"
        ).fetchall()
        assert historical == [(None, None, None)] * 17


def test_wrong_candidate_is_refused_before_mutation(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    first = state / "first.sqlite3"
    other = state / "other.sqlite3"
    _v17(first)
    other.write_bytes(first.read_bytes())
    before = other.read_bytes()
    with deployment_lock(state) as lock:
        authorization = _issue(lock, first)
        with pytest.raises(CompactCutoverRequiredError, match="WRONG_CANDIDATE"):
            migrate_database(other, lock=lock, cutover=authorization)
    assert other.read_bytes() == before
    with sqlite3.connect(other) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_reused_authorization_is_refused(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    first = state / "first.sqlite3"
    second = state / "second.sqlite3"
    _v17(first)
    second.write_bytes(first.read_bytes())
    before = second.read_bytes()
    with deployment_lock(state) as lock:
        authorization = _issue(lock, first)
        migrate_database(first, lock=lock, cutover=authorization)
        with pytest.raises(CompactCutoverRequiredError, match="REUSED"):
            migrate_database(second, lock=lock, cutover=authorization)
    assert second.read_bytes() == before
    with sqlite3.connect(second) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_expired_lock_cannot_redeem_authorization(tmp_path: Path) -> None:
    database = tmp_path / "cand.sqlite3"
    _v17(database)
    before = database.read_bytes()
    with deployment_lock(database.parent) as lock:
        authorization = _issue(lock, database)
    with pytest.raises(CompactCutoverRequiredError, match="EXPIRED_LOCK"):
        migrate_database(database, lock=lock, cutover=authorization)
    assert database.read_bytes() == before


def test_candidate_changed_after_issuance_is_refused(tmp_path: Path) -> None:
    database = tmp_path / "cand.sqlite3"
    _v17(database)
    with deployment_lock(database.parent) as lock:
        authorization = _issue(lock, database)
        database.write_bytes(database.read_bytes() + b"\x00")
        before = database.read_bytes()
        with pytest.raises(CompactCutoverRequiredError, match="CANDIDATE_CHANGED"):
            migrate_database(database, lock=lock, cutover=authorization)
    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_fresh_install_still_walks_to_schema18_without_authorization(tmp_path: Path) -> None:
    database = tmp_path / "fresh.sqlite3"

    result = migrate_database(database)

    assert result.previous_version == 0
    assert result.current_version == 18
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT source_schema_version, source_db_sha256, reconciliation_sha256 "
            "FROM schema_migrations WHERE version = 18"
        ).fetchone()
        assert row == (None, None, None)


def _insert_event(connection: sqlite3.Connection, state: str) -> None:
    owner = "worker:drain" if state == "IN_FLIGHT" else None
    expires = 2.0 if state == "IN_FLIGHT" else None
    connection.execute(
        "INSERT INTO evidence_events "
        "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at,"
        "lease_owner,lease_expires_at) VALUES "
        "('event:drain','2026-08-24T00:00:00Z','{}',?,1,1,?,?)",
        (state, owner, expires),
    )


def _insert_clip(connection: sqlite3.Connection, *, local: str, publish: str) -> None:
    connection.execute(
        "INSERT INTO evidence_clips (clip_id, local_state, publish_state) VALUES (?,?,?)",
        ("clip:drain", local, publish),
    )


def _insert_retention_pending(connection: sqlite3.Connection) -> None:
    _insert_clip(connection, local="VERIFIED", publish="WAITING")
    connection.execute(
        "INSERT INTO evidence_retention_states "
        "(clip_id,state,reason,revision,requested_at,updated_at) "
        "VALUES ('clip:drain','PENDING',NULL,1,'2026-08-24T00:00:00Z','2026-08-24T00:00:00Z')"
    )


def _insert_incident_graph(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO evidence_events "
        "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at) "
        "VALUES ('event:drain','2026-08-24T00:00:00Z','{}','ACKED',1,1)"
    )
    connection.execute(
        "INSERT INTO evidence_incidents "
        "(incident_id,edge_event_id,camera_id,event_type,detected_at,"
        "provenance_state,provenance_missing_reason,lifecycle_state,created_at,updated_at) "
        "VALUES ('incident:drain','event:drain','cam','fall','2026-08-24T00:00:00Z',"
        "'MISSING','NOT_RECORDED','STAGING','2026-08-24T00:00:00Z','2026-08-24T00:00:00Z')"
    )


def _insert_derivative_job(connection: sqlite3.Connection, state: str) -> None:
    _insert_incident_graph(connection)
    connection.execute(
        "INSERT INTO derivative_jobs "
        "(incident_id,derivative_kind,request_id,state,created_at,updated_at) "
        "VALUES ('incident:drain','STILL',?,?,'2026-08-24T00:00:00Z','2026-08-24T00:00:00Z')",
        ("a" * 64, state),
    )


def _insert_derivative_slot(connection: sqlite3.Connection) -> None:
    _insert_incident_graph(connection)
    connection.execute(
        "INSERT INTO derivative_evidence_slots "
        "(incident_id,derivative_kind,state,media_id,reason,revision,created_at,updated_at) "
        "VALUES ('incident:drain','ANNOTATED_CLIP','PENDING',NULL,NULL,1,"
        "'2026-08-24T00:00:00Z','2026-08-24T00:00:00Z')"
    )


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(lambda c: _insert_event(c, "STAGED"), id="event_staged"),
        pytest.param(lambda c: _insert_event(c, "READY"), id="event_ready"),
        pytest.param(lambda c: _insert_event(c, "IN_FLIGHT"), id="event_in_flight"),
        pytest.param(
            lambda c: _insert_clip(c, local="AWAITING_FINALIZE", publish="WAITING"),
            id="clip_awaiting_finalize",
        ),
        pytest.param(
            lambda c: _insert_clip(c, local="VERIFIED", publish="IN_FLIGHT"),
            id="clip_publish_in_flight",
        ),
        pytest.param(lambda c: _insert_derivative_job(c, "PENDING"), id="derivative_job_pending"),
        pytest.param(lambda c: _insert_derivative_job(c, "RUNNING"), id="derivative_job_running"),
        pytest.param(_insert_derivative_slot, id="derivative_slot_pending"),
        pytest.param(_insert_retention_pending, id="retention_pending"),
    ],
)
def test_schema18_drain_sentinels_still_refuse_without_mutation(
    tmp_path: Path,
    seed: Callable[[sqlite3.Connection], None],
) -> None:
    database = tmp_path / "drain.sqlite3"
    _v17(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        seed(connection)
        connection.commit()
    before = database.read_bytes()

    with pytest.raises(SchemaV18MigrationError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        migrate_database(database)

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)
