from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from compact_cutover_fixtures import cutover_request, sha256

from backend.app.edge_db.compact_cutover import (
    CompactCutoverRequest,
    CutoverPhase,
    run_compact_cutover,
)
from backend.app.edge_db.compatibility import CANONICAL_MIGRATION_LEDGER

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")
Mutation = Callable[[sqlite3.Connection], None]
_EXPECTED_V18 = CANONICAL_MIGRATION_LEDGER[17]


def _version(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _valid_interrupted_candidate(tmp_path: Path) -> CompactCutoverRequest:
    request = cutover_request(tmp_path)

    def interrupt(phase: CutoverPhase) -> None:
        if phase is CutoverPhase.PRE_RENAME_DIRECTORY_SYNCED:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        run_compact_cutover(request, on_phase=interrupt)
    assert request.candidate.exists()
    assert _version(request.live) == 17
    return request


def _wrong_name(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE schema_migrations SET name='EVIL_FALSE_IDENTITY' WHERE version=18")


def _wrong_checksum(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE schema_migrations SET checksum=? WHERE version=18", ("0" * 64,))


def _missing_row18(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM schema_migrations WHERE version=18")


def _extra_v18_equivalent(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO schema_migrations "
        "(version,name,applied_at,checksum,source_schema_version,source_db_sha256,"
        "reconciliation_sha256) SELECT 19,name || '_duplicate',applied_at,checksum,"
        "source_schema_version,"
        "source_db_sha256,reconciliation_sha256 FROM schema_migrations WHERE version=18"
    )


def _wrong_version_pair(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE schema_migrations SET version=19 WHERE version=18")


def _wrong_source_schema(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE schema_migrations SET source_schema_version=16 WHERE version=18")


def _wrong_source_hash(connection: sqlite3.Connection) -> None:
    connection.execute(
        "UPDATE schema_migrations SET source_db_sha256=? WHERE version=18", ("0" * 64,)
    )


def _wrong_receipt_hash(connection: sqlite3.Connection) -> None:
    connection.execute(
        "UPDATE schema_migrations SET reconciliation_sha256=? WHERE version=18", ("0" * 64,)
    )


@pytest.mark.parametrize(
    "mutation",
    (
        _wrong_name,
        _wrong_checksum,
        _missing_row18,
        _extra_v18_equivalent,
        _wrong_version_pair,
        _wrong_source_schema,
        _wrong_source_hash,
        _wrong_receipt_hash,
    ),
    ids=lambda mutation: mutation.__name__,
)
def test_invalid_resumed_candidate_is_discarded_and_rebuilt(
    tmp_path: Path, mutation: Mutation
) -> None:
    request = _valid_interrupted_candidate(tmp_path)
    source_hash = sha256(request.source)
    with sqlite3.connect(request.candidate) as connection:
        mutation(connection)
        connection.commit()
    observed: list[CutoverPhase] = []

    def prove_rebuild(phase: CutoverPhase) -> None:
        observed.append(phase)
        if phase is CutoverPhase.CANDIDATE_WRITTEN:
            assert _version(request.live) == 17
            assert sha256(request.source) == source_hash == sha256(request.archive)

    run_compact_cutover(request, on_phase=prove_rebuild)

    assert CutoverPhase.CANDIDATE_WRITTEN in observed
    assert sha256(request.source) == source_hash == sha256(request.archive)
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        assert (
            connection.execute(
                "SELECT version,name,checksum FROM schema_migrations WHERE version=18"
            ).fetchone()
            == _EXPECTED_V18
        )
        assert connection.execute(
            "SELECT count(*) FROM schema_migrations WHERE version>18"
        ).fetchone() == (0,)


def test_identity_is_rechecked_immediately_before_replace(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    source_hash = sha256(request.source)

    def mutate_after_reconciliation(phase: CutoverPhase) -> None:
        if phase is CutoverPhase.CANDIDATE_FILE_SYNCED:
            with sqlite3.connect(request.candidate) as connection:
                _wrong_name(connection)
                connection.commit()

    with pytest.raises(sqlite3.DatabaseError, match="CANONICAL_IDENTITY"):
        run_compact_cutover(request, on_phase=mutate_after_reconciliation)

    assert _version(request.live) == 17
    assert sha256(request.source) == source_hash == sha256(request.archive)
    observed: list[CutoverPhase] = []
    run_compact_cutover(request, on_phase=observed.append)
    assert CutoverPhase.CANDIDATE_WRITTEN in observed
    assert _version(request.live) == 18


def test_valid_resumed_candidate_installs_without_rebuild(tmp_path: Path) -> None:
    request = _valid_interrupted_candidate(tmp_path)
    observed: list[CutoverPhase] = []

    run_compact_cutover(request, on_phase=observed.append)

    assert CutoverPhase.CANDIDATE_WRITTEN not in observed
    with sqlite3.connect(request.live) as connection:
        assert (
            connection.execute(
                "SELECT version,name,checksum FROM schema_migrations WHERE version=18"
            ).fetchone()
            == _EXPECTED_V18
        )
