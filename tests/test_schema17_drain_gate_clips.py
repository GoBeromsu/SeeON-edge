"""The schema-17 drain gate must block unfinalized clips without bricking cutover.

Two opposite mistakes are possible here, and both are load-bearing.

Gating too narrowly lets the migration run over clips whose media never
finalized, which is exactly the state the 1053 stalled live clips are in.

Gating too widely is worse and less obvious: `publish_state` defaults to
`'WAITING'`, so a clip that finalized perfectly and was simply never published
upstream also reads as `WAITING`. A gate on that condition blocks every real
database permanently and the cutover can never happen. That regression reached
the suite once and is pinned here.

The discriminator is therefore `local_state`, which describes whether the local
evidence is complete, plus `publish_state = 'IN_FLIGHT'`, which means a delivery
is mid-flight and must not be interrupted.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.app.edge_db.migrator import MIGRATIONS, migrate_database
from backend.app.edge_db.schema import SchemaV17MigrationError

_SCHEMA_16 = 16


def _seeded(database: Path, seed: Callable[[sqlite3.Connection], None]) -> Path:
    migrate_database(database, migrations=MIGRATIONS[:_SCHEMA_16])
    connection = sqlite3.connect(database)
    try:
        seed(connection)
        connection.commit()
    finally:
        connection.close()
    return database


def _insert_clip(
    connection: sqlite3.Connection, local_state: str, publish_state: str = "WAITING"
) -> None:
    connection.execute(
        "INSERT INTO evidence_clips (clip_id, local_state, publish_state, state_version) "
        "VALUES ('clip:probe', ?, ?, 1)",
        (local_state, publish_state),
    )


def _migrates(database: Path) -> bool:
    try:
        migrate_database(database)
    except SchemaV17MigrationError:
        return False
    return True


def test_a_clip_awaiting_finalize_blocks_the_migration(tmp_path: Path) -> None:
    """The live stall condition: media never finalized."""
    database = _seeded(
        tmp_path / "edge.sqlite3", lambda c: _insert_clip(c, "AWAITING_FINALIZE")
    )
    assert not _migrates(database)


def test_a_publish_in_flight_blocks_the_migration(tmp_path: Path) -> None:
    """Interrupting a delivery mid-flight would strand the upload."""
    database = _seeded(
        tmp_path / "edge.sqlite3", lambda c: _insert_clip(c, "VERIFIED", "IN_FLIGHT")
    )
    assert not _migrates(database)


@pytest.mark.parametrize("local_state", ["VERIFIED", "UNAVAILABLE", "CORRUPT"])
def test_a_finalized_but_unpublished_clip_does_not_block(
    tmp_path: Path, local_state: str
) -> None:
    """`publish_state` defaults to WAITING; gating on it would brick every cutover."""
    database = _seeded(
        tmp_path / "edge.sqlite3", lambda c: _insert_clip(c, local_state, "WAITING")
    )
    assert _migrates(database), (
        f"a clip with local_state={local_state!r} has complete local evidence; "
        "its unpublished publish_state must not block the migration, or no real "
        "database can ever be migrated"
    )


def test_an_empty_database_migrates(tmp_path: Path) -> None:
    """Baseline: the gate must not refuse when there is nothing to drain."""
    database = _seeded(tmp_path / "edge.sqlite3", lambda _: None)
    assert _migrates(database)
