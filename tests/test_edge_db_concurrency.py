from __future__ import annotations

import multiprocessing
import os
import sqlite3
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.snapshot import export_snapshot
from tests_support.sqlite_source import create_schema19_source, hold_runtime_lock


def _hold_runtime_lock(database: str, channel: Connection) -> None:
    with hold_runtime_lock(Path(database)):
        channel.send("LOCKED")
        assert channel.recv() == "RELEASE"
    channel.send("RELEASED")
    channel.close()


def _prepare_database(path: Path) -> None:
    create_schema19_source(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO edge_site (id,updated_at) VALUES (1,'2026-08-24T00:00:00Z')"
        )
        connection.commit()
    finally:
        connection.close()


def test_snapshot_export_refuses_while_another_process_holds_the_deployment_lock(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "edge" / "edge.sqlite3"
    _prepare_database(database_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    context = multiprocessing.get_context("spawn")
    parent_channel, child_channel = context.Pipe()
    runtime = context.Process(
        target=_hold_runtime_lock,
        args=(os.fspath(database_path), child_channel),
    )
    runtime.start()
    assert parent_channel.poll(10), "runtime did not take the deployment lock"
    assert parent_channel.recv() == "LOCKED"

    try:
        with pytest.raises(
            MigrationError, match="^source database is in use by a running runtime$"
        ):
            export_snapshot(database_path, snapshots / "edge.snapshot.sqlite3")
    finally:
        parent_channel.send("RELEASE")
        assert parent_channel.poll(10), "runtime did not release the deployment lock"
        assert parent_channel.recv() == "RELEASED"
        runtime.join(10)
    assert runtime.exitcode == 0
    assert list(snapshots.iterdir()) == []


# The diagnostics-file isolation this used to cover (a writer on
# edge-diagnostics.sqlite3 never blocking edge.sqlite3) is now exercised
# through the real production wiring instead -- see
# tests/test_api_execution_records.py::
# test_lifespan_diagnostics_query_and_product_write_skip_a_pending_diagnostics_write
# (#579/#580, S4).
