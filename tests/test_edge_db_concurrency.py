from __future__ import annotations

import multiprocessing
import os
import sqlite3
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.connection import (
    RuntimeActor,
    best_effort_zero_wait_write,
    open_runtime_database,
    write_transaction,
)
from backend.app.edge_db.migrator import migrate_database


def _hold_worker_write(database: str, channel: Connection) -> None:
    connection = open_runtime_database(Path(database), actor=RuntimeActor.API)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO runtime_contention VALUES (1, 'worker')")
        channel.send("LOCKED")
        assert channel.recv() == "COMMIT"
        connection.commit()
        channel.send("COMMITTED")
    finally:
        connection.close()
        channel.close()


def _write_api_after_barrier(database: str, channel: Connection) -> None:
    connection = open_runtime_database(Path(database), actor=RuntimeActor.API)
    try:
        channel.send("STARTING")
        with write_transaction(connection):
            connection.execute("INSERT INTO control_contention VALUES (1, 'api')")
        channel.send("COMMITTED")
    except Exception as error:  # noqa: BLE001 - process boundary returns the actual failure
        channel.send(f"ERROR:{type(error).__name__}:{error}")
    finally:
        connection.close()
        channel.close()


def _zero_wait_fault_write(connection: sqlite3.Connection) -> None:
    connection.execute("INSERT INTO runtime_contention VALUES (2, 'fault')")


def _hold_runtime_open(database: str, actor: str, channel: Connection) -> None:
    connection = open_runtime_database(Path(database), actor=RuntimeActor(actor))
    try:
        channel.send("RUNTIME_OPEN")
        assert channel.recv() == "CLOSE"
    finally:
        connection.close()
        channel.send("CLOSED")
        channel.close()


def _prepare_database(path: Path) -> None:
    migrate_database(path)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE control_contention (
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT;
            CREATE TABLE runtime_contention (
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT;
            """
        )
    finally:
        connection.close()


@pytest.mark.parametrize("actor", [RuntimeActor.API, RuntimeActor.API])
def test_public_migration_refuses_while_runtime_holds_deployment_lock(
    tmp_path: Path,
    actor: RuntimeActor,
) -> None:
    database_path = tmp_path / "edge" / "edge.sqlite3"
    _prepare_database(database_path)
    context = multiprocessing.get_context("spawn")
    parent_channel, child_channel = context.Pipe()
    runtime = context.Process(
        target=_hold_runtime_open,
        args=(os.fspath(database_path), actor.value, child_channel),
    )
    runtime.start()
    assert parent_channel.poll(10), f"{actor.value} runtime did not open"
    assert parent_channel.recv() == "RUNTIME_OPEN"

    try:
        with pytest.raises(EdgeDatabaseError, match="deployment lock.*running runtime"):
            migrate_database(database_path)
    finally:
        parent_channel.send("CLOSE")
        assert parent_channel.poll(10), f"{actor.value} runtime did not close"
        assert parent_channel.recv() == "CLOSED"
        runtime.join(10)
    assert runtime.exitcode == 0


def test_module_cli_refuses_while_runtime_holds_deployment_lock(tmp_path: Path) -> None:
    database_path = tmp_path / "edge" / "edge.sqlite3"
    _prepare_database(database_path)
    context = multiprocessing.get_context("spawn")
    parent_channel, child_channel = context.Pipe()
    runtime = context.Process(
        target=_hold_runtime_open,
        args=(os.fspath(database_path), RuntimeActor.API.value, child_channel),
    )
    runtime.start()
    assert parent_channel.poll(10), "API runtime did not open"
    assert parent_channel.recv() == "RUNTIME_OPEN"

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "backend.app.edge_db", "--database", os.fspath(database_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        parent_channel.send("CLOSE")
        assert parent_channel.poll(10), "API runtime did not close"
        assert parent_channel.recv() == "CLOSED"
        runtime.join(10)
    assert runtime.exitcode == 0
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "EDGE_DB_MIGRATION_FAILED" in completed.stderr
    assert "deployment lock" in completed.stderr
    assert "running runtime" in completed.stderr


def test_two_process_api_and_worker_writes_serialize_and_keep_integrity(tmp_path: Path) -> None:
    database_path = tmp_path / "edge" / "edge.sqlite3"
    _prepare_database(database_path)
    context = multiprocessing.get_context("spawn")
    holder_parent, holder_child = context.Pipe()
    contender_parent, contender_child = context.Pipe()
    holder = context.Process(
        target=_hold_worker_write,
        args=(os.fspath(database_path), holder_child),
    )
    contender = context.Process(
        target=_write_api_after_barrier,
        args=(os.fspath(database_path), contender_child),
    )

    holder.start()
    assert holder_parent.poll(10), "worker did not acquire the write lock"
    assert holder_parent.recv() == "LOCKED"
    contender.start()
    assert contender_parent.poll(10), "API contender did not reach its transaction"
    assert contender_parent.recv() == "STARTING"
    holder_parent.send("COMMIT")

    assert holder_parent.poll(10), "worker did not commit"
    assert holder_parent.recv() == "COMMITTED"
    assert contender_parent.poll(10), "API did not finish within the fixed busy timeout"
    assert contender_parent.recv() == "COMMITTED"
    holder.join(10)
    contender.join(10)
    assert holder.exitcode == 0
    assert contender.exitcode == 0

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("SELECT * FROM runtime_contention").fetchall() == [(1, "worker")]
        assert connection.execute("SELECT * FROM control_contention").fetchall() == [(1, "api")]
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        connection.close()


def test_fatal_fault_best_effort_write_returns_without_waiting_for_writer(tmp_path: Path) -> None:
    database_path = tmp_path / "edge" / "edge.sqlite3"
    _prepare_database(database_path)
    context = multiprocessing.get_context("spawn")
    holder_parent, holder_child = context.Pipe()
    holder = context.Process(
        target=_hold_worker_write,
        args=(os.fspath(database_path), holder_child),
    )
    holder.start()
    assert holder_parent.poll(10), "worker did not acquire the write lock"
    assert holder_parent.recv() == "LOCKED"

    written = best_effort_zero_wait_write(
        database_path,
        actor=RuntimeActor.API,
        write=_zero_wait_fault_write,
    )
    assert written is False

    holder_parent.send("COMMIT")
    assert holder_parent.poll(10)
    assert holder_parent.recv() == "COMMITTED"
    holder.join(10)
    assert holder.exitcode == 0

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute(
            "SELECT count(*) FROM runtime_contention WHERE id = 2"
        ).fetchone() == (0,)
    finally:
        connection.close()
