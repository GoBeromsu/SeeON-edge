"""Filesystem inventory gate must retire once central schema 17 is installed."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db import inventory


def _database(tmp_path: Path, version: int) -> Path:
    database = tmp_path / "edge-state" / "edge.sqlite3"
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
    return database


def _pending_entry(runtime_state: Path) -> None:
    queue = runtime_state / "delivery-queue"
    queue.mkdir(parents=True)
    (queue / "event-1.json").write_text("{}", encoding="utf-8")


def test_pre_cutover_pending_delivery_queue_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = _database(tmp_path, 16)
    runtime_state = tmp_path / "worker-state"
    _pending_entry(runtime_state)

    exit_code = inventory.main(
        [
            "--database",
            str(database),
            "--runtime-state-dir",
            str(runtime_state),
            "--clip-store-dir",
            str(tmp_path / "clip-store"),
        ]
    )

    assert exit_code == 1
    assert "delivery-queue entries=1" in capsys.readouterr().err


def test_pre_cutover_empty_filesystem_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = _database(tmp_path, 16)

    exit_code = inventory.main(
        [
            "--database",
            str(database),
            "--runtime-state-dir",
            str(tmp_path / "worker-state"),
            "--clip-store-dir",
            str(tmp_path / "clip-store"),
        ]
    )

    assert exit_code == 0
    assert "EDGE_FS_INVENTORY_OK schema=16" in capsys.readouterr().out


def test_pre_cutover_staged_clip_blocks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    database = _database(tmp_path, 16)
    clip_store = tmp_path / "clip-store"
    (clip_store / "clips" / ".staging" / "clip-1").mkdir(parents=True)

    exit_code = inventory.main(
        [
            "--database",
            str(database),
            "--runtime-state-dir",
            str(tmp_path / "worker-state"),
            "--clip-store-dir",
            str(clip_store),
        ]
    )

    assert exit_code == 1
    assert "clip staging entries=1" in capsys.readouterr().err


def test_post_cutover_bypasses_pending_delivery_queue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = _database(tmp_path, 17)
    runtime_state = tmp_path / "worker-state"
    _pending_entry(runtime_state)

    exit_code = inventory.main(
        [
            "--database",
            str(database),
            "--runtime-state-dir",
            str(runtime_state),
            "--clip-store-dir",
            str(tmp_path / "clip-store"),
        ]
    )

    assert exit_code == 0
    assert "gate retired post-cutover" in capsys.readouterr().out


def test_schema_version_connection_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _database(tmp_path, 16)
    connect = sqlite3.connect
    calls: list[tuple[str, bool]] = []

    def recording_connect(database_name: str, *, uri: bool = False) -> sqlite3.Connection:
        calls.append((database_name, uri))
        return connect(database_name, uri=uri)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)

    assert inventory.read_schema_version(database) == 16
    assert calls == [(f"file:{database}?mode=ro", True)]


def test_retained_refused_evidence_blocks_the_pre_v17_gate(tmp_path: Path) -> None:
    """Retained evidence is undelivered evidence, so the gate must see it.

    A 422 retains the entry rather than deleting it. The gate counted only the
    live queue, so a deployment holding refused evidence nobody had reviewed
    reported all-clear and the cutover proceeded over it.
    """
    state = tmp_path / "runtime-state"
    (state / "delivery-queue").mkdir(parents=True)
    retention = state / "delivery-queue-dead-letter"
    retention.mkdir(parents=True)
    (retention / "422.0.event-abc.json").write_bytes(b"{}")
    clip_store = tmp_path / "clip-store"
    (clip_store / "clips" / ".staging").mkdir(parents=True)

    result = inventory.inspect_filesystem(state, clip_store)

    assert result.retained_refused_entries == 1
    assert not result.is_empty, (
        "the gate reported all-clear while refused evidence sat undelivered"
    )
    assert "retained refused entries=1" in result.describe_pending()


def test_a_clear_deployment_still_passes_the_gate(tmp_path: Path) -> None:
    """Guard the guard: the gate must remain clearable."""
    state = tmp_path / "runtime-state"
    (state / "delivery-queue").mkdir(parents=True)
    clip_store = tmp_path / "clip-store"
    (clip_store / "clips" / ".staging").mkdir(parents=True)

    assert inventory.inspect_filesystem(state, clip_store).is_empty


def test_an_unpublished_snapshot_staging_blocks_the_gate(tmp_path: Path) -> None:
    """A crash between stage() and publish() leaves evidence in limbo.

    The snapshot was written to `.snapshot-staging` but no attachment was queued
    and no disposition was tagged, so it is neither delivered nor discarded.
    Nothing in production reconciles those records, so the cutover gate must at
    minimum refuse to migrate over them rather than silently leaving them behind
    on a forward-only schema change.
    """
    state = tmp_path / "runtime-state"
    (state / "delivery-queue").mkdir(parents=True)
    clip_store = tmp_path / "clip-store"
    (clip_store / "clips" / ".staging").mkdir(parents=True)
    staged = clip_store / ".snapshot-staging" / "camera-1"
    staged.mkdir(parents=True)
    (staged / "abc.json").write_bytes(b"{}")

    result = inventory.inspect_filesystem(state, clip_store)

    assert result.pending_snapshot_stagings == 1
    assert not result.is_empty
    assert "unpublished snapshots=1" in result.describe_pending()


def test_a_fresh_deployment_is_not_blocked_by_the_gate(tmp_path: Path) -> None:
    """A clean install must be able to start.

    This gate runs before the migrator, so on a fresh deployment the database
    does not exist yet. Failing closed there made a clean install impossible to
    bring up at all: the inventory service exited 1 with
    EDGE_FS_INVENTORY_FAILED and the migrator never ran. There is no legacy
    schema-16 evidence to drain when there is no database, so this is the
    over-broad direction of the gate, and it is as damaging as letting a dirty
    deployment through.

    Found by actually tearing down and rebuilding the live stack, not by
    reading.
    """
    state = tmp_path / "runtime-state"
    (state / "delivery-queue").mkdir(parents=True)
    clip_store = tmp_path / "clip-store"
    (clip_store / "clips" / ".staging").mkdir(parents=True)

    cleared, detail = inventory.check_filesystem_drain(
        tmp_path / "absent.sqlite3", state, clip_store
    )

    assert cleared, f"a fresh deployment was blocked from starting: {detail}"
    assert "FRESH" in detail
