from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from backend.app.features.connection.store import ConnectionSettingsStore
from tests_support.compact_authority_db import prepare_compact_database


def _enrollment(*, generation: int = 1, token: str = "token-1") -> dict[str, str | int]:
    return {
        "facility_code": "NH-7H2K9M4QXP",
        "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
        "facility_id": "facility-1",
        "facility_token": token,
        "edge_installation_id": "edge-1",
        "enrollment_generation": generation,
    }


def _store(tmp_path: Path) -> ConnectionSettingsStore:
    path = prepare_compact_database(tmp_path / "edge.sqlite3")
    return ConnectionSettingsStore(path)


def test_schema18_backup_is_integrity_checked_and_owner_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_enrollment())

    backup = store.create_pre_v1_backup()

    assert stat.S_IMODE(store.rollback_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup.path.stat().st_mode) == 0o600
    assert store.integrity_check(backup.path) == "ok"
    assert len(backup.sha256) == 64
    assert backup.size_bytes == backup.path.stat().st_size


def test_online_backup_includes_committed_schema18_wal_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_enrollment())
    with sqlite3.connect(store.path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("UPDATE edge_site SET facility_token='wal-token' WHERE id=1")

    backup = store.create_pre_v1_backup()

    with sqlite3.connect(backup.path) as connection:
        assert connection.execute("SELECT facility_token FROM edge_site WHERE id=1").fetchone() == (
            "wal-token",
        )


def test_restore_reverts_enrollment_but_preserves_sibling_edge_site_authority(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.save(_enrollment())
    backup = store.create_pre_v1_backup()
    store.save(_enrollment(generation=2, token="token-2"))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE edge_site SET clip_export_enabled=1,runtime_settings_version=4 WHERE id=1"
        )

    store.restore_pre_v1_backup(backup.path)

    restored = ConnectionSettingsStore(store.path).load()
    with sqlite3.connect(store.path) as connection:
        runtime = connection.execute(
            "SELECT clip_export_enabled,runtime_settings_version FROM edge_site WHERE id=1"
        ).fetchone()
    assert restored.enrollment_generation == 1
    assert restored.facility_token == "token-1"
    assert runtime == (1, 4)


def test_corrupt_backup_restore_leaves_enrollment_intact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_enrollment())
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not sqlite")

    with pytest.raises(sqlite3.DatabaseError):
        store.restore_pre_v1_backup(corrupt)

    assert ConnectionSettingsStore(store.path).load().facility_token == "token-1"


def test_interrupted_backup_removes_partial_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.save(_enrollment())

    def interrupt(_source: str | Path, _destination: str | Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            store.create_pre_v1_backup()

    assert list(store.rollback_directory.glob("*.tmp")) == []
    assert ConnectionSettingsStore(store.path).load().facility_token == "token-1"
