from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

import pytest

from backend.app.edge_db.compatibility import MigrationRequiredError
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    ConnectionSettingsStore,
    mask_facility_token,
)
from tests_support.compact_authority_db import prepare_compact_database


def _enrollment() -> dict[str, str | int]:
    return {
        "facility_code": "NH-7H2K9M4QXP",
        "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
        "facility_id": "facility-1",
        "facility_token": "super-secret-facility-token",
        "edge_installation_id": "edge-1",
        "enrollment_generation": 1,
    }


def _store(tmp_path: Path) -> ConnectionSettingsStore:
    path = prepare_compact_database(tmp_path / "connection_settings.sqlite3")
    return ConnectionSettingsStore(path)


class TestReprSafety:
    def test_repr_does_not_contain_the_raw_facility_token(self, tmp_path: Path) -> None:
        settings = _store(tmp_path).save(_enrollment())

        assert "super-secret-facility-token" not in repr(settings)
        assert "facility_token" not in repr(settings)


class TestAtomicWriteAndCorruption:
    def test_write_leaves_db_file_0600(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save(_enrollment())

        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600

    def test_write_leaves_no_leftover_tmp_file(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save(_enrollment())

        assert list(tmp_path.glob("*.tmp")) == []

    def test_row_persists_via_real_sqlite_connection(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save(_enrollment())

        with sqlite3.connect(store.path) as connection:
            row = connection.execute(
                "SELECT facility_id,facility_token FROM edge_site WHERE id=1"
            ).fetchone()

        assert row == ("facility-1", "super-secret-facility-token")

    def test_corrupt_db_file_fails_closed_as_unconfigured(self, tmp_path: Path) -> None:
        path = tmp_path / "connection_settings.sqlite3"
        path.write_bytes(b"not a valid sqlite database, just garbage bytes")

        settings = ConnectionSettingsStore(path).load()

        assert settings.facility_id is None
        assert settings.facility_token is None
        assert settings.edge_installation_id is None
        assert settings.enrollment_generation is None

    def test_missing_file_requires_external_migration(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "connection_settings.sqlite3"

        with pytest.raises(MigrationRequiredError):
            ConnectionSettingsStore(path).load()

        assert not path.exists()


class TestMasking:
    def test_mask_facility_token_short_token_fully_masked(self) -> None:
        assert mask_facility_token("abcd") == "****"
        assert mask_facility_token("ab") == "****"

    def test_mask_facility_token_long_token_shows_last_four(self) -> None:
        assert mask_facility_token("supersecrettoken1234") == "****1234"

    def test_mask_facility_token_none_or_empty(self) -> None:
        assert mask_facility_token(None) is None
        assert mask_facility_token("") is None

    def test_masked_dict_never_contains_raw_token(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save(_enrollment())

        public = store.masked()

        assert public["facility_token_masked"] == "****oken"
        assert public["facility_token_set"] is True
        assert "super-secret-facility-token" not in json.dumps(public)
        assert "facility_token" not in public

    def test_masked_dict_when_token_unset(self, tmp_path: Path) -> None:
        public = _store(tmp_path).masked()

        assert public["facility_token_masked"] is None
        assert public["facility_token_set"] is False

    def test_masked_dict_reflects_derived_urls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://backend.example/api")
        store = _store(tmp_path)
        store.save(_enrollment() | {"facility_id": "facility-42"})

        public = store.masked()

        assert public["events_url"] == "https://backend.example/api/v1/events"
        assert public["config_url"] == "https://backend.example/api/v1/ml-config"
        assert public["facility_id"] == "facility-42"
        assert public["updated_at"] is not None
