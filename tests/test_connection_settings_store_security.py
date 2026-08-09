from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path
from typing import cast

from backend.app.features.connection.store import (
    ConnectionSettings,
    ConnectionSettingsStore,
    mask_facility_token,
)


def _store(tmp_path: Path) -> ConnectionSettingsStore:
    return ConnectionSettingsStore(tmp_path / "connection_settings.sqlite3")


class TestReprSafety:
    def test_repr_does_not_contain_the_raw_facility_token(self, tmp_path: Path) -> None:
        settings = _store(tmp_path).save({"facility_token": "super-secret-facility-token"})

        assert "super-secret-facility-token" not in repr(settings)
        assert "facility_token" not in repr(settings)


class TestAtomicWriteAndCorruption:
    def test_write_leaves_db_file_0600(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _ = store.save({"facility_id": "facility-1"})

        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600

    def test_write_leaves_no_leftover_tmp_file(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _ = store.save({"facility_id": "facility-1"})

        assert list(tmp_path.glob("*.tmp")) == []

    def test_row_persists_via_real_sqlite_connection(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _ = store.save({"facility_id": "facility-1", "facility_token": "token-abcd"})

        with sqlite3.connect(store.path) as connection:
            row = cast(
                tuple[str, str] | None,
                connection.execute(
                    "SELECT facility_id, facility_token FROM connection_settings WHERE id = 1"
                ).fetchone(),
            )

        assert row == ("facility-1", "token-abcd")

    def test_corrupt_db_file_treated_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "connection_settings.sqlite3"
        _ = path.write_bytes(b"not a valid sqlite database, just garbage bytes")

        settings = ConnectionSettingsStore(path).load()

        assert settings == ConnectionSettings(
            events_url=None,
            config_url=None,
            facility_id=None,
            facility_token=None,
            updated_at=None,
        )

    def test_missing_file_does_not_crash_and_save_still_works(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "connection_settings.sqlite3"
        store = ConnectionSettingsStore(path)

        assert store.load().facility_id is None
        _ = store.save({"facility_id": "facility-1"})

        assert store.load().facility_id == "facility-1"


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
        _ = store.save({"facility_token": "supersecrettoken1234"})

        public = store.masked()

        assert public["facility_token_masked"] == "****1234"
        assert public["facility_token_set"] is True
        assert "supersecrettoken1234" not in json.dumps(public)
        assert "facility_token" not in public

    def test_masked_dict_when_token_unset(self, tmp_path: Path) -> None:
        public = _store(tmp_path).masked()

        assert public["facility_token_masked"] is None
        assert public["facility_token_set"] is False

    def test_masked_dict_reflects_other_fields(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _ = store.save(
            {
                "events_url": "https://backend.example/events",
                "config_url": "https://backend.example/ml-config",
                "facility_id": "facility-42",
            }
        )

        public = store.masked()

        assert public["events_url"] == "https://backend.example/events"
        assert public["config_url"] == "https://backend.example/ml-config"
        assert public["facility_id"] == "facility-42"
        assert public["updated_at"] is not None
