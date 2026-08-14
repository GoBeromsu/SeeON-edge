"""SQLite-backed runtime settings shared by API projection and relay policy."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from fastapi import FastAPI

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from backend.app.shared.state_dir import resolve_state_dir

_CREATE_RUNTIME_SETTINGS_TABLE = (
    "CREATE TABLE IF NOT EXISTS runtime_settings ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), "
    "clip_export_enabled INTEGER NOT NULL CHECK (clip_export_enabled IN (0, 1)), "
    "version INTEGER NOT NULL CHECK (version >= 0)) STRICT"
)


@dataclass(frozen=True, slots=True)
class RuntimeSetting:
    clip_export_enabled: bool = False
    version: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "clip_export_enabled": self.clip_export_enabled,
            "version": self.version,
        }


class RuntimeSettingsVersionConflict(RuntimeError):
    def __init__(self, current: RuntimeSetting) -> None:
        super().__init__("runtime settings version conflict")
        self.current = current


def _check_expected_version(current: RuntimeSetting, expected: int | None) -> None:
    if expected is not None and expected != current.version:
        raise RuntimeSettingsVersionConflict(current)


class RuntimeSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def from_env(cls) -> RuntimeSettingsStore:
        return cls(resolve_state_dir("ml-api") / "catalog.sqlite3")

    def get(self) -> RuntimeSetting:
        with self._lock:
            row = (
                self._connect()
                .execute("SELECT clip_export_enabled, version FROM runtime_settings WHERE id = 1")
                .fetchone()
            )
        if row is None:
            return RuntimeSetting()
        return RuntimeSetting(clip_export_enabled=bool(row[0]), version=int(row[1]))

    def set_clip_export_enabled(
        self,
        enabled: bool,
        *,
        expected_version: int | None = None,
    ) -> RuntimeSetting:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT clip_export_enabled, version FROM runtime_settings WHERE id = 1"
                ).fetchone()
                current = (
                    RuntimeSetting() if row is None else RuntimeSetting(bool(row[0]), int(row[1]))
                )
                _check_expected_version(current, expected_version)
                if row is None and not enabled:
                    setting = current
                elif row is None:
                    setting = RuntimeSetting(clip_export_enabled=True, version=1)
                    connection.execute(
                        "INSERT INTO runtime_settings "
                        "(id, clip_export_enabled, version) VALUES (1, ?, ?)",
                        (1, setting.version),
                    )
                elif bool(row[0]) == enabled:
                    setting = RuntimeSetting(clip_export_enabled=enabled, version=int(row[1]))
                else:
                    setting = RuntimeSetting(
                        clip_export_enabled=enabled,
                        version=int(row[1]) + 1,
                    )
                    connection.execute(
                        "UPDATE runtime_settings SET clip_export_enabled = ?, version = ? "
                        "WHERE id = 1",
                        (int(enabled), setting.version),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return setting

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = connect_catalog_store(self.path, (_CREATE_RUNTIME_SETTINGS_TABLE,))
        return self._connection


def get_runtime_settings_store(app: FastAPI) -> RuntimeSettingsStore:
    state = app.state
    store = getattr(state, "runtime_settings_store", None)
    if not isinstance(store, RuntimeSettingsStore):
        store = RuntimeSettingsStore.from_env()
        state.runtime_settings_store = store
    return store


__all__ = [
    "RuntimeSetting",
    "RuntimeSettingsStore",
    "RuntimeSettingsVersionConflict",
    "get_runtime_settings_store",
]
