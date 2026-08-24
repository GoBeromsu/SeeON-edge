"""Schema-18 runtime settings authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from fastapi import FastAPI

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.configuration import (
    ensure_edge_site,
    open_configuration_database,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class RuntimeSetting:
    clip_export_enabled: bool = False
    version: int = 0

    def as_dict(self) -> dict[str, object]:
        return {"clip_export_enabled": self.clip_export_enabled, "version": self.version}


class RuntimeSettingsVersionConflict(RuntimeError):
    def __init__(self, current: RuntimeSetting) -> None:
        super().__init__("runtime settings version conflict")
        self.current = current


class RuntimeSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection = open_configuration_database(self.path)

    @classmethod
    def from_env(cls) -> RuntimeSettingsStore:
        return cls(EDGE_DATABASE_PATH)

    def get(self) -> RuntimeSetting:
        with self._lock:
            return self._get_unlocked()

    def set_clip_export_enabled(
        self, enabled: bool, *, expected_version: int | None = None
    ) -> RuntimeSetting:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._get_unlocked()
                _require_expected_version(current, expected_version)
                if current.clip_export_enabled == enabled:
                    setting = current
                else:
                    ensure_edge_site(self._connection)
                    setting = RuntimeSetting(enabled, current.version + 1)
                    self._connection.execute(
                        "UPDATE edge_site SET clip_export_enabled=?,runtime_settings_version=?,"
                        "updated_at=? WHERE id=1",
                        (int(enabled), setting.version, utc_now()),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return setting

    def _get_unlocked(self) -> RuntimeSetting:
        row = self._connection.execute(
            "SELECT clip_export_enabled,runtime_settings_version FROM edge_site WHERE id=1"
        ).fetchone()
        return RuntimeSetting() if row is None else RuntimeSetting(bool(row[0]), int(row[1]))


def _require_expected_version(current: RuntimeSetting, expected: int | None) -> None:
    if expected is not None and expected != current.version:
        raise RuntimeSettingsVersionConflict(current)


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
