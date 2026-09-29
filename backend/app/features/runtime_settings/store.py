"""Central/export settings on the API-owned PostgreSQL pool.

The export control does not disable local clip recording. Bootstrap owns the
site singleton; lifespan owns the pool and injects this store.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
from fastapi import FastAPI

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase, PostgresError


class RuntimeSettingsNotInitialized(PostgresError):
    def __init__(self) -> None:
        super().__init__("runtime settings bootstrap row is missing")


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
    """Borrow transactions only; pool lifecycle belongs to the API."""

    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        self.database = database
        self.authority = authority

    def get(self) -> RuntimeSetting:
        return self.database.read(_read_setting)

    def set_clip_export_enabled(
        self,
        enabled: bool,
        *,
        expected_version: int | None = None,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> RuntimeSetting:
        def persist(connection: psycopg.Connection) -> RuntimeSetting:
            require_authority(connection, self.authority)
            current = _read_setting(connection, for_update=True)
            _require_expected_version(current, expected_version)
            if current.clip_export_enabled == enabled:
                setting = current
            else:
                setting = RuntimeSetting(enabled, current.version + 1)
                now = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                connection.execute(
                    "UPDATE edge_site SET clip_export_enabled=%s,runtime_settings_version=%s,"
                    "updated_at=%s WHERE id=1",
                    (int(enabled), setting.version, now),
                )
            if after_write is not None:
                after_write(connection)
            return setting

        # Release only this transaction's candidate after known COMMIT and pool return.
        return self.database.transact(persist)


def _read_setting(connection: psycopg.Connection, *, for_update: bool = False) -> RuntimeSetting:
    row = connection.execute(
        "SELECT clip_export_enabled,runtime_settings_version FROM edge_site WHERE id=1"
        + (" FOR UPDATE" if for_update else "")
    ).fetchone()
    if row is None:
        raise RuntimeSettingsNotInitialized()
    return RuntimeSetting(bool(row[0]), int(row[1]))


def _require_expected_version(current: RuntimeSetting, expected: int | None) -> None:
    if expected is not None and expected != current.version:
        raise RuntimeSettingsVersionConflict(current)


def get_runtime_settings_store(app: FastAPI) -> RuntimeSettingsStore:
    store = getattr(app.state, "runtime_settings_store", None)
    if store is None:
        raise RuntimeError("runtime settings store is not injected")
    if not isinstance(store, RuntimeSettingsStore):
        raise TypeError("runtime settings store has invalid type")
    return store


__all__ = [
    "RuntimeSetting",
    "RuntimeSettingsNotInitialized",
    "RuntimeSettingsStore",
    "RuntimeSettingsVersionConflict",
    "get_runtime_settings_store",
]
