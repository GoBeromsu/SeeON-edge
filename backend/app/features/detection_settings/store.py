"""Persisted per-domain detection on/off + time-window overrides for ml-api.

Backs ``GET``/``PUT /api/v1/detection-settings`` (see ``router.py``): an
operator-set global toggle and optional detection window per domain (``fall``,
``bed_exit``), independent of whatever the backend externally pulls. Local
settings, once saved, take precedence over the externally pulled
``detection_windows``/domain-enabled state at ``worker_config_snapshot()``
time (see ``cameras/router.py``) -- this store never mutates
``app.state.pulled_config`` itself.

Storage lives in its own ``detection_settings`` table of the shared
``catalog.sqlite3`` database (see ``sqlite_bootstrap.connect_catalog_store``),
one row per domain, mirroring ``BedZoneStore``'s per-key-row shape (a
CameraRegistryStore-style single JSON blob would make a "change just one
domain" PUT into a read-modify-write of an opaque blob for no benefit here).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from shared.edge_db import EDGE_DATABASE_PATH

DOMAINS: tuple[str, ...] = ("fall", "bed_exit")

_CREATE_DETECTION_SETTINGS_TABLE = (
    "CREATE TABLE IF NOT EXISTS detection_settings ("
    "domain TEXT PRIMARY KEY, "
    "on_flag INTEGER NOT NULL, "
    "mode TEXT NOT NULL, "
    "start_time TEXT, "
    "end_time TEXT) STRICT"
)


@dataclass(frozen=True, slots=True)
class DomainDetectionSetting:
    on: bool
    mode: str  # "always" | "window"
    start: str | None
    end: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "on": self.on,
            "mode": self.mode,
            "start": self.start,
            "end": self.end,
        }


class DetectionSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def from_env(cls) -> DetectionSettingsStore:
        return cls(EDGE_DATABASE_PATH)

    def get_all(self) -> dict[str, DomainDetectionSetting]:
        with self._lock:
            connection = self._connect()
            rows = connection.execute(
                "SELECT domain, on_flag, mode, start_time, end_time FROM detection_settings"
            ).fetchall()
        result: dict[str, DomainDetectionSetting] = {}
        for domain, on_flag, mode, start_time, end_time in rows:
            result[str(domain)] = DomainDetectionSetting(
                on=bool(on_flag),
                mode=str(mode),
                start=None if start_time is None else str(start_time),
                end=None if end_time is None else str(end_time),
            )
        return result

    def replace_all(self, settings: dict[str, DomainDetectionSetting]) -> None:
        """Full replace: every domain in ``settings`` is upserted. Domains
        omitted from ``settings`` keep whatever row (or absence of one) they
        already had -- the router always builds a complete map for both
        known domains before calling this, so a genuine partial write never
        happens in practice, but this store itself makes no such assumption.
        """
        with self._lock:
            connection = self._connect()
            for domain, setting in settings.items():
                connection.execute(
                    "INSERT INTO detection_settings "
                    "(domain, on_flag, mode, start_time, end_time) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(domain) DO UPDATE SET "
                    "on_flag = excluded.on_flag, "
                    "mode = excluded.mode, "
                    "start_time = excluded.start_time, "
                    "end_time = excluded.end_time",
                    (domain, int(setting.on), setting.mode, setting.start, setting.end),
                )

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = connect_catalog_store(
                self.path, (_CREATE_DETECTION_SETTINGS_TABLE,)
            )
        return self._connection


__all__ = ["DOMAINS", "DetectionSettingsStore", "DomainDetectionSetting"]
