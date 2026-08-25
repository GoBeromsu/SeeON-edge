"""Schema-18 local detection override authority."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.configuration import (
    ensure_edge_site,
    open_configuration_database,
    utc_now,
)

DOMAINS: tuple[str, ...] = ("fall", "bed_exit")


@dataclass(frozen=True, slots=True)
class DomainDetectionSetting:
    on: bool
    mode: str
    start: str | None
    end: str | None

    def as_dict(self) -> dict[str, object]:
        return {"on": self.on, "mode": self.mode, "start": self.start, "end": self.end}


class DetectionSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection = open_configuration_database(self.path)

    @classmethod
    def from_env(cls) -> DetectionSettingsStore:
        return cls(EDGE_DATABASE_PATH)

    def get_all(self) -> dict[str, DomainDetectionSetting]:
        with self._lock:
            row = self._connection.execute(
                "SELECT fall_on,fall_mode,fall_start_time,fall_end_time,"
                "bed_exit_on,bed_exit_mode,bed_exit_start_time,bed_exit_end_time "
                "FROM edge_site WHERE id=1"
            ).fetchone()
        if row is None:
            return {}
        result: dict[str, DomainDetectionSetting] = {}
        for domain, offset in (("fall", 0), ("bed_exit", 4)):
            if row[offset] is not None:
                result[domain] = DomainDetectionSetting(
                    bool(row[offset]),
                    str(row[offset + 1]),
                    None if row[offset + 2] is None else str(row[offset + 2]),
                    None if row[offset + 3] is None else str(row[offset + 3]),
                )
        return result

    def replace_all(
        self,
        settings: dict[str, DomainDetectionSetting],
        *,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> None:
        _require_known_domains(settings)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                ensure_edge_site(self._connection)
                for domain, setting in settings.items():
                    self._connection.execute(
                        f"UPDATE edge_site SET {domain}_on=?,{domain}_mode=?,"
                        f"{domain}_start_time=?,{domain}_end_time=?,updated_at=? WHERE id=1",
                        (int(setting.on), setting.mode, setting.start, setting.end, utc_now()),
                    )
                if after_write is not None:
                    after_write(self._connection)
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise


def _require_known_domains(settings: dict[str, DomainDetectionSetting]) -> None:
    unknown = set(settings) - set(DOMAINS)
    if unknown:
        raise KeyError(min(unknown))


__all__ = ["DOMAINS", "DetectionSettingsStore", "DomainDetectionSetting"]
