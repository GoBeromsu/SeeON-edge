"""Local detection overrides on the API-owned PostgreSQL pool."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase, PostgresError

DOMAINS: tuple[str, ...] = ("fall", "bed_exit")


class DetectionSettingsNotInitialized(PostgresError):
    def __init__(self) -> None:
        super().__init__("detection settings bootstrap row is missing")


@dataclass(frozen=True, slots=True)
class DomainDetectionSetting:
    on: bool
    mode: str
    start: str | None
    end: str | None

    def as_dict(self) -> dict[str, object]:
        return {"on": self.on, "mode": self.mode, "start": self.start, "end": self.end}


class DetectionSettingsStore:
    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        self.database = database
        self.authority = authority

    def get_all(self) -> dict[str, DomainDetectionSetting]:
        def read(connection: psycopg.Connection) -> dict[str, DomainDetectionSetting]:
            row = connection.execute(
                "SELECT fall_on,fall_mode,fall_start_time,fall_end_time,"
                "bed_exit_on,bed_exit_mode,bed_exit_start_time,bed_exit_end_time "
                "FROM edge_site WHERE id=1"
            ).fetchone()
            if row is None:
                raise DetectionSettingsNotInitialized()
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

        return self.database.read(read)

    def replace_all(
        self,
        settings: dict[str, DomainDetectionSetting],
        *,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> None:
        def persist(connection: psycopg.Connection) -> None:
            require_authority(connection, self.authority)
            row = connection.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE").fetchone()
            if row is None:
                raise DetectionSettingsNotInitialized()
            _require_known_domains(settings)
            now = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            for domain, setting in settings.items():
                connection.execute(
                    f"UPDATE edge_site SET {domain}_on=%s,{domain}_mode=%s,"
                    f"{domain}_start_time=%s,{domain}_end_time=%s,updated_at=%s WHERE id=1",
                    (int(setting.on), setting.mode, setting.start, setting.end, now),
                )
            if after_write is not None:
                after_write(connection)

        self.database.transact(persist)


def _require_known_domains(settings: dict[str, DomainDetectionSetting]) -> None:
    unknown = set(settings) - set(DOMAINS)
    if unknown:
        raise KeyError(min(unknown))


__all__ = [
    "DOMAINS",
    "DetectionSettingsNotInitialized",
    "DetectionSettingsStore",
    "DomainDetectionSetting",
]
