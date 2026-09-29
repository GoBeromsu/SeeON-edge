"""Native credential persistence; app ownership and HTTP activation live elsewhere.

The supplied database owns every complete transaction and pool lease. This
store never bootstraps storage, selects authority or constructs a fallback.
"""

from __future__ import annotations

from collections.abc import Callable

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase, PostgresError
from backend.app.shared.dashboard_credentials import (
    DashboardCredentialsStoreError,
    PersistedDashboardCredentials,
)

_UNREADABLE = "dashboard credentials store unreadable"


class PostgresDashboardCredentialsStore:
    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("native credentials require a PostgreSQL owner") from None
        if not isinstance(authority, AuthorityToken):
            raise TypeError("native credentials require deployment authority") from None
        self._database, self._authority = database, authority

    @property
    def database(self) -> PostgresDatabase:
        return self._database

    @property
    def authority(self) -> AuthorityToken:
        return self._authority

    def load(self) -> PersistedDashboardCredentials | None:
        def read(connection: psycopg.Connection) -> PersistedDashboardCredentials | None:
            rows = connection.execute(
                "SELECT id,username,algorithm,salt,password_hash,updated_at,"
                "seeon_utc_timestamp(updated_at) FROM credentials ORDER BY id LIMIT 2"
            ).fetchall()
            if not rows:
                return None
            if len(rows) != 1 or type(rows[0][0]) is not int or rows[0][0] != 1:
                raise DashboardCredentialsStoreError(_UNREADABLE) from None
            _, username, algorithm, salt, password_hash, updated_at, timestamp_valid = rows[0]
            # Match the native row format, including its own UTC validator;
            # corrupt present rows must never become bootstrap-eligible absence.
            if (
                not isinstance(username, str)
                or not 1 <= len(username) <= 128
                or algorithm != "scrypt"
                or not isinstance(salt, (bytes, bytearray, memoryview))
                or memoryview(salt).nbytes != 16
                or not isinstance(password_hash, (bytes, bytearray, memoryview))
                or memoryview(password_hash).nbytes != 64
                or not isinstance(updated_at, str)
                or timestamp_valid is not True
            ):
                raise DashboardCredentialsStoreError(_UNREADABLE) from None
            return PersistedDashboardCredentials(
                username, algorithm, bytes(salt), bytes(password_hash), updated_at
            )

        try:
            return self._database.read(read)
        except (PostgresError, psycopg.Error, OSError, TypeError, ValueError):
            raise DashboardCredentialsStoreError(_UNREADABLE) from None

    def save(
        self,
        *,
        username: str,
        password: str,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> PersistedDashboardCredentials:
        # Scrypt runs before acquisition, never while holding the authority lock.
        record = PersistedDashboardCredentials.from_password(username=username, password=password)

        def write(connection: psycopg.Connection) -> PersistedDashboardCredentials:
            require_authority(connection, self._authority)
            connection.execute(
                "INSERT INTO credentials(id,username,algorithm,salt,password_hash,updated_at) "
                "VALUES(1,%s,%s,%s,%s,%s) ON CONFLICT(id) DO UPDATE SET "
                "username=excluded.username,algorithm=excluded.algorithm,salt=excluded.salt,"
                "password_hash=excluded.password_hash,updated_at=excluded.updated_at",
                (
                    record.username,
                    record.algorithm,
                    record.salt,
                    record.password_hash,
                    record.updated_at,
                ),
            )
            if after_write is not None:
                after_write(connection)
            return record

        # Do not translate or retry write failures, particularly unknown COMMIT.
        return self._database.transact(write)


__all__ = ["PostgresDashboardCredentialsStore"]
