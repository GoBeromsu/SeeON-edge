"""Persisted dashboard login credentials (scrypt-hashed, SQLite-backed).

Separate from the in-memory ``DashboardSessionStore`` in ``dashboard_auth.py``:
this module only knows how to durably store and verify one username/password
pair. ``dashboard_auth.py`` decides *when* to consult it (persisted row wins
over a fully-set env bootstrap pair). There is no built-in password default.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.shared.sqlite_bootstrap import connect_catalog_store

_CREATE_CREDENTIALS_TABLE = (
    "CREATE TABLE IF NOT EXISTS credentials (id INTEGER PRIMARY KEY CHECK (id = 1), "
    "username TEXT NOT NULL, algorithm TEXT NOT NULL, salt BLOB NOT NULL, "
    "password_hash BLOB NOT NULL, updated_at TEXT NOT NULL) STRICT"
)

_ALGORITHM_SCRYPT = "scrypt"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64
_SALT_BYTES = 16


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )


@dataclass(frozen=True, slots=True)
class PersistedDashboardCredentials:
    username: str
    algorithm: str
    salt: bytes
    password_hash: bytes
    updated_at: str

    def verify_password(self, password: str) -> bool:
        if self.algorithm != _ALGORITHM_SCRYPT:
            return False
        candidate = _hash_password(password, self.salt)
        return hmac.compare_digest(candidate, self.password_hash)


class DashboardCredentialsStoreError(RuntimeError):
    """Persisted credential state exists but cannot be read safely.

    Callers must fail closed: never fall back to env or any default pair after
    a rotation-capable store has become unreadable or corrupt.
    """


class DashboardCredentialsStore:
    """SQLite-backed persistence for dashboard login credentials.

    Reads and writes the single-row ``credentials`` table (``id = 1``) in the
    catalog database. A missing table row is \"no persisted credentials\" (env
    bootstrap may still apply). An unreadable or corrupt database raises
    ``DashboardCredentialsStoreError`` so auth fails closed instead of
    silently falling back to env/default after rotation.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def from_env(cls) -> DashboardCredentialsStore:
        return cls(EDGE_DATABASE_PATH)

    def load(self) -> PersistedDashboardCredentials | None:
        with self._lock:
            return self._load_unlocked()

    def save(self, *, username: str, password: str) -> PersistedDashboardCredentials:
        salt = os.urandom(_SALT_BYTES)
        record = PersistedDashboardCredentials(
            username=username,
            algorithm=_ALGORITHM_SCRYPT,
            salt=salt,
            password_hash=_hash_password(password, salt),
            updated_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        )
        with self._lock:
            self._write_unlocked(record)
        return record

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = connect_catalog_store(self.path, (_CREATE_CREDENTIALS_TABLE,))
        return self._connection

    def _load_unlocked(self) -> PersistedDashboardCredentials | None:
        try:
            connection = self._connect()
            row = connection.execute(
                "SELECT username, algorithm, salt, password_hash, updated_at "
                "FROM credentials WHERE id = 1"
            ).fetchone()
        except (OSError, sqlite3.Error) as exc:
            message = (
                f"dashboard credentials store unreadable at {self.path}: {exc!r}"
            )
            print(message, file=sys.stderr)
            raise DashboardCredentialsStoreError(message) from exc
        if row is None:
            return None
        username, algorithm, salt, password_hash, updated_at = row
        return PersistedDashboardCredentials(
            username=username,
            algorithm=algorithm,
            salt=bytes(salt),
            password_hash=bytes(password_hash),
            updated_at=updated_at,
        )

    def _write_unlocked(self, record: PersistedDashboardCredentials) -> None:
        connection = self._connect()
        connection.execute(
            "INSERT INTO credentials (id, username, algorithm, salt, password_hash, updated_at) "
            "VALUES (1, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "username = excluded.username, algorithm = excluded.algorithm, "
            "salt = excluded.salt, password_hash = excluded.password_hash, "
            "updated_at = excluded.updated_at",
            (
                record.username,
                record.algorithm,
                record.salt,
                record.password_hash,
                record.updated_at,
            ),
        )


__all__ = [
    "DashboardCredentialsStore",
    "DashboardCredentialsStoreError",
    "PersistedDashboardCredentials",
]
