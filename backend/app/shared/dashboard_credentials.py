"""Persisted dashboard login credentials (scrypt-hashed, SQLite-backed).

Separate from the in-memory ``DashboardSessionStore`` in ``dashboard_auth.py``:
this module only knows how to durably store and verify one username/password
pair. ``dashboard_auth.py`` decides *when* to consult it (persisted row wins
over env vars, which win over the built-in ``admin``/``admin`` default).

Storage lives in the single-row ``credentials`` table of the shared
``catalog.sqlite3`` database (see ``backend/app/features/clips/catalog.py``),
alongside the ``camera_registry`` and ``runtime_latency`` tables. This module
cannot import ``CatalogStore`` to bootstrap that table -- the import-linter
contract "backend base (core/shared) does not import upper layers"
(``pyproject.toml``) forbids ``backend.app.shared`` from importing
``backend.app.features`` -- so it opens and migrates its own single table via
an idempotent ``CREATE TABLE IF NOT EXISTS`` (identical statement text to
``catalog.py``'s ``_V3_TABLE_STATEMENTS``), independent of whichever store
opens the database file first, and never touches ``PRAGMA user_version``
(that stays exclusively ``CatalogStore``'s responsibility).
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

from backend.app.shared.state_dir import resolve_state_dir

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


class DashboardCredentialsStore:
    """SQLite-backed persistence for dashboard login credentials.

    Reads and writes the single-row ``credentials`` table (``id = 1``) in the
    catalog database. A missing table row, an unreadable/corrupt database, or
    any other read-time failure is logged to stderr and treated as "no
    persisted credentials" -- it must never crash boot or brick login (the
    operator can always fall back to env vars or the built-in default).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def from_env(cls) -> DashboardCredentialsStore:
        return cls(resolve_state_dir("ml-api") / "catalog.sqlite3")

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
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path, timeout=5.0, isolation_level=None, check_same_thread=False
            )
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(_CREATE_CREDENTIALS_TABLE)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            self._connection = connection
        return self._connection

    def _load_unlocked(self) -> PersistedDashboardCredentials | None:
        try:
            connection = self._connect()
            row = connection.execute(
                "SELECT username, algorithm, salt, password_hash, updated_at "
                "FROM credentials WHERE id = 1"
            ).fetchone()
        except (OSError, sqlite3.Error) as exc:  # noqa: BLE001 - never brick login on a bad db
            print(
                f"dashboard credentials store unreadable at {self.path}, "
                f"falling back to env/default: {exc!r}",
                file=sys.stderr,
            )
            return None
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
    "PersistedDashboardCredentials",
]
