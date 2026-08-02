"""Persistent storage for the external-backend connection settings.

Historically the ml-api -> backend link (Event API URL, ml-config pull URL,
facility id, facility bearer token) was configured exclusively through env
vars read in ``backend/app/lifespan.py`` (``API_BACKEND_EVENTS_URL``,
``EDGE_FACILITY_TOKEN``, ``API_FACILITY_ID``, ``API_BACKEND_CONFIG_URL``).
This module adds a persistent sqlite3-backed store so a technician can
configure the same link from the dashboard UI without env vars or a restart.
Per the "no JSON state stores" rule (AGENTS.md), mutable runtime state that
is read-modify-written belongs in a table, not a hand-rolled atomic-write
JSON file -- this store keeps its own dedicated sqlite3 database rather than
sharing ``catalog.py``'s (single-DB consolidation is issue #35's job).

Precedence (``load()``): **the saved row wins over env, field by field**.
Env values are only used to *seed* a field that has never been saved -- once
a field is saved via the store (e.g. through the future settings UI), the
corresponding env var is ignored for that field even if it is still set in
the process environment. This lets an operator override a bad/legacy env
value from the UI without needing to touch the deployment's env file.

Below the two specific env vars sits one more, lower-priority seed:
``API_BACKEND_BASE_URL``. Packaging already knows the backend host at
build/deploy time, so a single base URL can be baked into the image and
this module derives ``events_url``/``config_url`` from it
(``{base}/v1/events`` and ``{base}/v1/ml-config``) whenever the specific
vars are unset. The technician-facing UI only ever needs to supply the
values packaging *cannot* know ahead of time: facility id and token.

This module is store-only (story G001 of the edge-onboarding plan): it does
not wire into ``lifespan.py`` and exposes no HTTP routes. The connection
timeout (``API_BACKEND_INGEST_TIMEOUT_SEC``) stays env-only for now.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

# Reuses lifespan.py's env-name constants rather than restringing them (same
# names the future settings UI/router will surface). NOTE for G002 (wiring
# this store into lifespan.py): lifespan.py must import this module lazily
# inside a function, not at module load, to avoid a circular import -- the
# same trick lifespan.py already applies for cameras.router, which imports
# these same constants at module level (see refresh_backend_config's lazy
# `from backend.app.features.cameras.router import ...`).
from backend.app.lifespan import (
    API_BACKEND_CONFIG_URL_ENV,
    API_BACKEND_EVENTS_URL_ENV,
    API_FACILITY_ID_ENV,
    EDGE_FACILITY_TOKEN_ENV,
)

API_CONNECTION_SETTINGS_PATH_ENV = "API_CONNECTION_SETTINGS_PATH"
DEFAULT_CONNECTION_SETTINGS_PATH = "/var/lib/ml-api/connection-settings.sqlite3"

logger = logging.getLogger(__name__)

_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS connection_settings ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), "
    "events_url TEXT, config_url TEXT, facility_id TEXT, "
    "facility_token TEXT, updated_at TEXT) STRICT"
)

# Packaging-time base URL: the company already knows the backend host when it
# bakes the deploy image, so this seeds `events_url`/`config_url` from
# `{base}/v1/events` and `{base}/v1/ml-config` whenever the more specific
# API_BACKEND_EVENTS_URL/API_BACKEND_CONFIG_URL vars are unset. Lowest
# priority in load()'s fallback chain -- see load() below.
API_BACKEND_BASE_URL_ENV = "API_BACKEND_BASE_URL"

# Saved-file field names -- also the accepted keys of the `updates` dict
# passed to `ConnectionSettingsStore.save()`.
_FIELDS = ("events_url", "config_url", "facility_id", "facility_token")


@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    """Effective connection settings after applying file-over-env precedence."""

    events_url: str | None
    config_url: str | None
    facility_id: str | None
    # repr=False -- a stray log/exception of the dataclass itself must never
    # print the raw bearer token; masked()/mask_facility_token() are the only
    # sanctioned way to surface it externally.
    facility_token: str | None = field(repr=False)
    updated_at: str | None


class ConnectionSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    @classmethod
    def from_env(cls) -> ConnectionSettingsStore:
        return cls(
            os.environ.get(API_CONNECTION_SETTINGS_PATH_ENV, DEFAULT_CONNECTION_SETTINGS_PATH)
        )

    def load(self) -> ConnectionSettings:
        """Return the effective settings: saved row wins, env seeds gaps.

        Precedence per field (highest to lowest): saved row > the
        field-specific env var (API_BACKEND_EVENTS_URL/API_BACKEND_CONFIG_URL)
        > API_BACKEND_BASE_URL-derived value > None. The base var lets a
        packaged image bake a single backend host without also setting the
        two specific vars; config_url is derived as exactly `{base}/v1/ml-config`
        (no trailing facility id) since the config-pull code already appends
        `/{facility_id}` itself.
        """
        with self._lock:
            saved = self._read_unlocked()
        base = os.environ.get(API_BACKEND_BASE_URL_ENV)
        base = base.rstrip("/") if base else None
        return ConnectionSettings(
            events_url=(
                saved.get("events_url")
                or os.environ.get(API_BACKEND_EVENTS_URL_ENV)
                or (f"{base}/v1/events" if base else None)
            ),
            config_url=(
                saved.get("config_url")
                or os.environ.get(API_BACKEND_CONFIG_URL_ENV)
                or (f"{base}/v1/ml-config" if base else None)
            ),
            facility_id=saved.get("facility_id") or os.environ.get(API_FACILITY_ID_ENV),
            facility_token=saved.get("facility_token") or os.environ.get(EDGE_FACILITY_TOKEN_ENV),
            updated_at=saved.get("updated_at"),
        )

    def save(self, updates: dict[str, str | None]) -> ConnectionSettings:
        """Partially update the saved row: only keys present in `updates` change.

        Passing a key with value ``None`` explicitly clears that saved field
        (falling back to env, if any, on the next `load()`); omitting a key
        leaves its stored value untouched.
        """
        unknown = set(updates) - set(_FIELDS)
        if unknown:
            raise ValueError(f"unknown connection setting field(s): {sorted(unknown)}")
        with self._lock:
            data = self._read_unlocked()
            for field_name in _FIELDS:
                if field_name in updates:
                    data[field_name] = updates[field_name]
            data["updated_at"] = utc_now_iso()
            self._write_unlocked(data)
        return self.load()

    def masked(self) -> dict[str, object]:
        """Return an API/log-safe view: `facility_token` is never present unmasked."""
        settings = self.load()
        return {
            "events_url": settings.events_url,
            "config_url": settings.config_url,
            "facility_id": settings.facility_id,
            "facility_token_masked": mask_facility_token(settings.facility_token),
            "facility_token_set": bool(settings.facility_token),
            "updated_at": settings.updated_at,
        }

    def _connect_unlocked(self, *, create: bool) -> sqlite3.Connection:
        # facility_token is a bearer credential stored in a local edge sqlite3
        # database by design; API responses and logs must use
        # masked()/mask_facility_token(). The db file is chmod'd 0600 right
        # after connecting (which creates it if missing) so the plaintext
        # token is never world-readable even transiently.
        #
        # `create` gates parent-directory creation: only the write path may
        # create it (mirrors the old JSON store, where reading a store that
        # was never saved touched no filesystem state beyond the read
        # itself). A read against a missing/unwritable directory must fail
        # into the same graceful "no saved settings" fallback as a corrupt
        # file -- it must never crash boot just because the configured
        # default path (e.g. `/var/lib/ml-api`) doesn't exist yet or isn't
        # writable in this environment (notably: local test runs).
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _read_unlocked(self) -> dict[str, str | None]:
        # Always returns all of `(*_FIELDS, "updated_at")` as keys (None for
        # unset fields), never a partial dict -- `save()` reuses this dict
        # for named sqlite parameter binding, so every placeholder must
        # resolve even when a field was never saved.
        keys = (*_FIELDS, "updated_at")
        try:
            conn = self._connect_unlocked(create=False)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("connection settings store unreadable at %s: %r", self.path, exc)
            return dict.fromkeys(keys)
        try:
            conn.execute(_SCHEMA_SQL)
            row = conn.execute(
                "SELECT events_url, config_url, facility_id, facility_token, updated_at "
                "FROM connection_settings WHERE id = 1"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            # Covers a corrupt db and a leftover non-sqlite file (e.g. an old
            # JSON store) at the configured path -- never crash boot.
            logger.warning("connection settings store unreadable at %s: %r", self.path, exc)
            return dict.fromkeys(keys)
        finally:
            conn.close()
        if row is None:
            return dict.fromkeys(keys)
        return {
            key: (value if isinstance(value, str) and value else None)
            for key, value in zip(keys, row, strict=True)
        }

    def _write_unlocked(self, data: dict[str, str | None]) -> None:
        conn = self._connect_unlocked(create=True)
        try:
            conn.execute(_SCHEMA_SQL)
            with conn:
                conn.execute(
                    "INSERT INTO connection_settings "
                    "(id, events_url, config_url, facility_id, facility_token, updated_at) "
                    "VALUES (1, :events_url, :config_url, :facility_id, :facility_token, "
                    ":updated_at) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "events_url = excluded.events_url, "
                    "config_url = excluded.config_url, "
                    "facility_id = excluded.facility_id, "
                    "facility_token = excluded.facility_token, "
                    "updated_at = excluded.updated_at",
                    data,
                )
        finally:
            conn.close()


def mask_facility_token(token: str | None) -> str | None:
    """Mask a facility bearer token, showing at most its last 4 characters.

    Uses a fixed-length "****" prefix (not one sized to the token) so the
    masked value never leaks the real token's length.
    """
    if not token:
        return None
    if len(token) <= 4:
        return "****"
    return f"****{token[-4:]}"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "API_BACKEND_BASE_URL_ENV",
    "API_CONNECTION_SETTINGS_PATH_ENV",
    "DEFAULT_CONNECTION_SETTINGS_PATH",
    "ConnectionSettings",
    "ConnectionSettingsStore",
    "mask_facility_token",
    "utc_now_iso",
]
