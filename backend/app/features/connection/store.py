"""Persistent storage for the external-backend connection settings.

Historically the ml-api -> backend link (Event API URL, ml-config pull URL,
facility id, facility bearer token) was configured exclusively through env
vars read in ``backend/app/lifespan.py`` (``API_BACKEND_EVENTS_URL``,
``EDGE_FACILITY_TOKEN``, ``API_FACILITY_ID``, ``API_BACKEND_CONFIG_URL``).
This module adds a persistent JSON-backed store so a technician can configure
the same link from the dashboard UI without env vars or a restart.

Precedence (``load()``): **the saved file wins over env, field by field**.
Env values are only used to *seed* a field that has never been saved -- once
a field is saved via the store (e.g. through the future settings UI), the
corresponding env var is ignored for that field even if it is still set in
the process environment. This lets an operator override a bad/legacy env
value from the UI without needing to touch the deployment's env file.

This module is store-only (story G001 of the edge-onboarding plan): it does
not wire into ``lifespan.py`` and exposes no HTTP routes. The connection
timeout (``API_BACKEND_INGEST_TIMEOUT_SEC``) stays env-only for now.
"""

from __future__ import annotations

import json
import os
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
DEFAULT_CONNECTION_SETTINGS_PATH = "/var/lib/ml-api/connection_settings.json"

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
        """Return the effective settings: saved file wins, env seeds gaps."""
        with self._lock:
            saved = self._read_unlocked()
        return ConnectionSettings(
            events_url=saved.get("events_url") or os.environ.get(API_BACKEND_EVENTS_URL_ENV),
            config_url=saved.get("config_url") or os.environ.get(API_BACKEND_CONFIG_URL_ENV),
            facility_id=saved.get("facility_id") or os.environ.get(API_FACILITY_ID_ENV),
            facility_token=saved.get("facility_token") or os.environ.get(EDGE_FACILITY_TOKEN_ENV),
            updated_at=saved.get("updated_at"),
        )

    def save(self, updates: dict[str, str | None]) -> ConnectionSettings:
        """Partially update the saved file: only keys present in `updates` change.

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

    def _read_unlocked(self) -> dict[str, str | None]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        data: dict[str, str | None] = {}
        for field_name in (*_FIELDS, "updated_at"):
            value = raw.get(field_name)
            data[field_name] = value if isinstance(value, str) and value else None
        return data

    def _write_unlocked(self, data: dict[str, str | None]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        serialized = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        # facility_token is a bearer credential stored in local edge JSON by
        # design; API responses and logs must use masked()/mask_facility_token().
        # The tmp file is created with mode 0600 from the very first byte
        # (O_EXCL so we never open/reuse an existing, differently-permissioned
        # file) rather than write-then-chmod, so the plaintext token is never
        # world-readable even transiently.
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(serialized)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        os.replace(tmp_path, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


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
    "API_CONNECTION_SETTINGS_PATH_ENV",
    "DEFAULT_CONNECTION_SETTINGS_PATH",
    "ConnectionSettings",
    "ConnectionSettingsStore",
    "mask_facility_token",
    "utc_now_iso",
]
