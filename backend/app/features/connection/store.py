"""Connection settings on the API-owned PostgreSQL pool.

Enrollment identity and credentials are DB-only. API_BACKEND_BASE_URL is
the sole deployment authority for Hub URLs, never a persisted site setting.
Bootstrap owns the edge_site singleton; serving neither creates nor restores it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, TypedDict

import psycopg
from fastapi import FastAPI

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.features.connection.hub_url import hub_url_transport_allowed
from backend.app.features.connection.repository import (
    MAX_ENROLLMENT_GENERATION,
    REQUIRED_ENROLLMENT_FIELDS,
    SAVE_FIELDS,
    TEXT_FIELD_LIMITS,
    ConnectionData,
    ConnectionValue,
    ConnectionWriteHook,
    read_settings,
    write_settings,
)

API_BACKEND_BASE_URL_ENV: Final = "API_BACKEND_BASE_URL"


@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    events_url: str | None
    config_url: str | None
    facility_id: str | None
    facility_token: str | None = field(repr=False)
    updated_at: str | None
    facility_code: str | None = None
    client_installation_ref: str | None = None
    edge_installation_id: str | None = None
    enrollment_generation: int | None = None
    enrollment_created_at: str | None = None
    enrollment_updated_at: str | None = None


class InvalidConnectionSettingError(ValueError):
    field_name: str
    reason: str

    def __init__(self, *, field_name: str, reason: str) -> None:
        self.field_name = field_name
        self.reason = reason
        super().__init__(f"{reason}: {field_name}")


class MaskedConnectionSettings(TypedDict):
    events_url: str | None
    config_url: str | None
    facility_id: str | None
    facility_token_masked: str | None
    facility_token_set: bool
    updated_at: str | None


def _normalize_api_base(base: str | None) -> str | None:
    if not base:
        return None
    trimmed = base.strip().rstrip("/")
    if not trimmed:
        return None
    # Reject cleartext public Hub bases before deriving events/config paths so a
    # mis-baked compose default cannot seed bearer-bearing outbound URLs.
    try:
        allowed = hub_url_transport_allowed(trimmed)
    except ValueError:
        return None
    if not allowed:
        return None
    return trimmed if trimmed.endswith("/api") else f"{trimmed}/api"


class ConnectionSettingsStore:
    """Borrow transactions only; pool lifecycle belongs to the API."""

    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        self.database = database
        self.authority = authority

    def load(self) -> ConnectionSettings:
        return self.database.read(lambda connection: _settings_from_data(read_settings(connection)))

    def save(
        self,
        updates: Mapping[str, ConnectionValue],
        *,
        after_write: ConnectionWriteHook | None = None,
    ) -> ConnectionSettings:
        def persist(connection: psycopg.Connection) -> ConnectionSettings:
            require_authority(connection, self.authority)
            data = read_settings(connection, for_update=True)
            previous_principal = (data["edge_installation_id"], data["enrollment_generation"])
            changes = dict(updates)
            _validate_updates(changes)
            data.update(changes)
            if _has_enrollment_state(data):
                _validate_complete_enrollment(data)
            timestamp = utc_now_iso()
            data["updated_at"] = timestamp
            if _has_enrollment_state(data):
                data["enrollment_created_at"] = data["enrollment_created_at"] or timestamp
                data["enrollment_updated_at"] = timestamp
            else:
                data["enrollment_created_at"] = None
                data["enrollment_updated_at"] = None
            current_principal = (data["edge_installation_id"], data["enrollment_generation"])
            write_settings(connection, data, reset_topology=previous_principal != current_principal)
            if after_write is not None:
                after_write(connection)
            return _settings_from_data(data)

        # The database releases this candidate only after known successful COMMIT.
        # No reread, local publication, or retry on an unknown commit outcome.
        return self.database.transact(persist)

    def masked(self) -> MaskedConnectionSettings:
        settings = self.load()
        return {
            "events_url": settings.events_url,
            "config_url": settings.config_url,
            "facility_id": settings.facility_id,
            "facility_token_masked": mask_facility_token(settings.facility_token),
            "facility_token_set": bool(settings.facility_token),
            "updated_at": settings.updated_at,
        }


def _settings_from_data(saved: ConnectionData) -> ConnectionSettings:
    base = _normalize_api_base(os.environ.get(API_BACKEND_BASE_URL_ENV))
    return ConnectionSettings(
        events_url=(f"{base}/v1/events" if base else None),
        config_url=(f"{base}/v1/ml-config" if base else None),
        facility_id=_text(saved["facility_id"]),
        facility_token=_text(saved["facility_token"]),
        updated_at=_text(saved["updated_at"]),
        facility_code=_text(saved["facility_code"]),
        client_installation_ref=_text(saved["client_installation_ref"]),
        edge_installation_id=_text(saved["edge_installation_id"]),
        enrollment_generation=_positive_int(saved["enrollment_generation"]),
        enrollment_created_at=_text(saved["enrollment_created_at"]),
        enrollment_updated_at=_text(saved["enrollment_updated_at"]),
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validate_updates(updates: Mapping[str, ConnectionValue]) -> None:
    unknown = set(updates) - set(SAVE_FIELDS)
    if unknown:
        raise InvalidConnectionSettingError(
            field_name="connection_settings",
            reason="unknown connection setting field(s)",
        )
    for field_name, max_length in TEXT_FIELD_LIMITS.items():
        value = updates.get(field_name)
        if value is not None and not _valid_text(value, max_length):
            raise InvalidConnectionSettingError(
                field_name=field_name,
                reason="invalid connection setting field",
            )
    generation = updates.get("enrollment_generation")
    if generation is not None and (
        type(generation) is not int or not 1 <= generation <= MAX_ENROLLMENT_GENERATION
    ):
        raise InvalidConnectionSettingError(
            field_name="enrollment_generation",
            reason="invalid connection setting field",
        )


def _valid_text(value: ConnectionValue, max_length: int) -> bool:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= max_length
        or not value.strip()
        or "\x00" in value
    ):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _has_enrollment_state(data: ConnectionData) -> bool:
    return any(data[field_name] is not None for field_name in REQUIRED_ENROLLMENT_FIELDS)


def _validate_complete_enrollment(data: ConnectionData) -> None:
    if any(data[field_name] is None for field_name in REQUIRED_ENROLLMENT_FIELDS):
        raise InvalidConnectionSettingError(
            field_name="runtime_enrollment",
            reason="runtime enrollment fields must be saved atomically",
        )


def _text(value: ConnectionValue) -> str | None:
    return value if isinstance(value, str) and value else None


def _positive_int(value: ConnectionValue) -> int | None:
    return value if type(value) is int and 1 <= value <= MAX_ENROLLMENT_GENERATION else None


def mask_facility_token(token: str | None) -> str | None:
    if not token:
        return None
    return "****" if len(token) <= 4 else f"****{token[-4:]}"


def get_connection_settings_store(app: FastAPI) -> ConnectionSettingsStore:
    store = getattr(app.state, "connection_settings_store", None)
    if store is None:
        raise RuntimeError("connection settings store is not injected")
    if not isinstance(store, ConnectionSettingsStore):
        raise TypeError("connection settings store has invalid type")
    return store


__all__ = [
    "API_BACKEND_BASE_URL_ENV",
    "ConnectionSettings",
    "ConnectionSettingsStore",
    "InvalidConnectionSettingError",
    "MaskedConnectionSettings",
    "get_connection_settings_store",
    "mask_facility_token",
    "utc_now_iso",
]
