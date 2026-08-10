from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Final, TypedDict

from backend.app.features.connection.sqlite_store import (
    ENROLLMENT_MARKER_FIELDS,
    REQUIRED_ENROLLMENT_FIELDS,
    SAVE_FIELDS,
    ConnectionData,
    ConnectionStoreBackup,
    ConnectionStoreDatabase,
    ConnectionValue,
    utc_now_iso,
)

API_CONNECTION_SETTINGS_PATH_ENV: Final = "API_CONNECTION_SETTINGS_PATH"
API_BACKEND_BASE_URL_ENV: Final = "API_BACKEND_BASE_URL"
API_BACKEND_CONFIG_URL_ENV: Final = "API_BACKEND_CONFIG_URL"
API_BACKEND_EVENTS_URL_ENV: Final = "API_BACKEND_EVENTS_URL"
DEFAULT_CONNECTION_SETTINGS_PATH: Final = "/var/lib/ml-api/connection-settings.sqlite3"


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
    return trimmed if trimmed.endswith("/api") else f"{trimmed}/api"


class ConnectionSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        self._database: ConnectionStoreDatabase = ConnectionStoreDatabase(self.path)
        self.rollback_directory: Path = self._database.rollback_directory
        self._lock: Lock = Lock()

    @classmethod
    def from_env(cls) -> ConnectionSettingsStore:
        return cls(
            os.environ.get(API_CONNECTION_SETTINGS_PATH_ENV, DEFAULT_CONNECTION_SETTINGS_PATH)
        )

    def load(self) -> ConnectionSettings:
        with self._lock:
            saved = self._database.read()
        base = _normalize_api_base(os.environ.get(API_BACKEND_BASE_URL_ENV))
        return ConnectionSettings(
            events_url=(
                _text(saved["events_url"])
                or os.environ.get(API_BACKEND_EVENTS_URL_ENV)
                or (f"{base}/v1/events" if base else None)
            ),
            config_url=(
                _text(saved["config_url"])
                or os.environ.get(API_BACKEND_CONFIG_URL_ENV)
                or (f"{base}/v1/ml-config" if base else None)
            ),
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

    def save(self, updates: Mapping[str, ConnectionValue]) -> ConnectionSettings:
        _validate_updates(updates)
        with self._lock:
            data = self._database.read()
            data.update(dict(updates))
            if _has_enrollment_state(data):
                _validate_complete_enrollment(data)
            timestamp = utc_now_iso()
            data["updated_at"] = timestamp
            if _has_enrollment_state(data):
                data["enrollment_created_at"] = data["enrollment_created_at"] or timestamp
                data["enrollment_updated_at"] = timestamp
            self._database.write(data)
        return self.load()

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

    def create_pre_v1_backup(self) -> ConnectionStoreBackup:
        with self._lock:
            return self._database.create_pre_v1_backup()

    def integrity_check(self, path: str | Path | None = None) -> str:
        database = Path(path) if path is not None else self.path
        with self._lock:
            return self._database.integrity_check(database)

    def restore_pre_v1_backup(self, backup_path: str | Path) -> None:
        with self._lock:
            self._database.restore_pre_v1_backup(Path(backup_path))


def _validate_updates(updates: Mapping[str, ConnectionValue]) -> None:
    unknown = set(updates) - set(SAVE_FIELDS)
    if unknown:
        raise InvalidConnectionSettingError(
            field_name=", ".join(sorted(unknown)),
            reason="unknown connection setting field(s)",
        )
    for field_name in (
        "facility_code",
        "client_installation_ref",
        "facility_id",
        "facility_token",
        "edge_installation_id",
    ):
        value = updates.get(field_name)
        invalid_text = value is not None and (
            not isinstance(value, str) or not value.strip()
        )
        if field_name in updates and invalid_text:
            raise InvalidConnectionSettingError(
                field_name=field_name,
                reason="invalid connection setting field",
            )
    generation = updates.get("enrollment_generation")
    if generation is not None and (type(generation) is not int or generation < 1):
        raise InvalidConnectionSettingError(
            field_name="enrollment_generation",
            reason="invalid connection setting field",
        )


def _has_enrollment_state(data: ConnectionData) -> bool:
    return any(data[field_name] is not None for field_name in ENROLLMENT_MARKER_FIELDS)


def _validate_complete_enrollment(data: ConnectionData) -> None:
    if any(data[field_name] is None for field_name in REQUIRED_ENROLLMENT_FIELDS):
        raise InvalidConnectionSettingError(
            field_name="runtime_enrollment",
            reason="runtime enrollment fields must be saved atomically",
        )


def _text(value: ConnectionValue) -> str | None:
    return value if isinstance(value, str) and value else None


def _positive_int(value: ConnectionValue) -> int | None:
    return value if type(value) is int and value > 0 else None


def mask_facility_token(token: str | None) -> str | None:
    if not token:
        return None
    return "****" if len(token) <= 4 else f"****{token[-4:]}"


__all__ = [
    "API_BACKEND_BASE_URL_ENV",
    "API_CONNECTION_SETTINGS_PATH_ENV",
    "DEFAULT_CONNECTION_SETTINGS_PATH",
    "ConnectionSettings",
    "ConnectionSettingsStore",
    "ConnectionStoreBackup",
    "InvalidConnectionSettingError",
    "MaskedConnectionSettings",
    "mask_facility_token",
    "utc_now_iso",
]
