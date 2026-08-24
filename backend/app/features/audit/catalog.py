"""Closed audit action and privacy-bounded detail catalogs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias, assert_never

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | Mapping[str, "JsonValue"]

MAX_DETAIL_BYTES: Final = 16 * 1024


class AuditActorType(StrEnum):
    USER = "user"
    SERVICE = "service"
    SYSTEM = "system"


class AuditAuthMechanism(StrEnum):
    DASHBOARD_SESSION = "dashboard_session"
    RELAY_TOKEN = "relay_token"
    INTERNAL = "internal"


class AuditAction(StrEnum):
    AUTH_LOGIN = "auth.login"
    AUTH_SESSION_READ = "auth.session.read"
    AUTH_LOGOUT = "auth.logout"
    CREDENTIAL_ROTATE = "credential.rotate"
    CAMERA_CREATE = "camera.create"
    CAMERA_UPDATE = "camera.update"
    CAMERA_DELETE = "camera.delete"
    LOCATION_CREATE = "location.create"
    LOCATION_UPDATE = "location.update"
    LOCATION_DELETE = "location.delete"
    BED_ZONE_UPDATE = "bed-zone.update"
    BED_ZONE_DELETE = "bed-zone.delete"
    CONNECTION_UPDATE = "connection.update"
    RUNTIME_SETTINGS_UPDATE = "runtime-settings.update"
    POLICY_APPLY = "policy.apply"
    POLICY_ROLLBACK = "policy.rollback"
    INCIDENT_LIST = "incident.list"
    INCIDENT_DETAIL = "incident.detail"
    INCIDENT_REVIEW = "incident.review"
    CLIP_LIST = "clip.list"
    CLIP_DETAIL = "clip.detail"
    CLIP_PLAY = "clip.play"
    CLIP_VIDEO = "clip.video"
    CLIP_THUMBNAIL = "clip.thumbnail"
    CLIP_ARTIFACT = "clip.artifact"
    CLIP_DELETE = "clip.delete"
    AUDIT_LIST = "audit.list"
    AUDIT_DETAIL = "audit.detail"
    RELAY_ALERT = "relay.alert"
    RECOVERY_FENCE = "audit.recovery-fence"


@dataclass(frozen=True, slots=True)
class AuditDetail:
    json: str | None


@dataclass(frozen=True, slots=True)
class AuditDetailError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


_FORBIDDEN_CLASSES: Final = frozenset(
    {
        "raw", "resident", "pose", "keypoint", "token", "password", "credential",
        "session", "cookie", "request", "response", "traceback", "path", "media",
        "bytes", "secret", "authorization", "rtsp", "image", "frame",
    }
)
_RECOVERY_KEYS: Final = frozenset({"failure_code", "ended_at"})


def _normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _forbidden(value: str) -> bool:
    normalized = _normalized(value)
    return any(term in normalized for term in _FORBIDDEN_CLASSES)


def _inspect(value: JsonValue) -> None:
    match value:
        case None | bool() | int() | float():
            return
        case str() as text:
            if _forbidden(text):
                raise AuditDetailError("audit detail contains a forbidden value class")
        case list() as items:
            for item in items:
                _inspect(item)
        case Mapping() as values:
            for key, item in values.items():
                if _forbidden(key):
                    raise AuditDetailError("audit detail contains a forbidden field class")
                _inspect(item)
        case unreachable:
            assert_never(unreachable)


def parse_detail(action: AuditAction, raw: Mapping[str, JsonValue]) -> AuditDetail:
    """Parse untrusted detail into canonical JSON for one registered action."""
    _inspect(raw)
    allowed = _RECOVERY_KEYS if action is AuditAction.RECOVERY_FENCE else frozenset()
    unknown = frozenset(raw) - allowed
    if unknown:
        raise AuditDetailError(f"detail fields are not registered for {action.value}")
    if not raw:
        return AuditDetail(json=None)
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode()) > MAX_DETAIL_BYTES:
        raise AuditDetailError("audit detail exceeds 16 KiB")
    return AuditDetail(json=encoded)


__all__ = [
    "MAX_DETAIL_BYTES", "AuditAction", "AuditActorType", "AuditAuthMechanism",
    "AuditDetail", "AuditDetailError", "JsonValue",
    "parse_detail",
]
