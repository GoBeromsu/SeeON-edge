"""Closed-catalog classification and recursive redaction for migrated audit facts."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from collections.abc import Mapping
from typing import TypeAlias

from pydantic import JsonValue, TypeAdapter

_PAYLOAD = TypeAdapter(dict[str, JsonValue])
_ACTIONS = frozenset(
    {
        "analysis-view",
        "artifact-view",
        "audit-view",
        "clip-delete-failed",
        "clip-view",
        "label",
        "list",
        "metadata-view",
        "play",
        "play-annotated",
    }
)
_SENSITIVE_TERMS = (
    "authorization",
    "cookie",
    "credential",
    "keypoint",
    "manifestpath",
    "mediapath",
    "password",
    "pose",
    "requestbody",
    "resident",
    "responsebody",
    "rtsp",
    "secret",
    "session",
    "sourcemedia",
    "token",
    "traceback",
)
_REDACTED = "[REDACTED]"
SqliteValue: TypeAlias = None | int | float | str | bytes


def parse_payload(raw: SqliteValue) -> dict[str, JsonValue]:
    return _PAYLOAD.validate_json(str(raw))


def classified_audit_id(
    audit_id: SqliteValue, action: SqliteValue, payload: Mapping[str, JsonValue]
) -> int | None:
    try:
        identifier = int(str(audit_id))
    except ValueError:
        return None
    required = (
        isinstance(payload.get("actor_type"), str)
        and payload.get("actor_type") in {"user", "service", "system"}
        and isinstance(payload.get("actor_id"), str)
        and bool(payload.get("actor_id"))
        and isinstance(payload.get("target_type"), str)
        and bool(payload.get("target_type"))
        and isinstance(payload.get("target_id"), str)
        and bool(payload.get("target_id"))
        and payload.get("outcome") in {"success", "denied", "failed"}
    )
    return identifier if identifier > 0 and str(action) in _ACTIONS and required else None


def collect_source_secrets(source: sqlite3.Connection) -> tuple[str, ...]:
    secrets: set[str] = set()
    credential = source.execute("SELECT salt,password_hash FROM credentials WHERE id=1").fetchone()
    if credential is not None:
        for value in credential:
            if isinstance(value, bytes):
                secrets.add(value.hex())
                secrets.add(base64.b64encode(value).decode("ascii"))
                if all(32 <= byte < 127 for byte in value):
                    secrets.add(value.decode("ascii"))
    connection = source.execute(
        "SELECT facility_token FROM connection_settings WHERE id=1"
    ).fetchone()
    if connection is not None and isinstance(connection[0], str) and connection[0]:
        secrets.add(connection[0])
    registry = source.execute("SELECT cameras_json FROM camera_registry WHERE id=1").fetchone()
    if registry is not None:
        cameras = _PAYLOAD.validate_python({"cameras": json.loads(str(registry[0]))})["cameras"]
        if isinstance(cameras, list):
            for camera in cameras:
                if isinstance(camera, dict):
                    rtsp_url = camera.get("rtsp_url")
                    if isinstance(rtsp_url, str):
                        match = re.match(r"rtsp://([^/@]+)@", rtsp_url)
                        if match is not None:
                            secrets.add(match.group(1))
    return tuple(sorted((value for value in secrets if value), key=lambda item: (-len(item), item)))


def _sensitive_key(key: str) -> bool:
    normalized = "".join(character for character in key.lower() if character.isalnum())
    return any(term in normalized for term in _SENSITIVE_TERMS)


def _redact_value(value: JsonValue, secrets: tuple[str, ...]) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _redact_value(item, secrets)
            for key, item in value.items()
            if not _sensitive_key(key)
        }
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered.startswith("bearer ") or lowered.startswith("rtsp://") or value.startswith("/"):
        return _REDACTED
    if "cookie=" in lowered or "session=" in lowered:
        return _REDACTED
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, _REDACTED)
    return redacted


def redacted_detail(payload: Mapping[str, JsonValue], secrets: tuple[str, ...]) -> str:
    safe = _redact_value(dict(payload), secrets)
    encoded = json.dumps(
        safe, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    if len(encoded.encode()) > 16384:
        return '{"redaction":"detail_omitted_over_cap"}'
    return encoded


__all__ = [
    "classified_audit_id",
    "collect_source_secrets",
    "parse_payload",
    "redacted_detail",
]
