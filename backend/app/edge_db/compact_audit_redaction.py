"""Two-pass sensitive-alias collection and redaction for migrated audit facts."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from typing import TypeAlias, assert_never

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
    "evidence",
    "file",
    "keypoint",
    "manifestpath",
    "media",
    "password",
    "path",
    "pose",
    "relpath",
    "requestbody",
    "resident",
    "responsebody",
    "rtsp",
    "secret",
    "session",
    "snapshot",
    "sourcemedia",
    "token",
    "traceback",
)
_REDACTED = "[REDACTED]"
SqliteValue: TypeAlias = None | int | float | str | bytes
SensitiveValue: TypeAlias = str | bytes


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


def _sensitive_key(key: str) -> bool:
    normalized = "".join(character for character in key.lower() if character.isalnum())
    return any(term in normalized for term in _SENSITIVE_TERMS)


def _nested_strings(value: JsonValue) -> Iterable[str]:
    match value:
        case str():
            yield value
        case list():
            for item in value:
                yield from _nested_strings(item)
        case dict():
            for item in value.values():
                yield from _nested_strings(item)
        case None | bool() | int() | float():
            return
        case unreachable:
            assert_never(unreachable)


def _nested_payload_values(value: JsonValue) -> Iterable[str]:
    match value:
        case dict():
            yield from _payload_sensitive_values(value)
        case list():
            for item in value:
                yield from _nested_payload_values(item)
        case None | bool() | int() | float() | str():
            return
        case unreachable:
            assert_never(unreachable)


def _payload_sensitive_values(payload: Mapping[str, JsonValue]) -> Iterable[str]:
    for key, value in payload.items():
        if _sensitive_key(key):
            yield from _nested_strings(value)
        yield from _nested_payload_values(value)


def _source_values(source: sqlite3.Connection) -> Iterable[SensitiveValue]:
    credential = source.execute("SELECT salt,password_hash FROM credentials WHERE id=1").fetchone()
    if credential is not None:
        for value in credential:
            if isinstance(value, bytes):
                yield value
    connection = source.execute(
        "SELECT facility_token FROM connection_settings WHERE id=1"
    ).fetchone()
    if connection is not None and isinstance(connection[0], str) and connection[0]:
        yield connection[0]
    path_queries = (
        "SELECT path FROM clips",
        "SELECT path FROM snapshots",
        "SELECT manifest_path,media_relpath FROM evidence_clips",
        "SELECT contained_relpath,basename FROM evidence_media_objects",
        "SELECT manifest_relpath FROM evidence_primary_clips",
    )
    for query in path_queries:
        for row in source.execute(query):
            for value in row:
                if isinstance(value, str) and value:
                    yield value
    registry = source.execute("SELECT cameras_json FROM camera_registry WHERE id=1").fetchone()
    if registry is not None:
        payload = json.loads(str(registry[0]))
        if isinstance(payload, list):
            for camera in payload:
                if not isinstance(camera, dict):
                    continue
                rtsp_url = camera.get("rtsp_url")
                if isinstance(rtsp_url, str):
                    yield rtsp_url
                    match = re.match(r"rtsp://([^/@]+)@", rtsp_url)
                    if match is not None:
                        yield match.group(1)


def _audit_values(source: sqlite3.Connection) -> Iterable[str]:
    for audit_id, action, raw_payload in source.execute(
        "SELECT audit_id,action,payload_json FROM audit ORDER BY audit_id"
    ):
        payload = parse_payload(raw_payload)
        if classified_audit_id(audit_id, action, payload) is not None:
            yield from _payload_sensitive_values(payload)


def _aliases(values: Iterable[SensitiveValue]) -> tuple[str, ...]:
    aliases: set[str] = set()
    for value in values:
        encoded = value if isinstance(value, bytes) else value.encode()
        if not encoded:
            continue
        if isinstance(value, str):
            aliases.add(value)
            printable = value
        else:
            try:
                printable = value.decode()
            except UnicodeDecodeError:
                printable = ""
            if printable:
                aliases.add(printable)
        hexadecimal = encoded.hex()
        aliases.update({hexadecimal, hexadecimal.upper()})
        for base64_value in (
            base64.b64encode(encoded).decode(),
            base64.urlsafe_b64encode(encoded).decode(),
        ):
            aliases.update({base64_value, base64_value.rstrip("=")})
        if printable:
            aliases.update(
                {
                    f"Bearer {printable}",
                    f"cookie={printable}",
                    f"session={printable}",
                }
            )
    return tuple(sorted(aliases, key=lambda item: (-len(item), item)))


def collect_source_secrets(source: sqlite3.Connection) -> tuple[str, ...]:
    """First pass: derive every audit alias before any detail is transformed."""
    return _aliases((*_source_values(source), *_audit_values(source)))


def _redact_value(value: JsonValue, aliases: tuple[str, ...]) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _redact_value(item, aliases)
            for key, item in value.items()
            if not _sensitive_key(key)
        }
    if isinstance(value, list):
        return [_redact_value(item, aliases) for item in value]
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if (
        value in aliases
        or value.startswith("/")
        or lowered.startswith("bearer ")
        or lowered.startswith("cookie=")
        or lowered.startswith("session=")
    ):
        return _REDACTED
    return value


def redacted_detail(payload: Mapping[str, JsonValue], aliases: tuple[str, ...]) -> str:
    """Second pass: recursively remove protected keys and exact aliases."""
    safe = _redact_value(dict(payload), aliases)
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
