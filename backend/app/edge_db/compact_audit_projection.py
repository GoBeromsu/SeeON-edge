"""Project redacted closed-catalog legacy audit rows into the immutable chain."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import TypeAlias

from pydantic import JsonValue, TypeAdapter

from backend.app.edge_db.functions import audit_record_hash

_PAYLOAD = TypeAdapter(dict[str, JsonValue])
_FORBIDDEN = frozenset({"password", "token", "session", "rtsp_url", "resident", "traceback"})
AuditValue: TypeAlias = None | str


def _text(payload: Mapping[str, JsonValue], key: str, default: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) and value else default


def _optional_text(payload: Mapping[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _redacted(payload: Mapping[str, JsonValue]) -> str:
    safe = {key: value for key, value in payload.items() if key.lower() not in _FORBIDDEN}
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def project_audit(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    """Preserve each safe legacy audit fact in source order and chain order."""
    previous = "0" * 64
    rows = source.execute(
        "SELECT audit_id,occurred_at,action,payload_json FROM audit ORDER BY audit_id"
    ).fetchall()
    for audit_id, occurred_at, action, raw_payload in rows:
        payload = _PAYLOAD.validate_json(str(raw_payload))
        values: tuple[AuditValue, ...] = (
            str(occurred_at),
            str(occurred_at),
            _text(payload, "clock_quality", "unknown"),
            _text(payload, "actor_type", "system"),
            _text(payload, "actor_id", "legacy-import"),
            _text(payload, "auth_mechanism", "legacy"),
            str(action),
            _text(payload, "target_type", "legacy"),
            _text(payload, "target_id", str(audit_id)),
            _text(payload, "outcome", "success"),
            _optional_text(payload, "reason"),
            _optional_text(payload, "request_id"),
            _optional_text(payload, "interaction_id"),
            _redacted(payload),
            previous,
            "standard",
            None,
        )
        hash_payload = dict(
            zip(
                (
                    "occurred_at",
                    "recorded_at",
                    "clock_quality",
                    "actor_type",
                    "actor_id",
                    "auth_mechanism",
                    "action",
                    "target_type",
                    "target_id",
                    "outcome",
                    "reason",
                    "request_id",
                    "interaction_id",
                    "detail_json",
                    "previous_hash",
                    "retention_class",
                    "hold_reference",
                ),
                values,
                strict=True,
            )
        )
        record_hash = audit_record_hash(
            previous,
            json.dumps(hash_payload, ensure_ascii=False, separators=(",", ":")),
        )
        target.execute(
            "INSERT INTO audit_events (audit_id,occurred_at,recorded_at,clock_quality,actor_type,"
            "actor_id,auth_mechanism,action,target_type,target_id,outcome,reason,request_id,"
            "interaction_id,detail_json,previous_hash,record_hash,retention_class,hold_reference) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (audit_id, *values[:15], record_hash, *values[15:]),
        )
        previous = record_hash


__all__ = ["project_audit"]
