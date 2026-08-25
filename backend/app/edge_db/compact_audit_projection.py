"""Project redacted closed-catalog and governed review audit rows."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from backend.app.edge_db.compact_audit_redaction import (
    classified_audit_id,
    collect_source_secrets,
    parse_payload,
    redacted_detail,
)
from backend.app.edge_db.functions import audit_record_hash


@dataclass(frozen=True, slots=True)
class AuditRecord:
    audit_id: int
    occurred_at: str
    actor_type: str
    actor_id: str
    auth_mechanism: str
    action: str
    target_type: str
    target_id: str
    outcome: str
    reason: str | None
    request_id: str | None
    interaction_id: str | None
    detail_json: str


def _text(payload: Mapping[str, JsonValue], key: str, default: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) and value else default


def _optional_text(payload: Mapping[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _legacy_records(
    source: sqlite3.Connection, secrets: tuple[str, ...]
) -> tuple[AuditRecord, ...]:
    records: list[AuditRecord] = []
    rows = source.execute(
        "SELECT audit_id,occurred_at,action,payload_json FROM audit ORDER BY audit_id"
    )
    for source_id, occurred_at, action, raw_payload in rows:
        payload = parse_payload(raw_payload)
        audit_id = classified_audit_id(source_id, action, payload)
        if audit_id is None:
            continue
        records.append(
            AuditRecord(
                audit_id,
                str(occurred_at),
                _text(payload, "actor_type", "system"),
                _text(payload, "actor_id", "legacy-import"),
                _text(payload, "auth_mechanism", "legacy"),
                str(action),
                _text(payload, "target_type", "legacy"),
                _text(payload, "target_id", str(source_id)),
                _text(payload, "outcome", "success"),
                _optional_text(payload, "reason"),
                _optional_text(payload, "request_id"),
                _optional_text(payload, "interaction_id"),
                redacted_detail(payload, secrets),
            )
        )
    return tuple(sorted(records, key=lambda record: record.audit_id))


def _review_records(source: sqlite3.Connection, first_id: int) -> tuple[AuditRecord, ...]:
    records: list[AuditRecord] = []
    rows = source.execute(
        "SELECT review_id,incident_id,review_version,actor_id,reviewed_at,disposition "
        "FROM control_evidence_review_revisions "
        "ORDER BY reviewed_at,incident_id,review_version,review_id"
    )
    for offset, row in enumerate(rows):
        review_id, incident_id, version, actor_id, reviewed_at, disposition = row
        detail: dict[str, JsonValue] = {
            "review_id": str(review_id),
            "review_version": int(version),
            "disposition": str(disposition),
        }
        records.append(
            AuditRecord(
                first_id + offset,
                str(reviewed_at),
                "user",
                str(actor_id),
                "legacy-review-migration",
                "incident-review-migrated",
                "incident",
                str(incident_id),
                "success",
                None,
                None,
                None,
                json.dumps(detail, sort_keys=True, separators=(",", ":")),
            )
        )
    return tuple(records)


def _insert(target: sqlite3.Connection, record: AuditRecord, previous: str) -> str:
    values: dict[str, str | None] = {
        "occurred_at": record.occurred_at,
        "recorded_at": record.occurred_at,
        "clock_quality": "unknown",
        "actor_type": record.actor_type,
        "actor_id": record.actor_id,
        "auth_mechanism": record.auth_mechanism,
        "action": record.action,
        "target_type": record.target_type,
        "target_id": record.target_id,
        "outcome": record.outcome,
        "reason": record.reason,
        "request_id": record.request_id,
        "interaction_id": record.interaction_id,
        "detail_json": record.detail_json,
        "previous_hash": previous,
        "retention_class": "standard",
        "hold_reference": None,
    }
    record_hash = audit_record_hash(
        previous,
        json.dumps(values, ensure_ascii=False, separators=(",", ":")),
    )
    ordered = tuple(values.values())
    target.execute(
        "INSERT INTO audit_events (audit_id,occurred_at,recorded_at,clock_quality,actor_type,"
        "actor_id,auth_mechanism,action,target_type,target_id,outcome,reason,request_id,"
        "interaction_id,detail_json,previous_hash,record_hash,retention_class,hold_reference) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (record.audit_id, *ordered[:15], record_hash, *ordered[15:]),
    )
    return record_hash


def project_audit(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    """Preserve classified facts and review history in one valid deterministic chain."""
    secrets = collect_source_secrets(source)
    legacy = _legacy_records(source, secrets)
    first_review_id = max((record.audit_id for record in legacy), default=0) + 1
    records = (*legacy, *_review_records(source, first_review_id))
    previous = "0" * 64
    for record in records:
        previous = _insert(target, record, previous)


__all__ = ["project_audit"]
