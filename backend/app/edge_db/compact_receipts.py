"""Canonical source-row dispositions for schema-18 reconciliation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias, assert_never

from backend.app.edge_db.compact_receipt_context import (
    ReceiptContext,
    build_receipt_context,
    disposition,
    require_map_target,
)

SqliteValue: TypeAlias = None | int | float | str | bytes


def canonical_json(payload: dict[str, str | int | list[str] | None]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _encoded(value: SqliteValue) -> str | int | float | None:
    match value:
        case None | int() | float() | str():
            return value
        case bytes():
            return "hex:" + value.hex()
        case unreachable:
            assert_never(unreachable)


def _pk(value: SqliteValue) -> str:
    encoded = _encoded(value)
    return "null" if encoded is None else str(encoded)


def _row_digest(columns: tuple[str, ...], row: tuple[SqliteValue, ...]) -> str:
    payload = {column: _encoded(value) for column, value in zip(columns, row, strict=True)}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _primary_key(
    columns: tuple[str, ...], row: tuple[SqliteValue, ...], pk_indices: tuple[int, ...]
) -> list[str]:
    if pk_indices:
        return [f"{columns[index]}={_pk(row[index])}" for index in pk_indices]
    return ["row_sha256=" + _row_digest(columns, row)]


def _target_pks(
    table: str,
    columns: tuple[str, ...],
    row: tuple[SqliteValue, ...],
    context: ReceiptContext,
) -> list[str]:
    values = dict(zip(columns, row, strict=True))
    if table == "schema_migrations":
        version_value = values["version"]
        if not isinstance(version_value, int):
            raise sqlite3.DatabaseError("migration receipt version is not an integer")
        version = version_value
        targets = [f"schema_migrations:version={version}"]
        return targets + (["schema_migrations:version=18"] if version == 17 else [])
    if table == "audit":
        audit_id = context.legacy_audit_ids.get(str(values["audit_id"]))
        return [] if audit_id is None else [f"audit_events:audit_id={audit_id}"]
    if table == "credentials":
        return [f"credentials:id={_pk(values['id'])}"]
    if table == "camera_registry":
        return ["edge_site:id=1", *(f"cameras:camera_id={item}" for item in context.camera_ids)]
    if table in {"camera_topology_cameras", "camera_bed_zone"}:
        return [f"cameras:camera_id={_pk(values['camera_id'])}"]
    if table in {"clips", "evidence_clips", "evidence_retention_states"}:
        clip_id = str(values["clip_id"])
        return [f"clips:clip_id={clip_id}"] if clip_id in context.clip_ids else []
    if table == "evidence_primary_clips":
        targets = [f"artifacts:incident_id={_pk(values['incident_id'])},kind=PRIMARY_CLIP"]
        clip_id = str(values["clip_id"])
        if clip_id in context.clip_ids:
            targets.append(f"clips:clip_id={clip_id}")
        return targets
    if table == "evidence_events":
        incident_id = context.event_incidents.get(str(values["edge_event_id"]))
        return [] if incident_id is None else [f"incidents:incident_id={incident_id}"]
    if table == "evidence_incident_snapshots":
        return [f"artifacts:incident_id={_pk(values['incident_id'])},kind=SNAPSHOT"]
    if table == "evidence_media_objects":
        return list(context.media_targets.get(str(values["media_id"]), ()))
    if table == "snapshots":
        incident_id = context.snapshot_incidents.get(str(values["snapshot_id"]))
        return [] if incident_id is None else [f"artifacts:incident_id={incident_id},kind=SNAPSHOT"]
    if table == "evidence_incidents":
        return [f"incidents:incident_id={_pk(values['incident_id'])}"]
    if table == "control_evidence_review_state":
        return [f"incidents:incident_id={_pk(values['incident_id'])}"]
    if table == "control_evidence_review_revisions":
        review_id = str(values["review_id"])
        targets = [f"audit_events:audit_id={context.review_audit_ids[review_id]}"]
        incident_id = context.review_incidents[review_id]
        if review_id in context.current_reviews:
            targets.append(f"incidents:incident_id={incident_id}")
        return targets
    if table == "labels":
        incident_id = context.label_incidents.get(str(values["clip_id"]))
        return [] if incident_id is None else [f"incidents:incident_id={incident_id}"]
    if table == "evidence_artifact_slots" and values["slot_name"] in {
        "PRIMARY_CLIP",
        "SNAPSHOT",
    }:
        return [
            f"artifacts:incident_id={_pk(values['incident_id'])},kind={_pk(values['slot_name'])}"
        ]
    if table == "camera_topology_floors":
        return [f"locations:location_id={_pk(values['edge_ref'])},kind=FLOOR"]
    if table == "camera_topology_rooms":
        return [f"locations:location_id={_pk(values['edge_ref'])},kind=ROOM"]
    if table == "control_detection_policy_activations":
        policy_id = context.policy_ids.get(str(values["activation_id"]))
        return [] if policy_id is None else [f"policies:policy_id={policy_id}"]
    if table == "control_detection_policy_revisions":
        policy_id = context.policy_revisions.get(str(values["revision_id"]))
        return [] if policy_id is None else [f"policies:policy_id={policy_id}"]
    if table == "control_detection_policy_state":
        policy_ids = context.policy_facilities.get(str(values["facility_id"]), ())
        return [f"policies:policy_id={policy_id}" for policy_id in policy_ids]
    if table == "clip_listing_generation":
        return [f"clips:clip_id={clip_id}" for clip_id in context.rebuilt_clip_ids]
    if table == "connection_store_migrations":
        return ["schema_migrations:version=18"]
    if table in {"schema_import_receipts", "schema_import_sources", "schema_metadata"}:
        return ["schema_migrations:version=18"]
    if table == "control_legacy_label_migrations":
        targets = ["schema_migrations:version=18"]
        if values["incident_id"] is not None:
            targets.append(f"incidents:incident_id={_pk(values['incident_id'])}")
        return targets
    if table in {
        "clip_storage_location",
        "connection_settings",
        "detection_settings",
        "edge_topology_confirmation_preview",
        "edge_topology_sync_state",
        "runtime_settings",
    }:
        return ["edge_site:id=1"]
    return []


def receipt_lines(
    source: Path, inventory_sha256: str, rebuilt_clip_ids: tuple[str, ...]
) -> Iterable[bytes]:
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        context = build_receipt_context(connection, rebuilt_clip_ids)
        tables = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        for table in tables:
            info = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            columns = tuple(str(item[1]) for item in info)
            pk_indices = tuple(
                columns.index(str(item[1]))
                for item in sorted((item for item in info if int(item[5]) > 0), key=lambda x: x[5])
            )
            quoted = ",".join('"' + column.replace('"', '""') + '"' for column in columns)
            order = ",".join(str(index + 1) for index in range(len(columns)))
            rows: Iterable[tuple[SqliteValue, ...]] = connection.execute(
                f'SELECT {quoted} FROM "{table}" ORDER BY {order}'
            )
            for row in rows:
                targets = _target_pks(table, columns, row, context)
                require_map_target(table, targets)
                empty_reasons = {
                    "audit": "unclassified_legacy_audit_archived",
                    "control_detection_policy_revisions": "superseded_policy_revision_archived",
                }
                action, reason = disposition(table, targets, empty_reasons.get(table))
                relation_kind = (
                    "CUTOVER_RECONCILIATION"
                    if table == "connection_store_migrations"
                    else {
                        "MAP": "ROW_AUTHORITY",
                        "REBUILD": "FILESYSTEM_AUTHORITY",
                        "NONE": "ARCHIVE_ONLY",
                    }[action]
                )
                yield (
                    canonical_json(
                        {
                            "action": action,
                            "inventory_sha256": inventory_sha256,
                            "reason": reason,
                            "relation_kind": relation_kind,
                            "source_pk": _primary_key(columns, row, pk_indices),
                            "source_row_sha256": _row_digest(columns, row),
                            "source_table": table,
                            "target_pks": targets,
                        }
                    ).encode()
                    + b"\n"
                )
    finally:
        connection.close()


def write_or_verify_receipts(
    source: Path,
    inventory_sha256: str,
    receipt: Path,
    rebuilt_clip_ids: tuple[str, ...],
    *,
    on_written: Callable[[], None] | None = None,
) -> tuple[int, str]:
    """Stream canonical receipts without retaining a large source in memory."""
    count = 0
    digest = hashlib.sha256()
    if receipt.exists():
        with receipt.open("rb") as existing:
            for line in receipt_lines(source, inventory_sha256, rebuilt_clip_ids):
                if existing.readline() != line:
                    raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_STALE_RECEIPT")
                digest.update(line)
                count += 1
            if existing.read(1):
                raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_STALE_RECEIPT")
            os.fsync(existing.fileno())
        return count, digest.hexdigest()
    descriptor = os.open(receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            for line in receipt_lines(source, inventory_sha256, rebuilt_clip_ids):
                stream.write(line)
                digest.update(line)
                count += 1
            stream.flush()
            if on_written is not None:
                on_written()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return count, digest.hexdigest()


__all__ = ["canonical_json", "receipt_lines", "write_or_verify_receipts"]
