"""Canonical source-row dispositions for schema-18 reconciliation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Final, TypeAlias, assert_never

SqliteValue: TypeAlias = None | int | float | str | bytes


_MAP_TABLES: Final = frozenset(
    {
        "audit",
        "camera_bed_zone",
        "camera_registry",
        "camera_topology_cameras",
        "camera_topology_floors",
        "camera_topology_rooms",
        "clip_storage_location",
        "clips",
        "connection_settings",
        "connection_store_migrations",
        "control_detection_policy_activations",
        "control_detection_policy_revisions",
        "control_detection_policy_state",
        "control_evidence_review_revisions",
        "control_evidence_review_state",
        "control_legacy_label_migrations",
        "credentials",
        "detection_settings",
        "edge_topology_confirmation_preview",
        "edge_topology_sync_state",
        "events",
        "evidence_artifact_slots",
        "evidence_clips",
        "evidence_events",
        "evidence_incident_snapshots",
        "evidence_incidents",
        "evidence_media_objects",
        "evidence_primary_clips",
        "evidence_retention_states",
        "labels",
        "runtime_settings",
        "schema_import_receipts",
        "schema_import_sources",
        "schema_metadata",
        "schema_migrations",
        "snapshots",
        "topology_dirty",
    }
)
_REBUILD_TABLES: Final = frozenset(
    {
        "clip_listing_generation",
        "clip_listing_rows",
        "clip_listing_summary",
        "clip_listing_thumbnails",
    }
)


def canonical_json(payload: dict[str, str | int | list[str]]) -> str:
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


def _action(table: str) -> str:
    if table in _MAP_TABLES:
        return "MAP"
    if table in _REBUILD_TABLES:
        return "REBUILD"
    return "NONE"


def _target_pks(table: str, columns: tuple[str, ...], row: tuple[SqliteValue, ...]) -> list[str]:
    values = dict(zip(columns, row, strict=True))
    if table == "schema_migrations":
        return [f"schema_migrations:version={_pk(values['version'])}"]
    if table == "credentials":
        return [f"credentials:id={_pk(values['id'])}"]
    if table == "evidence_incidents":
        return [f"incidents:incident_id={_pk(values['incident_id'])}"]
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
    if table in {
        "camera_registry",
        "clip_storage_location",
        "connection_settings",
        "detection_settings",
        "edge_topology_confirmation_preview",
        "edge_topology_sync_state",
        "runtime_settings",
        "topology_dirty",
    }:
        return ["edge_site:id=1"]
    return []


def _receipt_lines(source: Path, inventory_sha256: str) -> Iterable[bytes]:
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
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
                yield (
                    canonical_json(
                        {
                            "action": _action(table),
                            "inventory_sha256": inventory_sha256,
                            "source_pk": _primary_key(columns, row, pk_indices),
                            "source_row_sha256": _row_digest(columns, row),
                            "source_table": table,
                            "target_pks": _target_pks(table, columns, row),
                        }
                    ).encode()
                    + b"\n"
                )
    finally:
        connection.close()


def write_or_verify_receipts(source: Path, inventory_sha256: str, receipt: Path) -> tuple[int, str]:
    """Stream canonical receipts without retaining a large source in memory."""
    count = 0
    digest = hashlib.sha256()
    if receipt.exists():
        with receipt.open("rb") as existing:
            for line in _receipt_lines(source, inventory_sha256):
                if existing.readline() != line:
                    raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_STALE_RECEIPT")
                digest.update(line)
                count += 1
            if existing.read(1):
                raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_STALE_RECEIPT")
        return count, digest.hexdigest()
    descriptor = os.open(receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            for line in _receipt_lines(source, inventory_sha256):
                stream.write(line)
                digest.update(line)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return count, digest.hexdigest()


__all__ = ["canonical_json", "write_or_verify_receipts"]
