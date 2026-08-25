"""Deterministic fan-in and fan-out context for source-row receipts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter

from backend.app.edge_db.compact_audit_redaction import classified_audit_id, parse_payload

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
    }
)
_STRICT_MAP_TABLES: Final = frozenset(
    {
        "clips",
        "connection_store_migrations",
        "control_detection_policy_state",
        "control_evidence_review_revisions",
        "evidence_events",
        "evidence_incident_snapshots",
        "evidence_media_objects",
        "snapshots",
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
_CAMERA_LIST = TypeAdapter(list[dict[str, JsonValue]])
SqliteValue: TypeAlias = None | int | float | str | bytes


@dataclass(frozen=True, slots=True)
class ReceiptContext:
    camera_ids: tuple[str, ...]
    clip_ids: tuple[str, ...]
    current_reviews: frozenset[str]
    event_incidents: dict[str, str]
    label_incidents: dict[str, str]
    legacy_audit_ids: dict[str, int]
    media_targets: dict[str, tuple[str, ...]]
    policy_facilities: dict[str, tuple[int, ...]]
    policy_ids: dict[str, int]
    policy_revisions: dict[str, int]
    review_audit_ids: dict[str, int]
    review_incidents: dict[str, str]
    snapshot_incidents: dict[str, str]
    rebuilt_clip_ids: tuple[str, ...]


def require_map_target(table: str, targets: list[str]) -> None:
    if table in _STRICT_MAP_TABLES and not targets:
        raise sqlite3.DatabaseError(f"locked MAP source row has no target authority: {table}")


def disposition(
    table: str, targets: list[str], empty_reason: str | None = None
) -> tuple[str, str | None]:
    if table in _MAP_TABLES and targets:
        return "MAP", None
    if table in _REBUILD_TABLES:
        return "REBUILD", "filesystem_manifest_authority"
    reasons = {
        "events": "catalog_duplicate_retired",
        "topology_dirty": "derived_dirty_marker_retired",
    }
    return "NONE", empty_reason or reasons.get(table, "source_archive_only")


def _camera_ids(connection: sqlite3.Connection) -> tuple[str, ...]:
    registry = connection.execute("SELECT cameras_json FROM camera_registry WHERE id=1").fetchone()
    if registry is None:
        return ()
    return tuple(
        str(item["id"])
        for item in _CAMERA_LIST.validate_json(str(registry[0]))
        if isinstance(item.get("id"), str)
    )


def _policy_context(
    connection: sqlite3.Connection,
) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[int, ...]]]:
    rows = connection.execute(
        "SELECT activation_id,facility_id,camera_id,module_id,activation_generation,"
        "active_revision_id,previous_revision_id FROM control_detection_policy_activations "
        "ORDER BY activation_generation,activation_id"
    ).fetchall()
    current: dict[tuple[str, str | None, str], tuple[SqliteValue, ...]] = {}
    for row in rows:
        current[(str(row[1]), row[2], str(row[3]))] = row
    activation_ids: dict[str, int] = {}
    revision_ids: dict[str, int] = {}
    facilities: dict[str, list[int]] = {}
    for policy_id, (_key, row) in enumerate(sorted(current.items()), start=1):
        activation_ids[str(row[0])] = policy_id
        revision_ids[str(row[5])] = policy_id
        if row[6] is not None:
            revision_ids[str(row[6])] = policy_id
        facilities.setdefault(str(row[1]), []).append(policy_id)
    return (
        activation_ids,
        revision_ids,
        {facility: tuple(ids) for facility, ids in facilities.items()},
    )


def _audit_context(
    connection: sqlite3.Connection,
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    legacy: dict[str, int] = {}
    for source_id, action, raw_payload in connection.execute(
        "SELECT audit_id,action,payload_json FROM audit ORDER BY audit_id"
    ):
        target_id = classified_audit_id(source_id, action, parse_payload(raw_payload))
        if target_id is not None:
            legacy[str(source_id)] = target_id
    review_rows = connection.execute(
        "SELECT review_id,incident_id FROM control_evidence_review_revisions "
        "ORDER BY reviewed_at,incident_id,review_version,review_id"
    ).fetchall()
    first = max(legacy.values(), default=0) + 1
    review_audits = {str(row[0]): first + index for index, row in enumerate(review_rows)}
    review_incidents = {str(row[0]): str(row[1]) for row in review_rows}
    return legacy, review_audits, review_incidents


def _media_targets(
    connection: sqlite3.Connection, clip_ids: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    targets: dict[str, list[str]] = {}
    for incident_id, clip_id, media_id in connection.execute(
        "SELECT incident_id,clip_id,media_id FROM evidence_primary_clips WHERE media_id IS NOT NULL"
    ):
        values = targets.setdefault(str(media_id), [])
        values.append(f"artifacts:incident_id={incident_id},kind=PRIMARY_CLIP")
        if str(clip_id) in clip_ids:
            values.append(f"clips:clip_id={clip_id}")
    for incident_id, media_id in connection.execute(
        "SELECT incident_id,media_id FROM evidence_incident_snapshots"
    ):
        targets.setdefault(str(media_id), []).append(
            f"artifacts:incident_id={incident_id},kind=SNAPSHOT"
        )
    return {key: tuple(dict.fromkeys(values)) for key, values in targets.items()}


def build_receipt_context(
    connection: sqlite3.Connection, rebuilt: tuple[str, ...]
) -> ReceiptContext:
    event_incidents = {
        str(event_id): str(incident_id)
        for incident_id, event_id in connection.execute(
            "SELECT incident_id,edge_event_id FROM evidence_incidents"
        )
    }
    label_incidents = {
        str(clip_id): str(incident_id)
        for incident_id, clip_id in connection.execute(
            "SELECT incident_id,clip_id FROM evidence_primary_clips"
        )
    }
    source_clips = {str(row[0]) for row in connection.execute("SELECT clip_id FROM clips")} | set(
        label_incidents
    )
    clip_ids = tuple(sorted(set(rebuilt) | source_clips))
    snapshot_incidents = {
        str(snapshot_id): str(incident_id)
        for incident_id, snapshot_id in connection.execute(
            "SELECT incident_id,snapshot_id FROM evidence_incident_snapshots"
        )
    }
    for snapshot_id, edge_event_id in connection.execute(
        "SELECT snapshot_id,edge_event_id FROM snapshots"
    ):
        incident_id = event_incidents.get(str(edge_event_id))
        if incident_id is not None:
            snapshot_incidents.setdefault(str(snapshot_id), incident_id)
    policy_ids, policy_revisions, policy_facilities = _policy_context(connection)
    legacy_audits, review_audits, review_incidents = _audit_context(connection)
    current_reviews = frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT revision.review_id FROM control_evidence_review_state AS state "
            "JOIN control_evidence_review_revisions AS revision "
            "ON revision.incident_id=state.incident_id "
            "AND revision.review_version=state.current_version"
        )
    )
    return ReceiptContext(
        _camera_ids(connection),
        clip_ids,
        current_reviews,
        event_incidents,
        label_incidents,
        legacy_audits,
        _media_targets(connection, clip_ids),
        policy_facilities,
        policy_ids,
        policy_revisions,
        review_audits,
        review_incidents,
        snapshot_incidents,
        rebuilt,
    )


__all__ = [
    "ReceiptContext",
    "build_receipt_context",
    "disposition",
    "require_map_target",
]
