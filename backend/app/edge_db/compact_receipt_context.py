"""Deterministic fan-out context for source-row receipts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter

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
_REBUILD_TABLES: Final = frozenset(
    {
        "clip_listing_generation",
        "clip_listing_rows",
        "clip_listing_summary",
        "clip_listing_thumbnails",
    }
)


_CAMERA_LIST = TypeAdapter(list[dict[str, JsonValue]])


@dataclass(frozen=True, slots=True)
class ReceiptContext:
    camera_ids: tuple[str, ...]
    label_incidents: dict[str, str]
    policy_ids: dict[str, int]
    rebuilt_clip_ids: tuple[str, ...]


def disposition(table: str, targets: list[str]) -> tuple[str, str | None]:
    if table in _MAP_TABLES and targets:
        return "MAP", None
    if table in _REBUILD_TABLES:
        return "REBUILD", "filesystem_manifest_authority"
    reasons = {
        "events": "catalog_duplicate_retired",
        "topology_dirty": "derived_dirty_marker_retired",
    }
    return "NONE", reasons.get(table, "source_archive_only")


def build_receipt_context(
    connection: sqlite3.Connection, rebuilt: tuple[str, ...]
) -> ReceiptContext:
    registry = connection.execute("SELECT cameras_json FROM camera_registry WHERE id=1").fetchone()
    cameras = (
        ()
        if registry is None
        else tuple(
            str(item["id"])
            for item in _CAMERA_LIST.validate_json(str(registry[0]))
            if isinstance(item.get("id"), str)
        )
    )
    rows = connection.execute(
        "SELECT activation_id,facility_id,camera_id,module_id,activation_generation "
        "FROM control_detection_policy_activations ORDER BY activation_generation,activation_id"
    ).fetchall()
    current: dict[tuple[str, str | None, str], str] = {}
    for activation_id, facility_id, camera_id, module_id, _generation in rows:
        current[(str(facility_id), camera_id, str(module_id))] = str(activation_id)
    label_incidents = {
        str(clip_id): str(incident_id)
        for incident_id, clip_id in connection.execute(
            "SELECT incident_id,clip_id FROM evidence_primary_clips"
        )
    }
    policy_ids = {
        activation_id: index
        for index, (_key, activation_id) in enumerate(sorted(current.items()), start=1)
    }
    return ReceiptContext(cameras, label_incidents, policy_ids, rebuilt)


__all__ = ["ReceiptContext", "build_receipt_context", "disposition"]
