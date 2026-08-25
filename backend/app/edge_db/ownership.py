"""Table-family writer ownership for the one local edge database."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from backend.app.edge_db.compact_schema import COMPACT_API_TABLES, COMPACT_APPLICATION_TABLES


class Writer(StrEnum):
    API = "api"
    MIGRATOR = "migrator"


@dataclass(frozen=True, slots=True)
class TableFamily:
    prefix: str
    writer: Writer
    purpose: str


API_LEGACY_TABLES: Final = frozenset(
    {
        "audit",
        "camera_bed_zone",
        "camera_registry",
        "camera_topology_cameras",
        "camera_topology_floors",
        "camera_topology_rooms",
        "cameras",
        "clip_listing_generation",
        "clip_listing_rows",
        "clip_listing_summary",
        "clip_listing_thumbnails",
        "clip_storage_location",
        "clips",
        "connection_settings",
        "connection_store_migrations",
        "credentials",
        "detection_settings",
        "edge_topology_confirmation_preview",
        "edge_topology_sync_state",
        "events",
        "labels",
        "runtime_latency",
        "runtime_settings",
        "snapshots",
        "topology_dirty",
    }
)
APPLICATION_LEGACY_TABLES: Final = frozenset(
    {
        "clip_events",
        "config_current",
        "config_history",
        "evidence_clips",
        "evidence_events",
        "faults",
    }
)

TABLE_FAMILIES: Final = (
    TableFamily("control_", Writer.API, "operator and deployment control state"),
    TableFamily("qa_", Writer.API, "internal replay and QA state"),
    TableFamily("runtime_", Writer.API, "applied worker runtime state"),
    TableFamily("evidence_", Writer.API, "event and evidence state"),
    TableFamily("derivative_", Writer.API, "derived media state"),
    TableFamily("schema_", Writer.MIGRATOR, "schema ledger and ownership metadata"),
)


def writer_for_table(table: str) -> Writer | None:
    """Return the declared writer for *table*, including released imported names."""
    if table in COMPACT_API_TABLES:
        return Writer.API
    if table in API_LEGACY_TABLES:
        return Writer.API
    if table in APPLICATION_LEGACY_TABLES:
        return Writer.API
    for family in TABLE_FAMILIES:
        if table.startswith(family.prefix):
            return family.writer
    return None


__all__ = [
    "API_LEGACY_TABLES",
    "APPLICATION_LEGACY_TABLES",
    "COMPACT_API_TABLES",
    "COMPACT_APPLICATION_TABLES",
    "TABLE_FAMILIES",
    "TableFamily",
    "Writer",
    "writer_for_table",
]
