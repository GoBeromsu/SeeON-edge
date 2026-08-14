"""Schema compatibility policy shared by DDL-free API and worker runtimes."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from shared.edge_db.ownership import TABLE_FAMILIES


class EdgeDatabaseError(RuntimeError):
    """Base failure for the edge database foundation."""


@dataclass(slots=True)
class MigrationRequiredError(EdgeDatabaseError):
    found: int
    minimum: int

    def __str__(self) -> str:
        return f"edge database schema {self.found} requires migration to at least {self.minimum}"


@dataclass(slots=True)
class NewerSchemaError(EdgeDatabaseError):
    found: int
    maximum: int

    def __str__(self) -> str:
        return f"edge database schema {self.found} is newer than supported {self.maximum}"


class SchemaLedgerError(EdgeDatabaseError):
    """The version marker and machine-readable schema ledger disagree."""


class OwnershipMapError(EdgeDatabaseError):
    """The on-disk table-family writer map differs from the compiled contract."""


@dataclass(frozen=True, slots=True)
class SchemaCompatibility:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError("schema compatibility must be a non-negative inclusive range")


class CompatibilityDisposition(StrEnum):
    MIGRATION_REQUIRED = "migration_required"
    COMPATIBLE = "compatible"
    NEWER_SCHEMA = "newer_schema"


MigrationIdentity = tuple[int, str, str]

# Runtime-safe canonical identities are deliberately separate from schema.py's
# DDL. The migrator and tests reject drift between this registry and the actual
# migration objects; runtime packages never need to import migration statements.
CANONICAL_MIGRATION_LEDGER: Final[tuple[MigrationIdentity, ...]] = (
    (
        1,
        "edge_database_foundation",
        "a4b4147ac858c3bdc9c4438e14b8165258e6d032c93f588aeda0067e4fdb20a5",
    ),
    (
        2,
        "single_edge_application_schema",
        "201bbc542e31350e3fdb76c57972b0d2b2e15aa70a9614ea959e2fb078e6123f",
    ),
    (
        3,
        "initialize_clip_listing_generation",
        "61189f418332f22918587b0f72395caef504b77bb93da08bf3d8a4979f613e08",
    ),
    (
        4,
        "versioned_numeric_detection_policies",
        "50021b0d36d25508cfdc9931f58e5f27a409c7b9b7757f70623dec76bd99dc35",
    ),
    (
        5,
        "applied_runtime_provenance_manifests",
        "57bbf42982b9c02307b87b4fbe04de6d25779c91178c6ca55a30b5ffd8b8ed57",
    ),
    (
        6,
        "bounded_analysis_decision_traces",
        "083dbb6457739d46e36248df9a65d1c505a49596395e0ae27eb1b3e43a306819",
    ),
    (
        7,
        "trace_persistence_integrity_and_bounds",
        "0296bbe4fa10eb324a606051ad57d05cd7415ece47c1643960761ab70e1a670a",
    ),
    (
        8,
        "truthful_trace_component_states",
        "f2190fb59a685aa60a13e439d90787bc489b3a27754168a7dbecdf568456a93d",
    ),
    (
        9,
        "authoritative_central_evidence_records",
        "c698903ad864a78ef134a91084afe9cf91488bf5c53141b72e3c6465305c0319",
    ),
    (
        10,
        "versioned_operator_evidence_reviews",
        "e52017eb2f393d4d654f60f1d1c7ac16bb441e069d9da86ce9265d439ca8ddc0",
    ),
    (
        11,
        "exhaustive_evidence_unavailable_reasons",
        "0b00e127d29bfd60202e96cd242d8926902e6e0f9bd4cec01eaf5b75eacdf257",
    ),
    (
        12,
        "retire_legacy_system_test_operator_state",
        "5bb3aca4d85ef7f0f448747dd6b6903c768d69c9be157da12e26c3dea095ff92",
    ),
    (
        13,
        "canonical_overlay_scenes_and_derivatives",
        "f77b5154932bfb8ecbe0fd1dd63a906c0d6f90e3541c347fc049a37cbc32809d",
    ),
    (
        14,
        "still_video_derivative_lifecycle",
        "dcdf072bda75f38169ca796e9b1d7c66c0c2e8a507b54c1eca590431ba073845",
    ),
    (
        15,
        "internal_replay_qa",
        "c29d47b081fc8920e5b0ca77bff51913cb38d6dd8f353df9b847dccb9d95d375",
    ),
    (
        16,
        "live_runtime_clip_export_settings",
        "a2a515c71e1ca57d62c423ed88c95ddec6a1bd5d4c22c6e02d4494312f4b8270",
    ),
)

# Explicit rolling-version matrix. Both runtimes require the complete cutover schema.
CURRENT_SCHEMA_RANGE: Final = SchemaCompatibility(minimum=16, maximum=16)
COMPATIBILITY_MATRIX: Final = (
    ("database_version < minimum", CompatibilityDisposition.MIGRATION_REQUIRED),
    ("minimum <= database_version <= maximum", CompatibilityDisposition.COMPATIBLE),
    ("database_version > maximum", CompatibilityDisposition.NEWER_SCHEMA),
)


def classify_schema(
    version: int,
    compatibility: SchemaCompatibility = CURRENT_SCHEMA_RANGE,
) -> CompatibilityDisposition:
    if version < compatibility.minimum:
        return CompatibilityDisposition.MIGRATION_REQUIRED
    if version > compatibility.maximum:
        return CompatibilityDisposition.NEWER_SCHEMA
    return CompatibilityDisposition.COMPATIBLE


def verify_runtime_schema(
    connection: sqlite3.Connection,
    compatibility: SchemaCompatibility = CURRENT_SCHEMA_RANGE,
    *,
    expected_migrations: Sequence[MigrationIdentity] = CANONICAL_MIGRATION_LEDGER,
) -> int:
    """Verify version, ledger continuity, format identity, and writer ownership."""
    row = connection.execute("PRAGMA user_version").fetchone()
    version = 0 if row is None else int(row[0])
    disposition = classify_schema(version, compatibility)
    if disposition is CompatibilityDisposition.MIGRATION_REQUIRED:
        raise MigrationRequiredError(found=version, minimum=compatibility.minimum)
    if disposition is CompatibilityDisposition.NEWER_SCHEMA:
        raise NewerSchemaError(found=version, maximum=compatibility.maximum)

    try:
        format_row = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'format'"
        ).fetchone()
        ledger = connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        families = connection.execute(
            "SELECT prefix, writer, purpose FROM schema_table_families ORDER BY prefix"
        ).fetchall()
    except sqlite3.Error as error:
        raise SchemaLedgerError("edge database schema ledger is missing or unreadable") from error

    if format_row != ("seeon-edge-v1",):
        raise SchemaLedgerError("edge database format identity is invalid")
    expected_ledger = list(expected_migrations[:version])
    if ledger != expected_ledger:
        raise SchemaLedgerError("applied migration ledger differs from this migrator")

    expected_families = sorted(
        (family.prefix, family.writer.value, family.purpose) for family in TABLE_FAMILIES
    )
    if families != expected_families:
        raise OwnershipMapError("edge database table-family ownership map is invalid")
    return version


__all__ = [
    "CANONICAL_MIGRATION_LEDGER",
    "COMPATIBILITY_MATRIX",
    "CURRENT_SCHEMA_RANGE",
    "CompatibilityDisposition",
    "EdgeDatabaseError",
    "MigrationIdentity",
    "MigrationRequiredError",
    "NewerSchemaError",
    "OwnershipMapError",
    "SchemaCompatibility",
    "SchemaLedgerError",
    "classify_schema",
    "verify_runtime_schema",
]
