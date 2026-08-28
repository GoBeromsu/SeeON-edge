"""Externally prepared schema-18 fixtures for durable-authority tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app.edge_db.bootstrap import bootstrap_database

TS = "2026-08-24T00:00:00Z"


def prepare_compact_database(path: Path) -> Path:
    bootstrap_database(path)
    return path


def seed_camera(path: Path, camera_id: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO cameras("
            "camera_id,label,rtsp_url,normalized_stream_identity,mapping_state,"
            "never_connected,revision,created_at,updated_at) "
            "VALUES (?,?,?,?, 'UNMAPPED',1,1,?,?)",
            (camera_id, camera_id, f"rtsp://fixture/{camera_id}", f"fixture:{camera_id}", TS, TS),
        )


def seed_enrollment(
    path: Path,
    *,
    edge_installation_id: str,
    enrollment_generation: int,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO edge_site("
            "id,facility_code,client_installation_ref,facility_id,facility_token,"
            "edge_installation_id,enrollment_generation,enrollment_created_at,"
            "enrollment_updated_at,updated_at) VALUES (1,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET facility_code=excluded.facility_code,"
            "client_installation_ref=excluded.client_installation_ref,"
            "facility_id=excluded.facility_id,facility_token=excluded.facility_token,"
            "edge_installation_id=excluded.edge_installation_id,"
            "enrollment_generation=excluded.enrollment_generation,"
            "enrollment_created_at=excluded.enrollment_created_at,"
            "enrollment_updated_at=excluded.enrollment_updated_at,updated_at=excluded.updated_at",
            (
                "NH-7H2K9M4QXP",
                "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
                "facility-fixture",
                "token-fixture",
                edge_installation_id,
                enrollment_generation,
                TS,
                TS,
                TS,
            ),
        )
