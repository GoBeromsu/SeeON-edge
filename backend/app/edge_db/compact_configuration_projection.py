"""Project v17 site, location, and camera authorities into schema 18."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import JsonValue, TypeAdapter

_TS = "2026-08-24T00:00:00Z"
_CAMERA_LIST = TypeAdapter(list[dict[str, JsonValue]])


def _copy_locations(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    for edge_ref, name, order_index in source.execute(
        "SELECT edge_ref,name,order_index FROM camera_topology_floors ORDER BY edge_ref"
    ):
        target.execute(
            "INSERT INTO locations VALUES (?,?,?,?,?,?,?,?,?,?)",
            (edge_ref, "FLOOR", None, None, name, order_index, None, None, _TS, _TS),
        )
    for row in source.execute(
        "SELECT edge_ref,floor_edge_ref,name,capacity,legacy_canonical_space_id "
        "FROM camera_topology_rooms ORDER BY edge_ref"
    ):
        target.execute(
            "INSERT INTO locations VALUES (?,?,?,?,?,?,?,?,?,?)",
            (row[0], "ROOM", row[1], "FLOOR", row[2], 0, row[3], row[4], _TS, _TS),
        )


def _stream_identity(rtsp_url: str) -> str:
    parsed = urlsplit(rtsp_url.strip())
    hostname = "" if parsed.hostname is None else parsed.hostname.lower()
    port = "" if parsed.port is None else f":{parsed.port}"
    netloc = hostname + port
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, parsed.query, ""))


def _copy_cameras(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    registry = source.execute("SELECT cameras_json FROM camera_registry WHERE id=1").fetchone()
    if registry is None:
        return
    cameras = _CAMERA_LIST.validate_json(str(registry[0]))
    topology = {
        str(row[0]): (row[1], row[2])
        for row in source.execute(
            "SELECT camera_id,edge_ref,room_edge_ref FROM camera_topology_cameras"
        )
    }
    zones = {
        str(row[0]): row[1:]
        for row in source.execute(
            "SELECT camera_id,polygon_json,image_width,image_height,recognized_at "
            "FROM camera_bed_zone"
        )
    }
    for record in sorted(cameras, key=lambda camera: str(camera.get("id", ""))):
        camera_id = str(record.get("id", ""))
        label = str(record.get("label", ""))
        rtsp_url = str(record.get("rtsp_url", ""))
        if not camera_id or not label or not rtsp_url:
            raise sqlite3.DatabaseError("camera registry contains incomplete current record")
        backend_id = record.get("backend_camera_id")
        backend_camera_id = backend_id if isinstance(backend_id, str) and backend_id else None
        edge_ref, room_ref = topology.get(camera_id, (None, None))
        mapping_state = (
            "PENDING"
            if record.get("mapping_pending") is True
            else "MAPPED"
            if backend_camera_id is not None
            else "UNMAPPED"
        )
        zone = zones.get(camera_id, (None, None, None, None))
        floor = record.get("floor")
        target.execute(
            "INSERT INTO cameras VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                camera_id,
                backend_camera_id,
                label,
                rtsp_url,
                _stream_identity(rtsp_url),
                record.get("space_id"),
                room_ref,
                "ROOM" if room_ref is not None else None,
                edge_ref,
                mapping_state,
                record.get("decode_backend"),
                str(floor) if isinstance(floor, int) and not isinstance(floor, bool) else None,
                int(record.get("never_connected") is not False),
                record.get("last_probed_at"),
                record.get("last_ok_at"),
                zone[0],
                zone[1],
                zone[2],
                zone[3],
                1,
                record.get("created_at") or _TS,
                record.get("updated_at") or record.get("created_at") or _TS,
            ),
        )


def _copy_edge_site(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    target.execute("INSERT INTO edge_site (id,updated_at) VALUES (1,?)", (_TS,))
    registry = source.execute("SELECT registry_version FROM camera_registry WHERE id=1").fetchone()
    if registry is not None:
        target.execute("UPDATE edge_site SET registry_version=? WHERE id=1", registry)
    runtime = source.execute(
        "SELECT clip_export_enabled,version FROM runtime_settings WHERE id=1"
    ).fetchone()
    if runtime is not None:
        target.execute(
            "UPDATE edge_site SET clip_export_enabled=?,runtime_settings_version=? WHERE id=1",
            runtime,
        )
    for domain, enabled, mode, start, end in source.execute(
        "SELECT domain,on_flag,mode,start_time,end_time FROM detection_settings"
    ):
        if domain not in {"fall", "bed-exit"}:
            continue
        prefix = "fall" if domain == "fall" else "bed_exit"
        target.execute(
            f"UPDATE edge_site SET {prefix}_on=?,{prefix}_mode=?,"
            f"{prefix}_start_time=?,{prefix}_end_time=? WHERE id=1",
            (enabled, mode, start, end),
        )
    connection = source.execute(
        "SELECT facility_code,client_installation_ref,facility_id,facility_token,"
        "edge_installation_id,enrollment_generation,enrollment_created_at,"
        "enrollment_updated_at,updated_at FROM connection_settings WHERE id=1"
    ).fetchone()
    if connection is not None and all(value is not None for value in connection[:8]):
        target.execute(
            "UPDATE edge_site SET facility_code=?,client_installation_ref=?,facility_id=?,"
            "facility_token=?,edge_installation_id=?,enrollment_generation=?,"
            "enrollment_created_at=?,enrollment_updated_at=?,updated_at=? WHERE id=1",
            connection,
        )
    selected = source.execute(
        "SELECT selected_path FROM clip_storage_location WHERE id=1"
    ).fetchone()
    if selected is not None:
        path = Path(str(selected[0]))
        if path.is_absolute() or ".." in path.parts:
            raise sqlite3.DatabaseError("clip store selection is not a contained subdirectory")
        target.execute("UPDATE edge_site SET clip_store_subdir=? WHERE id=1", (path.as_posix(),))


def project_configuration(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    """Project canonical current configuration in FK-safe order."""
    _copy_edge_site(source, target)
    _copy_locations(source, target)
    _copy_cameras(source, target)


__all__ = ["project_configuration"]
