from __future__ import annotations

from typing import Final

CREATE_TOPOLOGY_FLOORS: Final = (
    "CREATE TABLE IF NOT EXISTS camera_topology_floors ("
    "edge_ref TEXT PRIMARY KEY, name TEXT NOT NULL, order_index INTEGER NOT NULL "
    "CHECK (order_index >= 0)) STRICT"
)
CREATE_TOPOLOGY_ROOMS: Final = (
    "CREATE TABLE IF NOT EXISTS camera_topology_rooms ("
    "edge_ref TEXT PRIMARY KEY, floor_edge_ref TEXT NOT NULL, name TEXT NOT NULL, "
    "room_type TEXT NOT NULL CHECK (room_type = 'ROOM'), capacity INTEGER NOT NULL "
    "CHECK (capacity > 0), legacy_canonical_space_id TEXT UNIQUE, "
    "FOREIGN KEY (floor_edge_ref) REFERENCES camera_topology_floors(edge_ref) "
    "ON UPDATE RESTRICT ON DELETE RESTRICT) STRICT"
)
CREATE_TOPOLOGY_CAMERAS: Final = (
    "CREATE TABLE IF NOT EXISTS camera_topology_cameras ("
    "camera_id TEXT PRIMARY KEY, edge_ref TEXT NOT NULL UNIQUE, "
    "room_edge_ref TEXT NOT NULL UNIQUE, "
    "FOREIGN KEY (room_edge_ref) REFERENCES camera_topology_rooms(edge_ref) "
    "ON UPDATE RESTRICT ON DELETE RESTRICT) STRICT"
)
CREATE_TOPOLOGY_DIRTY: Final = (
    "CREATE TABLE IF NOT EXISTS topology_dirty (id INTEGER PRIMARY KEY CHECK (id = 1), "
    "registry_version INTEGER NOT NULL CHECK (registry_version >= 1), "
    "created_at TEXT NOT NULL) STRICT"
)
TOPOLOGY_SCHEMA: Final = (
    CREATE_TOPOLOGY_FLOORS,
    CREATE_TOPOLOGY_ROOMS,
    CREATE_TOPOLOGY_CAMERAS,
    CREATE_TOPOLOGY_DIRTY,
)

__all__ = ["TOPOLOGY_SCHEMA"]
