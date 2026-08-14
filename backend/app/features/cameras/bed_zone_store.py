"""Persisted per-camera bed-zone polygon storage for ml-api.

Backs the on-demand bed-zone recognition endpoint (see
``bed_zone_router.recognize_bed_zone``): once an operator triggers
recognition and the worker returns a polygon, it is persisted here so it
survives restarts and is fed back to the worker (via
``cameras.router.worker_config_snapshot``) as the authoritative bed region
for bed-exit detection, instead of relying on live per-frame segmentation.

Storage lives in its own ``camera_bed_zone`` table of the shared
``catalog.sqlite3`` database (see ``sqlite_bootstrap.connect_catalog_store``),
distinct from ``CameraRegistryStore``'s single-row JSON-blob ``camera_registry``
table: a bed zone is naturally keyed one-row-per-camera, not a single blob, so
a real primary key (rather than read-modify-write-whole-blob) is the simpler
and more concurrency-friendly shape here.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TypeAlias

from pydantic import TypeAdapter, ValidationError

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from shared.edge_db import EDGE_DATABASE_PATH

_CREATE_BED_ZONE_TABLE = (
    "CREATE TABLE IF NOT EXISTS camera_bed_zone ("
    "camera_id TEXT PRIMARY KEY, "
    "polygon_json TEXT NOT NULL, "
    "image_width INTEGER NOT NULL, "
    "image_height INTEGER NOT NULL, "
    "recognized_at TEXT NOT NULL) STRICT"
)


@dataclass(frozen=True, slots=True)
class BedZone:
    polygon: tuple[tuple[int, int], ...]
    image_width: int
    image_height: int
    recognized_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "polygon": [[x, y] for x, y in self.polygon],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "recognized_at": self.recognized_at,
        }


class BedZoneStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def from_env(cls) -> BedZoneStore:
        return cls(EDGE_DATABASE_PATH)

    def get(self, camera_id: str) -> BedZone | None:
        with self._lock:
            connection = self._connect()
            row = connection.execute(
                "SELECT polygon_json, image_width, image_height, recognized_at "
                "FROM camera_bed_zone WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_bed_zone(row)

    def get_all(self) -> dict[str, BedZone]:
        with self._lock:
            connection = self._connect()
            rows = connection.execute(
                "SELECT camera_id, polygon_json, image_width, image_height, recognized_at "
                "FROM camera_bed_zone"
            ).fetchall()
        result: dict[str, BedZone] = {}
        for camera_id, *rest in rows:
            bed_zone = _row_to_bed_zone(tuple(rest))
            if bed_zone is not None:
                result[str(camera_id)] = bed_zone
        return result

    def put(
        self,
        camera_id: str,
        *,
        polygon: list[list[int]],
        image_width: int,
        image_height: int,
        recognized_at: str,
    ) -> BedZone:
        polygon_json = json.dumps(polygon, separators=(",", ":"))
        with self._lock:
            connection = self._connect()
            connection.execute(
                "INSERT INTO camera_bed_zone "
                "(camera_id, polygon_json, image_width, image_height, recognized_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(camera_id) DO UPDATE SET "
                "polygon_json = excluded.polygon_json, "
                "image_width = excluded.image_width, "
                "image_height = excluded.image_height, "
                "recognized_at = excluded.recognized_at",
                (camera_id, polygon_json, image_width, image_height, recognized_at),
            )
        return BedZone(
            polygon=tuple((int(x), int(y)) for x, y in polygon),
            image_width=image_width,
            image_height=image_height,
            recognized_at=recognized_at,
        )

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            connection = self._connect()
            cursor = connection.execute(
                "DELETE FROM camera_bed_zone WHERE camera_id = ?", (camera_id,)
            )
            return cursor.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = connect_catalog_store(self.path, (_CREATE_BED_ZONE_TABLE,))
        return self._connection


BedZoneRow: TypeAlias = tuple[str, int, int, str]
_BED_ZONE_ROW = TypeAdapter(BedZoneRow)


def _row_to_bed_zone(row: object) -> BedZone | None:
    try:
        polygon_json, image_width, image_height, recognized_at = _BED_ZONE_ROW.validate_python(row)
    except ValidationError:
        return None
    try:
        raw_polygon = json.loads(polygon_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(raw_polygon, list):
        return None
    polygon = _parse_polygon_points(raw_polygon)
    if polygon is None:
        return None
    return BedZone(
        polygon=polygon,
        image_width=image_width,
        image_height=image_height,
        recognized_at=recognized_at,
    )


def _parse_polygon_points(raw_polygon: list[object]) -> tuple[tuple[int, int], ...] | None:
    points: list[tuple[int, int]] = []
    for point in raw_polygon:
        parsed = _parse_point(point)
        if parsed is None:
            return None
        points.append(parsed)
    return tuple(points)


def _parse_point(point: object) -> tuple[int, int] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    x = _as_finite_int(point[0])
    y = _as_finite_int(point[1])
    if x is None or y is None:
        return None
    return x, y


def _as_finite_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_int = int(value)
    if float(value) != float(as_int):
        return None
    return as_int


__all__ = ["BedZone", "BedZoneStore"]
