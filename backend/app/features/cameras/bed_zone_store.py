"""Bed-zone projection stored on schema-18 camera rows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from pydantic import TypeAdapter, ValidationError

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.configuration import open_configuration_database, utc_now


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
        self._connection = open_configuration_database(self.path)

    @classmethod
    def from_env(cls) -> BedZoneStore:
        return cls(EDGE_DATABASE_PATH)

    def get(self, camera_id: str) -> BedZone | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT bed_polygon_json,bed_image_width,bed_image_height,bed_recognized_at "
                "FROM cameras WHERE camera_id=?",
                (camera_id,),
            ).fetchone()
        return None if row is None or row[0] is None else _row_to_bed_zone(row)

    def get_all(self) -> dict[str, BedZone]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT camera_id,bed_polygon_json,bed_image_width,bed_image_height,"
                "bed_recognized_at FROM cameras WHERE bed_polygon_json IS NOT NULL"
            ).fetchall()
        result: dict[str, BedZone] = {}
        for row in rows:
            bed_zone = _row_to_bed_zone(row[1:])
            if bed_zone is not None:
                result[str(row[0])] = bed_zone
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
        encoded = json.dumps(polygon, separators=(",", ":"))
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE cameras SET bed_polygon_json=?,bed_image_width=?,bed_image_height=?,"
                "bed_recognized_at=?,revision=revision+1,updated_at=? WHERE camera_id=?",
                (encoded, image_width, image_height, recognized_at, utc_now(), camera_id),
            )
        if cursor.rowcount != 1:
            raise sqlite3.IntegrityError("bed-zone camera does not exist")
        return BedZone(
            polygon=tuple((int(x), int(y)) for x, y in polygon),
            image_width=image_width,
            image_height=image_height,
            recognized_at=recognized_at,
        )

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE cameras SET bed_polygon_json=NULL,bed_image_width=NULL,"
                "bed_image_height=NULL,bed_recognized_at=NULL,revision=revision+1,updated_at=? "
                "WHERE camera_id=? AND bed_polygon_json IS NOT NULL",
                (utc_now(), camera_id),
            )
        return cursor.rowcount > 0


_BED_ZONE_ROW = TypeAdapter(tuple[str, int, int, str])


def _row_to_bed_zone(row: tuple[object, ...]) -> BedZone | None:
    try:
        polygon_json, image_width, image_height, recognized_at = _BED_ZONE_ROW.validate_python(row)
        raw_polygon = TypeAdapter(list[tuple[int, int]]).validate_json(polygon_json)
    except ValidationError:
        return None
    return BedZone(tuple(raw_polygon), image_width, image_height, recognized_at)


__all__ = ["BedZone", "BedZoneStore"]
