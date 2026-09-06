"""Canonical multi-region bed zones stored on camera rows."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import ConfigDict, TypeAdapter, ValidationError

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.configuration import open_configuration_database, utc_now
from backend.app.edge_db.connection import write_transaction

BedZoneOrigin = Literal["manual", "model"]
BedZoneWriteHook = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True, slots=True)
class BedZoneRegion:
    id: str
    polygon: tuple[tuple[int, int], ...]
    origin: BedZoneOrigin

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "polygon": [[x, y] for x, y in self.polygon],
            "origin": self.origin,
        }


@dataclass(frozen=True, slots=True)
class BedZone:
    regions: tuple[BedZoneRegion, ...]
    image_width: int
    image_height: int
    recognized_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "regions": [region.as_dict() for region in self.regions],
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

    def camera_exists(self, camera_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM cameras WHERE camera_id=?",
                (camera_id,),
            ).fetchone()
        return row is not None

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
        regions: Sequence[BedZoneRegion],
        image_width: int,
        image_height: int,
        recognized_at: str,
        after_write: BedZoneWriteHook | None = None,
    ) -> BedZone:
        bed_zone, encoded = validate_bed_zone(
            regions,
            image_width=image_width,
            image_height=image_height,
            recognized_at=recognized_at,
        )
        with self._lock, write_transaction(self._connection):
            cursor = self._connection.execute(
                "UPDATE cameras SET bed_polygon_json=?,bed_image_width=?,bed_image_height=?,"
                "bed_recognized_at=?,revision=revision+1,updated_at=? WHERE camera_id=?",
                (encoded, image_width, image_height, recognized_at, utc_now(), camera_id),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError("bed-zone camera does not exist")
            if after_write is not None:
                after_write(self._connection)
        return bed_zone

    def delete(
        self,
        camera_id: str,
        *,
        after_write: BedZoneWriteHook | None = None,
    ) -> bool:
        with self._lock, write_transaction(self._connection):
            cursor = self._connection.execute(
                "UPDATE cameras SET bed_polygon_json=NULL,bed_image_width=NULL,"
                "bed_image_height=NULL,bed_recognized_at=NULL,revision=revision+1,updated_at=? "
                "WHERE camera_id=? AND bed_polygon_json IS NOT NULL",
                (utc_now(), camera_id),
            )
            changed = cursor.rowcount > 0
            if changed and after_write is not None:
                after_write(self._connection)
        return changed


_BED_ZONE_ROW = TypeAdapter(tuple[str, int, int, str], config=ConfigDict(strict=True))
_REGIONS_JSON = TypeAdapter(list[dict[str, object]])
_MAX_REGIONS_JSON_BYTES = 4096


def validate_bed_zone(
    regions: Sequence[BedZoneRegion],
    *,
    image_width: int,
    image_height: int,
    recognized_at: str,
) -> tuple[BedZone, str]:
    if (
        not isinstance(image_width, int)
        or not isinstance(image_height, int)
        or isinstance(image_width, bool)
        or isinstance(image_height, bool)
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("bed-zone image dimensions must be positive integers")
    if not isinstance(recognized_at, str) or not recognized_at:
        raise ValueError("bed-zone recognized_at must not be empty")
    if len(regions) > 8:
        raise ValueError("bed-zone may contain at most 8 regions")

    ids: set[str] = set()
    normalized: list[BedZoneRegion] = []
    for region in regions:
        if not isinstance(region, BedZoneRegion):
            raise TypeError("bed-zone region has invalid shape")
        if not isinstance(region.id, str) or not region.id or len(region.id) > 64:
            raise ValueError("bed-zone region id must contain 1 to 64 characters")
        if region.id in ids:
            raise ValueError("bed-zone region ids must be distinct")
        ids.add(region.id)
        if region.origin not in ("manual", "model"):
            raise ValueError("bed-zone region origin is invalid")
        _validate_polygon(region.polygon, image_width=image_width, image_height=image_height)
        normalized.append(region)

    bed_zone = BedZone(tuple(normalized), image_width, image_height, recognized_at)
    encoded = json.dumps(
        [region.as_dict() for region in bed_zone.regions],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > _MAX_REGIONS_JSON_BYTES:
        raise ValueError("encoded bed-zone regions exceed 4096 bytes")
    return bed_zone, encoded


def _validate_polygon(
    polygon: Sequence[tuple[int, int]],
    *,
    image_width: int,
    image_height: int,
) -> None:
    if not 3 <= len(polygon) <= 16:
        raise ValueError("bed-zone polygon must contain 3 to 16 vertices")
    for point in polygon:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or isinstance(point[0], bool)
            or isinstance(point[1], bool)
            or not isinstance(point[0], int)
            or not isinstance(point[1], int)
        ):
            raise ValueError("bed-zone polygon coordinates must be integers")
        if not 0 <= point[0] < image_width or not 0 <= point[1] < image_height:
            raise ValueError("bed-zone polygon coordinate is outside the image")
    if len(set(polygon)) != len(polygon):
        raise ValueError("bed-zone polygon vertices must be distinct")

    twice_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
    )
    if twice_area == 0:
        raise ValueError("bed-zone polygon must be nondegenerate")
    if _self_intersects(polygon):
        raise ValueError("bed-zone polygon must not self-intersect")


def _self_intersects(polygon: Sequence[tuple[int, int]]) -> bool:
    edge_count = len(polygon)
    for first in range(edge_count):
        a = polygon[first]
        b = polygon[(first + 1) % edge_count]
        for second in range(first + 1, edge_count):
            if second in {first, (first + 1) % edge_count}:
                continue
            if first == 0 and second == edge_count - 1:
                continue
            c = polygon[second]
            d = polygon[(second + 1) % edge_count]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _segments_intersect(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    d: tuple[int, int],
) -> bool:
    def orientation(first: tuple[int, int], second: tuple[int, int], third: tuple[int, int]) -> int:
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
            third[0] - first[0]
        )

    def on_segment(first: tuple[int, int], second: tuple[int, int], point: tuple[int, int]) -> bool:
        return min(first[0], second[0]) <= point[0] <= max(first[0], second[0]) and min(
            first[1], second[1]
        ) <= point[1] <= max(first[1], second[1])

    orientations = (
        orientation(a, b, c),
        orientation(a, b, d),
        orientation(c, d, a),
        orientation(c, d, b),
    )
    if orientations[0] == 0 and on_segment(a, b, c):
        return True
    if orientations[1] == 0 and on_segment(a, b, d):
        return True
    if orientations[2] == 0 and on_segment(c, d, a):
        return True
    if orientations[3] == 0 and on_segment(c, d, b):
        return True
    return (orientations[0] > 0) != (orientations[1] > 0) and (orientations[2] > 0) != (
        orientations[3] > 0
    )


def _row_to_bed_zone(row: tuple[object, ...]) -> BedZone | None:
    try:
        regions_json, image_width, image_height, recognized_at = _BED_ZONE_ROW.validate_python(row)
        raw_regions = _REGIONS_JSON.validate_json(regions_json)
        regions = tuple(_region_from_json(raw) for raw in raw_regions)
        bed_zone, _ = validate_bed_zone(
            regions,
            image_width=image_width,
            image_height=image_height,
            recognized_at=recognized_at,
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        return None
    return bed_zone


def _region_from_json(raw: dict[str, object]) -> BedZoneRegion:
    if set(raw) != {"id", "polygon", "origin"}:
        raise ValueError("bed-zone region has invalid fields")
    region_id = raw["id"]
    origin = raw["origin"]
    raw_polygon = raw["polygon"]
    if not isinstance(region_id, str) or origin not in ("manual", "model"):
        raise ValueError("bed-zone region has invalid fields")
    if not isinstance(raw_polygon, list):
        raise TypeError("bed-zone polygon has invalid shape")
    polygon: list[tuple[int, int]] = []
    for point in raw_polygon:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("bed-zone polygon has invalid shape")
        x, y = point
        if (
            not isinstance(x, int)
            or isinstance(x, bool)
            or not isinstance(y, int)
            or isinstance(y, bool)
        ):
            raise TypeError("bed-zone polygon coordinates must be integers")
        polygon.append((x, y))
    return BedZoneRegion(id=region_id, polygon=tuple(polygon), origin=origin)


__all__ = [
    "BedZone",
    "BedZoneOrigin",
    "BedZoneRegion",
    "BedZoneStore",
    "validate_bed_zone",
]
