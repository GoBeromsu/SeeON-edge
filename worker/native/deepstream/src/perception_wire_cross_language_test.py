#!/usr/bin/env python3
"""Decode the representative C++ perception payload with Python wire primitives."""

from __future__ import annotations

import struct
import subprocess
import sys
import uuid
from typing import cast

Scalar = int | float


class Reader:
    def __init__(self, payload: bytes) -> None:
        self.payload: bytes = payload
        self.offset: int = 0

    def take(self, size: int) -> bytes:
        value = self.payload[self.offset : self.offset + size]
        if len(value) != size:
            raise AssertionError("truncated payload")
        self.offset += size
        return value

    def unpack(self, pattern: str) -> tuple[Scalar, ...]:
        size = struct.calcsize(pattern)
        return cast(tuple[Scalar, ...], struct.unpack(pattern, self.take(size)))

    def text(self) -> str:
        (size,) = self.unpack("<H")
        return self.take(int(size)).decode("utf-8")


def identity(reader: Reader) -> tuple[str, str, int, int, int]:
    boot = str(uuid.UUID(bytes=reader.take(16)))
    camera = reader.text()
    epoch, pts, sequence = reader.unpack("<QQQ")
    return boot, camera, int(epoch), int(pts), int(sequence)


def main() -> None:
    encoded = subprocess.check_output(
        [sys.argv[1], "--emit-nonempty"], text=True
    ).strip()
    reader = Reader(bytes.fromhex(encoded))
    assert reader.take(4) == b"PFV1"
    expected_identity = ("12345678-1234-5678-1234-567812345678", "camera-a", 7, 123456, 11)
    assert identity(reader) == expected_identity
    assert reader.unpack("<BBBB") == (1, 1, 1, 1)

    (box_count,) = reader.unpack("<H")
    boxes: list[tuple[Scalar, ...]] = [
        reader.unpack("<iiiid") for _ in range(int(box_count))
    ]
    assert boxes == [(1, 2, 30, 40, 0.75), (5, 6, 50, 60, 0.5)]

    (pose_count,) = reader.unpack("<H")
    poses: list[list[tuple[Scalar, ...]]] = []
    for _ in range(int(pose_count)):
        (point_count,) = reader.unpack("<H")
        poses.append([reader.unpack("<iid") for _ in range(int(point_count))])
    assert poses == [[(3, 4, 0.9), (7, 8, 0.8)], [(10, 11, 0.7)]]

    (bed_count,) = reader.unpack("<H")
    beds: list[tuple[tuple[Scalar, ...], list[tuple[Scalar, ...]]]] = []
    for _ in range(int(bed_count)):
        box = reader.unpack("<iiiid")
        (point_count,) = reader.unpack("<H")
        polygon = [reader.unpack("<ii") for _ in range(int(point_count))]
        beds.append((box, polygon))
    assert beds == [((0, 0, 100, 80, 0.95), [(0, 0), (100, 0), (100, 80)])]

    assert identity(reader) == expected_identity
    assert reader.text() == "legacy-greedy-bbox-iou.v1"
    assert reader.text() == "person_box"
    (pair_count,) = reader.unpack("<H")
    pairs = [reader.unpack("<qH") for _ in range(int(pair_count))]
    assert pairs == [(41, 0), (42, 1)]
    assert reader.offset == len(reader.payload)


if __name__ == "__main__":
    main()
