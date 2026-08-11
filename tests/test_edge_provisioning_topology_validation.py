from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest

from contracts.edge_provisioning_models import JsonRecord, JsonValue
from contracts.edge_provisioning_v1 import (
    ContractViolation,
    MachinePrincipal,
    TopologyCamera,
    TopologyFloor,
    TopologyRoom,
    TopologySnapshot,
    parse_topology_manifest,
    parse_topology_snapshot,
    serialize_topology_snapshot,
)

INSTALLATION_ID = "c72bd9a7-3e04-47ba-a8cd-a56e54f98152"
LEGACY_SPACE_ID = "a2222222-2222-4222-8222-222222222222"


def _records(value: JsonValue) -> list[JsonRecord]:
    assert isinstance(value, list)
    records: list[JsonRecord] = []
    for item in value:
        assert isinstance(item, dict)
        records.append(item)
    return records


def _root(body: JsonRecord) -> JsonRecord:
    return body


def _floor(body: JsonRecord) -> JsonRecord:
    return _records(body["floors"])[0]


def _room(body: JsonRecord) -> JsonRecord:
    return _records(_floor(body)["rooms"])[0]


def _camera(body: JsonRecord) -> JsonRecord:
    return _records(_room(body)["cameras"])[0]


_TARGETS: Final[dict[str, Callable[[JsonRecord], JsonRecord]]] = {
    "root": _root,
    "floor": _floor,
    "room": _room,
    "camera": _camera,
}


def test_snapshot_serializer_sorts_every_entity_kind_without_mutating_source() -> None:
    camera_b = TopologyCamera("camera-b", "Camera B")
    camera_a = TopologyCamera("camera-a", "Camera A")
    room_b = TopologyRoom("room-b", "Room B", "ROOM", 1, (camera_b, camera_a))
    room_a = TopologyRoom("room-a", "Room A", "ROOM", 1, ())
    floor_b = TopologyFloor("floor-b", "Floor B", 2, (room_b, room_a))
    floor_a = TopologyFloor("floor-a", "Floor A", 1, ())
    snapshot = TopologySnapshot(MachinePrincipal(INSTALLATION_ID, 1), 1, 0, (floor_b, floor_a))

    serialized = serialize_topology_snapshot(snapshot)

    floors = _records(serialized["floors"])
    assert [floor["edgeRef"] for floor in floors] == ["floor-a", "floor-b"]
    assert [room["edgeRef"] for room in _records(floors[1]["rooms"])] == ["room-a", "room-b"]
    rooms = _records(floors[1]["rooms"])
    assert [camera["edgeRef"] for camera in _records(rooms[1]["cameras"])] == [
        "camera-a",
        "camera-b",
    ]
    assert snapshot.floors == (floor_b, floor_a)
    assert room_b.cameras == (camera_b, camera_a)


def test_snapshot_parser_allows_same_ref_across_different_entity_kinds() -> None:
    body = _snapshot_body()
    floor = _records(body["floors"])[0]
    room = _records(floor["rooms"])[0]
    floor["edgeRef"] = "shared-ref"
    room["edgeRef"] = "shared-ref"
    _records(room["cameras"])[0]["edgeRef"] = "shared-ref"

    parsed = parse_topology_snapshot(body)

    assert parsed.floors[0].rooms[0].cameras[0].edge_ref == "shared-ref"


def test_snapshot_parser_rejects_duplicate_room_refs_across_floors() -> None:
    body = _snapshot_body()
    floor = _records(body["floors"])[0]
    duplicate = dict(floor)
    duplicate["edgeRef"] = "floor-3"
    body["floors"] = [floor, duplicate]

    with pytest.raises(ContractViolation):
        _ = parse_topology_snapshot(body)


def test_snapshot_parser_rejects_duplicate_camera_refs_across_rooms() -> None:
    body = _snapshot_body()
    floor = _floor(body)
    room = _room(body)
    duplicate = dict(room)
    duplicate["edgeRef"] = "room-202"
    floor["rooms"] = [room, duplicate]

    with pytest.raises(ContractViolation):
        _ = parse_topology_snapshot(body)


def test_manifest_parser_allows_same_ref_across_entity_kinds() -> None:
    floor_id = "f1111111-1111-4111-8111-111111111111"
    room_id = "a2222222-2222-4222-8222-222222222222"
    manifest: list[JsonRecord] = [
        dict(kind="FLOOR", edgeRef="shared", canonicalId=floor_id, parentCanonicalId=None),
        dict(kind="ROOM", edgeRef="shared", canonicalId=room_id, parentCanonicalId=floor_id),
        dict(
            kind="CAMERA",
            edgeRef="shared",
            canonicalId="b3333333-3333-4333-8333-333333333333",
            parentCanonicalId=room_id,
        ),
    ]

    assert len(parse_topology_manifest(manifest)) == 3


def test_manifest_parser_rejects_missing_parent_relationship() -> None:
    manifest: list[JsonRecord] = [
        dict(
            kind="ROOM",
            edgeRef="room-201",
            canonicalId="a2222222-2222-4222-8222-222222222222",
            parentCanonicalId="b3333333-3333-4333-8333-333333333333",
        )
    ]

    with pytest.raises(ContractViolation):
        _ = parse_topology_manifest(manifest)


def test_room_parser_accepts_defaults_and_optional_legacy_alias() -> None:
    body = _snapshot_body()
    room = _records(_records(body["floors"])[0]["rooms"])[0]
    del room["type"]
    del room["capacity"]
    room["legacyCanonicalSpaceId"] = LEGACY_SPACE_ID

    parsed = parse_topology_snapshot(body)
    serialized = serialize_topology_snapshot(parsed)

    parsed_room = parsed.floors[0].rooms[0]
    assert (parsed_room.room_type, parsed_room.capacity) == ("ROOM", 1)
    assert parsed_room.legacy_canonical_space_id == LEGACY_SPACE_ID
    output_room = _records(_records(serialized["floors"])[0]["rooms"])[0]
    assert output_room["legacyCanonicalSpaceId"] == LEGACY_SPACE_ID


@pytest.mark.parametrize("field", ["enrollmentGeneration", "clientRevision"])
def test_alias_parser_rejects_noninitial_generation_or_revision(field: str) -> None:
    body = _snapshot_body()
    _room(body)["legacyCanonicalSpaceId"] = LEGACY_SPACE_ID
    body[field] = 2

    with pytest.raises(ContractViolation):
        _ = parse_topology_snapshot(body)


@pytest.mark.parametrize(("generation", "revision"), [(2, 1), (1, 2)])
def test_alias_serializer_rejects_noninitial_generation_or_revision(
    generation: int,
    revision: int,
) -> None:
    room = TopologyRoom(
        "room-201",
        "Room 201",
        "ROOM",
        1,
        (TopologyCamera("camera-001", "Camera 1"),),
        LEGACY_SPACE_ID,
    )
    snapshot = TopologySnapshot(
        MachinePrincipal(INSTALLATION_ID, generation),
        revision,
        0,
        (TopologyFloor("floor-2", "Floor 2", 2, (room,)),),
    )

    with pytest.raises(ContractViolation):
        _ = serialize_topology_snapshot(snapshot)


@pytest.mark.parametrize(
    ("target_name", "field", "value"),
    [
        ("root", "schemaVersion", True),
        ("root", "edgeInstallationId", "not-a-uuid"),
        ("floor", "edgeRef", " invalid "),
        ("floor", "edgeRef", "a" * 65),
        ("floor", "name", ""),
        ("room", "name", "x" * 121),
        ("room", "type", "WARD"),
        ("room", "legacyCanonicalSpaceId", "invalid"),
        ("room", "legacyCanonicalSpaceId", None),
        ("camera", "label", "x" * 121),
        ("root", "clientRevision", True),
    ],
)
def test_snapshot_parser_rejects_invalid_frozen_formats(
    target_name: str,
    field: str,
    value: JsonValue,
) -> None:
    body = _snapshot_body()
    _TARGETS[target_name](body)[field] = value

    with pytest.raises(ContractViolation):
        _ = parse_topology_snapshot(body)


def _snapshot_body() -> JsonRecord:
    return serialize_topology_snapshot(
        TopologySnapshot(
            MachinePrincipal(INSTALLATION_ID, 1),
            1,
            0,
            (
                TopologyFloor(
                    "floor-2",
                    "Floor 2",
                    2,
                    (
                        TopologyRoom(
                            "room-201",
                            "Room 201",
                            "ROOM",
                            1,
                            (TopologyCamera("camera-001", "Camera 1"),),
                        ),
                    ),
                ),
            ),
        )
    )
