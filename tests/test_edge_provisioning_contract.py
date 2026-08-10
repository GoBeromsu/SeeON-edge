from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue as PydanticJsonValue
from pydantic import TypeAdapter

from backend.app.shared.backend_mapping import BackendCameraMapper
from contracts.edge_provisioning_models import JsonRecord, JsonValue
from contracts.edge_provisioning_v1 import (
    CANONICAL_SCHEMA_VERSION,
    ContractViolation,
    EnrollmentVerification,
    MachinePrincipal,
    TopologyCamera,
    TopologyConfirmation,
    TopologyFloor,
    TopologyRoom,
    TopologySnapshot,
    parse_enrollment_verification,
    parse_enrollment_verification_result,
    parse_error_envelope,
    parse_machine_principal,
    parse_topology_confirmation,
    parse_topology_manifest,
    parse_topology_snapshot,
    parse_topology_success_envelope,
    serialize_enrollment_verification,
    serialize_topology_confirmation,
    serialize_topology_snapshot,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
FIXTURE_PATH: Final = REPO_ROOT / "contracts/edge-provisioning-v1/contract-fixtures.json"
PROVENANCE_PATH: Final = REPO_ROOT / "contracts/edge-provisioning-v1/provenance.json"
SOURCE_COMMIT: Final = "17b4f0a24bcd806c1f8c8f71d9576309e25bd87e"
CANONICAL_DIGEST: Final = "5e64609fdeba5968864b9c78807ae71864a54143c70bffb82924788957c3f2ff"
JSON_RECORD_ADAPTER: Final = TypeAdapter(dict[str, PydanticJsonValue])


def test_vendored_fixture_has_declared_source_and_semantic_jcs_digest() -> None:
    provenance = JSON_RECORD_ADAPTER.validate_json(PROVENANCE_PATH.read_bytes())
    document = JSON_RECORD_ADAPTER.validate_json(FIXTURE_PATH.read_bytes())

    assert provenance["sourceRepository"] == "eldercare-fall-ai"
    assert provenance["sourcePath"] == (
        "backend/test/fixtures/edge-provisioning-v1/contract-fixtures.json"
    )
    assert provenance["sourceCommit"] == SOURCE_COMMIT
    assert provenance["canonicalSha256"] == CANONICAL_DIGEST
    metadata = document["metadata"]
    assert isinstance(metadata, dict)
    assert metadata.pop("canonicalSha256") == CANONICAL_DIGEST
    canonical = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == CANONICAL_DIGEST


def test_enrollment_verification_serializer_is_frozen_to_v1_fields() -> None:
    request = EnrollmentVerification(
        "NH-7H2K9M4QXP", "8b0f5ba2-d359-4d8e-948f-e386ac40c347"
    )

    serialized = serialize_enrollment_verification(request)
    assert serialized == {
        "schemaVersion": CANONICAL_SCHEMA_VERSION,
        "facilityCode": "NH-7H2K9M4QXP",
        "clientInstallationRef": "8b0f5ba2-d359-4d8e-948f-e386ac40c347",
    }
    assert parse_enrollment_verification(serialized) == request
    response = parse_enrollment_verification_result(
        {
            "schemaVersion": 1,
            "edgeInstallationId": "c72bd9a7-3e04-47ba-a8cd-a56e54f98152",
            "enrollmentGeneration": 1,
            "facility": {"id": "a5ff4ed1-7e63-4a4f-9ef0-42e807d74a64", "displayName": "Test"},
            "serverRevision": 0,
        }
    )
    assert response.facility.display_name == "Test"


def test_topology_snapshot_round_trips_without_facility_or_secret_fields() -> None:
    snapshot = _snapshot()

    serialized = serialize_topology_snapshot(snapshot)
    parsed = parse_topology_snapshot(serialized)

    assert parsed == snapshot
    expected = "schemaVersion edgeInstallationId enrollmentGeneration clientRevision"
    assert set(serialized) == set(f"{expected} expectedServerRevision floors".split())
    serialized_text = json.dumps(serialized).lower()
    assert all(term not in serialized_text for term in ("facility", "rtsp", "password", "secret"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("facilityId", "synthetic-facility"),
        ("rtspUrl", "rtsp://forbidden"),
        ("cameraPassword", "forbidden"),
        ("secret", "forbidden"),
    ],
)
def test_topology_snapshot_parser_rejects_forbidden_or_unknown_fields(
    field: str, value: JsonValue
) -> None:
    body = dict(serialize_topology_snapshot(_snapshot()))
    body[field] = value

    with pytest.raises(ContractViolation):
        _ = parse_topology_snapshot(body)


def test_topology_snapshot_parser_rejects_duplicate_refs() -> None:
    body = serialize_topology_snapshot(_snapshot())
    floors = body["floors"]
    assert isinstance(floors, list)
    floors.append(floors[0])

    with pytest.raises(ContractViolation):
        _ = parse_topology_snapshot(body)


def test_topology_snapshot_parser_rejects_multiple_cameras_per_room() -> None:
    body = serialize_topology_snapshot(_snapshot())
    floors = body["floors"]
    assert isinstance(floors, list)
    floor = floors[0]
    assert isinstance(floor, dict)
    rooms = floor["rooms"]
    assert isinstance(rooms, list)
    room = rooms[0]
    assert isinstance(room, dict)
    cameras = room["cameras"]
    assert isinstance(cameras, list)
    cameras.append({"edgeRef": "camera-002", "label": "Camera 2"})

    with pytest.raises(ContractViolation):
        _ = parse_topology_snapshot(body)


@pytest.mark.parametrize("field, value", [("clientRevision", 0), ("expectedServerRevision", -1)])
def test_topology_snapshot_parser_rejects_invalid_revisions(field: str, value: int) -> None:
    body = dict(serialize_topology_snapshot(_snapshot()))
    body[field] = value

    with pytest.raises(ContractViolation):
        _ = parse_topology_snapshot(body)


def test_topology_confirmation_serializer_uses_only_confirmation_state() -> None:
    confirmation = TopologyConfirmation(
        "0197f671-3a31-7a6c-a6e4-83ed412de81b", "a" * 64, 1
    )

    serialized = serialize_topology_confirmation(confirmation)

    assert serialized == {
        "schemaVersion": 1,
        "confirmationId": "0197f671-3a31-7a6c-a6e4-83ed412de81b",
        "digest": "a" * 64,
        "expectedServerRevision": 1,
    }
    assert parse_topology_confirmation(serialized) == confirmation


def test_topology_success_envelope_parser_accepts_snapshot_and_confirmation_shapes() -> None:
    response: JsonRecord = {
        "schemaVersion": 1,
        "snapshotId": "0197f671-3a31-7a6c-a6e4-83ed412de81a",
        "clientRevision": 1,
        "serverRevision": 1,
        "result": {
            "floors": {"created": 1, "updated": 0, "unchanged": 0},
            "rooms": {"created": 1, "updated": 0, "unchanged": 0},
            "cameras": {"created": 1, "updated": 0, "unchanged": 0},
        },
        "omissions": {
            "confirmationId": "0197f671-3a31-7a6c-a6e4-83ed412de81b",
            "digest": "a" * 64,
            "expiresAt": "2026-01-01T00:15:00.000Z",
            "cameras": ["camera-000-old"],
            "rooms": ["room-200-old"],
            "floors": ["floor-1-old"],
        },
    }

    parsed = parse_topology_success_envelope(response)
    response["omissions"] = None
    confirmed = parse_topology_success_envelope(response)

    assert parsed.omissions is not None
    assert parsed.omissions.cameras == ("camera-000-old",)
    assert confirmed.omissions is None


def test_machine_principal_parser_rejects_facility_identity() -> None:
    principal = parse_machine_principal(
        {"edgeInstallationId": "c72bd9a7-3e04-47ba-a8cd-a56e54f98152", "enrollmentGeneration": 1}
    )

    assert principal == _snapshot().principal
    with pytest.raises(ContractViolation):
        _ = parse_machine_principal(
            {
                "edgeInstallationId": "c72bd9a7-3e04-47ba-a8cd-a56e54f98152",
                "enrollmentGeneration": 1,
                "facilityId": "synthetic-facility",
            }
        )


def test_manifest_parser_rejects_missing_parent_relationship() -> None:
    manifest: list[JsonRecord] = [
        dict(kind="FLOOR", edgeRef="floor-2", canonicalId="floor-id", parentCanonicalId=None),
        dict(
            kind="ROOM", edgeRef="room-201", canonicalId="room-id", parentCanonicalId="missing"
        ),
    ]

    with pytest.raises(ContractViolation):
        _ = parse_topology_manifest(manifest)


def test_error_envelope_parser_rejects_malformed_envelopes() -> None:
    error = parse_error_envelope(
        {
            "schemaVersion": 1,
            "error": {
                "code": "INVALID_SCHEMA",
                "message": "Request does not match edge provisioning v1.",
                "retryable": False,
                "requestId": "request-test-001",
            },
        }
    )

    assert error.code == "INVALID_SCHEMA"
    with pytest.raises(ContractViolation):
        _ = parse_error_envelope({"schemaVersion": 1, "error": {"code": "INVALID_SCHEMA"}})


def test_current_mapper_remains_per_camera_legacy_client_without_network() -> None:
    mapper = _CharacterizedMapper(
        endpoint="https://api.example/api/v1/edge/cameras",
        token="fixture-token",
        facility_id="legacy-facility",
    )

    request = mapper.request(edge_camera_ref="camera-001", label="Camera Test", space_id="room-201")

    assert request.full_url == "https://api.example/api/v1/edge/cameras"
    assert request.get_method() == "PUT"
    assert request.get_header("Authorization") == "Bearer fixture-token"
    assert request.get_header("X-facility-id") == "legacy-facility"
    request_data = request.data
    assert isinstance(request_data, bytes)
    assert json.loads(request_data.decode("utf-8")) == {
        "edge_camera_ref": "camera-001",
        "label": "Camera Test",
        "spaceId": "room-201",
    }


class _CharacterizedMapper(BackendCameraMapper):
    def request(
        self, *, edge_camera_ref: str, label: str, space_id: str
    ) -> urllib.request.Request:
        return self._mapping_request(
            edge_camera_ref=edge_camera_ref,
            label=label,
            space_id=space_id,
        )


def _snapshot() -> TopologySnapshot:
    return TopologySnapshot(
        principal=MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 1),
        client_revision=1,
        expected_server_revision=0,
        floors=(
            TopologyFloor(
                edge_ref="floor-2",
                name="2F Test",
                order_index=2,
                rooms=(
                    TopologyRoom(
                        edge_ref="room-201", name="Room Test", room_type="ROOM", capacity=1,
                        cameras=(TopologyCamera(edge_ref="camera-001", label="Camera Test"),),
                    ),
                ),
            ),
        ),
    )
