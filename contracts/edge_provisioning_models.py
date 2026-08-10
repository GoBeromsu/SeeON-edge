from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonRecord: TypeAlias = dict[str, JsonValue]
TopologyKind: TypeAlias = Literal["FLOOR", "ROOM", "CAMERA"]


@dataclass(frozen=True, slots=True)
class ContractViolation(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class EnrollmentVerification:
    facility_code: str
    client_installation_ref: str


@dataclass(frozen=True, slots=True)
class FacilityIdentity:
    facility_id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class MachinePrincipal:
    edge_installation_id: str
    enrollment_generation: int


@dataclass(frozen=True, slots=True)
class EnrollmentVerificationResult:
    principal: MachinePrincipal
    facility: FacilityIdentity
    server_revision: int


@dataclass(frozen=True, slots=True)
class TopologyCamera:
    edge_ref: str
    label: str


@dataclass(frozen=True, slots=True)
class TopologyRoom:
    edge_ref: str
    name: str
    room_type: str
    capacity: int
    cameras: tuple[TopologyCamera, ...]


@dataclass(frozen=True, slots=True)
class TopologyFloor:
    edge_ref: str
    name: str
    order_index: int
    rooms: tuple[TopologyRoom, ...]


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    principal: MachinePrincipal
    client_revision: int
    expected_server_revision: int
    floors: tuple[TopologyFloor, ...]


@dataclass(frozen=True, slots=True)
class TopologyConfirmation:
    confirmation_id: str
    digest: str
    expected_server_revision: int


@dataclass(frozen=True, slots=True)
class TopologyManifestEntry:
    kind: TopologyKind
    edge_ref: str
    canonical_id: str
    parent_canonical_id: str | None


@dataclass(frozen=True, slots=True)
class ErrorEnvelope:
    code: str
    message: str
    retryable: bool
    request_id: str


@dataclass(frozen=True, slots=True)
class MutationCounts:
    created: int
    updated: int
    unchanged: int


@dataclass(frozen=True, slots=True)
class TopologyMutationResult:
    floors: MutationCounts
    rooms: MutationCounts
    cameras: MutationCounts


@dataclass(frozen=True, slots=True)
class OmissionPreview:
    confirmation_id: str
    digest: str
    expires_at: str
    cameras: tuple[str, ...]
    rooms: tuple[str, ...]
    floors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopologySuccessEnvelope:
    snapshot_id: str
    client_revision: int
    server_revision: int
    result: TopologyMutationResult
    omissions: OmissionPreview | None


class EnrollmentVerificationBody(TypedDict):
    schemaVersion: int
    facilityCode: str
    clientInstallationRef: str


class MachinePrincipalBody(TypedDict):
    edgeInstallationId: str
    enrollmentGeneration: int


class TopologyCameraBody(TypedDict):
    edgeRef: str
    label: str


class TopologyRoomBody(TypedDict):
    edgeRef: str
    name: str
    type: str
    capacity: int
    cameras: list[TopologyCameraBody]


class TopologyFloorBody(TypedDict):
    edgeRef: str
    name: str
    orderIndex: int
    rooms: list[TopologyRoomBody]


class TopologyConfirmationBody(TypedDict):
    schemaVersion: int
    confirmationId: str
    digest: str
    expectedServerRevision: int


class ErrorDetailBody(TypedDict):
    code: str
    message: str
    retryable: bool
    requestId: str


class ErrorEnvelopeBody(TypedDict):
    schemaVersion: int
    error: ErrorDetailBody


__all__ = [
    "ContractViolation",
    "EnrollmentVerification",
    "EnrollmentVerificationBody",
    "EnrollmentVerificationResult",
    "ErrorDetailBody",
    "ErrorEnvelope",
    "ErrorEnvelopeBody",
    "JsonRecord",
    "JsonValue",
    "FacilityIdentity",
    "MachinePrincipal",
    "MachinePrincipalBody",
    "MutationCounts",
    "OmissionPreview",
    "TopologyCamera",
    "TopologyCameraBody",
    "TopologyConfirmation",
    "TopologyConfirmationBody",
    "TopologyFloor",
    "TopologyFloorBody",
    "TopologyKind",
    "TopologyManifestEntry",
    "TopologyRoom",
    "TopologyRoomBody",
    "TopologySnapshot",
    "TopologyMutationResult",
    "TopologySuccessEnvelope",
]
