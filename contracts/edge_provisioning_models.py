from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Literal, NotRequired, TypeAlias, TypedDict, final

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonRecord: TypeAlias = dict[str, JsonValue]
TopologyKind: TypeAlias = Literal["FLOOR", "ROOM", "CAMERA"]


@unique
class EdgeErrorCode(StrEnum):
    INVALID_SCHEMA = "INVALID_SCHEMA"
    INVALID_TOPOLOGY = "INVALID_TOPOLOGY"
    EDGE_CREDENTIAL_REQUIRED = "EDGE_CREDENTIAL_REQUIRED"
    EDGE_CREDENTIAL_INVALID = "EDGE_CREDENTIAL_INVALID"
    EDGE_CREDENTIAL_INACTIVE = "EDGE_CREDENTIAL_INACTIVE"
    FACILITY_BINDING_MISMATCH = "FACILITY_BINDING_MISMATCH"
    INSTALLATION_CONFLICT = "INSTALLATION_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    CLIENT_REVISION_OUT_OF_SEQUENCE = "CLIENT_REVISION_OUT_OF_SEQUENCE"
    STALE_SERVER_REVISION = "STALE_SERVER_REVISION"
    STALE_ENROLLMENT_GENERATION = "STALE_ENROLLMENT_GENERATION"
    TOPOLOGY_CONFLICT = "TOPOLOGY_CONFLICT"
    TOPOLOGY_TRANSFER_CONFLICT = "TOPOLOGY_TRANSFER_CONFLICT"
    LEGACY_MAPPING_REQUIRED = "LEGACY_MAPPING_REQUIRED"
    CONFIRMATION_STALE = "CONFIRMATION_STALE"
    CONFIRMATION_EXPIRED = "CONFIRMATION_EXPIRED"
    ENROLLMENT_RATE_LIMITED = "ENROLLMENT_RATE_LIMITED"
    EDGE_AUTH_NOT_CONFIGURED = "EDGE_AUTH_NOT_CONFIGURED"


@final
class ContractViolation(Exception):
    __slots__ = ("detail",)
    detail: str

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


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
    legacy_canonical_space_id: str | None = None


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
    code: EdgeErrorCode
    message: str
    retryable: bool
    request_id: str


@dataclass(frozen=True, slots=True)
class MutationCounts:
    created: int
    updated: int
    unchanged: int
    reactivated: int = 0
    deactivated: int = 0


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
class OwnershipTransferPreview:
    manifest_digest: str
    items: tuple[TopologyManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class TopologySuccessEnvelope:
    snapshot_id: str
    client_revision: int
    server_revision: int
    result: TopologyMutationResult
    omissions: OmissionPreview | None
    ownership_transfer_required: OwnershipTransferPreview | None = None


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
    cameras: list[TopologyCameraBody]
    type: NotRequired[str]
    capacity: NotRequired[int]
    legacyCanonicalSpaceId: NotRequired[str]


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
    code: EdgeErrorCode
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
    "EdgeErrorCode",
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
    "OwnershipTransferPreview",
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
