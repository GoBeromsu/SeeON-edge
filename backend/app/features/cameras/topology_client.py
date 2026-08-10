from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from backend.app.features.cameras.edge_topology_sync_state import (
    PendingTopologySnapshot,
    TopologyPauseReason,
)
from backend.app.features.cameras.topology_query import RegistryTopologySnapshot
from backend.app.features.connection.enrollment import (
    EnrollmentCredentials,
    EnrollmentVerificationFailure,
    verify_enrollment,
)
from backend.app.shared.backend_client_bundle import BackendClientBundle
from contracts.edge_provisioning_v1 import (
    ContractViolation,
    MachinePrincipal,
    TopologySnapshot,
    TopologySuccessEnvelope,
    parse_topology_success_envelope,
    serialize_topology_snapshot,
)
from shared.events.evidence_export_contract import DeliveryFailure
from shared.events.evidence_http_transport import bounded_request

TopologyRetryError: TypeAlias = Literal["timeout", "unreachable"]
_RESPONSE_ADAPTER = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class TopologyAccepted:
    response: TopologySuccessEnvelope


@dataclass(frozen=True, slots=True)
class TopologyRetryable:
    error_class: TopologyRetryError


@dataclass(frozen=True, slots=True)
class TopologyPaused:
    reason: TopologyPauseReason
    status_code: int


TopologyPutResult: TypeAlias = TopologyAccepted | TopologyRetryable | TopologyPaused


@dataclass(frozen=True, slots=True)
class TopologySnapshotBuilder:
    topology: RegistryTopologySnapshot
    principal: MachinePrincipal
    snapshot_id: str

    @property
    def registry_version(self) -> int:
        return self.topology.registry_version

    def build(self, client_revision: int, expected_server_revision: int) -> bytes:
        snapshot = TopologySnapshot(
            self.principal,
            client_revision,
            expected_server_revision,
            self.topology.floors,
        )
        return json.dumps(
            serialize_topology_snapshot(snapshot),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()


@dataclass(frozen=True, slots=True)
class TopologyClient:
    events_url: str
    bearer_token: str = field(repr=False)
    principal: MachinePrincipal
    facility_code: str
    client_installation_ref: str
    timeout_sec: float

    @classmethod
    def from_bundle(cls, bundle: BackendClientBundle) -> TopologyClient:
        return cls(
            bundle.events_url,
            bundle.facility_token,
            MachinePrincipal(bundle.edge_installation_id, bundle.enrollment_generation),
            bundle.facility_code,
            bundle.client_installation_ref,
            bundle.ingest_client.timeout_sec,
        )

    def put(self, pending: PendingTopologySnapshot) -> TopologyPutResult:
        result = bounded_request(
            f"{_topology_base(self.events_url)}/{pending.snapshot_id}",
            "PUT",
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            },
            pending.body,
            self.timeout_sec,
        )
        if isinstance(result, DeliveryFailure):
            detail = result.transport_error or ""
            error_class: TopologyRetryError = (
                "timeout" if "timeout" in detail.casefold() else "unreachable"
            )
            return TopologyRetryable(error_class)
        status_code, _headers, response_body = result
        match status_code:
            case 401:
                return TopologyPaused(TopologyPauseReason.AUTH, status_code)
            case 403:
                return TopologyPaused(TopologyPauseReason.FORBIDDEN, status_code)
            case 409:
                return TopologyPaused(TopologyPauseReason.CONFLICT, status_code)
            case code if 200 <= code < 300:
                try:
                    payload = _RESPONSE_ADAPTER.validate_json(response_body)
                    response = parse_topology_success_envelope(payload)
                except (ContractViolation, ValidationError):
                    return TopologyRetryable("unreachable")
                if response.snapshot_id != pending.snapshot_id:
                    return TopologyRetryable("unreachable")
                return TopologyAccepted(response)
            case code if code in {408, 425, 429} or 500 <= code < 600:
                return TopologyRetryable("unreachable")
            case _:
                return TopologyPaused(TopologyPauseReason.CONFLICT, status_code)

    def refresh_server_revision(self) -> int | None:
        credentials = EnrollmentCredentials(
            self.facility_code,
            self.client_installation_ref,
            self.bearer_token,
        )
        try:
            result = verify_enrollment(self.events_url, credentials, timeout_sec=self.timeout_sec)
        except EnrollmentVerificationFailure:
            return None
        if result.principal != self.principal:
            return None
        return result.server_revision


def _topology_base(events_url: str) -> str:
    parsed = urllib.parse.urlsplit(events_url)
    origin = urllib.parse.urlunsplit(parsed._replace(path="", query="", fragment=""))
    return f"{origin.rstrip('/')}/api/v1/edge/topology-snapshots"


__all__ = [
    "TopologyAccepted",
    "TopologyClient",
    "TopologyPaused",
    "TopologyPutResult",
    "TopologyRetryError",
    "TopologyRetryable",
    "TopologySnapshotBuilder",
]
