from __future__ import annotations

from typing import assert_never

from fastapi import APIRouter, Request, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.http import append_transactional
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.features.cameras.edge_topology_sync_state import TopologyPauseReason
from backend.app.features.cameras.router import _authorize
from backend.app.features.cameras.topology_client import (
    TopologyAccepted,
    TopologyPaused,
    TopologyRetryable,
)
from backend.app.features.cameras.topology_confirmation import (
    TopologyConfirmationCommand,
    TopologyConfirmationRejected,
)
from backend.app.features.connection.topology_retry_coordinator import (
    topology_retry_coordinator,
)
from contracts.edge_provisioning_v1 import EdgeErrorCode, MutationCounts

router = APIRouter(prefix="/connection", tags=["connection"])


class TopologyConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmation_id: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    client_revision: int = Field(gt=0)
    server_revision: int = Field(ge=0)


class TopologyPreviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmation_id: str
    digest: str
    expires_at: str
    snapshot_id: str
    client_revision: int
    server_revision: int
    cameras: int
    rooms: int
    floors: int
    confirmed: bool


class TopologyPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    preview: TopologyPreviewPayload | None


class MutationCountsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    created: int
    updated: int
    unchanged: int
    reactivated: int
    deactivated: int


class TopologyMutationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    floors: MutationCountsResponse
    rooms: MutationCountsResponse
    cameras: MutationCountsResponse


class TopologyConfirmResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str
    client_revision: int
    server_revision: int
    result: TopologyMutationResponse


@router.get("/topology-preview", response_model=TopologyPreviewResponse)
def topology_preview(request: Request) -> TopologyPreviewResponse:
    _authorize(request)
    preview = topology_retry_coordinator(request.app).preview()
    if preview is None:
        return TopologyPreviewResponse(preview=None)
    return TopologyPreviewResponse(
        preview=TopologyPreviewPayload(
            confirmation_id=preview.confirmation_id,
            digest=preview.digest,
            expires_at=preview.expires_at,
            snapshot_id=preview.snapshot_id,
            client_revision=preview.client_revision,
            server_revision=preview.server_revision,
            cameras=preview.cameras,
            rooms=preview.rooms,
            floors=preview.floors,
            confirmed=preview.confirmed,
        )
    )


@router.post("/topology-preview/confirm", response_model=TopologyConfirmResponse)
def confirm_topology_preview(
    payload: TopologyConfirmRequest,
    request: Request,
) -> TopologyConfirmResponse:
    actor = _authorize(request)
    outcome = topology_retry_coordinator(request.app).confirm(
        TopologyConfirmationCommand(
            payload.confirmation_id,
            payload.digest,
            payload.client_revision,
            payload.server_revision,
        ),
        after_write=lambda connection: append_transactional(
            request,
            connection,
            AuditEvent(
                occurred_at=utc_now(), actor_id=actor,
                action=AuditAction.TOPOLOGY_CONFIRM,
                target_id=payload.confirmation_id,
                detail=empty_detail(AuditAction.TOPOLOGY_CONFIRM),
            ),
        ),
    )
    match outcome:
        case TopologyAccepted(response=response):
            return TopologyConfirmResponse(
                snapshot_id=response.snapshot_id,
                client_revision=response.client_revision,
                server_revision=response.server_revision,
                result=TopologyMutationResponse(
                    floors=_counts(response.result.floors),
                    rooms=_counts(response.result.rooms),
                    cameras=_counts(response.result.cameras),
                ),
            )
        case TopologyConfirmationRejected(status_code=status_code, error_code=error_code):
            raise HTTPException(status_code=status_code, detail={"code": error_code.value})
        case TopologyPaused(reason=reason, status_code=status_code):
            raise HTTPException(
                status_code=status_code,
                detail={"code": _pause_code(reason).value},
            )
        case TopologyRetryable():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "TOPOLOGY_CONFIRMATION_UNREACHABLE"},
            )
        case unreachable:
            assert_never(unreachable)


def _counts(counts: MutationCounts) -> MutationCountsResponse:
    return MutationCountsResponse(
        created=counts.created,
        updated=counts.updated,
        unchanged=counts.unchanged,
        reactivated=counts.reactivated,
        deactivated=counts.deactivated,
    )


def _pause_code(reason: TopologyPauseReason) -> EdgeErrorCode:
    match reason:
        case TopologyPauseReason.AUTH:
            return EdgeErrorCode.EDGE_CREDENTIAL_INVALID
        case TopologyPauseReason.FORBIDDEN:
            return EdgeErrorCode.FACILITY_BINDING_MISMATCH
        case TopologyPauseReason.CONFLICT:
            return EdgeErrorCode.CONFIRMATION_STALE
        case unreachable:
            assert_never(unreachable)


__all__ = ["router"]
