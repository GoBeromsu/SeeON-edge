"""Keyset HTTP projection for immutable audit history."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.history_repository import get_history, list_history
from backend.app.features.audit.http import append_governed, audit_runtime
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditListQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: int = Field(default=50, ge=1, le=100)
    before_id: int | None = Field(default=None, ge=1)


class AuditEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: int
    occurred_at: str
    recorded_at: str
    actor_type: str
    actor_id: str
    action: AuditAction
    target_type: str
    target_id: str
    outcome: str
    detail_json: str | None
    previous_hash: str
    record_hash: str


class AuditListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    events: tuple[AuditEventResponse, ...]
    next_before_id: int | None



@router.get("", response_model=AuditListResponse)
def list_audit(request: Request, filters: Annotated[AuditListQuery, Query()]) -> AuditListResponse:
    actor = authorize_dashboard(request)
    rows = list_history(audit_runtime(request).database, filters.limit, filters.before_id)
    page = rows[: filters.limit]
    append_governed(request, actor_id=actor, action=AuditAction.AUDIT_LIST, target_id="audit")
    return AuditListResponse(
        events=tuple(AuditEventResponse(**asdict(event)) for event in page),
        next_before_id=page[-1].audit_id if len(rows) > filters.limit else None,
    )


@router.get("/{audit_id}", response_model=AuditEventResponse)
def get_audit(audit_id: int, request: Request) -> AuditEventResponse:
    actor = authorize_dashboard(request)
    row = get_history(audit_runtime(request).database, audit_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audit event not found")
    response = AuditEventResponse(**asdict(row))
    append_governed(
        request, actor_id=actor, action=AuditAction.AUDIT_DETAIL, target_id=str(audit_id)
    )
    return response


__all__ = ["router"]
