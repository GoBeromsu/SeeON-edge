"""Keyset HTTP projection for immutable audit history."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.http import append_governed, audit_runtime
from backend.app.shared.dashboard_auth import authorize_dashboard

SqlValue = str | int | float | bytes | None

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


class AuditStoredIdentityError(ValueError):
    """A verified audit row contains a non-integer identity."""


_SELECT = (
    "SELECT audit_id,occurred_at,recorded_at,actor_type,actor_id,action,target_type,"
    "target_id,outcome,detail_json,previous_hash,record_hash FROM audit_events"
)


def _audit_id(value: SqlValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AuditStoredIdentityError
    return value


def _response(row: tuple[SqlValue, ...]) -> AuditEventResponse:
    return AuditEventResponse(
        audit_id=_audit_id(row[0]),
        occurred_at=str(row[1]),
        recorded_at=str(row[2]),
        actor_type=str(row[3]),
        actor_id=str(row[4]),
        action=AuditAction(str(row[5])),
        target_type=str(row[6]),
        target_id=str(row[7]),
        outcome=str(row[8]),
        detail_json=None if row[9] is None else str(row[9]),
        previous_hash=str(row[10]),
        record_hash=str(row[11]),
    )


@router.get("", response_model=AuditListResponse)
def list_audit(request: Request, filters: Annotated[AuditListQuery, Query()]) -> AuditListResponse:
    actor = authorize_dashboard(request)
    parameters: tuple[int, ...]
    if filters.before_id is None:
        query = _SELECT + " ORDER BY audit_id DESC LIMIT %s"
        parameters = (filters.limit + 1,)
    else:
        query = _SELECT + " WHERE audit_id < %s ORDER BY audit_id DESC LIMIT %s"
        parameters = (filters.before_id, filters.limit + 1)
    rows = audit_runtime(request).database.read(
        lambda connection: connection.execute(query, parameters).fetchall()
    )
    page = rows[: filters.limit]
    append_governed(request, actor_id=actor, action=AuditAction.AUDIT_LIST, target_id="audit")
    return AuditListResponse(
        events=tuple(_response(row) for row in page),
        next_before_id=_audit_id(page[-1][0]) if len(rows) > filters.limit else None,
    )


@router.get("/{audit_id}", response_model=AuditEventResponse)
def get_audit(audit_id: int, request: Request) -> AuditEventResponse:
    actor = authorize_dashboard(request)
    row = audit_runtime(request).database.read(
        lambda connection: connection.execute(
            _SELECT + " WHERE audit_id = %s", (audit_id,)
        ).fetchone()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audit event not found")
    response = _response(row)
    append_governed(
        request, actor_id=actor, action=AuditAction.AUDIT_DETAIL, target_id=str(audit_id)
    )
    return response


__all__ = ["router"]
