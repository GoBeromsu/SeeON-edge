"""Authenticated, privacy-bounded central evidence operator API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.http import append_governed, append_transactional
from backend.app.features.audit.store import AuditEvent
from backend.app.features.evidence.record_store import (
    CentralEvidenceQuery,
    CentralEvidenceReviewStore,
    CentralEvidenceSummary,
    EvidenceReviewConflictError,
    ReviewDisposition,
)
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["evidence"])

_DEFAULT_INCIDENT_LIMIT = 50
_MAX_INCIDENT_LIMIT = 100


class IncidentReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=0)
    disposition: Literal["TRUE_POSITIVE", "FALSE_POSITIVE"]
    notes: str | None = Field(default=None, max_length=1000)


class IncidentListQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=_DEFAULT_INCIDENT_LIMIT, ge=1, le=_MAX_INCIDENT_LIMIT)
    cursor: str | None = Field(default=None, min_length=1, max_length=384)


@router.get("/incidents")
def list_incidents(
    request: Request,
    filters: Annotated[IncidentListQuery, Query()],
) -> dict[str, object]:
    actor = _authorize(request)
    try:
        incidents, next_cursor = _query(request).list(limit=filters.limit, cursor=filters.cursor)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    append_governed(
        request, actor_id=actor, action=AuditAction.INCIDENT_LIST, target_id="incidents"
    )
    return {
        "incidents": [_summary_response(value) for value in incidents],
        "pagination": {
            "limit": filters.limit,
            "next_cursor": next_cursor,
            "has_more": next_cursor is not None,
        },
    }


@router.get("/incidents/{incident_id:path}")
def get_incident(
    incident_id: str,
    request: Request,
) -> dict[str, object]:
    actor = _authorize(request)
    summary = _query(request).get(incident_id)
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="incident not found")
    append_governed(
        request, actor_id=actor, action=AuditAction.INCIDENT_DETAIL,
        target_id=summary.incident_id,
    )
    return _summary_response(summary)


@router.put("/incident-reviews/{incident_id:path}")
def review_incident(
    incident_id: str,
    payload: IncidentReviewRequest,
    request: Request,
) -> dict[str, object]:
    actor = _authorize(request)
    summary = _query(request).get(incident_id)
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="incident not found")
    event = AuditEvent(
        occurred_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        actor_id=actor, action=AuditAction.INCIDENT_REVIEW, target_id=summary.incident_id,
        detail=empty_detail(AuditAction.INCIDENT_REVIEW),
    )
    try:
        _reviews(request).update(
            incident_id=summary.incident_id,
            expected_version=payload.expected_version,
            actor_id=actor,
            reviewed_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            disposition=ReviewDisposition(payload.disposition),
            notes=payload.notes,
            after_write=lambda connection: append_transactional(request, connection, event),
        )
    except EvidenceReviewConflictError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    refreshed = _query(request).get(incident_id)
    assert refreshed is not None
    return _summary_response(refreshed)


def _summary_response(summary: CentralEvidenceSummary) -> dict[str, object]:
    review = summary.review
    return {
        "incident_id": summary.incident_id,
        "edge_event_id": summary.edge_event_id,
        "schema_version": summary.schema_version,
        "camera_id": summary.camera_id,
        "event_type": summary.event_type,
        "detected_at": summary.detected_at,
        "lifecycle_state": summary.lifecycle_state,
        "revision": summary.revision,
        "failure_reason": summary.failure_reason,
        "runtime_manifest_sha256": summary.runtime_manifest_sha256,
        "decision_trace_id": summary.decision_trace_id,
        "module_qualified_id": summary.module_qualified_id,
        "policy_qualified_id": summary.policy_qualified_id,
        "primary_clip_id": summary.primary_clip_id,
        "primary_artifact_state": summary.primary_artifact_state,
        "snapshot_artifact_state": summary.snapshot_artifact_state,
        "event_delivery_state": summary.event_delivery_state,
        "clip_publish_state": summary.clip_publish_state,
        "retention_state": summary.retention_state,
        "review": None
        if review is None
        else {
            "version": review.version,
            "disposition": review.disposition.value,
            "reviewed_at": review.reviewed_at,
            "notes": review.notes,
        },
    }


def _query(request: Request) -> CentralEvidenceQuery:
    query = getattr(request.app.state, "central_evidence_query", None)
    if isinstance(query, CentralEvidenceQuery):
        return query
    return CentralEvidenceQuery(EDGE_DATABASE_PATH)


def _reviews(request: Request) -> CentralEvidenceReviewStore:
    store = getattr(request.app.state, "central_evidence_review_store", None)
    if isinstance(store, CentralEvidenceReviewStore):
        return store
    return CentralEvidenceReviewStore(EDGE_DATABASE_PATH)


def _authorize(request: Request) -> str:
    return authorize_dashboard(request)


__all__ = ["router"]
