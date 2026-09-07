"""Dashboard relay and artifact reads for stored-clip reanalysis."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.app.features.clips.analysis_relay import relay
from backend.app.features.clips.analysis_status import assemble_clip_analysis_status
from backend.app.features.clips.router import _get_located_clip_or_404
from backend.app.features.clips.schemas import (
    ClipAnalysisCancelResponse,
    ClipAnalysisResponse,
)
from backend.app.features.clips.store import ClipStore
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["clips"])


@router.post(
    "/clips/{clip_id}/analysis",
    response_model=ClipAnalysisResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_clip_analysis(
    clip_id: str, request: Request, response: Response
) -> ClipAnalysisResponse:
    authorize_dashboard(request)
    located = _get_located_clip_or_404(request, clip_id)
    store = _clip_store(request)
    digest = store.manifest_video_sha256(located)
    if digest is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="clip manifest identity unavailable",
        )
    relay_response = relay(
        request,
        clip_id,
        "analysis",
        body={"clip_sha256": digest},
        accepted=frozenset({HTTPStatus.ACCEPTED, HTTPStatus.CONFLICT, HTTPStatus.NOT_FOUND}),
    )
    response.status_code = relay_response.status_code
    return assemble_clip_analysis_status(request, clip_id, located, store)


@router.get(
    "/clips/{clip_id}/analysis",
    response_model=ClipAnalysisResponse,
    response_model_exclude_none=True,
)
def get_clip_analysis(clip_id: str, request: Request) -> ClipAnalysisResponse:
    authorize_dashboard(request)
    located = _get_located_clip_or_404(request, clip_id)
    return assemble_clip_analysis_status(request, clip_id, located, _clip_store(request))


@router.post("/clips/{clip_id}/analysis/cancel", response_model=ClipAnalysisCancelResponse)
def cancel_clip_analysis(clip_id: str, request: Request) -> ClipAnalysisCancelResponse:
    authorize_dashboard(request)
    _ = _get_located_clip_or_404(request, clip_id)
    relay_response = relay(
        request,
        clip_id,
        "analysis/cancel",
        body={},
        accepted=frozenset({HTTPStatus.OK}),
    )
    try:
        return ClipAnalysisCancelResponse.model_validate_json(relay_response.body)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker_unreachable",
        ) from exc


def _clip_store(request: Request) -> ClipStore:
    try:
        store = request.app.state.clip_store
    except AttributeError:
        store = ClipStore.from_env()
        request.app.state.clip_store = store
    if not isinstance(store, ClipStore):
        raise TypeError("clip_store is invalid")
    return store


__all__ = ["router"]
