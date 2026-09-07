"""Dashboard relay and artifact reads for stored-clip reanalysis."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.app.features.clips.analysis_relay import relay
from backend.app.features.clips.analysis_status import assemble_clip_analysis_status
from backend.app.features.clips.router import _clip_store, _get_located_clip_or_404
from backend.app.features.clips.schemas import ClipAnalysisResponse
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
        accepted=frozenset(
            {
                HTTPStatus.ACCEPTED,
                HTTPStatus.CONFLICT,
                HTTPStatus.NOT_FOUND,
                HTTPStatus.SERVICE_UNAVAILABLE,
            }
        ),
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


@router.post("/clips/{clip_id}/analysis/cancel", status_code=status.HTTP_204_NO_CONTENT)
def cancel_clip_analysis(clip_id: str, request: Request) -> Response:
    authorize_dashboard(request)
    _ = _get_located_clip_or_404(request, clip_id)
    relay_response = relay(
        request,
        clip_id,
        "analysis/cancel",
        body={},
        accepted=frozenset({HTTPStatus.NO_CONTENT}),
    )
    return Response(status_code=relay_response.status_code)


__all__ = ["router"]
