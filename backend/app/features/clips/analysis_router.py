"""Dashboard relay and artifact reads for stored-clip reanalysis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.app.core.config import get_settings
from backend.app.features.clips.analysis_relay import (
    AnalysisRelayError,
    AnalysisRelayResponse,
    AnalysisRelaySettings,
    AnalysisTransport,
    relay,
)
from backend.app.features.clips.analysis_status import assemble_clip_analysis_status
from backend.app.features.clips.router import _clip_store, _get_located_clip_or_404
from backend.app.features.clips.schemas import ClipAnalysisResponse
from backend.app.features.clips.store import ClipStore, LocatedClip
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["clips"])


def _transport(request: Request) -> AnalysisTransport:
    def send(
        clip_id: str,
        suffix: str,
        *,
        body: Mapping[str, str] | None,
        accepted: frozenset[HTTPStatus | int],
        method: str = "POST",
    ) -> AnalysisRelayResponse:
        settings = get_settings()
        token = request.app.state.edge_relay_token
        configuration = AnalysisRelaySettings(
            settings.worker_stream_origin,
            token if isinstance(token, str) else None,
            settings.worker_stream_timeout_s,
        )
        return relay(
            configuration, clip_id, suffix, body=body, accepted=accepted, method=method
        )

    return send


def _relay(
    request: Request,
    clip_id: str,
    suffix: str,
    *,
    body: Mapping[str, str] | None,
    accepted: frozenset[HTTPStatus | int],
) -> AnalysisRelayResponse:
    try:
        return _transport(request)(clip_id, suffix, body=body, accepted=accepted)
    except AnalysisRelayError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error


def _analysis_status(
    request: Request, clip_id: str, located: LocatedClip, store: ClipStore
) -> ClipAnalysisResponse:
    try:
        return assemble_clip_analysis_status(_transport(request), clip_id, located, store)
    except AnalysisRelayError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error


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
    relay_response = _relay(
        request,
        clip_id,
        "analysis",
        body={"clip_sha256": digest},
        accepted=frozenset(
            {
                HTTPStatus.ACCEPTED,
                HTTPStatus.OK,
                HTTPStatus.CONFLICT,
                HTTPStatus.NOT_FOUND,
                HTTPStatus.SERVICE_UNAVAILABLE,
                HTTPStatus.UNPROCESSABLE_ENTITY,
            }
        ),
    )
    response.status_code = relay_response.status_code
    if relay_response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        try:
            rejection = json.loads(relay_response.body)
        except (TypeError, json.JSONDecodeError):
            rejection = {}
        reason = rejection.get("reason") if isinstance(rejection, dict) else None
        if isinstance(reason, str) and reason:
            return ClipAnalysisResponse(
                state="failed",
                served_media_sha256=digest,
                reason=reason,
            )
    return _analysis_status(request, clip_id, located, store)


@router.get(
    "/clips/{clip_id}/analysis",
    response_model=ClipAnalysisResponse,
    response_model_exclude_none=True,
)
def get_clip_analysis(clip_id: str, request: Request) -> ClipAnalysisResponse:
    authorize_dashboard(request)
    located = _get_located_clip_or_404(request, clip_id)
    return _analysis_status(request, clip_id, located, _clip_store(request))


@router.post("/clips/{clip_id}/analysis/cancel", status_code=status.HTTP_204_NO_CONTENT)
def cancel_clip_analysis(clip_id: str, request: Request) -> Response:
    authorize_dashboard(request)
    _ = _get_located_clip_or_404(request, clip_id)
    relay_response = _relay(
        request,
        clip_id,
        "analysis/cancel",
        body={},
        accepted=frozenset({HTTPStatus.NO_CONTENT}),
    )
    return Response(status_code=relay_response.status_code)


__all__ = ["router"]
