"""Dashboard relay and artifact reads for stored-clip reanalysis."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.app.core.config import get_settings
from backend.app.features.clips.router import _get_located_clip_or_404
from backend.app.features.clips.schemas import (
    ClipAnalysisCancelResponse,
    ClipAnalysisResponse,
    ClipAnalysisTriggerResponse,
)
from backend.app.features.clips.store import PLAYBACK_H264_FILENAME, ClipStore, LocatedClip
from backend.app.shared.dashboard_auth import authorize_dashboard
from shared.events.clip_analysis_wire import ClipAnalysisWireError, decode_clip_analysis

router = APIRouter(tags=["clips"])
_RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"
_WORKER_STATES = frozenset({"idle", "running", "available", "failed"})


@router.post(
    "/clips/{clip_id}/analysis",
    response_model=ClipAnalysisTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_clip_analysis(clip_id: str, request: Request) -> Response:
    authorize_dashboard(request)
    located = _get_located_clip_or_404(request, clip_id)
    digest = _clip_store(request).manifest_video_sha256(located)
    if digest is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="clip manifest identity unavailable",
        )
    return _relay(
        request,
        clip_id,
        "analysis",
        body={"clip_sha256": digest},
        accepted=frozenset({HTTPStatus.ACCEPTED, HTTPStatus.CONFLICT, HTTPStatus.NOT_FOUND}),
    )


@router.get(
    "/clips/{clip_id}/analysis",
    response_model=ClipAnalysisResponse,
    response_model_exclude_none=True,
)
def get_clip_analysis(clip_id: str, request: Request) -> ClipAnalysisResponse:
    authorize_dashboard(request)
    located = _get_located_clip_or_404(request, clip_id)
    store = _clip_store(request)
    try:
        payload = store.read_clip_analysis(located)
    except ValueError:
        return ClipAnalysisResponse(state="unavailable", reason="artifact_invalid")
    if payload is not None:
        try:
            result = decode_clip_analysis(payload)
        except ClipAnalysisWireError:
            return ClipAnalysisResponse(state="unavailable", reason="artifact_invalid")
        digest = store.manifest_video_sha256(located)
        if digest is None or result.clip_sha256 != digest:
            return ClipAnalysisResponse(state="unavailable", reason="identity_mismatch")
        return ClipAnalysisResponse(
            state="available",
            served_timing_identical=_served_timing_identical(store, located),
            result=result.as_dict(),
        )
    try:
        upstream = _relay(
            request,
            clip_id,
            "analysis",
            body=None,
            accepted=frozenset({HTTPStatus.OK}),
            method="GET",
        )
    except HTTPException as exc:
        if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            return ClipAnalysisResponse(state="unavailable", reason="worker_unreachable")
        raise
    try:
        body = json.loads(upstream.body)
    except (TypeError, json.JSONDecodeError):
        return ClipAnalysisResponse(state="unavailable", reason="worker_unreachable")
    if not isinstance(body, dict):
        return ClipAnalysisResponse(state="unavailable", reason="worker_unreachable")
    state_value = body.get("state")
    reason = body.get("reason")
    if state_value not in _WORKER_STATES or (
        reason is not None and (not isinstance(reason, str) or not reason)
    ):
        return ClipAnalysisResponse(state="unavailable", reason="worker_unreachable")
    return ClipAnalysisResponse(
        state=cast(Literal["idle", "running", "available", "failed"], state_value),
        reason=reason,
    )


@router.post("/clips/{clip_id}/analysis/cancel", response_model=ClipAnalysisCancelResponse)
def cancel_clip_analysis(clip_id: str, request: Request) -> Response:
    authorize_dashboard(request)
    _ = _get_located_clip_or_404(request, clip_id)
    return _relay(
        request,
        clip_id,
        "analysis/cancel",
        body={},
        accepted=frozenset({HTTPStatus.OK, HTTPStatus.NOT_FOUND}),
    )


def _relay(
    request: Request,
    clip_id: str,
    suffix: str,
    *,
    body: dict[str, str] | dict[object, object] | None,
    accepted: frozenset[HTTPStatus],
    method: str = "POST",
) -> Response:
    settings = get_settings()
    origin = settings.worker_stream_origin.strip().rstrip("/")
    if not origin:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    url = f"{origin}/clips/{urllib.parse.quote(clip_id, safe='')}/{suffix}"
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {"X-Edge-Relay-Token": request.app.state.edge_relay_token}
    if data is not None:
        headers["Content-Type"] = "application/json"
    upstream_request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        upstream = urllib.request.urlopen(
            upstream_request, timeout=settings.worker_stream_timeout_s
        )
    except urllib.error.HTTPError as exc:
        if HTTPStatus(exc.code) not in accepted:
            raise HTTPException(status_code=exc.code) from exc
        raw = exc.read()
        return Response(content=raw, status_code=exc.code, media_type="application/json")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE) from exc
    try:
        raw = upstream.read()
        upstream_status = HTTPStatus(upstream.status)
    except (OSError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE) from exc
    finally:
        upstream.close()
    if upstream_status not in accepted:
        raise HTTPException(status_code=int(upstream_status))
    return Response(content=raw, status_code=int(upstream_status), media_type="application/json")


def _clip_store(request: Request) -> ClipStore:
    try:
        store = request.app.state.clip_store
    except AttributeError:
        store = ClipStore.from_env()
        request.app.state.clip_store = store
    if not isinstance(store, ClipStore):
        raise TypeError("clip_store is invalid")
    return store


def _served_timing_identical(store: ClipStore, located: LocatedClip) -> bool:
    try:
        identity = store.open_located_playback_identity(located)
    except (ValueError, FileNotFoundError):
        return False
    try:
        return identity.opened.path.name != PLAYBACK_H264_FILENAME or identity.served_pts_identical
    finally:
        identity.opened.handle.close()


__all__ = ["router"]
