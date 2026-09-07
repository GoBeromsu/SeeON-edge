"""Clip analysis status assembly for dashboard responses."""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Literal, cast

from fastapi import HTTPException, Request, status

from backend.app.features.clips.analysis_relay import relay
from backend.app.features.clips.schemas import ClipAnalysisResponse
from backend.app.features.clips.store import ClipStore, LocatedClip
from shared.events.clip_analysis_wire import ClipAnalysisWireError, decode_clip_analysis

_WORKER_STATES = frozenset({"idle", "running", "available", "failed"})


def assemble_clip_analysis_status(
    request: Request, clip_id: str, located: LocatedClip, store: ClipStore
) -> ClipAnalysisResponse:
    served_media_sha256 = served_media_sha256_for(store, located)
    try:
        payload = store.read_clip_analysis(located)
    except ValueError:
        return _unavailable(served_media_sha256, "artifact_invalid")
    if payload is not None:
        try:
            result = decode_clip_analysis(payload)
        except ClipAnalysisWireError:
            return _unavailable(served_media_sha256, "artifact_invalid")
        digest = store.manifest_video_sha256(located)
        if digest is None or result.clip_sha256 != digest:
            return _unavailable(served_media_sha256, "identity_mismatch")
        if not served_timing_identical(store, located):
            return _unavailable(served_media_sha256, "timing_unverified", timing_identical=False)
        return ClipAnalysisResponse(
            state="available",
            served_media_sha256=served_media_sha256,
            served_timing_identical=True,
            result=result.as_dict(),
        )
    try:
        upstream = relay(
            request,
            clip_id,
            "analysis",
            body=None,
            accepted=frozenset({HTTPStatus.OK}),
            method="GET",
        )
    except HTTPException as exc:
        if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            return _unavailable(served_media_sha256, "worker_unreachable")
        raise
    try:
        body = json.loads(upstream.body)
    except (TypeError, json.JSONDecodeError):
        return _unavailable(served_media_sha256, "worker_unreachable")
    if not isinstance(body, dict):
        return _unavailable(served_media_sha256, "worker_unreachable")
    state_value = body.get("state")
    reason = body.get("reason")
    if state_value not in _WORKER_STATES or (
        reason is not None and (not isinstance(reason, str) or not reason)
    ):
        return _unavailable(served_media_sha256, "worker_unreachable")
    if state_value == "available" and not served_timing_identical(store, located):
        return _unavailable(served_media_sha256, "timing_unverified", timing_identical=False)
    return ClipAnalysisResponse(
        state=cast(Literal["idle", "running", "available", "failed"], state_value),
        served_media_sha256=served_media_sha256,
        reason=reason,
    )


def served_timing_identical(store: ClipStore, located: LocatedClip) -> bool:
    try:
        identity = store.open_located_playback_identity(located)
    except (ValueError, FileNotFoundError):
        return False
    try:
        return identity.served_pts_identical is True
    finally:
        identity.opened.handle.close()


def served_media_sha256_for(store: ClipStore, located: LocatedClip) -> str | None:
    try:
        identity = store.open_located_playback_identity(located)
    except (ValueError, FileNotFoundError):
        return None
    try:
        return identity.served_media_sha256
    finally:
        identity.opened.handle.close()


def _unavailable(
    served_media_sha256: str | None,
    reason: str,
    *,
    timing_identical: bool | None = None,
) -> ClipAnalysisResponse:
    return ClipAnalysisResponse(
        state="unavailable",
        served_media_sha256=served_media_sha256,
        reason=reason,
        served_timing_identical=timing_identical,
    )
