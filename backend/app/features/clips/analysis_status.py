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
    try:
        identity = store.open_located_playback_identity(located)
    except (ValueError, FileNotFoundError):
        served_media_sha256 = None
        served_timing_identical = False
    else:
        try:
            served_media_sha256 = identity.served_media_sha256
            served_timing_identical = (
                identity.served_kind == "original" or identity.served_pts_identical
            )
        finally:
            identity.opened.handle.close()
    try:
        payloads = store.read_clip_analysis_candidates(located)
    except ValueError:
        return _unavailable(served_media_sha256, "artifact_invalid")
    if payloads:
        digest = store.manifest_video_sha256(located)
        identity_mismatch = False
        artifact_invalid = False
        for payload in payloads:
            try:
                result = decode_clip_analysis(payload)
            except ClipAnalysisWireError:
                artifact_invalid = True
                continue
            if digest is None or result.clip_sha256 != digest:
                identity_mismatch = True
                continue
            if not served_timing_identical:
                return _unavailable(
                    served_media_sha256, "timing_unverified", timing_identical=False
                )
            return ClipAnalysisResponse(
                state="available",
                served_media_sha256=served_media_sha256,
                served_timing_identical=True,
                result=result.as_dict(),
            )
        if identity_mismatch:
            return _unavailable(served_media_sha256, "identity_mismatch")
        if artifact_invalid:
            return _unavailable(served_media_sha256, "artifact_invalid")
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
    if state_value == "available" and not served_timing_identical:
        return _unavailable(served_media_sha256, "timing_unverified", timing_identical=False)
    return ClipAnalysisResponse(
        state=cast(Literal["idle", "running", "available", "failed"], state_value),
        served_media_sha256=served_media_sha256,
        reason=reason,
    )


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
