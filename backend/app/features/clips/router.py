"""Evidence clip playback, labeling, and audit routes."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal, cast

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response
from pydantic import ValidationError

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.clips.artifacts import (
    CentralClipArtifactQuery,
    open_verified_annotated,
)
from backend.app.features.clips.audit_log import (
    AUDIT_NO_CLIP_ID,
    AuditLogStore,
    post_backend_backup,
    utc_now_iso,
)
from backend.app.features.clips.compact_listing import (
    CompactClipConflictError,
    CompactClipListing,
    CompactClipQuery,
)
from backend.app.features.clips.deletion_control import control_clip_deletion
from backend.app.features.clips.derivative_control import control_derivative
from backend.app.features.clips.manifest import is_valid_clip_id
from backend.app.features.clips.media_response import media_response, media_type
from backend.app.features.clips.responses import clip_response, resolved_video_size
from backend.app.features.clips.schemas import (
    ArtifactState,
    AuditResponse,
    ClipAnalysisResponse,
    ClipAnalysisValueResponse,
    ClipArtifactViewsResponse,
    ClipDeleteStatus,
    ClipDerivativeResponse,
    ClipListQuery,
    ClipManifestResponse,
    ClipsPaginationResponse,
    DeleteClipRequest,
    DeleteClipResponse,
    LabelClipRequest,
    LabelClipResponse,
    ListClipsResponse,
)
from backend.app.features.clips.store import (
    ClipStore,
    DuplicateClipIdError,
    LabelRecord,
    LabelStore,
    LocatedClip,
)
from backend.app.features.evidence.receipt_store import (
    ArtifactReceiptStore,
    ArtifactReceiptVerificationError,
    verify_artifact,
)
from backend.app.shared.backend_client_bundle import backend_client_bundle
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["clips"])


@router.get("/clips", response_model=ListClipsResponse)
def list_clips(
    request: Request,
    filters: Annotated[ClipListQuery, Query()],
) -> ListClipsResponse:
    actor = _authorize(request)
    store = _clip_store(request)
    try:
        page = CompactClipListing(EDGE_DATABASE_PATH).rebuild_and_page(
            store,
            CompactClipQuery(
                camera_id=filters.camera_id,
                event_type=filters.event_type,
                limit=filters.limit or 100,
                cursor=filters.cursor,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DuplicateClipIdError as exc:
        raise _duplicate_clip_http_error(exc) from exc
    except (CompactClipConflictError, OSError, sqlite3.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="clip listing rebuild unavailable",
        ) from exc
    clips = [
        clip_response(
            manifest,
            resolved_video_size(store, manifest),
            store.thumbnail_available(manifest.clip_id),
        )
        for manifest in page.manifests
    ]
    response = ListClipsResponse(
        clips=clips,
        pagination=ClipsPaginationResponse(
            limit=filters.limit,
            offset=filters.offset,
            total=page.total,
            has_more=page.has_more,
            next_cursor=getattr(page, "next_cursor", None),
        ),
        event_type_counts=dict(page.event_type_counts),
    )
    _ = _audit_store(request).append(
        actor=actor,
        action="list",
        clip_id=AUDIT_NO_CLIP_ID,
        backend_token=_backend_token(request),
    )
    return response


@router.get("/clips/{clip_id}/metadata", response_model=ClipManifestResponse)
def get_clip_metadata(
    clip_id: str,
    request: Request,
) -> ClipManifestResponse:
    actor = _authorize(request)
    store = _clip_store(request)
    located = _get_located_clip_or_404(request, clip_id)
    manifest = located.manifest
    response = clip_response(
        manifest,
        resolved_video_size(store, located),
        store.thumbnail_available(located),
    )
    _ = _audit_store(request).append(
        actor=actor,
        action="metadata-view",
        clip_id=manifest.clip_id,
        backend_token=_backend_token(request),
    )
    return response


@router.get("/clips/{clip_id}/artifacts", response_model=ClipArtifactViewsResponse)
def clip_artifacts(
    clip_id: str,
    request: Request,
) -> ClipArtifactViewsResponse:
    actor = _authorize(request)
    manifest = _get_located_clip_or_404(request, clip_id).manifest
    artifacts = _artifact_query(request).get(clip_id)
    clean_state = (
        "AVAILABLE" if manifest.video_available and manifest.path is not None else "UNAVAILABLE"
    )
    analysis_state = "AVAILABLE" if artifacts is not None and artifacts.analysis else "UNAVAILABLE"
    annotated_state = cast(
        ArtifactState,
        "NOT_REQUESTED" if artifacts is None else artifacts.annotated_state,
    )
    annotated_available = annotated_state == "AVAILABLE"
    _ = _audit_store(request).append(
        actor=actor,
        action="artifact-view",
        clip_id=manifest.clip_id,
        backend_token=_backend_token(request),
    )
    return ClipArtifactViewsResponse(
        clip_id=manifest.clip_id,
        clean=clean_state,
        analysis=analysis_state,
        annotated=annotated_state,
        playback_view="annotated" if annotated_available else "clean",
        annotated_fallback_to_clean=not annotated_available,
    )


@router.post(
    "/clips/{clip_id}/derivatives/{kind}",
    response_model=ClipDerivativeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_clip_derivative(
    clip_id: str,
    kind: Literal["still", "video"],
    request: Request,
) -> ClipDerivativeResponse:
    actor = _authorize(request)
    manifest = _get_located_clip_or_404(request, clip_id).manifest
    payload = control_derivative(request, clip_id, kind, "request")
    _ = _audit_store(request).append(
        actor=actor,
        action=f"derivative-{kind}-request",
        clip_id=manifest.clip_id,
        backend_token=_backend_token(request),
    )
    return _derivative_response(request, clip_id, kind, payload)


@router.get(
    "/clips/{clip_id}/derivatives/{kind}",
    response_model=ClipDerivativeResponse,
)
def get_clip_derivative(
    clip_id: str,
    kind: Literal["still", "video"],
    request: Request,
) -> ClipDerivativeResponse:
    actor = _authorize(request)
    manifest = _get_located_clip_or_404(request, clip_id).manifest
    payload = control_derivative(request, clip_id, kind, "status")
    _ = _audit_store(request).append(
        actor=actor,
        action=f"derivative-{kind}-status",
        clip_id=manifest.clip_id,
        backend_token=_backend_token(request),
    )
    return _derivative_response(request, clip_id, kind, payload)


@router.delete(
    "/clips/{clip_id}/derivatives/{kind}",
    response_model=ClipDerivativeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def cancel_clip_derivative(
    clip_id: str,
    kind: Literal["still", "video"],
    request: Request,
) -> ClipDerivativeResponse:
    actor = _authorize(request)
    manifest = _get_located_clip_or_404(request, clip_id).manifest
    payload = control_derivative(request, clip_id, kind, "cancel")
    _ = _audit_store(request).append(
        actor=actor,
        action=f"derivative-{kind}-cancel",
        clip_id=manifest.clip_id,
        backend_token=_backend_token(request),
    )
    return _derivative_response(request, clip_id, kind, payload)


def _derivative_response(
    request: Request,
    clip_id: str,
    kind: Literal["still", "video"],
    payload: dict[str, object],
) -> ClipDerivativeResponse:
    artifacts = _artifact_query(request).get(clip_id)
    projection = None
    if artifacts is not None:
        projection = artifacts.still if kind == "still" else artifacts.video
    if projection is not None:
        payload = payload | {
            "mime_type": projection.mime_type,
            "sha256": projection.sha256,
            "size_bytes": projection.size_bytes,
            "width": projection.width,
            "height": projection.height,
            "start_time_ms": projection.start_time_ms,
            "end_time_ms": projection.end_time_ms,
            "render_backend": projection.render_backend,
            "render_version": projection.render_version,
            "scene_id": projection.scene_id,
            "primary_clip_id": projection.primary_clip_id,
            "decision_trace_id": projection.decision_trace_id,
            "runtime_manifest_sha256": projection.runtime_manifest_sha256,
        }
    return ClipDerivativeResponse.model_validate(payload)


_CLIP_DELETE_AUDIT_ACTION: dict[ClipDeleteStatus, str] = {
    "PURGED": "clip-delete-completed",
    "HELD": "clip-delete-held",
    "MISSING": "clip-delete-failed",
    "UNVERIFIABLE": "clip-delete-failed",
    "DELETE_FAILED": "clip-delete-failed",
    "VERIFICATION_FAILED": "clip-delete-failed",
}


@router.delete(
    "/clips/{clip_id}",
    response_model=DeleteClipResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def delete_clip(
    clip_id: str,
    payload: DeleteClipRequest,
    request: Request,
) -> DeleteClipResponse:
    """Operator-only, explicitly-confirmed primary clip deletion.

    Routes through the worker's control seam (``control_clip_deletion``,
    mirroring ``control_derivative``) rather than deleting anything itself --
    the backend never touches worker-owned evidence tables or clip-store
    files. The response is always ``202`` with a typed, truthful ``status``.

    Deliberately does **not** gate on ``_get_located_clip_or_404`` (the
    filesystem-backed clip catalog) the way the read/derivative routes do: a
    successful delete removes exactly what that catalog looks up, so gating
    on it would make the second half of ``duplicate PENDING/PURGED requests
    are idempotent`` (the contract this route exists to satisfy) impossible --
    a duplicate request after a real purge would wrongly 404 instead of
    reporting the still-true ``PURGED`` tombstone. The worker is the
    authoritative owner and already reports a clip it has never heard of as
    the truthful, distinct ``MISSING`` status instead.
    """
    actor = _authorize(request)
    if not is_valid_clip_id(clip_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip not found")
    if payload.confirm_clip_id != clip_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="clip id confirmation does not match",
        )
    backend_token = _backend_token(request)
    try:
        result = control_clip_deletion(request, clip_id)
    except HTTPException:
        _ = _audit_store(request).append(
            actor=actor,
            action="clip-delete-failed",
            clip_id=clip_id,
            backend_token=backend_token,
        )
        raise
    try:
        response = DeleteClipResponse.model_validate(result)
    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="worker clip deletion response is invalid",
        ) from error
    audit = _audit_store(request)
    action = _CLIP_DELETE_AUDIT_ACTION[response.status]
    # A converged PURGED retry is not a second operator action. Keep one
    # durable completion record while still returning the truthful result.
    if not (
        response.status == "PURGED"
        and any(
            entry.get("action") == action and entry.get("clip_id") == clip_id
            for entry in audit.list_entries()
        )
    ):
        _ = audit.append(
            actor=actor,
            action=action,
            clip_id=clip_id,
            backend_token=backend_token,
        )
    return response


@router.get("/clips/{clip_id}/analysis", response_model=ClipAnalysisResponse)
def clip_analysis(
    clip_id: str,
    request: Request,
) -> ClipAnalysisResponse:
    actor = _authorize(request)
    manifest = _get_located_clip_or_404(request, clip_id).manifest
    artifacts = _artifact_query(request).get(clip_id)
    if artifacts is None or artifacts.analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="clip analysis not available",
        )
    analysis = artifacts.analysis
    _ = _audit_store(request).append(
        actor=actor,
        action="analysis-view",
        clip_id=manifest.clip_id,
        backend_token=_backend_token(request),
    )
    return ClipAnalysisResponse(
        clip_id=manifest.clip_id,
        decision_trace_id=analysis.decision_trace_id,
        module_qualified_id=analysis.module_qualified_id,
        policy_qualified_id=analysis.policy_qualified_id,
        effective_policy_id=analysis.effective_policy_id,
        runtime_manifest_sha256=analysis.runtime_manifest_sha256,
        reason=analysis.reason,
        previous_state=analysis.previous_state,
        current_state=analysis.current_state,
        triggered=analysis.triggered,
        track_id=analysis.track_id,
        bed_id=analysis.bed_id,
        values=[
            ClipAnalysisValueResponse(name=name, value=value, missing_reason=missing_reason)
            for name, value, missing_reason in analysis.values
        ],
    )


@router.get("/clips/{clip_id}/video")
def clip_video(
    clip_id: str,
    request: Request,
    view: Annotated[Literal["clean", "annotated"], Query()] = "clean",
) -> Response:
    actor = _authorize(request)
    located = _get_located_clip_or_404(request, clip_id)
    manifest = located.manifest
    opened = None
    actual_view = "clean"
    if view == "annotated":
        artifacts = _artifact_query(request).get(clip_id)
        if artifacts is not None:
            opened = open_verified_annotated(_clip_store(request).root, artifacts)
            if opened is not None:
                actual_view = "annotated"
    if opened is None:
        if not manifest.video_available or manifest.path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="clip video not available",
            )
        try:
            opened = _clip_store(request).open_located_video(located)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="clip video not found",
            ) from exc
    if actual_view == "clean":
        receipt_store = getattr(request.app.state, "artifact_receipt_store", None)
        receipt = (
            receipt_store.get(manifest.clip_id)
            if isinstance(receipt_store, ArtifactReceiptStore)
            else None
        )
        # A receipt is proof the served bytes are the recorded ones, so when one
        # exists it is enforced below without exception. Its ABSENCE is not
        # evidence of tampering: a receipt is only committed after a successful
        # upstream export, which needs clip export enabled (Hub-owned config,
        # off by default) and a Hub-issued camera id. Requiring one before an
        # operator may review local footage made every clip on this deployment
        # permanently unplayable -- verified media on disk, thumbnail and all,
        # answering "영상을 재생하지 못했습니다" forever. Evidence a carer cannot
        # watch is evidence the system did not capture.
        if receipt is not None and not receipt.accepted:
            opened.handle.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="clip video receipt not accepted",
            )
        try:
            if receipt is not None:
                verify_artifact(opened.path, receipt)
        except ArtifactReceiptVerificationError as exc:
            opened.handle.close()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="clip video receipt verification failed",
            ) from exc
    _ = _audit_store(request).append(
        actor=actor,
        action="play-annotated" if actual_view == "annotated" else "play",
        clip_id=manifest.clip_id,
        backend_token=_backend_token(request),
    )
    response = media_response(
        opened,
        request.headers.get("range"),
        media_type(opened.path.name),
    )
    response.headers["X-Clip-View"] = actual_view
    if view != actual_view:
        response.headers["X-Clip-View-Fallback"] = actual_view
    return response


@router.get("/clips/{clip_id}/thumbnail")
def clip_thumbnail(
    clip_id: str,
    request: Request,
) -> Response:
    _ = _authorize(request)
    store = _clip_store(request)
    located = _get_located_clip_or_404(request, clip_id)
    try:
        content = store.read_thumbnail(located)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="clip thumbnail not found",
        ) from exc
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store"},
    )


@router.put("/clips/{clip_id}/label", response_model=LabelClipResponse)
def label_clip(
    clip_id: str,
    payload: LabelClipRequest,
    request: Request,
) -> dict[str, object]:
    actor = _authorize(request)
    manifest = _get_located_clip_or_404(request, clip_id).manifest
    reviewer = payload.reviewer or actor
    record = LabelRecord(
        clip_id=manifest.clip_id,
        label=payload.label,
        reviewer=reviewer,
        reviewed_at=utc_now_iso(),
    )
    persisted = _label_store(request).save(record)
    if not persisted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="label store unavailable",
        )
    backend_token = _backend_token(request)
    post_backend_backup("clip_label", record.as_response(), backend_token=backend_token)
    _audit_store(request).append(
        actor=reviewer,
        action="label",
        clip_id=manifest.clip_id,
        backend_token=backend_token,
    )
    return record.as_response()


@router.get("/audit", response_model=AuditResponse)
def list_audit(
    request: Request,
) -> dict[str, object]:
    actor = _authorize(request)
    store = _audit_store(request)
    # Snapshot entries before recording this view so the response reflects
    # the log state prior to this request, matching how "play"/"label" audit
    # entries never appear until after their triggering action completes.
    entries = store.list_entries()
    store.append(
        actor=actor,
        action="audit-view",
        clip_id=AUDIT_NO_CLIP_ID,
        backend_token=_backend_token(request),
    )
    return {"entries": entries}


def _get_located_clip_or_404(request: Request, clip_id: str) -> LocatedClip:
    store = _clip_store(request)
    try:
        located = store.locate_manifest(clip_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DuplicateClipIdError as exc:
        raise _duplicate_clip_http_error(exc) from exc
    if located is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip not found")
    return located


def _duplicate_clip_http_error(exc: DuplicateClipIdError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=f"duplicate clip_id: {exc.clip_id}",
    )


def _clip_store(request: Request) -> ClipStore:
    store = getattr(request.app.state, "clip_store", None)
    if not isinstance(store, ClipStore):
        store = ClipStore.from_env()
        request.app.state.clip_store = store
    return store


def _artifact_query(request: Request) -> CentralClipArtifactQuery:
    query = getattr(request.app.state, "central_clip_artifact_query", None)
    if not isinstance(query, CentralClipArtifactQuery):
        query = CentralClipArtifactQuery()
        request.app.state.central_clip_artifact_query = query
    return query


def _label_store(request: Request) -> LabelStore:
    store = getattr(request.app.state, "clip_label_store", None)
    if not isinstance(store, LabelStore):
        store = LabelStore.from_env()
        request.app.state.clip_label_store = store
    return store


def _audit_store(request: Request) -> AuditLogStore:
    store = getattr(request.app.state, "clip_audit_log", None)
    if not isinstance(store, AuditLogStore):
        store = AuditLogStore.from_env()
        request.app.state.clip_audit_log = store
    return store


def _backend_token(request: Request) -> str | None:
    bundle = backend_client_bundle(request.app)
    return None if bundle is None else bundle.facility_token


def _authorize(request: Request) -> str:
    return authorize_dashboard(request)


__all__ = ["router"]
