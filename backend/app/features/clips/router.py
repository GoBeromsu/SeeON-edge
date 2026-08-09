"""Evidence clip playback, labeling, and audit routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import Response

from backend.app.features.clips.audit_log import (
    AUDIT_NO_CLIP_ID,
    AuditLogStore,
    post_backend_backup,
    utc_now_iso,
)
from backend.app.features.clips.listing import select_clip_page
from backend.app.features.clips.listing_index import ClipListingIndex, ClipListingReconcileError
from backend.app.features.clips.media_response import media_response, media_type
from backend.app.features.clips.responses import clip_response, resolved_video_size
from backend.app.features.clips.schemas import (
    AuditResponse,
    ClipListQuery,
    ClipManifestResponse,
    ClipsPaginationResponse,
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
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["clips"])

@router.get("/clips", response_model=ListClipsResponse)
def list_clips(
    request: Request,
    filters: Annotated[ClipListQuery, Query()],
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> ListClipsResponse:
    actor = _authorize(request, authorization)
    if filters.limit is None:
        store = _clip_store(request)
        page = select_clip_page(store.list_manifests(), filters)
        try:
            clips = [
                clip_response(
                    manifest,
                    resolved_video_size(store, manifest),
                    store.thumbnail_available(manifest.clip_id),
                )
                for manifest in page.manifests
            ]
        except DuplicateClipIdError as exc:
            raise _duplicate_clip_http_error(exc) from exc
    else:
        index = getattr(request.app.state, "clip_listing_index", None)
        if not isinstance(index, ClipListingIndex):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="clip listing index unavailable",
            )
        try:
            page = index.page(filters)
        except ClipListingReconcileError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="clip listing index unavailable",
            ) from exc
        clips = [
            clip_response(
                manifest,
                manifest.size_bytes,
                manifest.thumbnail_available,
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
        ),
        event_type_counts=dict(page.event_type_counts),
    )
    _ = _audit_store(request).append(actor=actor, action="list", clip_id=AUDIT_NO_CLIP_ID)
    return response


@router.get("/clips/{clip_id}/metadata", response_model=ClipManifestResponse)
def get_clip_metadata(
    clip_id: str,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> ClipManifestResponse:
    actor = _authorize(request, authorization)
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
    )
    return response


@router.get("/clips/{clip_id}/video")
def clip_video(
    clip_id: str,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    token: Annotated[str | None, Query()] = None,
) -> Response:
    actor = _authorize(request, authorization, query_token=token)
    located = _get_located_clip_or_404(request, clip_id)
    manifest = located.manifest
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
    _audit_store(request).append(actor=actor, action="play", clip_id=manifest.clip_id)
    return media_response(
        opened,
        request.headers.get("range"),
        media_type(opened.path.name),
    )


@router.get("/clips/{clip_id}/thumbnail")
def clip_thumbnail(
    clip_id: str,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> Response:
    _ = _authorize(request, authorization)
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
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    actor = _authorize(request, authorization)
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
    post_backend_backup("clip_label", record.as_response())
    _audit_store(request).append(actor=reviewer, action="label", clip_id=manifest.clip_id)
    return record.as_response()


@router.get("/audit", response_model=AuditResponse)
def list_audit(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    actor = _authorize(request, authorization)
    store = _audit_store(request)
    # Snapshot entries before recording this view so the response reflects
    # the log state prior to this request, matching how "play"/"label" audit
    # entries never appear until after their triggering action completes.
    entries = store.list_entries()
    store.append(actor=actor, action="audit-view", clip_id=AUDIT_NO_CLIP_ID)
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


def _authorize(
    request: Request,
    authorization: str | None,
    *,
    query_token: str | None = None,
) -> str:
    supplied = _bearer_token(authorization) or query_token
    actor = authorize_dashboard(request, legacy_token=supplied)
    if actor != "legacy-dashboard":
        return actor
    return "operator" if query_token is not None else "bearer"


def _bearer_token(value: str | None) -> str | None:
    if value is None or not value.startswith("Bearer "):
        return None
    return value.removeprefix("Bearer ").strip() or None


__all__ = ["router"]
