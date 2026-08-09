"""Evidence clip playback, labeling, and audit routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from backend.app.features.clips.audit_log import (
    AUDIT_NO_CLIP_ID,
    AuditLogStore,
    post_backend_backup,
    utc_now_iso,
)
from backend.app.features.clips.listing import select_clip_page
from backend.app.features.clips.listing_index import (
    ClipListingIndex,
    ClipListingReconcileError,
)
from backend.app.features.clips.schemas import (
    AuditResponse,
    ClipListQuery,
    ClipManifestResponse,
    ClipsPaginationResponse,
    LabelClipRequest,
    LabelClipResponse,
    ListClipsResponse,
)
from backend.app.features.clips.store import ClipManifest, ClipStore, LabelRecord, LabelStore
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["clips"])

_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
}


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
        clips = [_clip_response(store, manifest) for manifest in page.manifests]
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
            ClipManifestResponse.model_validate(manifest.as_response())
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


def _clip_response(store: ClipStore, manifest: ClipManifest) -> ClipManifestResponse:
    response = manifest.as_response()
    response["size_bytes"] = _clip_size_bytes(store, manifest)
    return ClipManifestResponse.model_validate(response)


def _clip_size_bytes(store: ClipStore, manifest: ClipManifest) -> int | None:
    if not manifest.video_available:
        return None
    try:
        video_path = store.resolve_video_path(manifest)
    except (ValueError, FileNotFoundError):
        return None
    try:
        return video_path.stat().st_size
    except OSError:
        return None


@router.get("/clips/{clip_id}/metadata", response_model=ClipManifestResponse)
def get_clip_metadata(
    clip_id: str,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> ClipManifestResponse:
    actor = _authorize(request, authorization)
    store = _clip_store(request)
    manifest = _get_manifest_or_404(request, clip_id)
    response = _clip_response(store, manifest)
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
) -> FileResponse:
    actor = _authorize(request, authorization, query_token=token)
    manifest = _get_manifest_or_404(request, clip_id)
    if not manifest.video_available or manifest.path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="clip video not available",
        )
    try:
        video_path = _clip_store(request).resolve_video_path(manifest)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="clip video not found",
        ) from exc
    _audit_store(request).append(actor=actor, action="play", clip_id=manifest.clip_id)
    return FileResponse(video_path, media_type=_media_type(video_path.name))


@router.put("/clips/{clip_id}/label", response_model=LabelClipResponse)
def label_clip(
    clip_id: str,
    payload: LabelClipRequest,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    actor = _authorize(request, authorization)
    manifest = _get_manifest_or_404(request, clip_id)
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


def _get_manifest_or_404(request: Request, clip_id: str):
    store = _clip_store(request)
    try:
        manifest = store.get_manifest(clip_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if manifest is None:
        manifest = next(
            (candidate for candidate in store.list_manifests() if candidate.clip_id == clip_id),
            None,
        )
    if manifest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip not found")
    return manifest


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


def _media_type(filename: str) -> str:
    suffix = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    return _MEDIA_TYPES.get(f".{suffix}", "application/octet-stream")


__all__ = ["router"]
