"""Evidence clip playback, labeling, and audit routes."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.app.features.clips.audit_log import AuditLogStore, post_backend_backup, utc_now_iso
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


class ClipManifestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    event_ref: str = Field(min_length=1)
    event_type: str | None = Field(default=None, min_length=1)
    started_at: str = Field(min_length=1)
    duration_s: float = Field(ge=0)
    codec: str = Field(default="")
    path: str | None = Field(default=None)
    video_available: bool
    video_error: str | None = Field(default=None)
    finalized: bool
    size_bytes: int | None = Field(default=None, ge=0)


class ListClipsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clips: list[ClipManifestResponse]


class LabelClipRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["TRUE_POSITIVE", "FALSE_POSITIVE"] | None
    reviewer: str | None = Field(default=None, min_length=1)


class LabelClipResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    label: Literal["TRUE_POSITIVE", "FALSE_POSITIVE"] | None
    reviewer: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)


class AuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[dict[str, object]]


@router.get("/clips", response_model=ListClipsResponse)
def list_clips(
    request: Request,
    camera_id: str | None = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    store = _clip_store(request)
    manifests = store.list_manifests(camera_id=camera_id)
    return {"clips": [_clip_response(store, manifest) for manifest in manifests]}


def _clip_response(store: ClipStore, manifest: ClipManifest) -> dict[str, object]:
    response = manifest.as_response()
    response["size_bytes"] = _clip_size_bytes(store, manifest)
    return response


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
    post_backend_backup("clip_label", record.as_response())
    _audit_store(request).append(actor=reviewer, action="label", clip_id=manifest.clip_id)
    if not persisted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="label store unavailable",
        )
    return record.as_response()


@router.get("/audit", response_model=AuditResponse)
def list_audit(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    return {"entries": _audit_store(request).list_entries()}


def _get_manifest_or_404(request: Request, clip_id: str):
    try:
        manifest = _clip_store(request).get_manifest(clip_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
