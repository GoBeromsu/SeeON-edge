"""Evidence clip playback and audit routes."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.http import AuditUnavailableError, append_governed
from backend.app.features.clips.artifacts import CentralClipArtifactQuery
from backend.app.features.clips.compact_listing import (
    CompactClipConflictError,
    CompactClipListing,
    CompactClipQuery,
)
from backend.app.features.clips.deletion_control import (
    control_clip_deletion,
    preflight_clip_deletion,
)
from backend.app.features.clips.deletion_lifecycle import ClipDeletionLifecycle
from backend.app.features.clips.manifest import is_valid_clip_id
from backend.app.features.clips.media_response import media_response, media_type
from backend.app.features.clips.responses import clip_response, resolved_video_size
from backend.app.features.clips.schemas import (
    CleanArtifactState,
    ClipArtifactViewsResponse,
    ClipListQuery,
    ClipManifestResponse,
    ClipsPaginationResponse,
    DeleteClipRequest,
    DeleteClipResponse,
    ListClipsResponse,
    SnapshotArtifactState,
)
from backend.app.features.clips.store import (
    ClipStore,
    DuplicateClipIdError,
    LocatedClip,
)
from backend.app.features.evidence.receipt_store import (
    ArtifactReceiptStore,
    ArtifactReceiptVerificationError,
    verify_artifact,
)
from backend.app.shared.dashboard_auth import authorize_dashboard
from backend.app.shared.head_response import HEAD_METHODS, drop_body_for_head


@dataclass(frozen=True, slots=True)
class _DeletionCommandResult:
    clip_id: str
    status: str


def _validate_deletion_payload(
    payload: dict[str, object], clip_id: str, *, allow_ready: bool
) -> _DeletionCommandResult:
    allowed = {
        "PURGED", "HELD", "MISSING", "UNVERIFIABLE", "DELETE_FAILED",
        "VERIFICATION_FAILED",
    }
    if allow_ready:
        allowed.add("READY")
    if (
        set(payload) != {"clip_id", "status"}
        or payload.get("clip_id") != clip_id
        or payload.get("status") not in allowed
    ):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="worker clip deletion response is invalid",
        )
    return _DeletionCommandResult(clip_id, str(payload["status"]))


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
            located.manifest,
            resolved_video_size(store, located),
            store.thumbnail_available(located),
        )
        for located in page.clips
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
    append_governed(
        request, actor_id=actor, action=AuditAction.CLIP_LIST, target_id="clips"
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
    append_governed(
        request, actor_id=actor, action=AuditAction.CLIP_DETAIL, target_id=manifest.clip_id
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
    clean_state: CleanArtifactState = (
        "AVAILABLE" if manifest.video_available and manifest.path is not None else "UNAVAILABLE"
    )
    snapshot_states: dict[str, SnapshotArtifactState] = {
        "PENDING": "PENDING",
        "AVAILABLE": "AVAILABLE",
        "UNAVAILABLE": "UNAVAILABLE",
        "CORRUPT": "CORRUPT",
        "PURGED": "PURGED",
    }
    snapshot = (
        snapshot_states.get(artifacts.snapshot_state)
        if artifacts is not None and artifacts.snapshot_state is not None
        else None
    )
    append_governed(
        request, actor_id=actor, action=AuditAction.CLIP_ARTIFACT, target_id=manifest.clip_id
    )
    return ClipArtifactViewsResponse(
        clip_id=manifest.clip_id,
        clean=clean_state,
        snapshot=snapshot,
    )


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

    Routes through the worker's control seam (``control_clip_deletion``)
    rather than deleting anything itself -- the backend never touches
    worker-owned evidence tables or clip-store files. The response is always
    ``202`` with a typed, truthful ``status``.

    Deliberately does **not** gate on ``_get_located_clip_or_404`` (the
    filesystem-backed clip catalog) the way the read routes do: a successful
    delete removes exactly what that catalog looks up, so gating on it would
    make the second half of ``duplicate PENDING/PURGED requests are
    idempotent`` (the contract this route exists to satisfy) impossible -- a
    duplicate request after a real purge would wrongly 404 instead of
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
    lifecycle = ClipDeletionLifecycle(EDGE_DATABASE_PATH, request.app)
    retention_state = lifecycle.state(clip_id)
    if retention_state == "PURGED":
        return DeleteClipResponse(clip_id=clip_id, status="PURGED")

    preflight = _validate_deletion_payload(
        preflight_clip_deletion(request, clip_id), clip_id, allow_ready=True
    )
    if preflight.status == "HELD":
        return DeleteClipResponse(clip_id=clip_id, status="HELD")
    if preflight.status == "MISSING":
        if retention_state == "PENDING":
            lifecycle.complete(clip_id, actor_id=actor)
            return DeleteClipResponse(clip_id=clip_id, status="PURGED")
        return DeleteClipResponse(clip_id=clip_id, status="MISSING")
    if preflight.status != "READY":
        return DeleteClipResponse.model_validate(
            {"clip_id": preflight.clip_id, "status": preflight.status}
        )

    pending_state = lifecycle.begin(clip_id, actor)
    if pending_state is None:
        return DeleteClipResponse(clip_id=clip_id, status="MISSING")
    if pending_state == "PURGED":
        return DeleteClipResponse(clip_id=clip_id, status="PURGED")

    response = _validate_deletion_payload(
        control_clip_deletion(request, clip_id), clip_id, allow_ready=False
    )
    if response.status in {"PURGED", "MISSING"}:
        lifecycle.complete(clip_id, actor_id=actor)
        return DeleteClipResponse(clip_id=clip_id, status="PURGED")
    return DeleteClipResponse.model_validate(
        {"clip_id": response.clip_id, "status": response.status}
    )


# HEAD answers with the GET header section and no body (issue #452): a
# player probes content-type/length/accept-ranges before it opens a clip,
# and FastAPI does not synthesise HEAD from GET. One endpoint serves both
# so the headers, the receipt gate, the range handling and the audit trail
# cannot drift between the methods; OpenedFileResponse suppresses the file
# reads for HEAD, so the probe stays cheap.
@router.api_route("/clips/{clip_id}/video", methods=HEAD_METHODS)
def clip_video(
    clip_id: str,
    request: Request,
) -> Response:
    actor = _authorize(request)
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
    response = media_response(
        opened,
        request.headers.get("range"),
        media_type(opened.path.name),
    )
    if response.status_code >= status.HTTP_400_BAD_REQUEST:
        return response
    try:
        append_governed(
            request, actor_id=actor, action=AuditAction.CLIP_PLAY, target_id=manifest.clip_id
        )
    except AuditUnavailableError:
        opened.handle.close()
        raise
    return response


@router.api_route("/clips/{clip_id}/thumbnail", methods=HEAD_METHODS)
def clip_thumbnail(
    clip_id: str,
    request: Request,
) -> Response:
    actor = _authorize(request)
    store = _clip_store(request)
    located = _get_located_clip_or_404(request, clip_id)
    try:
        content = store.read_thumbnail(located)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="clip thumbnail not found",
        ) from exc
    append_governed(
        request, actor_id=actor, action=AuditAction.CLIP_THUMBNAIL, target_id=clip_id
    )
    # A thumbnail's Content-Length is only knowable from the bytes themselves
    # (they arrive through one bounded, containment-checked read), so HEAD runs
    # the identical path and drops the body last -- headers stay byte-identical
    # to the GET, and the read stays capped at MAX_THUMBNAIL_BYTES.
    return drop_body_for_head(
        request,
        Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, no-store"},
        ),
    )


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


def _authorize(request: Request) -> str:
    return authorize_dashboard(request)


__all__ = ["router"]
