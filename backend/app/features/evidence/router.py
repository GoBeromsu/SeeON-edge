"""Authenticated ml-worker to backend evidence export relay."""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Never, Protocol, runtime_checkable
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.http import mutation_audit
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.features.clips.store import CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR
from backend.app.features.evidence.postgres_receipts import PostgresArtifactReceiptStore
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    ArtifactReceiptPersistenceError,
    ArtifactReceiptVerificationError,
    VerifiedArtifact,
    verified_artifact,
)
from backend.app.features.relay.auth import authorize_relay as _authorize
from backend.app.features.relay.router import RELAY_TOKEN_HEADER, _camera_binding
from backend.app.features.runtime_settings.store import get_runtime_settings_store
from backend.app.shared.backend_client_bundle import backend_client_bundle
from shared.events.clip_identity import is_clip_id
from shared.events.evidence_export_client import ReadyClipRequest, UnavailableClipRequest
from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    DeliveryFailureCode,
)

_LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/relay", tags=["relay"])


class CapabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_idempotency: Literal[1]
    clip_export: Literal[0, 1]


class _ClipBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    event_refs: list[str] = Field(min_length=1)
    state_version: int = Field(ge=1)

    @field_validator("event_refs")
    @classmethod
    def valid_event_refs(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("event_refs must be unique")
        for value in values:
            try:
                parsed = UUID(value)
            except ValueError as exc:
                raise ValueError("event_refs must be UUIDv4") from exc
            if parsed.version != 4 or str(parsed) != value:
                raise ValueError("event_refs must be canonical UUIDv4")
        return values


class ReadyClipPayload(_ClipBase):
    state: Literal["READY"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    mime_type: Literal["video/mp4"]
    codec: str = Field(min_length=1)
    duration_ms: int = Field(ge=1, le=120000)
    clip_start_at: str = Field(min_length=1)
    clip_end_at: str = Field(min_length=1)
    finalized_at: str = Field(min_length=1)


class UnavailableClipPayload(_ClipBase):
    state: Literal["UNAVAILABLE"]
    reason: Literal["CAPTURE_FAILED", "QUEUE_FULL", "CORRUPT", "UPLOAD_TIMEOUT"]


ClipExportPayload = Annotated[
    ReadyClipPayload | UnavailableClipPayload,
    Field(discriminator="state"),
]


class ClipReceiptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_id: str
    state: Literal["READY", "UNAVAILABLE", "EXPIRED"]
    state_version: int
    sha256: str | None
    size_bytes: int | None


@runtime_checkable
class BackendEvidenceClient(Protocol):
    def probe_capabilities(self, camera_id: str) -> BackendCapabilities | DeliveryFailure: ...

    def publish_ready(
        self, request: ReadyClipRequest, media: BinaryIO
    ) -> ClipReceipt | DeliveryFailure: ...

    def report_unavailable(
        self, request: UnavailableClipRequest
    ) -> ClipReceipt | DeliveryFailure: ...


@runtime_checkable
class CameraScopedEvidenceClient(Protocol):
    def for_camera(self, _camera_id: str) -> BackendEvidenceClient: ...


@dataclass(frozen=True, slots=True)
class _ReadyRequest:
    clip_id: str
    camera_id: str
    event_refs: tuple[str, ...]
    state_version: int
    sha256: str
    size_bytes: int
    mime_type: str
    codec: str
    duration_ms: int
    clip_start_at: str
    clip_end_at: str
    finalized_at: str


@dataclass(frozen=True, slots=True)
class _UnavailableRequest:
    clip_id: str
    camera_id: str
    event_refs: tuple[str, ...]
    state_version: int
    reason: str


@router.get("/capabilities", response_model=CapabilityResponse)
def capabilities(
    camera_id: str,
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> CapabilityResponse:
    _authorize(request, relay_token)
    if not _enabled(request):
        return CapabilityResponse(event_idempotency=1, clip_export=0)
    # Fourth Hub egress path. The worker supplies camera_id straight off the query
    # string, so without this the edge-local id addressed the backend even for a
    # mapped camera (issue #308). Resolve to the Hub-issued id, and when there is
    # none fall back to the same conservative answer the disabled path returns
    # rather than probing under an id the Hub never issued.
    binding = _camera_binding(request, camera_id, "")
    bound_camera_id = binding.get("backend_camera_id")
    if not isinstance(bound_camera_id, str) or not bound_camera_id.strip():
        _LOGGER.warning(
            "capabilities probe skipped: camera %s has no Hub mapping yet",
            camera_id,
        )
        return CapabilityResponse(event_idempotency=1, clip_export=0)
    client = _backend_client(request, bound_camera_id)
    result = client.probe_capabilities(bound_camera_id)
    if isinstance(result, DeliveryFailure):
        if result.disposition is DeliveryDisposition.COMPATIBILITY:
            return CapabilityResponse(event_idempotency=1, clip_export=0)
        _raise_failure(result)
    return CapabilityResponse(
        event_idempotency=result.event_idempotency,
        clip_export=result.clip_export,
    )


@router.put("/clips/{clip_id}", response_model=ClipReceiptResponse)
def export_clip(
    clip_id: str,
    payload: ClipExportPayload,
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> ClipReceiptResponse:
    _authorize(request, relay_token)
    if not _enabled(request) or not is_clip_id(clip_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip export unavailable")
    binding = _camera_binding(request, payload.camera_id, payload.facility_id)
    bound_camera_id = binding.get("backend_camera_id")
    if not isinstance(bound_camera_id, str) or not bound_camera_id.strip():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": DeliveryFailureCode.CAMERA_MAPPING_MISSING,
                "message": "camera has no backend mapping; clip export cannot address the backend",
            },
        )
    receipt_store = _receipt_store(request)
    audit = mutation_audit(
        request,
        lambda: AuditEvent(
            occurred_at=utc_now(),
            actor_id="worker-relay",
            action=AuditAction.EVIDENCE_RECEIPT,
            target_id=clip_id,
            detail=empty_detail(AuditAction.EVIDENCE_RECEIPT),
            actor_type=AuditActorType.SERVICE,
            auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
        ),
    )
    audit.require_admission(receipt_store)
    client = _backend_client(request, bound_camera_id)
    if isinstance(payload, ReadyClipPayload):
        media = _ready_media(request, clip_id, payload)
        # This route owns the supplied descriptor through persistence AND
        # network egress, including cancellation and post-COMMIT failures.
        with media.handle:
            receipt = ArtifactReceipt(clip_id, payload.sha256, payload.size_bytes)
            with _receipt_errors(ready=True):
                audit.apply(
                    receipt_store,
                    lambda append: receipt_store.commit_verified(
                        receipt, media, after_write=append
                    ),
                )
            request_payload = _ReadyRequest(
                clip_id=clip_id,
                camera_id=bound_camera_id,
                event_refs=tuple(payload.event_refs),
                state_version=payload.state_version,
                sha256=payload.sha256,
                size_bytes=payload.size_bytes,
                mime_type=payload.mime_type,
                codec=payload.codec,
                duration_ms=payload.duration_ms,
                clip_start_at=payload.clip_start_at,
                clip_end_at=payload.clip_end_at,
                finalized_at=payload.finalized_at,
            )
            audit.require_admission(receipt_store)
            result = client.publish_ready(request_payload, media.handle)
    else:
        audit.require_admission(receipt_store)
        result = client.report_unavailable(
            _UnavailableRequest(
                clip_id=clip_id,
                camera_id=bound_camera_id,
                event_refs=tuple(payload.event_refs),
                state_version=payload.state_version,
                reason=payload.reason,
            )
        )
        if not isinstance(result, DeliveryFailure):
            # Keep the explicit unavailable-receipt zero-audit contract.
            # Admission is rechecked after the external effect; refusing
            # local persistence cannot undo a report already sent upstream.
            with _receipt_errors(ready=False):
                audit.apply(
                    receipt_store,
                    lambda _append: receipt_store.commit_unavailable(clip_id, payload.reason),
                    expects_audit=lambda _result: False,
                )
    if isinstance(result, DeliveryFailure):
        _raise_failure(result)
    return ClipReceiptResponse(
        clip_id=result.clip_id,
        state=result.state,
        state_version=result.state_version,
        sha256=result.sha256,
        size_bytes=result.size_bytes,
    )


@contextmanager
def _receipt_errors(*, ready: bool) -> Iterator[None]:
    """Map persistence errors only; external transport is not a write owner."""
    try:
        yield
    except ArtifactReceiptVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="clip media changed before receipt commit"
            if ready
            else "clip manifest unavailable",
        ) from exc
    except ArtifactReceiptConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="artifact receipt conflicts"
        ) from exc
    except (ArtifactReceiptPersistenceError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="artifact receipt persistence unavailable",
        ) from exc


def _ready_media(request: Request, clip_id: str, payload: ReadyClipPayload) -> VerifiedArtifact:
    try:
        return _verified_media(request, clip_id, payload)
    except ArtifactReceiptVerificationError as exc:
        unavailable = str(exc) == "artifact is unavailable"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if unavailable else status.HTTP_409_CONFLICT,
            detail="clip media unavailable" if unavailable else "clip media mismatch",
        ) from exc


def _verified_media(
    request: Request,
    clip_id: str,
    payload: ReadyClipPayload,
) -> VerifiedArtifact:
    root_value = getattr(request.app.state, "clip_store_root", None)
    root = Path(root_value or os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))
    with ExitStack() as media_owner:
        try:
            with ExitStack() as directories:
                directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                directories.callback(os.close, directory_fd)
                for component in ("clips", clip_id):
                    directory_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    directories.callback(os.close, directory_fd)
                media_fd = os.open(
                    "clip.mp4",
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=directory_fd,
                )
                try:
                    handle = os.fdopen(media_fd, "rb", closefd=True)
                except BaseException:
                    os.close(media_fd)
                    raise
                media_owner.enter_context(handle)
                file_stat = os.fstat(handle.fileno())
        except OSError as exc:
            raise ArtifactReceiptVerificationError("artifact is unavailable") from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise ArtifactReceiptVerificationError("artifact is not regular")
        verified = verified_artifact(handle)
        if (verified.sha256, verified.size_bytes) != (payload.sha256, payload.size_bytes):
            raise ArtifactReceiptVerificationError("artifact does not match receipt")
        media_owner.pop_all()
        return verified


def _enabled(request: Request) -> bool:
    return get_runtime_settings_store(request.app).get().clip_export_enabled


def _receipt_store(request: Request) -> PostgresArtifactReceiptStore:
    store = getattr(request.app.state, "artifact_receipt_store", None)
    if store is None:
        raise RuntimeError("artifact receipt store is not injected")
    if not isinstance(store, PostgresArtifactReceiptStore):
        raise TypeError("artifact receipt store has invalid type")
    return store


def _backend_client(request: Request, camera_id: str) -> BackendEvidenceClient:
    bundle = backend_client_bundle(request.app)
    if bundle is not None:
        return _camera_client(bundle.evidence_client, camera_id)
    return _camera_client(
        getattr(request.app.state, "backend_evidence_client", None),
        camera_id,
    )


def _camera_client(client: object, camera_id: str) -> BackendEvidenceClient:
    if isinstance(client, CameraScopedEvidenceClient):
        return client.for_camera(camera_id)
    if isinstance(client, BackendEvidenceClient):
        return client
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="backend evidence export unavailable",
    )


def _raise_failure(failure: DeliveryFailure) -> Never:
    if failure.disposition is DeliveryDisposition.RETRY:
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif failure.disposition is DeliveryDisposition.COMPATIBILITY:
        code = status.HTTP_404_NOT_FOUND
    else:
        code = failure.status_code if failure.status_code in {400, 401, 403, 413, 415, 422} else 502
    headers = None
    if failure.retry_after_seconds is not None:
        headers = {"Retry-After": str(max(0, int(failure.retry_after_seconds)))}
    raise HTTPException(status_code=code, detail="backend evidence export failed", headers=headers)


__all__ = ["router"]
