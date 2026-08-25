"""Authenticated ml-worker to backend evidence export relay."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Never, Protocol, runtime_checkable
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.http import AuditUnavailableError, append_transactional
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.features.clips.store import CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR
from backend.app.features.evidence.compact_receipts import CompactArtifactReceiptStore
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    ArtifactReceiptPersistenceError,
    ArtifactReceiptStore,
    ArtifactReceiptVerificationError,
    VerifiedArtifact,
    verified_artifact,
)
from backend.app.features.relay.auth import authorize_relay as _authorize
from backend.app.features.relay.router import RELAY_TOKEN_HEADER, _camera_binding
from backend.app.features.runtime_settings.store import get_runtime_settings_store
from backend.app.shared.backend_client_bundle import backend_client_bundle
from shared.events.evidence_export_client import ReadyClipRequest, UnavailableClipRequest
from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
)

_LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/relay", tags=["relay"])
CLIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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
    if not _enabled(request) or CLIP_ID_PATTERN.fullmatch(clip_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip export unavailable")
    binding = _camera_binding(request, payload.camera_id, payload.facility_id)
    bound_camera_id = binding.get("backend_camera_id")
    if not isinstance(bound_camera_id, str) or not bound_camera_id.strip():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="camera has no backend mapping; clip export cannot address the backend",
        )
    client = _backend_client(request, bound_camera_id)
    if isinstance(payload, ReadyClipPayload):
        try:
            media = _verified_media(request, clip_id, payload)
        except ArtifactReceiptVerificationError as exc:
            raise HTTPException(
                status_code=(
                    status.HTTP_404_NOT_FOUND
                    if str(exc) == "artifact is unavailable"
                    else status.HTTP_409_CONFLICT
                ),
                detail="clip media unavailable"
                if str(exc) == "artifact is unavailable"
                else "clip media mismatch",
            ) from exc
        try:
            receipt = ArtifactReceipt(clip_id, payload.sha256, payload.size_bytes)
            receipt_store = _receipt_store(request)
            if isinstance(receipt_store, CompactArtifactReceiptStore):
                _ = receipt_store.commit_verified(
                    receipt,
                    media,
                    after_write=lambda connection: append_transactional(
                        request,
                        connection,
                        AuditEvent(
                            occurred_at=utc_now(), actor_id="worker-relay",
                            action=AuditAction.EVIDENCE_RECEIPT, target_id=clip_id,
                            detail=empty_detail(AuditAction.EVIDENCE_RECEIPT),
                            actor_type=AuditActorType.SERVICE,
                            auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
                        ),
                    ),
                )
            else:
                _ = receipt_store.commit(receipt)
        except AuditUnavailableError:
            media.handle.close()
            raise
        except ArtifactReceiptVerificationError as exc:
            media.handle.close()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="clip media changed before receipt commit",
            ) from exc
        except ArtifactReceiptConflictError as exc:
            media.handle.close()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="artifact receipt conflicts"
            ) from exc
        except (ArtifactReceiptPersistenceError, OSError, sqlite3.Error) as exc:
            media.handle.close()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="artifact receipt persistence unavailable",
            ) from exc
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
        with media.handle:
            result = client.publish_ready(request_payload, media.handle)
    else:
        result = client.report_unavailable(
            _UnavailableRequest(
                clip_id=clip_id,
                camera_id=bound_camera_id,
                event_refs=tuple(payload.event_refs),
                state_version=payload.state_version,
                reason=payload.reason,
            )
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


def _verified_media(
    request: Request,
    clip_id: str,
    payload: ReadyClipPayload,
) -> VerifiedArtifact:
    root_value = getattr(request.app.state, "clip_store_root", None)
    root = Path(root_value or os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))
    directory_fds: list[int] = []
    media_fd: int | None = None
    try:
        directory_fds.append(os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        for component in ("clips", clip_id):
            directory_fds.append(
                os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fds[-1],
                )
            )
        media_fd = os.open(
            "clip.mp4",
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fds[-1],
        )
        file_stat = os.fstat(media_fd)
    except OSError as exc:
        raise ArtifactReceiptVerificationError("artifact is unavailable") from exc
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    if media_fd is None:
        raise ArtifactReceiptVerificationError("artifact is unavailable")
    handle = os.fdopen(media_fd, "rb", closefd=True)
    if not stat.S_ISREG(file_stat.st_mode):
        handle.close()
        raise ArtifactReceiptVerificationError("artifact is not regular")
    verified = verified_artifact(handle)
    if (verified.sha256, verified.size_bytes) != (payload.sha256, payload.size_bytes):
        handle.close()
        raise ArtifactReceiptVerificationError("artifact does not match receipt")
    return verified


def _enabled(request: Request) -> bool:
    return get_runtime_settings_store(request.app).get().clip_export_enabled


def _receipt_store(request: Request) -> ArtifactReceiptStore:
    store = getattr(request.app.state, "artifact_receipt_store", None)
    if not isinstance(store, ArtifactReceiptStore):
        root_value = getattr(request.app.state, "clip_store_root", None)
        root = Path(root_value or os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))
        store = CompactArtifactReceiptStore(EDGE_DATABASE_PATH, root)
        request.app.state.artifact_receipt_store = store
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
