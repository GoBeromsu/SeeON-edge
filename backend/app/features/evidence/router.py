"""Authenticated ml-worker to backend evidence export relay."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Protocol
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.features.clips.store import CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR
from backend.app.features.relay.router import RELAY_TOKEN_HEADER, _authorize, _camera_binding
from backend.app.shared.backend_client_bundle import backend_client_bundle
from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
)

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


class BackendEvidenceClient(Protocol):
    def for_camera(self, camera_id: str) -> BackendEvidenceClient: ...

    def probe_capabilities(self, camera_id: str) -> BackendCapabilities | DeliveryFailure: ...

    def publish_ready(
        self, request: _ReadyRequest, media: BinaryIO
    ) -> ClipReceipt | DeliveryFailure: ...

    def report_unavailable(self, request: _UnavailableRequest) -> ClipReceipt | DeliveryFailure: ...


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
    client = _backend_client(request, camera_id)
    result = client.probe_capabilities(camera_id)
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
    camera_id = str(binding.get("camera_id") or payload.camera_id)
    client = _backend_client(request, camera_id)
    if isinstance(payload, ReadyClipPayload):
        media = _verified_media(request, clip_id, payload)
        request_payload = _ReadyRequest(
            clip_id=clip_id,
            camera_id=camera_id,
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
        with media:
            result = client.publish_ready(request_payload, media)
    else:
        result = client.report_unavailable(
            _UnavailableRequest(
                clip_id=clip_id,
                camera_id=camera_id,
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


def _verified_media(request: Request, clip_id: str, payload: ReadyClipPayload) -> BinaryIO:
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="clip media unavailable",
        ) from exc
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    if media_fd is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip media unavailable")
    handle = os.fdopen(media_fd, "rb", closefd=True)
    mismatch = not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != payload.size_bytes
    try:
        if not mismatch:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            mismatch = digest.hexdigest() != payload.sha256
            handle.seek(0)
    except OSError as exc:
        handle.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="clip media mismatch",
        ) from exc
    if mismatch:
        handle.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="clip media mismatch",
        )
    return handle


def _enabled(request: Request) -> bool:
    value = getattr(request.app.state, "event_clip_export_enabled", None)
    if isinstance(value, bool):
        return value
    return os.environ.get("ML_API_EVENT_CLIP_EXPORT_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _backend_client(request: Request, camera_id: str) -> BackendEvidenceClient:
    bundle = backend_client_bundle(request.app)
    client = (
        bundle.evidence_client
        if bundle is not None
        else getattr(request.app.state, "backend_evidence_client", None)
    )
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend evidence export unavailable",
        )
    if hasattr(client, "for_camera"):
        return client.for_camera(camera_id)
    return client


def _raise_failure(failure: DeliveryFailure) -> None:
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
