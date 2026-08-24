"""Worker-to-api ingest relay routes."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, NotRequired, Protocol, TypedDict

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.types import Message, Receive

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.http import append_transactional
from backend.app.features.audit.store import AuditEvent
from backend.app.features.audit.store import utc_now as audit_now
from backend.app.features.cameras.router import (
    acknowledge_applied_detection_policies,
    worker_config_snapshot,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogConflictError, get_catalog_store
from backend.app.features.evidence.relay_projection import (
    RelayEvent,
    RelayEvidenceProjection,
    RelayEvidenceProjectionConflict,
    RelayEvidenceProjectionError,
    RelayEvidenceProjectionMissingEvent,
    RelaySnapshot,
)
from backend.app.features.relay.auth import authorize_relay
from backend.app.features.status.heartbeat_store import get_heartbeat_store
from backend.app.features.status.runtime_status_store import get_runtime_status_store
from contracts import AlertEventType
from contracts.decode_diagnostics import DECODE_BACKENDS, DECODE_FALLBACK_REASONS
from contracts.worker_config import RESTART_EPOCH_KEY
from shared.events import envelope_limits
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from shared.events.replay_wire import MAX_REPLAY_BODY_BYTES as _REPLAY_BODY_LIMIT
from shared.events.replay_wire import ReplayWireError, decode_replay_trace

RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"

logger = logging.getLogger(__name__)

# Catalog records are an auxiliary index, not the alert delivery path. A typical
# clip manifest is about 600 bytes; 16 KiB leaves ample room for event evidence
# while preventing a worker mistake from growing the SQLite catalog without bound.
MAX_CATALOG_PAYLOAD_BYTES = 16 * 1024
# The relay accepts at most 200 KiB of decoded inline evidence. Limit encoded
# input before decoding so an oversized Base64 string cannot trigger allocation.
MAX_INLINE_SNAPSHOT_BYTES = 200 * 1024
MAX_INLINE_SNAPSHOT_BASE64_CHARS = 4 * ((MAX_INLINE_SNAPSHOT_BYTES + 2) // 3)
# Eight container levels supports structured detector output without accepting
# arbitrarily recursive JSON from a trusted-but-fallible worker.
MAX_CATALOG_PAYLOAD_DEPTH = 8
# Bound the entire HTTP body before JSON parse / Pydantic validation. Alerts may
# carry a ~200 KiB base64 snapshot plus envelope fields; 512 KiB leaves margin
# without accepting multi-megabyte worker mistakes as DoS amplification.
MAX_RELAY_REQUEST_BODY_BYTES = 512 * 1024
MAX_RELAY_HEARTBEAT_BODY_BYTES = 4 * 1024
MAX_RELAY_RUNTIME_STATUS_BODY_BYTES = 64 * 1024
# Analysis frames are image-free but may contain a full pose/keypoint timeline.
# The worker sends at most this many frames per request; this is deliberately
# aligned with its trace writer batch bound rather than relying on a proxy's
# incidental request limit.
MAX_RELAY_ANALYSIS_TRACE_FRAMES = 16
# Derived, not chosen: see shared/events/replay_wire.py. Worker and backend
# read the same value so the two ends cannot drift apart.
MAX_RELAY_ANALYSIS_TRACE_BODY_BYTES = _REPLAY_BODY_LIMIT
MAX_RELAY_SNAPSHOT_ATTACHMENT_BODY_BYTES = 8 * 1024
MAX_RELAY_SNAPSHOT_DISPOSITION_BODY_BYTES = 8 * 1024

# Per-endpoint hard body caps, keyed by route path suffix. BoundedBodyRoute
# consults this before any body byte is buffered.
_MAX_BODY_BYTES_BY_SUFFIX: dict[str, int] = {
    "/alerts": MAX_RELAY_REQUEST_BODY_BYTES,
    "/heartbeat": MAX_RELAY_HEARTBEAT_BODY_BYTES,
    "/runtime-status": MAX_RELAY_RUNTIME_STATUS_BODY_BYTES,
    "/analysis-traces": MAX_RELAY_ANALYSIS_TRACE_BODY_BYTES,
    "/snapshot-attachments": MAX_RELAY_SNAPSHOT_ATTACHMENT_BODY_BYTES,
    "/snapshot-dispositions": MAX_RELAY_SNAPSHOT_DISPOSITION_BODY_BYTES,
}


def _oversized_body_error(max_bytes: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        detail=f"request body exceeds maximum of {max_bytes} bytes",
    )


def _bounded_receive(receive: Receive, max_bytes: int) -> Receive:
    """Wrap an ASGI ``receive`` so total body bytes can never exceed ``max_bytes``.

    Counts each ``http.request`` chunk as it arrives and raises 413 the moment
    the running total crosses the cap -- so a chunked / missing / lying
    Content-Length body is rejected mid-stream, before Starlette ever finishes
    buffering it for the Pydantic parse. This is the real bound; the
    Content-Length header pre-check in the auth dependency is only a fast path
    for honest oversized declarations.
    """
    total = 0

    async def wrapped() -> Message:
        nonlocal total
        message = await receive()
        if message["type"] == "http.request":
            body = message.get("body", b"")
            total += len(body)
            if total > max_bytes:
                raise _oversized_body_error(max_bytes)
        return message

    return wrapped


class BoundedBodyRoute(APIRoute):
    """Route class that caps request-body reads before FastAPI buffers them.

    FastAPI reads the whole body (``await request.body()``) *before* it solves
    route dependencies, so a dependency cannot bound the read. Wrapping
    ``receive`` at the route boundary enforces the cap during that read instead,
    independent of Content-Length. The auth dependency still runs first for the
    401/403 decision on within-limit bodies (auth-before-parse is preserved).
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()
        max_bytes = next(
            (
                limit
                for suffix, limit in _MAX_BODY_BYTES_BY_SUFFIX.items()
                if self.path.endswith(suffix)
            ),
            None,
        )
        if max_bytes is None:
            return original

        async def bounded_handler(request: Request) -> Response:
            request._receive = _bounded_receive(request.receive, max_bytes)  # noqa: SLF001 - wrap ASGI receive at the route boundary
            return await original(request)

        return bounded_handler


_LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/relay", tags=["relay"], route_class=BoundedBodyRoute)


def _reject_oversized_body(request: Request, *, max_bytes: int) -> None:
    """Cheap Content-Length pre-check.

    Rejects an *honest* oversized declaration. A missing or lying Content-Length
    is caught by the BoundedBodyRoute streaming bound, so this is a fast-path
    guard, not the authority on body size.
    """
    raw = request.headers.get("content-length")
    if raw is None:
        return
    try:
        length = int(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid Content-Length",
        ) from exc
    if length < 0 or length > max_bytes:
        raise _oversized_body_error(max_bytes)


def _authorize_relay_body(
    request: Request,
    *,
    max_bytes: int,
    relay_token: str | None,
    authorization: str | None = None,
) -> None:
    """Auth + Content-Length guard before Pydantic parse (FastAPI dep order)."""

    _reject_oversized_body(request, max_bytes=max_bytes)
    authorize_relay(request, relay_token or _bearer_token(authorization))


def require_relay_alert(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> None:
    _authorize_relay_body(request, max_bytes=MAX_RELAY_REQUEST_BODY_BYTES, relay_token=relay_token)


def require_relay_heartbeat(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> None:
    _authorize_relay_body(
        request, max_bytes=MAX_RELAY_HEARTBEAT_BODY_BYTES, relay_token=relay_token
    )


def require_relay_runtime_status(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    _authorize_relay_body(
        request,
        max_bytes=MAX_RELAY_RUNTIME_STATUS_BODY_BYTES,
        relay_token=relay_token,
        authorization=authorization,
    )


def require_relay_analysis_trace(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    _authorize_relay_body(
        request,
        max_bytes=MAX_RELAY_ANALYSIS_TRACE_BODY_BYTES,
        relay_token=relay_token,
        authorization=authorization,
    )


def require_relay_snapshot_attachment(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> None:
    _authorize_relay_body(
        request, max_bytes=MAX_RELAY_SNAPSHOT_ATTACHMENT_BODY_BYTES, relay_token=relay_token
    )


def require_relay_snapshot_disposition(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> None:
    _authorize_relay_body(
        request, max_bytes=MAX_RELAY_SNAPSHOT_DISPOSITION_BODY_BYTES, relay_token=relay_token
    )


class RelayAuditEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int | None = None
    model_version: str | None = None
    detector_version: str | None = None
    operating_threshold: float | None = None
    clock_source: str | None = None
    runtime_manifest_sha256: str | None = None
    """Digest of the runtime manifest that produced this event.

    The worker has always emitted this, but the envelope never declared it and
    ``extra="forbid"`` turned that omission into a permanent HTTP 422. Because
    the outbox treats 422 as non-retryable, every affected event was rejected
    for good rather than retried -- 41 bed-exit events were stranded this way in
    production before the field was declared here.
    """
    decision_trace_id: str | None = None
    """Pointer to the decision trace this event was derived from.

    Attached by ``TraceCapture._attach_trace``. This is the third field found to
    be emitted by the worker and undeclared here, after ``runtime_manifest_sha256``
    and a truncation marker, each producing the same permanent 422 and the same
    silent deletion by the outbox. It is declared rather than stripped because it
    is the only link from a delivered event back to the basis for the decision.

    The recurrence is the point: the guard against it is no longer a hand-written
    key list, which is exactly what let this one through, but a test that derives
    the emitted keys from the producer itself.
    """


class RelayAnalysisTraceRequest(BaseModel):
    """Closed shared replay envelope received from the inference runtime."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(...)
    frames: list[dict[str, object]] = Field(...)
    truncation: dict[str, object] = Field(...)


class RelayAnalysisTraceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool = Field(...)
    frame_count: int = Field(...)


class RelaySnapshotMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    mime_type: str = Field(min_length=1)
    captured_at: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    edge_event_id: str | None = None


class RelaySnapshotAttachmentRequest(BaseModel):
    """An immutable snapshot reference; snapshot bytes never cross this route."""

    model_config = ConfigDict(extra="forbid")

    edge_event_id: str = Field(min_length=1, max_length=envelope_limits.EDGE_EVENT_ID_MAX_CHARS)
    snapshot_id: str = Field(min_length=1, max_length=envelope_limits.SNAPSHOT_ID_MAX_CHARS)
    sha256: str = Field(
        min_length=envelope_limits.SHA256_MAX_CHARS,
        max_length=envelope_limits.SHA256_MAX_CHARS,
        pattern=r"^[0-9a-f]{64}$",
    )
    media_reference: str = Field(
        min_length=1, max_length=envelope_limits.MEDIA_REFERENCE_MAX_CHARS
    )
    size_bytes: int = Field(ge=0, le=envelope_limits.SNAPSHOT_SIZE_BYTES_MAX)
    mime_type: str = Field(min_length=1, max_length=envelope_limits.MIME_TYPE_MAX_CHARS)
    audit: RelayAuditEnvelope | None = None


class RelaySnapshotDispositionRequest(BaseModel):
    """A terminal, explicit statement that a snapshot cannot be delivered."""

    model_config = ConfigDict(extra="forbid")

    edge_event_id: str = Field(min_length=1, max_length=envelope_limits.EDGE_EVENT_ID_MAX_CHARS)
    snapshot_id: str = Field(min_length=1, max_length=envelope_limits.SNAPSHOT_ID_MAX_CHARS)
    disposition: str = Field(
        min_length=1, max_length=envelope_limits.DISPOSITION_MAX_CHARS
    )
    reason: str = Field(min_length=1, max_length=envelope_limits.DISPOSITION_REASON_MAX_CHARS)
    audit: RelayAuditEnvelope | None = None


class RelayAlertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_event_id: str | None = Field(default=None, pattern=r"^[0-9a-f-]{36}$")
    event_type: AlertEventType
    probability: float = Field(ge=0.0, le=1.0)
    detected_at: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    resident_id: str | None = None
    evidence: dict[str, Any] | None = None
    audit: RelayAuditEnvelope | None = None
    snapshot_jpeg_base64: str | None = Field(
        default=None, max_length=MAX_INLINE_SNAPSHOT_BASE64_CHARS
    )
    attempt_ordinal: int | None = Field(default=None, ge=1)
    snapshot: RelaySnapshotMetadata | None = None

    @model_validator(mode="after")
    def snapshot_matches_inline_evidence(self) -> RelayAlertRequest:
        snapshot_bytes = _decode_snapshot(self.snapshot_jpeg_base64)
        if self.snapshot is not None:
            if self.snapshot.camera_id != self.camera_id:
                raise ValueError("snapshot camera_id must match alert camera_id")
            if self.snapshot.edge_event_id != self.edge_event_id:
                raise ValueError("snapshot edge_event_id must match alert edge_event_id")
            if snapshot_bytes is not None:
                if self.edge_event_id is None or self.snapshot.edge_event_id is None:
                    raise ValueError("inline snapshot requires an edge_event_id")
                if self.snapshot.mime_type != "image/jpeg":
                    raise ValueError("inline snapshot MIME type must be image/jpeg")
                if self.snapshot.size_bytes <= 0 or self.snapshot.size_bytes != len(snapshot_bytes):
                    raise ValueError("inline snapshot size_bytes must exactly match decoded bytes")
                if self.snapshot.sha256 != hashlib.sha256(snapshot_bytes).hexdigest():
                    raise ValueError("inline snapshot sha256 must match decoded bytes")
        return self


class RelayHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    config_version: int | None = None


class RelayDecodeDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: str = Field(min_length=1)
    selected: str | None = Field(default=None)
    fallback_count: int = Field(ge=0)
    last_reason: str | None = Field(default=None)
    updated_at_sec: float = Field()

    @field_validator("requested", "selected")
    @classmethod
    def valid_backend(cls, value: str | None) -> str | None:
        if value is not None and value not in DECODE_BACKENDS:
            raise ValueError("decode backend is invalid")
        return value

    @field_validator("last_reason")
    @classmethod
    def valid_last_reason(cls, value: str | None) -> str | None:
        if value is not None and value not in DECODE_FALLBACK_REASONS:
            raise ValueError("last_reason is not a decode fallback reason")
        return value


class RelayDetectionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: bool = Field()
    inference_admitted: int = Field(ge=0)
    inference_succeeded: int = Field(ge=0)
    inference_overwritten: int = Field(ge=0)
    decision_completed: int = Field(ge=0)

    @model_validator(mode="after")
    def counters_are_ordered(self) -> RelayDetectionStatus:
        if self.inference_succeeded > self.inference_admitted:
            raise ValueError("inference_succeeded cannot exceed inference_admitted")
        if self.decision_completed > self.inference_succeeded:
            raise ValueError("decision_completed cannot exceed inference_succeeded")
        return self


class RelayRuntimeStatusCamera(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1)
    decode: RelayDecodeDiagnostics
    measured_fps: float | None = Field(default=None, ge=0.0)
    detection: RelayDetectionStatus | None = Field(default=None)


class RelayGpuStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nvml_available: bool = Field()
    cuda_context_ok: bool = Field()
    driver_version: str | None = Field(default=None)
    device_name: str | None = Field(default=None)
    captured_at_sec: float = Field()
    nvml_error: str | None = Field(default=None)


class RelayWorkerStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alive: bool = Field()
    pid: int | None = Field(default=None, ge=0)
    started_at_sec: float | None = Field(default=None)
    profile_boot_error: str | None = Field(default=None)


class RelayClipExportStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = Field()
    version: int = Field(ge=0)


class RelayClipRecorderStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = Field()
    dropped_frames: int | None = Field(default=None, ge=0)
    dropped_events: int | None = Field(default=None, ge=0)
    failed_writes: int | None = Field(default=None, ge=0)
    finalized_clips: int | None = Field(default=None, ge=0)
    video_unavailable_clips: int | None = Field(default=None, ge=0)
    active_clips: int | None = Field(default=None, ge=0)
    encoder: str | None = Field(default=None)


class RelayDeliveryQueueStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted_count: int = Field(ge=0)
    accepted_bytes: int = Field(ge=0)
    max_accepted_entries: int = Field(gt=0)
    max_accepted_bytes: int = Field(gt=0)
    by_kind: dict[str, int] = Field()
    # Evidence the backend refused or that exhausted delivery. Retained on disk,
    # not delivered, and needing operator action -- a deployment cannot act on
    # what it never reports. Defaulted so a worker predating this field is not
    # answered 422, which is how 41 real events were destroyed here.
    dead_lettered_count: int = Field(default=0, ge=0)
    dead_lettered_bytes: int = Field(default=0, ge=0)


class RelayRuntimeStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facility_id: str = Field(min_length=1)
    generation: int | None = Field(default=None, ge=0)
    seq: int = Field(ge=0)
    cameras: list[RelayRuntimeStatusCamera] = Field()
    clip_recorder: RelayClipRecorderStatus
    clip_export: RelayClipExportStatus | None = Field(default=None)
    gpu: RelayGpuStatus | None = Field(default=None)
    worker: RelayWorkerStatus | None = Field(default=None)
    delivery_queue: RelayDeliveryQueueStatus | None = Field(default=None)


class RelayRuntimeStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    generation: int


class _AlertKwargs(TypedDict):
    event_type: AlertEventType
    detected_at: str
    probability: float
    audit: NotRequired[dict[str, object]]
    snapshot_bytes: NotRequired[bytes]
    clip_id: NotRequired[str]


class BackendIngestClient(Protocol):
    def send_alert(
        self,
        *,
        event_type: AlertEventType,
        detected_at: str,
        probability: float,
        audit: dict[str, object] | None = None,
        snapshot_bytes: bytes | None = None,
        clip_id: str | None = None,
    ) -> bool: ...

    def send_alert_receipt(
        self,
        *,
        edge_event_id: str,
        event_type: AlertEventType,
        detected_at: str,
        probability: float,
        audit: dict[str, object] | None = None,
        snapshot_bytes: bytes | None = None,
        clip_id: str | None = None,
        on_accepted: Callable[[float], None] | None = None,
    ) -> EventReceipt | DeliveryFailure: ...

    def send_heartbeat(self) -> bool: ...


@router.get("/config")
def worker_config(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> dict[str, object]:
    authorize_relay(request, relay_token)
    return worker_config_snapshot(request, require_available=True)


@router.post("/restart", status_code=status.HTTP_202_ACCEPTED)
def bump_restart(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> dict[str, int]:
    authorize_relay(request, relay_token)
    request.app.state.restart_epoch = int(getattr(request.app.state, "restart_epoch", 0)) + 1
    return {RESTART_EPOCH_KEY: request.app.state.restart_epoch}


@router.post("/alerts", status_code=status.HTTP_202_ACCEPTED)
def relay_alert(
    payload: RelayAlertRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_alert)],
) -> dict[str, str]:
    # Local catalog recording is edge-local audit trail, not backend egress --
    # it must not depend on registry binding or the backend call's outcome
    # (see #183, #202). Recording it up front means ml-api keeps its own
    # record of every alert attempt even when the camera can't yet be
    # resolved or the backend can't be reached, instead of the attempt
    # leaving no local trace at all when _camera_binding() 403s below.
    projected = _project_relay_event(request, payload)
    catalog_result = None if projected else _record_catalog(request, payload)
    binding = _camera_binding(request, payload.camera_id, payload.facility_id)
    # Only a Hub-issued id may address the upstream ingest API. The previous
    # `or payload.camera_id` fallback sent the worker's edge-local id, which the
    # Hub never issued and rejects with FACILITY_BINDING_MISMATCH; on the edge
    # that surfaced as an opaque relay 502 and was repeatedly misdiagnosed as an
    # auth failure (issue #308). This mirrors the periodic heartbeat relay, which
    # already refuses to push under an unmapped id -- see
    # backend_heartbeat_relay._canonical_backend_camera_id.
    bound_camera_id = binding.get("backend_camera_id")
    if not isinstance(bound_camera_id, str) or not bound_camera_id.strip():
        # Coverage is untouched: the camera keeps streaming, and the local audit
        # record was already written above. Only the guaranteed-reject upstream
        # push is skipped, and the reason is named instead of arriving as a 502.
        _LOGGER.warning(
            "relay alert: skipping backend ingest, camera %s has no Hub mapping yet",
            payload.camera_id,
            extra={"local_camera_id": payload.camera_id},
        )
        return _alert_response({"status": "accepted"}, catalog_result)
    canonical_camera_id = bound_camera_id
    client = _optional_backend_ingest_client(request, camera_id=canonical_camera_id)
    if client is None:
        # Registry-bound local accept; cloud only when store built a client.
        return _alert_response({"status": "accepted"}, catalog_result)
    alert_kwargs: _AlertKwargs = {
        "event_type": payload.event_type,
        "detected_at": payload.detected_at,
        "probability": payload.probability,
    }
    clip_id = _payload_clip_id(payload)
    if clip_id is not None:
        alert_kwargs["clip_id"] = clip_id
    # Envelope-less alerts forward the exact prior 3-field shape; audit/snapshot
    # kwargs are added ONLY when present (backward-compat with the route contract).
    if payload.audit is not None:
        alert_kwargs["audit"] = payload.audit.model_dump(exclude_none=True)
    snapshot_bytes = _decode_snapshot(payload.snapshot_jpeg_base64)
    if snapshot_bytes is not None:
        alert_kwargs["snapshot_bytes"] = snapshot_bytes
    if payload.edge_event_id is not None:
        result = client.send_alert_receipt(
            edge_event_id=payload.edge_event_id,
            on_accepted=lambda accepted_at: _record_alert_latency(request, payload, accepted_at),
            **alert_kwargs,
        )
        if isinstance(result, DeliveryFailure):
            if result.disposition is DeliveryDisposition.RETRY:
                code = status.HTTP_503_SERVICE_UNAVAILABLE
            elif result.disposition is DeliveryDisposition.COMPATIBILITY:
                code = status.HTTP_404_NOT_FOUND
            else:
                code = result.status_code or status.HTTP_502_BAD_GATEWAY
            headers = None
            if result.retry_after_seconds is not None:
                headers = {"Retry-After": str(max(0, int(result.retry_after_seconds)))}
            raise HTTPException(
                status_code=code,
                detail="backend ingest rejected alert",
                headers=headers,
            )
        response = {
            "status": result.status,
            "edge_event_id": result.edge_event_id,
            "event_id": result.event_id,
        }
        return _alert_response(response, catalog_result)
    accepted = client.send_alert(**alert_kwargs)
    if not accepted:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="backend ingest rejected alert",
        )
    return _alert_response({"status": "accepted"}, catalog_result)


@router.post("/snapshot-attachments", status_code=status.HTTP_202_ACCEPTED)
def relay_snapshot_attachment(
    payload: RelaySnapshotAttachmentRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_snapshot_attachment)],
) -> dict[str, str]:
    """Record one immutable media reference without accepting media bytes."""

    if _project_snapshot_attachment(request, payload):
        return {"status": "accepted"}
    store = get_catalog_store(request.app)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="snapshot attachment storage unavailable",
        )
    try:
        store.record(
            "snapshots",
            _snapshot_delivery_key(payload.edge_event_id, payload.snapshot_id),
            payload.model_dump(exclude_none=True),
        )
    except CatalogConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="snapshot attachment conflicts with existing content identity",
        ) from error
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="snapshot attachment storage unavailable",
        ) from error
    return {"status": "accepted"}


@router.post("/snapshot-dispositions", status_code=status.HTTP_202_ACCEPTED)
def relay_snapshot_disposition(
    payload: RelaySnapshotDispositionRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_snapshot_disposition)],
) -> dict[str, str]:
    """Durably record an unavailable or failed snapshot without touching its event."""

    _project_snapshot_disposition(request, payload)
    return {"status": "accepted"}


def _snapshot_delivery_key(edge_event_id: str, snapshot_id: str) -> str:
    return hashlib.sha256(f"{edge_event_id}\0{snapshot_id}".encode()).hexdigest()


def _relay_evidence_projection(request: Request) -> RelayEvidenceProjection | None:
    projection = getattr(request.app.state, "relay_evidence_projection", None)
    if isinstance(projection, RelayEvidenceProjection):
        return projection
    if not EDGE_DATABASE_PATH.is_file():
        return None
    return RelayEvidenceProjection(EDGE_DATABASE_PATH)


def _project_relay_event(request: Request, payload: RelayAlertRequest) -> bool:
    if payload.edge_event_id is None:
        return False
    projection = _relay_evidence_projection(request)
    if projection is None:
        return False
    snapshot = None
    if payload.snapshot is not None and payload.snapshot_jpeg_base64 is not None:
        snapshot = RelaySnapshot(
            snapshot_id=payload.snapshot.snapshot_id,
            path=payload.snapshot.path,
            sha256=payload.snapshot.sha256,
            size_bytes=payload.snapshot.size_bytes,
            mime_type=payload.snapshot.mime_type,
            captured_at=payload.snapshot.captured_at,
        )
    try:
        projection.project_event(
            RelayEvent(
                edge_event_id=payload.edge_event_id,
                event_type=str(payload.event_type),
                probability=payload.probability,
                detected_at=payload.detected_at,
                camera_id=payload.camera_id,
                facility_id=payload.facility_id,
                resident_id=payload.resident_id,
                evidence=payload.evidence,
                audit=None
                if payload.audit is None
                else payload.audit.model_dump(exclude_none=True),
            ),
            snapshot,
            after_write=lambda connection: append_transactional(
                request,
                connection,
                AuditEvent(
                    occurred_at=audit_now(), actor_id="worker-relay",
                    action=AuditAction.RELAY_ALERT, target_id=str(payload.edge_event_id),
                    detail=empty_detail(AuditAction.RELAY_ALERT),
                    actor_type=AuditActorType.SERVICE,
                    auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
                ),
            ),
        )
    except RelayEvidenceProjectionConflict as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except RelayEvidenceProjectionError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="central evidence projection unavailable",
        ) from error
    return True


def _project_snapshot_attachment(
    request: Request, payload: RelaySnapshotAttachmentRequest
) -> bool:
    try:
        projection = _relay_evidence_projection(request)
        if projection is None:
            return False
        projection.attach_snapshot(
            edge_event_id=payload.edge_event_id,
            snapshot_id=payload.snapshot_id,
            sha256=payload.sha256,
            media_reference=payload.media_reference,
            size_bytes=payload.size_bytes,
            mime_type=payload.mime_type,
            after_write=lambda connection: append_transactional(
                request,
                connection,
                AuditEvent(
                    occurred_at=audit_now(), actor_id="worker-relay",
                    action=AuditAction.RELAY_SNAPSHOT_ATTACHMENT,
                    target_id=payload.snapshot_id,
                    detail=empty_detail(AuditAction.RELAY_SNAPSHOT_ATTACHMENT),
                    actor_type=AuditActorType.SERVICE,
                    auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
                ),
            ),
        )
    except RelayEvidenceProjectionMissingEvent as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except RelayEvidenceProjectionConflict as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except RelayEvidenceProjectionError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="central evidence projection unavailable",
        ) from error
    else:
        return True


def _project_snapshot_disposition(
    request: Request, payload: RelaySnapshotDispositionRequest
) -> None:
    try:
        projection = _relay_evidence_projection(request)
        if projection is None:
            return
        projection.record_snapshot_disposition(
            edge_event_id=payload.edge_event_id,
            snapshot_id=payload.snapshot_id,
            disposition=payload.disposition,
            reason=payload.reason,
            after_write=lambda connection: append_transactional(
                request,
                connection,
                AuditEvent(
                    occurred_at=audit_now(), actor_id="worker-relay",
                    action=AuditAction.RELAY_SNAPSHOT_DISPOSITION,
                    target_id=payload.snapshot_id,
                    detail=empty_detail(AuditAction.RELAY_SNAPSHOT_DISPOSITION),
                    actor_type=AuditActorType.SERVICE,
                    auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
                ),
            ),
        )
    except RelayEvidenceProjectionMissingEvent as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except RelayEvidenceProjectionConflict as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="central evidence projection unavailable",
        ) from error


@router.post("/heartbeat", status_code=status.HTTP_202_ACCEPTED)
def relay_heartbeat(
    payload: RelayHeartbeatRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_heartbeat)],
) -> dict[str, str]:
    # Stamp local liveness right after auth, BEFORE camera binding, so /status
    # reflects edge-local truth even when the registry can't yet resolve this
    # camera -- not just when backend egress later fails (see #183, #202). A
    # worker holding a valid relay token recording a heartbeat for camera X is
    # real local truth regardless of whether X is registered yet; registry
    # binding is for backend-id translation on egress, not admission to
    # ml-api's own liveness bookkeeping.
    get_heartbeat_store(request.app).record(
        payload.camera_id,
        payload.facility_id,
        config_version=payload.config_version,
    )
    acknowledge_applied_detection_policies(
        request,
        facility_id=payload.facility_id,
        config_version=payload.config_version,
    )
    _clear_never_connected_on_first_heartbeat(request, payload.camera_id)
    binding = _camera_binding(request, payload.camera_id, payload.facility_id)
    # Same Hub-boundary rule as relay_alert above: only a Hub-issued id may address
    # the upstream ingest API. The backend only knows its own camera ids, and an id
    # it never issued comes back as FACILITY_BINDING_MISMATCH, surfacing on the edge
    # as an opaque 502 that reads like an auth failure (issue #308). All local
    # bookkeeping above -- liveness, policy ack, never_connected -- has already run,
    # so skipping the push costs no local state. This also matches the periodic tick
    # in backend_heartbeat_relay, which likewise refuses to send under an unmapped id.
    bound_camera_id = binding.get("backend_camera_id")
    if not isinstance(bound_camera_id, str) or not bound_camera_id.strip():
        _LOGGER.warning(
            "relay heartbeat: skipping backend ingest, camera %s has no Hub mapping yet",
            payload.camera_id,
            extra={"local_camera_id": payload.camera_id},
        )
        return {"status": "accepted"}
    client = _optional_backend_ingest_client(request, camera_id=bound_camera_id)
    if client is None:
        return {"status": "accepted"}
    if not client.send_heartbeat():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="backend ingest rejected heartbeat",
        )
    return {"status": "accepted"}


@router.post("/runtime-status", response_model=RelayRuntimeStatusResponse)
def relay_runtime_status(
    payload: RelayRuntimeStatusRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_runtime_status)],
) -> RelayRuntimeStatusResponse:
    _runtime_status_facility_binding(request, payload.facility_id)
    _log_unresolved_runtime_status_cameras(request, payload)
    data = payload.model_dump()
    if data["worker"] is not None and data["worker"]["profile_boot_error"] is None:
        data["worker"].pop("profile_boot_error")
    result = get_runtime_status_store(request.app).record(data)
    if not result.accepted:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=result.reason)
    return RelayRuntimeStatusResponse(accepted=True, generation=result.generation)


@router.post("/analysis-traces", response_model=RelayAnalysisTraceResponse)
def relay_analysis_trace(
    payload: RelayAnalysisTraceRequest,
    _: Annotated[None, Depends(require_relay_analysis_trace)],
) -> RelayAnalysisTraceResponse:
    if len(payload.frames) > MAX_RELAY_ANALYSIS_TRACE_FRAMES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "analysis trace transfer exceeds maximum of "
                f"{MAX_RELAY_ANALYSIS_TRACE_FRAMES} frames"
            ),
        )
    try:
        trace = decode_replay_trace(payload.model_dump())
    except ReplayWireError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return RelayAnalysisTraceResponse(accepted=True, frame_count=len(trace.frames))


def _record_catalog(request: Request, payload: RelayAlertRequest) -> str | None:
    try:
        event = payload.model_dump(
            exclude={"attempt_ordinal", "snapshot_jpeg_base64"}, exclude_none=True
        )
        rejection_reason = _catalog_payload_rejection_reason(event)
        if rejection_reason is not None:
            # Do not reject the relay request: the catalog is a secondary index, and
            # an index limit must never prevent a safety alert reaching the backend.
            # The accepted response names the catalog failure and app state/logs make
            # it observable to local operators.
            request.app.state.catalog_error = rejection_reason
            logger.warning("catalog record skipped: %s", rejection_reason)
            return rejection_reason

        store = get_catalog_store(request.app)
        if store is None:
            return getattr(request.app.state, "catalog_error", "catalog unavailable")

        records: list[tuple[str, str, dict[str, Any]]] = []
        if payload.edge_event_id is not None:
            records.append(("events", payload.edge_event_id, event))
        if payload.snapshot is not None:
            snapshot = payload.snapshot.model_dump(exclude_none=True)
            records.append(("snapshots", payload.snapshot.snapshot_id, snapshot))
        if records:
            store.record_many(tuple(records))
    except CatalogConflictError:
        reason = "catalog idempotency conflict"
    except (OSError, sqlite3.Error) as exc:
        reason = f"catalog operational failure: {exc}"
    except Exception as exc:  # noqa: BLE001 - auxiliary catalog failures must never block alert egress
        reason = f"catalog unexpected failure: {exc}"
    else:
        return None

    request.app.state.catalog_error = reason
    logger.warning("catalog record skipped: %s", reason)
    return reason


def _catalog_payload_rejection_reason(event: dict[str, Any]) -> str | None:
    evidence = event.get("evidence")
    if _json_depth(evidence) > MAX_CATALOG_PAYLOAD_DEPTH:
        return f"catalog evidence exceeds maximum depth of {MAX_CATALOG_PAYLOAD_DEPTH}"
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    if len(encoded) > MAX_CATALOG_PAYLOAD_BYTES:
        return f"catalog payload exceeds maximum size of {MAX_CATALOG_PAYLOAD_BYTES} bytes"
    return None


def _json_depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _alert_response(response: dict[str, str], catalog_error: str | None) -> dict[str, str]:
    if catalog_error is not None:
        return {
            **response,
            "catalog": "not_recorded",
            "catalog_reason": catalog_error,
        }
    return response


def _decode_snapshot(snapshot_jpeg_base64: str | None) -> bytes | None:
    if snapshot_jpeg_base64 is None:
        return None
    try:
        snapshot_bytes = base64.b64decode(snapshot_jpeg_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("snapshot_jpeg_base64 must be valid Base64") from exc
    if len(snapshot_bytes) > MAX_INLINE_SNAPSHOT_BYTES:
        raise ValueError(
            "snapshot_jpeg_base64 exceeds maximum decoded size of "
            f"{MAX_INLINE_SNAPSHOT_BYTES} bytes"
        )
    return snapshot_bytes


def _record_alert_latency(request: Request, payload: RelayAlertRequest, received_at: float) -> None:
    if payload.attempt_ordinal != 1:
        return
    get_runtime_status_store(request.app).record_latency(
        payload.facility_id, payload.detected_at, received_at=received_at
    )


def _bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and token:
        return token
    return None


def _runtime_status_facility_binding(request: Request, facility_id: str) -> None:
    """No-op facility gate for local runtime-status recording.

    Runtime-status is purely local dashboard state (no cloud egress). Site
    facility identity lives in ConnectionSettingsStore and is not compared
    against the worker payload or any env var here.
    """
    del request, facility_id


def _log_unresolved_runtime_status_cameras(
    request: Request, payload: RelayRuntimeStatusRequest
) -> None:
    """Best-effort observability only -- never blocks the snapshot.

    relay_runtime_status has no backend egress, so an unresolved camera_id
    here is not a reason to drop the whole snapshot (see #183, #202): this
    loop used to call the same _camera_binding() that relay_alert/
    relay_heartbeat use to gate backend egress, whose return value was never
    even used here. One camera missing from camera_registry could blank the
    dashboard for every camera in the payload, even the ones that resolved fine.
    """
    for camera in payload.cameras:
        try:
            _camera_binding(request, camera.camera_id, payload.facility_id)
        except HTTPException as exc:
            logger.warning(
                "runtime-status camera unresolved (recorded anyway): camera_id=%s detail=%s",
                camera.camera_id,
                exc.detail,
            )


def _payload_clip_id(payload: RelayAlertRequest) -> str | None:
    if payload.evidence is None:
        return None
    value = payload.evidence.get("clip_id")
    if isinstance(value, str) and value.strip() != "":
        return value
    return None


def _camera_binding(request: Request, camera_id: str, facility_id: str) -> dict[str, str | None]:
    """Resolve egress camera binding from the dashboard registry only.

    ``facility_id`` is accepted on the worker→ml-api wire (may be the local
    placeholder ``"local"``) but is not compared to env or used for admission.
    """
    return _camera_binding_from_registry(request, camera_id, facility_id)


def _camera_binding_from_registry(
    request: Request,
    camera_id: str,
    facility_id: str,
) -> dict[str, str | None]:
    store = getattr(request.app.state, "camera_registry", None)
    if not isinstance(store, CameraRegistryStore):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="unknown camera")
    snapshot = store.snapshot()
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="unknown camera")
    for record in cameras:
        if not isinstance(record, dict):
            continue
        local_id = record.get("id")
        backend_id = record.get("backend_camera_id")
        if camera_id in {local_id, backend_id}:
            canonical_id = backend_id or local_id
            return {
                # Keeps its local-id fallback on purpose: this field gates local
                # ADMISSION, and a worker may legitimately report under either id.
                "camera_id": str(canonical_id),
                "facility_id": facility_id,
                "resident_id": None,
                # Hub-issued id only, None when unmapped. EGRESS must use this
                # field, never camera_id above -- sending an id the Hub never
                # issued comes back as FACILITY_BINDING_MISMATCH and reaches the
                # edge as an opaque 502 (issue #308).
                "backend_camera_id": (
                    backend_id if isinstance(backend_id, str) and backend_id.strip() else None
                ),
            }
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="unknown camera")


def _clear_never_connected_on_first_heartbeat(request: Request, camera_id: str) -> None:
    """Flip a registry record's never_connected off on its FIRST heartbeat.

    One-way: never reverts to True once cleared. Looked up by either the
    registry's local id or its backend_camera_id, matching payload.camera_id
    against whichever one the worker is currently configured to send (see
    _camera_binding_from_registry). A no-op once already False, so this stays
    a single extra write per camera lifetime rather than one per heartbeat.
    """
    store = getattr(request.app.state, "camera_registry", None)
    if not isinstance(store, CameraRegistryStore):
        return
    record = _find_registry_record(store, camera_id)
    if record is None or record.get("never_connected") is not True:
        return
    local_id = record.get("id")
    if isinstance(local_id, str):
        store.update(local_id, {"never_connected": False})


def _find_registry_record(store: CameraRegistryStore, camera_id: str) -> dict[str, object] | None:
    snapshot = store.snapshot()
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, list):
        return None
    for record in cameras:
        if not isinstance(record, dict):
            continue
        if camera_id in {record.get("id"), record.get("backend_camera_id")}:
            return record
    return None


def _optional_backend_ingest_client(
    request: Request, *, camera_id: str
) -> BackendIngestClient | None:
    """Return the cloud ingest client when connection settings built one.

    Missing client means unconfigured cloud path: local accept still OK.
    """
    client: BackendIngestClient | None = getattr(
        request.app.state, "backend_ingest_client", None
    )
    if client is None:
        return None
    for_camera = getattr(client, "for_camera", None)
    if for_camera is not None:
        scoped: BackendIngestClient = for_camera(camera_id)
        return scoped
    return client


def _backend_ingest_client(request: Request, *, camera_id: str) -> BackendIngestClient:
    client = _optional_backend_ingest_client(request, camera_id=camera_id)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend ingest client is not configured",
        )
    return client


__all__ = ["router"]
