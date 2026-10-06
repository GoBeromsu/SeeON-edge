"""Worker-to-api ingest relay routes."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.types import Message, Receive

from backend.app.edge_db import CheckViolation, DataError, NotNullViolation
from backend.app.edge_db.authority import AuthorityFenced
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.http import audit_runtime, mutation_audit
from backend.app.features.audit.store import AuditEvent
from backend.app.features.audit.store import utc_now as audit_now
from backend.app.features.cameras.router import (
    acknowledge_applied_detection_policies,
    worker_config_snapshot,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.update_command import CameraUpdate
from backend.app.features.evidence.event_outbox import (
    AcceptedEvent,
    EventIdentityConflict,
    EventOutbox,
    OutboxCapacityExceeded,
)
from backend.app.features.evidence.outbox_delivery import DeliveryStatus, OutboxDelivery
from backend.app.features.evidence.outbox_dispatch import RELAY_OUTBOX_BUDGET, dispatch
from backend.app.features.evidence.postgres_relay_projection import (
    PostgresRelayEvidenceProjection,
)
from backend.app.features.evidence.relay_projection import (
    RelayEvent,
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
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.execution_records import MAX_EXECUTION_RECORD_BODY_BYTES
from shared.events.relay_failure_log import RelayFailureLog

RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"

logger = logging.getLogger(__name__)

# The relay accepts at most 200 KiB of decoded inline evidence. Limit encoded
# input before decoding so an oversized Base64 string cannot trigger allocation.
MAX_INLINE_SNAPSHOT_BYTES = 200 * 1024
MAX_INLINE_SNAPSHOT_BASE64_CHARS = 4 * ((MAX_INLINE_SNAPSHOT_BYTES + 2) // 3)
# Bound the entire HTTP body before JSON parse / Pydantic validation. Alerts may
# carry a ~200 KiB base64 snapshot plus envelope fields; 512 KiB leaves margin
# without accepting multi-megabyte worker mistakes as DoS amplification.
MAX_RELAY_REQUEST_BODY_BYTES = 512 * 1024
MAX_RELAY_HEARTBEAT_BODY_BYTES = 4 * 1024
MAX_RELAY_RUNTIME_STATUS_BODY_BYTES = 64 * 1024
MAX_RELAY_SNAPSHOT_ATTACHMENT_BODY_BYTES = 8 * 1024
MAX_RELAY_SNAPSHOT_DISPOSITION_BODY_BYTES = 8 * 1024

# Per-endpoint hard body caps, keyed by route path suffix. BoundedBodyRoute
# consults this before any body byte is buffered.
_MAX_BODY_BYTES_BY_SUFFIX: dict[str, int] = {
    "/alerts": MAX_RELAY_REQUEST_BODY_BYTES,
    "/heartbeat": MAX_RELAY_HEARTBEAT_BODY_BYTES,
    "/runtime-status": MAX_RELAY_RUNTIME_STATUS_BODY_BYTES,
    "/snapshot-attachments": MAX_RELAY_SNAPSHOT_ATTACHMENT_BODY_BYTES,
    "/snapshot-dispositions": MAX_RELAY_SNAPSHOT_DISPOSITION_BODY_BYTES,
    "/execution-records": MAX_EXECUTION_RECORD_BODY_BYTES,
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

# Rate-limited/classified logging for ml-api's own outbound call to the Hub's
# backend ingest API (mirrors the worker-side RelayFailureLog channels in
# shared.events.evidence_export_client). Never logs the alert payload,
# facility/relay token, or any Hub response body -- only disposition, reason
# code, and status (see #579/#580).
_backend_ingest_alert_failures = RelayFailureLog(
    _LOGGER, channel="backend ingest alerts", method="POST"
)

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


def require_relay_execution_records(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    _authorize_relay_body(
        request,
        max_bytes=MAX_EXECUTION_RECORD_BODY_BYTES,
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

    Stamped by ``worker.runtime.flow.policy_pump._with_decision_trace_id`` via
    ``worker.types.trace.decision_trace_id``. This is the third field found to
    be emitted by the worker and undeclared here, after ``runtime_manifest_sha256``
    and a truncation marker, each producing the same permanent 422 and the same
    silent deletion by the outbox. It is declared rather than stripped because it
    is the only link from a delivered event back to the basis for the decision.

    The recurrence is the point: the guard against it is no longer a hand-written
    key list, which is exactly what let this one through, but a test that derives
    the emitted keys from the producer itself.
    """


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
    media_reference: str = Field(min_length=1, max_length=envelope_limits.MEDIA_REFERENCE_MAX_CHARS)
    size_bytes: int = Field(ge=0, le=envelope_limits.SNAPSHOT_SIZE_BYTES_MAX)
    mime_type: str = Field(min_length=1, max_length=envelope_limits.MIME_TYPE_MAX_CHARS)
    audit: RelayAuditEnvelope | None = None


class RelaySnapshotDispositionRequest(BaseModel):
    """A terminal, explicit statement that a snapshot cannot be delivered."""

    model_config = ConfigDict(extra="forbid")

    edge_event_id: str = Field(min_length=1, max_length=envelope_limits.EDGE_EVENT_ID_MAX_CHARS)
    snapshot_id: str = Field(min_length=1, max_length=envelope_limits.SNAPSHOT_ID_MAX_CHARS)
    disposition: str = Field(min_length=1, max_length=envelope_limits.DISPOSITION_MAX_CHARS)
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
    # Oldest live EVENT entry's acceptance time (ISO-8601 UTC), or None. Defaulted
    # for the same reason as dead_lettered_count above: a worker predating this
    # field must not be answered 422.
    oldest_event_accepted_at: str | None = Field(default=None)


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


class BackendIngestClient(Protocol):
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
    """Commit the incident and its delivery obligation, then answer.

    Every response is built after the admission COMMIT, so a failure before it
    is never acknowledged. A worker retry after a lost response lands on the
    same committed row instead of creating a second incident.
    """
    binding = _camera_binding(request, payload.camera_id, payload.facility_id)
    # Only a Hub-issued id may address the upstream ingest API. The previous
    # `or payload.camera_id` fallback sent the worker's edge-local id, which the
    # Hub never issued and rejects with FACILITY_BINDING_MISMATCH; on the edge
    # that surfaced as an opaque relay 502 and was repeatedly misdiagnosed as an
    # auth failure (issue #308). This mirrors the periodic heartbeat relay, which
    # already refuses to push under an unmapped id -- see
    # backend_heartbeat_relay._canonical_backend_camera_id.
    bound_camera_id = binding.get("backend_camera_id")
    backend_camera_id = (
        bound_camera_id if isinstance(bound_camera_id, str) and bound_camera_id.strip() else None
    )
    if backend_camera_id is None:
        # Coverage is untouched: the camera keeps streaming and the incident is
        # still recorded locally below. Only the guaranteed-reject upstream push
        # is skipped, and the reason is named instead of arriving as a 502.
        _LOGGER.warning(
            "relay alert: skipping backend ingest, camera %s has no Hub mapping yet",
            payload.camera_id,
            extra={"local_camera_id": payload.camera_id},
        )
    client = getattr(request.app.state, "backend_ingest_client", None)
    delivery = getattr(request.app.state, "event_outbox_delivery", None)
    forward = backend_camera_id is not None and client is not None
    if forward and not isinstance(delivery, OutboxDelivery):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend outbox delivery is not configured",
        )
    edge_event_id = _alert_event_id(payload)
    accepted = _accept_alert(
        request, payload, edge_event_id, backend_camera_id=backend_camera_id, forward=forward
    )
    if accepted.delivery_state == "LOCAL_ONLY":
        return _local_receipt(payload, edge_event_id)
    if not isinstance(delivery, OutboxDelivery):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend outbox delivery is not configured",
        )
    claim = None
    if client is not None and accepted.delivery_state in {"PENDING", "IN_FLIGHT"}:
        # The admission transaction is already committed: the Hub request below
        # runs with no SQL transaction open, under its own committed lease.
        try:
            claim = delivery.claim_event(edge_event_id)
        except AuthorityFenced as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="edge authority is fenced",
            ) from error
    if claim is None:
        return _replayed_alert_response(payload, edge_event_id, delivery.status(edge_event_id))
    result = dispatch(
        client,
        claim,
        delivery,
        on_accepted=lambda accepted_at: _record_alert_latency(request, payload, accepted_at),
    )
    if isinstance(result, DeliveryFailure):
        _backend_ingest_alert_failures.record_failure(result, path="alerts")
        raise _delivery_failure_error(result)
    _backend_ingest_alert_failures.record_success(path="alerts")
    if not result.event_id:
        return _local_receipt(payload, edge_event_id)
    return _central_receipt(payload, edge_event_id, result.event_id)


# Id-less alerts (pre-envelope workers) get a content-derived id so a resend of
# the same alert lands on the same committed row instead of a second incident.
_IDLESS_ALERT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "urn:seeon-edge:relay-alert")


def _alert_event_id(payload: RelayAlertRequest) -> str:
    if payload.edge_event_id is not None:
        return payload.edge_event_id
    canonical = json.dumps(
        payload.model_dump(
            exclude={"edge_event_id", "attempt_ordinal", "snapshot_jpeg_base64"},
            exclude_none=True,
            mode="json",
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(_IDLESS_ALERT_NAMESPACE, canonical))


def _accept_alert(
    request: Request,
    payload: RelayAlertRequest,
    edge_event_id: str,
    *,
    backend_camera_id: str | None,
    forward: bool,
) -> AcceptedEvent:
    runtime = audit_runtime(request)
    outbox = EventOutbox(
        runtime.database, runtime.authority, RELAY_OUTBOX_BUDGET, audit_runtime=runtime
    )
    runtime.require_mutation_admission(outbox)
    try:
        snapshot_bytes = _decode_snapshot(payload.snapshot_jpeg_base64)
        return outbox.accept(
            _relay_event(payload, edge_event_id),
            backend_camera_id=backend_camera_id,
            forward=forward,
            snapshot=_relay_snapshot(payload),
            snapshot_bytes=snapshot_bytes,
        )
    except (EventIdentityConflict, RelayEvidenceProjectionConflict) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except OutboxCapacityExceeded as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend outbox capacity exceeded",
            headers={"Retry-After": "5"},
        ) from error
    except AuthorityFenced as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="edge authority is fenced",
        ) from error
    except (ValueError, RelayEvidenceProjectionError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except _REJECTED_FACTS as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="relay alert violates a stored fact constraint",
        ) from error


# Constraint failures describe the request, not the database: answering 503
# would make the worker resend a payload that can never be stored.
_REJECTED_FACTS = (
    CheckViolation,
    NotNullViolation,
    DataError,
)


def _relay_event(payload: RelayAlertRequest, edge_event_id: str) -> RelayEvent:
    return RelayEvent(
        edge_event_id=edge_event_id,
        event_type=str(payload.event_type),
        probability=payload.probability,
        detected_at=payload.detected_at,
        camera_id=payload.camera_id,
        facility_id=payload.facility_id,
        resident_id=payload.resident_id,
        evidence=payload.evidence,
        audit=None if payload.audit is None else payload.audit.model_dump(exclude_none=True),
    )


def _relay_snapshot(payload: RelayAlertRequest) -> RelaySnapshot | None:
    if payload.snapshot is None or payload.snapshot_jpeg_base64 is None:
        return None
    return RelaySnapshot(
        snapshot_id=payload.snapshot.snapshot_id,
        path=payload.snapshot.path,
        sha256=payload.snapshot.sha256,
        size_bytes=payload.snapshot.size_bytes,
        mime_type=payload.snapshot.mime_type,
        captured_at=payload.snapshot.captured_at,
    )


def _local_receipt(payload: RelayAlertRequest, edge_event_id: str) -> dict[str, str]:
    if payload.edge_event_id is None:
        return {"status": "accepted"}
    return {"status": "accepted_local", "edge_event_id": edge_event_id}


def _central_receipt(
    payload: RelayAlertRequest, edge_event_id: str, event_id: str
) -> dict[str, str]:
    if payload.edge_event_id is None:
        return {"status": "accepted"}
    return {"status": "accepted", "edge_event_id": edge_event_id, "event_id": event_id}


def _delivery_failure_error(result: DeliveryFailure) -> HTTPException:
    if result.disposition is DeliveryDisposition.RETRY:
        code = status.HTTP_503_SERVICE_UNAVAILABLE
        # Names the failing side explicitly: this is the Hub/backend ingest API
        # declining or timing out, not the local PostgreSQL admission, which
        # already committed the incident and its delivery obligation.
        detail = f"backend ingest retryable failure: {result.code}"
    elif result.disposition is DeliveryDisposition.COMPATIBILITY:
        code = status.HTTP_404_NOT_FOUND
        detail = "backend ingest rejected alert"
    else:
        code = result.status_code or status.HTTP_502_BAD_GATEWAY
        detail = "backend ingest rejected alert"
    headers = None
    if result.retry_after_seconds is not None:
        headers = {"Retry-After": str(max(0, int(result.retry_after_seconds)))}
    return HTTPException(status_code=code, detail=detail, headers=headers)


def _replayed_alert_response(
    payload: RelayAlertRequest, edge_event_id: str, stored: DeliveryStatus | None
) -> dict[str, str]:
    """Answer a resend from the committed delivery state instead of resending."""
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="backend outbox delivery state unavailable",
        )
    if stored.state == "SENT" and stored.backend_event_id:
        return _central_receipt(payload, edge_event_id, stored.backend_event_id)
    if stored.state in {"LOCAL_ONLY", "EXHAUSTED"} or (
        stored.state == "REJECTED" and stored.reason == "ACCEPTED_LOCAL"
    ):
        # The incident is committed here; only the Hub copy is missing.
        return _local_receipt(payload, edge_event_id)
    if stored.state == "REJECTED":
        raise _delivery_failure_error(
            DeliveryFailure(
                DeliveryDisposition.PERMANENT,
                stored.reason or "REJECTED",
                status_code=stored.http_status,
            )
        )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="backend ingest delivery pending",
        headers={"Retry-After": "5"},
    )


@router.post("/snapshot-attachments", status_code=status.HTTP_202_ACCEPTED)
def relay_snapshot_attachment(
    payload: RelaySnapshotAttachmentRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_snapshot_attachment)],
) -> dict[str, str]:
    """Record one immutable media reference without accepting media bytes."""

    projection = _snapshot_projection(request)
    audit = mutation_audit(
        request,
        lambda: _relay_audit_event(AuditAction.RELAY_SNAPSHOT_ATTACHMENT, payload.snapshot_id),
    )
    with _snapshot_projection_errors():
        audit.apply(
            projection,
            lambda append: projection.attach_snapshot(
                edge_event_id=payload.edge_event_id,
                snapshot_id=payload.snapshot_id,
                sha256=payload.sha256,
                media_reference=payload.media_reference,
                size_bytes=payload.size_bytes,
                mime_type=payload.mime_type,
                after_write=append,
            ),
        )
    return {"status": "accepted"}


@router.post("/snapshot-dispositions", status_code=status.HTTP_202_ACCEPTED)
def relay_snapshot_disposition(
    payload: RelaySnapshotDispositionRequest,
    request: Request,
    _: Annotated[None, Depends(require_relay_snapshot_disposition)],
) -> dict[str, str]:
    """Durably record an unavailable or failed snapshot without touching its event."""

    projection = _snapshot_projection(request)
    audit = mutation_audit(
        request,
        lambda: _relay_audit_event(AuditAction.RELAY_SNAPSHOT_DISPOSITION, payload.snapshot_id),
    )
    with _snapshot_projection_errors():
        audit.apply(
            projection,
            lambda append: projection.record_snapshot_disposition(
                edge_event_id=payload.edge_event_id,
                snapshot_id=payload.snapshot_id,
                disposition=payload.disposition,
                reason=payload.reason,
                after_write=append,
            ),
        )
    return {"status": "accepted"}


def _snapshot_projection(request: Request) -> PostgresRelayEvidenceProjection:
    projection = getattr(request.app.state, "relay_snapshot_projection", None)
    if not isinstance(projection, PostgresRelayEvidenceProjection):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="snapshot projection is not configured",
        )
    return projection


@contextmanager
def _snapshot_projection_errors() -> Iterator[None]:
    try:
        yield
    except (RelayEvidenceProjectionMissingEvent, RelayEvidenceProjectionConflict) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except RelayEvidenceProjectionError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except AuthorityFenced as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="edge authority is fenced",
        ) from error
    except _REJECTED_FACTS as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="snapshot record violates a stored fact constraint",
        ) from error


def _relay_audit_event(action: AuditAction, target_id: str) -> AuditEvent:
    return AuditEvent(
        occurred_at=audit_now(),
        actor_id="worker-relay",
        action=action,
        target_id=target_id,
        detail=empty_detail(action),
        actor_type=AuditActorType.SERVICE,
        auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
    )


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
        store.update(local_id, CameraUpdate(never_connected=False))


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
    client: BackendIngestClient | None = getattr(request.app.state, "backend_ingest_client", None)
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
