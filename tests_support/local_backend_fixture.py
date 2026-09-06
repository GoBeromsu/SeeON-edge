from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import Any, Final, get_args
from uuid import UUID, uuid5

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from contracts.edge_provisioning_enrollment import parse_enrollment_verification
from contracts.edge_provisioning_models import ContractViolation
from contracts.edge_provisioning_parse import (
    parse_topology_confirmation,
    parse_topology_snapshot,
)
from contracts.relay import AlertEventType

_MAX_JSON_BYTES: Final = 256 * 1024
_MAX_SNAPSHOT_BYTES: Final = 512 * 1024
_MAX_ROUTE_RECORDS: Final = 4096
_FIXTURE_NAMESPACE: Final = UUID("6c5d6ed1-f8c2-4f69-9836-722b7d1fd671")
_EVENT_FIELDS: Final = frozenset(
    {
        "camera_id",
        "type",
        "detected_at",
        "confidence",
        "config_version",
        "model_version",
        "detector_version",
        "operating_threshold",
        "clock_source",
        "edge_event_id",
        "clip_id",
    }
)
_EVENT_REQUIRED: Final = frozenset(
    {"camera_id", "type", "detected_at", "confidence", "edge_event_id"}
)
_EVENT_TYPES: Final = frozenset(get_args(AlertEventType))


@dataclass(frozen=True, slots=True)
class RouteRecord:
    method: str
    path: str
    status_code: int
    content_length: int
    content_type: str
    actual_body_bytes: int


@dataclass(frozen=True, slots=True)
class EventRecord:
    edge_event_id: str
    event_id: str
    camera_id: str


@dataclass(frozen=True, slots=True)
class TopologyRecord:
    request_fingerprint: str
    client_revision: int
    server_revision: int
    confirmation_id: str
    digest: str


class LocalBackendFixture:
    """Memory-only Hub fixture for deterministic edge delivery diagnostics."""

    def __init__(
        self,
        *,
        bearer_token: str = "fixture-token",
        facility_id: str = "a5ff4ed1-7e63-4a4f-9ef0-42e807d74a64",
        edge_installation_id: str = "c72bd9a7-3e04-47ba-a8cd-a56e54f98152",
        faulty_event_identity: bool = False,
    ) -> None:
        self._bearer_token = bearer_token
        self.facility_id = facility_id
        self.edge_installation_id = edge_installation_id
        self.faulty_event_identity = faulty_event_identity
        self._fingerprint_key = secrets.token_bytes(32)
        self._events: dict[str, EventRecord] = {}
        self._event_fingerprints: dict[str, str] = {}
        self._accepted_event_ids: dict[str, list[str]] = {}
        self._topologies: dict[str, TopologyRecord] = {}
        self._snapshot_bytes_discarded = 0
        self._route_ledger: list[RouteRecord] = []
        self.app = self._build_app()

    @property
    def route_ledger(self) -> tuple[RouteRecord, ...]:
        return tuple(self._route_ledger)

    @property
    def events(self) -> tuple[EventRecord, ...]:
        return tuple(self._events.values())

    @property
    def snapshot_bytes_discarded(self) -> int:
        return self._snapshot_bytes_discarded

    @property
    def retained_media_bytes(self) -> int:
        return 0

    def event_for_edge_id(self, edge_event_id: str) -> EventRecord | None:
        return self._events.get(edge_event_id)

    def accepted_event_ids(self, edge_event_id: str) -> tuple[str, ...]:
        return tuple(self._accepted_event_ids.get(edge_event_id, ()))

    def reset_observations(self) -> None:
        self._route_ledger.clear()
        self._snapshot_bytes_discarded = 0

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.middleware("http")
        async def record_route(request: Request, call_next: Any) -> Response:
            if len(self._route_ledger) >= _MAX_ROUTE_RECORDS:
                response: Response = JSONResponse(
                    {"detail": "fixture route ledger capacity exhausted"}, status_code=507
                )
            else:
                response = await call_next(request)
            content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
            actual_bytes = getattr(request.state, "actual_body_bytes", 0)
            self._route_ledger.append(
                RouteRecord(
                    request.method,
                    request.url.path,
                    response.status_code,
                    _safe_content_length(request.headers.get("content-length")),
                    content_type,
                    actual_bytes,
                )
            )
            return response

        @app.post("/api/v1/edge/enrollments/verify")
        async def verify_enrollment(request: Request) -> JSONResponse:
            self._require_auth(request)
            body = await _bounded_json(request)
            try:
                parse_enrollment_verification(body)
            except ContractViolation as exc:
                raise HTTPException(422, exc.detail) from exc
            return JSONResponse(
                {
                    "schemaVersion": 1,
                    "edgeInstallationId": self.edge_installation_id,
                    "enrollmentGeneration": 1,
                    "facility": {"id": self.facility_id, "displayName": "Diagnostic Fixture"},
                    "serverRevision": 0,
                }
            )

        @app.get("/api/v1/ml-config/{facility_id}")
        async def ml_config(facility_id: str, request: Request) -> JSONResponse:
            self._require_auth(request)
            if facility_id != self.facility_id:
                raise HTTPException(404, "unknown facility")
            return JSONResponse({"configVersion": 1, "detectionWindows": {}, "cameras": []})

        @app.get("/api/v1/events/capabilities")
        async def capabilities(camera_id: str, request: Request) -> JSONResponse:
            self._require_auth(request)
            if not _bounded_text(camera_id, 128):
                raise HTTPException(422, "invalid camera_id")
            return JSONResponse({"event_idempotency": 1, "clip_export": 0})

        @app.post("/api/v1/events/heartbeat")
        async def heartbeat(request: Request) -> JSONResponse:
            self._require_auth(request)
            body = await _bounded_json(request)
            _require_exact_keys(body, {"camera_id"})
            if not _bounded_text(body["camera_id"], 128):
                raise HTTPException(422, "invalid camera_id")
            return JSONResponse({"ok": True})

        @app.post("/api/v1/events", status_code=201)
        async def ingest_event(request: Request) -> JSONResponse:
            self._require_auth(request)
            body = await _bounded_json(request)
            _validate_event(body)
            edge_event_id = body["edge_event_id"]
            camera_id = body["camera_id"]
            fingerprint = self._fingerprint(body)
            existing = self._events.get(edge_event_id)
            if existing is not None:
                if self._event_fingerprints[edge_event_id] != fingerprint:
                    raise HTTPException(409, "idempotency conflict")
                accepted = self._accepted_event_ids[edge_event_id]
                if self.faulty_event_identity:
                    event_id = str(uuid5(_FIXTURE_NAMESPACE, f"{edge_event_id}:{len(accepted)}"))
                    accepted.append(event_id)
                else:
                    event_id = existing.event_id
                return JSONResponse(
                    {
                        "status": "accepted",
                        "edge_event_id": edge_event_id,
                        "event_id": event_id,
                    },
                    status_code=201,
                )
            event_id = str(uuid5(_FIXTURE_NAMESPACE, edge_event_id))
            record = EventRecord(edge_event_id, event_id, camera_id)
            self._events[edge_event_id] = record
            self._event_fingerprints[edge_event_id] = fingerprint
            self._accepted_event_ids[edge_event_id] = [event_id]
            return JSONResponse(
                {"status": "accepted", "edge_event_id": edge_event_id, "event_id": event_id},
                status_code=201,
            )

        @app.put("/api/v1/events/{event_id}/snapshot", status_code=201)
        async def snapshot(event_id: str, request: Request) -> JSONResponse:
            self._require_auth(request)
            if request.headers.get("content-type", "").split(";", 1)[0] != "image/jpeg":
                raise HTTPException(415, "snapshot must be image/jpeg")
            if event_id not in {event.event_id for event in self._events.values()}:
                raise HTTPException(404, "unknown event")
            content_length = _safe_content_length(request.headers.get("content-length"))
            if content_length > _MAX_SNAPSHOT_BYTES:
                raise HTTPException(413, "invalid snapshot size")
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > _MAX_SNAPSHOT_BYTES:
                    raise HTTPException(413, "invalid snapshot size")
            request.state.actual_body_bytes = total
            if total == 0:
                raise HTTPException(413, "invalid snapshot size")
            self._snapshot_bytes_discarded += total
            return JSONResponse({"snapshotKey": f"events/{event_id}/snapshot.jpg"}, status_code=201)

        @app.put("/api/v1/edge/topology-snapshots/{snapshot_id}")
        async def topology(snapshot_id: str, request: Request) -> JSONResponse:
            self._require_auth(request)
            _require_uuid(snapshot_id)
            body = await _bounded_json(request)
            try:
                snapshot = parse_topology_snapshot(body)
            except ContractViolation as exc:
                raise HTTPException(422, exc.detail) from exc
            if (
                snapshot.principal.edge_installation_id != self.edge_installation_id
                or snapshot.principal.enrollment_generation != 1
            ):
                raise HTTPException(422, "topology principal mismatch")
            fingerprint = self._fingerprint(body)
            existing = self._topologies.get(snapshot_id)
            if existing is not None:
                if existing.request_fingerprint != fingerprint:
                    raise HTTPException(409, "topology replay conflict")
                record = existing
            else:
                digest = hashlib.sha256(
                    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                record = TopologyRecord(
                    fingerprint,
                    snapshot.client_revision,
                    snapshot.expected_server_revision + 1,
                    snapshot_id,
                    digest,
                )
                self._topologies[snapshot_id] = record
            return JSONResponse(_topology_success(snapshot_id, record, include_omissions=True))

        @app.post("/api/v1/edge/topology-snapshots/{snapshot_id}/confirm")
        async def confirm_topology(snapshot_id: str, request: Request) -> JSONResponse:
            self._require_auth(request)
            _require_uuid(snapshot_id)
            record = self._topologies.get(snapshot_id)
            if record is None:
                raise HTTPException(404, "unknown snapshot")
            body = await _bounded_json(request)
            try:
                confirmation = parse_topology_confirmation(body)
            except ContractViolation as exc:
                raise HTTPException(422, exc.detail) from exc
            if (
                confirmation.confirmation_id != record.confirmation_id
                or confirmation.digest != record.digest
                or confirmation.expected_server_revision != record.server_revision
            ):
                raise HTTPException(409, "topology confirmation conflict")
            return JSONResponse(_topology_success(snapshot_id, record, include_omissions=False))

        return app

    def _require_auth(self, request: Request) -> None:
        if request.headers.get("authorization") != f"Bearer {self._bearer_token}":
            raise HTTPException(401, "invalid bearer")

    def _fingerprint(self, body: dict[str, Any]) -> str:
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2s(encoded, key=self._fingerprint_key).hexdigest()


async def _bounded_json(request: Request) -> dict[str, Any]:
    content_length = _safe_content_length(request.headers.get("content-length"))
    if content_length > _MAX_JSON_BYTES:
        raise HTTPException(413, "invalid request size")
    raw = await request.body()
    request.state.actual_body_bytes = len(raw)
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise HTTPException(413, "invalid request size")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(422, "invalid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(422, "request must be an object")
    return body


def _validate_event(body: dict[str, Any]) -> None:
    if not _EVENT_REQUIRED.issubset(body) or not set(body).issubset(_EVENT_FIELDS):
        raise HTTPException(422, "invalid event fields")
    if (
        not _bounded_text(body["camera_id"], 128)
        or body["type"] not in _EVENT_TYPES
        or not _bounded_text(body["detected_at"], 64)
        or isinstance(body["confidence"], bool)
        or not isinstance(body["confidence"], int | float)
        or not 0.0 <= float(body["confidence"]) <= 1.0
        or not _bounded_text(body["edge_event_id"], 128)
    ):
        raise HTTPException(422, "invalid event")
    for key in ("model_version", "detector_version", "clock_source", "clip_id"):
        if key in body and not _bounded_text(body[key], 128):
            raise HTTPException(422, f"invalid {key}")
    if "config_version" in body and not _nonnegative_int(body["config_version"]):
        raise HTTPException(422, "invalid config_version")
    if "operating_threshold" in body and (
        isinstance(body["operating_threshold"], bool)
        or not isinstance(body["operating_threshold"], int | float)
        or not 0.0 <= float(body["operating_threshold"]) <= 1.0
    ):
        raise HTTPException(422, "invalid operating_threshold")


def _require_exact_keys(body: dict[str, Any], expected: set[str]) -> None:
    if set(body) != expected:
        raise HTTPException(422, "invalid fields")


def _bounded_text(value: object, limit: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit


def _nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _safe_content_length(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return min(max(int(value), 0), _MAX_SNAPSHOT_BYTES + 1)
    except ValueError:
        return 0


def _require_uuid(value: str) -> None:
    try:
        UUID(value)
    except ValueError as exc:
        raise HTTPException(422, "invalid identifier") from exc


def _topology_success(
    snapshot_id: str,
    record: TopologyRecord,
    *,
    include_omissions: bool,
) -> dict[str, Any]:
    counts = {"created": 0, "updated": 0, "unchanged": 0}
    omissions = (
        {
            "confirmationId": record.confirmation_id,
            "digest": record.digest,
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "cameras": [],
            "rooms": [],
            "floors": [],
        }
        if include_omissions
        else None
    )
    return {
        "schemaVersion": 1,
        "snapshotId": snapshot_id,
        "clientRevision": record.client_revision,
        "serverRevision": record.server_revision,
        "result": {"floors": counts, "rooms": counts, "cameras": counts},
        "omissions": omissions,
    }


__all__ = ["EventRecord", "LocalBackendFixture", "RouteRecord"]
