"""Evidence clients for the local relay and backend event API."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import BinaryIO, Final, Protocol, TypeVar, cast

from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from shared.events.evidence_http_transport import (
    EvidenceClientConfigurationError,
    JsonObject,
    JsonValue,
    bounded_file_request,
    bounded_request,
    classify_http_failure,
    encode_json,
    join_http_url,
    normalize_http_base,
    parse_capabilities,
    parse_clip_result,
    parse_event_result,
    parse_json_object,
)
from shared.events.relay_failure_log import RelayFailureLog

LOGGER: Final = logging.getLogger(__name__)

EdgeEventId = str
PayloadValue = TypeVar("PayloadValue")
MAX_LOCAL_EVENT_PAYLOAD_BYTES: Final = 512 * 1024


class ClaimedClipRequest(Protocol):
    @property
    def clip_id(self) -> str: ...

    @property
    def local_state(self) -> str: ...

    @property
    def state_version(self) -> int: ...

    @property
    def sha256(self) -> str | None: ...

    @property
    def size_bytes(self) -> int | None: ...

    @property
    def mime_type(self) -> str | None: ...

    @property
    def codec(self) -> str | None: ...

    @property
    def duration_ms(self) -> int | None: ...

    @property
    def clip_start_at(self) -> str | None: ...

    @property
    def clip_end_at(self) -> str | None: ...

    @property
    def finalized_at(self) -> str | None: ...

    @property
    def unavailable_reason(self) -> str | None: ...

    @property
    def event_ids(self) -> tuple[str, ...]: ...

    @property
    def event_payload_json(self) -> str: ...


class ReadyClipRequest(Protocol):
    @property
    def clip_id(self) -> str: ...

    @property
    def camera_id(self) -> str: ...

    @property
    def event_refs(self) -> Sequence[str]: ...

    @property
    def state_version(self) -> int: ...

    @property
    def sha256(self) -> str: ...

    @property
    def size_bytes(self) -> int: ...

    @property
    def mime_type(self) -> str: ...

    @property
    def codec(self) -> str: ...

    @property
    def duration_ms(self) -> int: ...

    @property
    def clip_start_at(self) -> str: ...

    @property
    def clip_end_at(self) -> str: ...

    @property
    def finalized_at(self) -> str: ...


class UnavailableClipRequest(Protocol):
    @property
    def clip_id(self) -> str: ...

    @property
    def camera_id(self) -> str: ...

    @property
    def event_refs(self) -> Sequence[str]: ...

    @property
    def state_version(self) -> int: ...

    @property
    def reason(self) -> str: ...


def _capabilities_failure_log() -> RelayFailureLog:
    return RelayFailureLog(LOGGER, channel="relay capabilities probe", method="GET")


def _alerts_failure_log() -> RelayFailureLog:
    return RelayFailureLog(LOGGER, channel="relay event delivery", method="POST")


def _clips_failure_log() -> RelayFailureLog:
    return RelayFailureLog(LOGGER, channel="relay clip delivery", method="PUT")


def _snapshot_attachment_failure_log() -> RelayFailureLog:
    return RelayFailureLog(LOGGER, channel="relay snapshot attachment delivery", method="POST")


def _snapshot_disposition_failure_log() -> RelayFailureLog:
    return RelayFailureLog(LOGGER, channel="relay snapshot disposition delivery", method="POST")


@dataclass(frozen=True, slots=True)
class RelayEvidenceClient:
    base_url: str
    relay_token: str = field(repr=False)
    timeout_sec: float = 2.0
    _capabilities_failures: RelayFailureLog = field(
        init=False, repr=False, compare=False, default_factory=_capabilities_failure_log
    )
    _alerts_failures: RelayFailureLog = field(
        init=False, repr=False, compare=False, default_factory=_alerts_failure_log
    )
    _clips_failures: RelayFailureLog = field(
        init=False, repr=False, compare=False, default_factory=_clips_failure_log
    )
    _snapshot_attachment_failures: RelayFailureLog = field(
        init=False, repr=False, compare=False, default_factory=_snapshot_attachment_failure_log
    )
    _snapshot_disposition_failures: RelayFailureLog = field(
        init=False, repr=False, compare=False, default_factory=_snapshot_disposition_failure_log
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", normalize_http_base(self.base_url))
        if not self.relay_token:
            raise EvidenceClientConfigurationError("relay token must be set")

    def probe_capabilities(self, camera_id: str) -> BackendCapabilities | DeliveryFailure:
        path = "api/v1/relay/capabilities"
        url = join_http_url(self.base_url, path)
        url = f"{url}?{urllib.parse.urlencode({'camera_id': camera_id})}"
        result = bounded_request(url, "GET", self._headers(), None, self.timeout_sec)
        if isinstance(result, DeliveryFailure):
            self._capabilities_failures.record_failure(result, path=path)
            return result
        status, headers, body = result
        if not 200 <= status < 300:
            failure = classify_http_failure(status, headers)
            self._capabilities_failures.record_failure(failure, path=path)
            return failure
        parsed = parse_capabilities(body)
        if isinstance(parsed, DeliveryFailure):
            self._capabilities_failures.record_failure(parsed, path=path)
        else:
            self._capabilities_failures.record_success(path=path)
        return parsed

    def send_event(
        self,
        payload_json: str,
        edge_event_id: EdgeEventId,
    ) -> EventReceipt | DeliveryFailure:
        path = "api/v1/relay/alerts"
        result = bounded_request(
            join_http_url(self.base_url, path),
            "POST",
            {"Content-Type": "application/json", **self._headers()},
            payload_json.encode("utf-8"),
            self.timeout_sec,
        )
        parsed = parse_event_result(result, edge_event_id)
        if isinstance(parsed, DeliveryFailure):
            self._alerts_failures.record_failure(parsed, path=path)
        else:
            self._alerts_failures.record_success(path=path)
        return parsed

    def send_clip(self, claim: ClaimedClipRequest) -> ClipReceipt | DeliveryFailure:
        identity = _local_event_identity(claim.event_payload_json)
        if identity is None:
            return DeliveryFailure(DeliveryDisposition.PERMANENT, "INVALID_EVENT_PAYLOAD")
        camera_id, facility_id = identity
        payload: JsonObject = {
            "state": "READY" if claim.local_state == "VERIFIED" else "UNAVAILABLE",
            "camera_id": camera_id,
            "facility_id": facility_id,
            "event_refs": list(claim.event_ids),
            "state_version": claim.state_version,
        }
        if claim.local_state == "VERIFIED":
            payload.update(
                sha256=claim.sha256,
                size_bytes=claim.size_bytes,
                mime_type=claim.mime_type,
                codec=claim.codec,
                duration_ms=claim.duration_ms,
                clip_start_at=claim.clip_start_at,
                clip_end_at=claim.clip_end_at,
                finalized_at=claim.finalized_at,
            )
        else:
            payload["reason"] = _backend_reason(claim)
        path = f"api/v1/relay/clips/{claim.clip_id}"
        result = bounded_request(
            join_http_url(self.base_url, path),
            "PUT",
            {"Content-Type": "application/json", **self._headers()},
            encode_json(payload),
            self.timeout_sec,
        )
        parsed = parse_clip_result(result, claim.clip_id, claim.state_version)
        if isinstance(parsed, DeliveryFailure):
            self._clips_failures.record_failure(parsed, path=path)
        else:
            self._clips_failures.record_success(path=path)
        return parsed

    def send_snapshot_attachment(
        self, payload: Mapping[str, PayloadValue]
    ) -> None | DeliveryFailure:
        path = "api/v1/relay/snapshot-attachments"
        result = bounded_request(
            join_http_url(self.base_url, path),
            "POST",
            {"Content-Type": "application/json", **self._headers()},
            encode_json(payload),
            self.timeout_sec,
        )
        parsed = _parse_relay_acceptance(result)
        if isinstance(parsed, DeliveryFailure):
            self._snapshot_attachment_failures.record_failure(parsed, path=path)
        else:
            self._snapshot_attachment_failures.record_success(path=path)
        return parsed

    def send_snapshot_disposition(
        self, payload: Mapping[str, PayloadValue]
    ) -> None | DeliveryFailure:
        path = "api/v1/relay/snapshot-dispositions"
        result = bounded_request(
            join_http_url(self.base_url, path),
            "POST",
            {"Content-Type": "application/json", **self._headers()},
            encode_json(payload),
            self.timeout_sec,
        )
        parsed = _parse_relay_acceptance(result)
        if isinstance(parsed, DeliveryFailure):
            self._snapshot_disposition_failures.record_failure(parsed, path=path)
        else:
            self._snapshot_disposition_failures.record_success(path=path)
        return parsed

    def _headers(self) -> dict[str, str]:
        return {"X-Edge-Relay-Token": self.relay_token}


@dataclass(frozen=True, slots=True)
class BackendEvidenceClient:
    events_url: str
    bearer_token: str | None = field(default=None, repr=False)
    timeout_sec: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "events_url", normalize_http_base(self.events_url))

    def for_camera(self, _camera_id: str) -> BackendEvidenceClient:
        return self

    def probe_capabilities(self, camera_id: str) -> BackendCapabilities | DeliveryFailure:
        url = join_http_url(self.events_url, "capabilities")
        url = f"{url}?{urllib.parse.urlencode({'camera_id': camera_id})}"
        result = bounded_request(url, "GET", self._headers(), None, self.timeout_sec)
        if isinstance(result, DeliveryFailure):
            return result
        status, headers, body = result
        if not 200 <= status < 300:
            return classify_http_failure(status, headers)
        return parse_capabilities(body)

    def send_event_payload(
        self,
        payload: Mapping[str, PayloadValue],
        edge_event_id: EdgeEventId,
        on_accepted: Callable[[float], None] | None = None,
    ) -> EventReceipt | DeliveryFailure:
        accepted_at: float | None = None

        def capture_accepted(_status: int) -> None:
            nonlocal accepted_at
            accepted_at = time.time()

        result = bounded_request(
            self.events_url,
            "POST",
            {"Content-Type": "application/json", **self._headers()},
            encode_json(payload),
            self.timeout_sec,
            on_response=capture_accepted,
        )
        parsed = parse_event_result(result, edge_event_id)
        if isinstance(parsed, EventReceipt) and accepted_at is not None and on_accepted is not None:
            on_accepted(accepted_at)
        return parsed

    def publish_ready(
        self,
        request: ReadyClipRequest,
        media: BinaryIO,
    ) -> ClipReceipt | DeliveryFailure:
        headers = {
            **self._headers(),
            "Content-Type": "video/mp4",
            "Content-Length": str(request.size_bytes),
            "X-Edge-Camera-Id": request.camera_id,
            "X-Edge-Event-Refs": json.dumps(list(request.event_refs), separators=(",", ":")),
            "X-Clip-Start-At": request.clip_start_at,
            "X-Clip-End-At": request.clip_end_at,
            "X-Clip-Finalized-At": request.finalized_at,
            "X-Clip-Sha256": request.sha256,
            "X-Clip-Size-Bytes": str(request.size_bytes),
            "X-Clip-Duration-Ms": str(request.duration_ms),
            "X-Clip-State-Version": str(request.state_version),
        }
        result = bounded_file_request(
            join_http_url(self.events_url, f"clips/{request.clip_id}"),
            headers,
            media,
            self.timeout_sec,
        )
        return parse_clip_result(result, request.clip_id, request.state_version)

    def report_unavailable(
        self,
        request: UnavailableClipRequest,
    ) -> ClipReceipt | DeliveryFailure:
        payload: JsonObject = {
            "camera_id": request.camera_id,
            "event_refs": list(request.event_refs),
            "state_version": request.state_version,
            "reason": request.reason,
        }
        result = bounded_request(
            join_http_url(self.events_url, f"clips/{request.clip_id}/state"),
            "PUT",
            {"Content-Type": "application/json", **self._headers()},
            encode_json(payload),
            self.timeout_sec,
        )
        return parse_clip_result(result, request.clip_id, request.state_version)

    def _headers(self) -> dict[str, str]:
        if not self.bearer_token:
            return {}
        return {"Authorization": f"Bearer {self.bearer_token}"}


def _local_event_identity(payload_json: str) -> tuple[str, str] | None:
    if len(payload_json.encode("utf-8")) > MAX_LOCAL_EVENT_PAYLOAD_BYTES:
        return None
    try:
        payload = cast(JsonValue, json.loads(payload_json))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    camera_id = payload.get("camera_id")
    facility_id = payload.get("facility_id")
    if (
        not isinstance(camera_id, str)
        or not camera_id.strip()
        or not isinstance(facility_id, str)
        or not facility_id.strip()
    ):
        return None
    return camera_id, facility_id


def _backend_reason(claim: ClaimedClipRequest) -> str:
    return "CORRUPT" if str(claim.unavailable_reason) == "CORRUPT" else "CAPTURE_FAILED"


def _parse_relay_acceptance(
    result: tuple[int, Mapping[str, str], bytes] | DeliveryFailure,
) -> None | DeliveryFailure:
    if isinstance(result, DeliveryFailure):
        return result
    status, headers, body = result
    if not 200 <= status < 300:
        return classify_http_failure(status, headers)
    if parse_json_object(body).get("status") != "accepted":
        return DeliveryFailure(DeliveryDisposition.RETRY, "MALFORMED_RECEIPT")
    return None


__all__ = ["BackendEvidenceClient", "RelayEvidenceClient"]
