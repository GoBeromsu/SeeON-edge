"""Backend Event API HTTP client shared by the API relay and edge worker."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, Literal, Protocol, TypedDict

from contracts.event import EventPayload
from contracts.relay import AlertEventType, EventApiPayload
from shared.events.evidence_export_client import BackendEvidenceClient
from shared.events.evidence_export_contract import DeliveryFailure, EventReceipt

DEFAULT_TIMEOUT_SEC: Final = 0.5
EdgeEventId = str
IngestErrorClass = Literal["auth", "timeout", "unreachable"]


class _PublishedEvent(Protocol):
    event_type: str
    evidence: EventPayload


@dataclass(frozen=True, slots=True)
class IngestSendResult:
    """Classified outcome of a single ingest POST, for callers (story G005's
    heartbeat relay) that need more than a bare bool to report *why* a send
    failed, not just that it did.
    """

    ok: bool
    error_class: IngestErrorClass | None


@dataclass(slots=True)
class EdgeIngestClient:
    events_url: str
    camera_id: str
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    bearer_token: str | None = field(default=None, repr=False)
    _failure_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.events_url = _parse_http_url(self.events_url)
        self.camera_id = _required(self.camera_id, "camera_id")
        self.bearer_token = _clean(self.bearer_token)

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def send_heartbeat(self) -> bool:
        return self.send_heartbeat_result().ok

    def send_heartbeat_result(self) -> IngestSendResult:
        return self._post_classified(
            _join_url(self.events_url, "heartbeat"), {"camera_id": self.camera_id}
        )

    def send_alert(
        self,
        *,
        event_type: AlertEventType,
        detected_at: str,
        probability: float,
        audit: dict[str, object] | None = None,
        snapshot_bytes: bytes | None = None,
        clip_id: str | None = None,
    ) -> bool:
        payload = EventApiPayload(
            camera_id=self.camera_id,
            type=event_type,
            detected_at=detected_at,
            confidence=probability,
            **_audit_payload_fields(audit),
        ).as_dict()
        if clip_id is not None and clip_id.strip() != "":
            payload["clip_id"] = clip_id
        # dict is invariant in its value type, so RelayAlertPayload
        # (dict[str, str | int | float]) is not a dict[str, object]. Widen
        # explicitly at the boundary rather than loosening _post_json.
        response = self._post_json(self.events_url, dict(payload))
        if response is None:
            self._increment_failure()
            return False
        event_id = response.get("id")
        if snapshot_bytes is not None and isinstance(event_id, str) and event_id != "":
            self._put_snapshot(_join_url(self.events_url, f"{event_id}/snapshot"), snapshot_bytes)
        return True

    def send_alert_receipt(
        self,
        *,
        edge_event_id: EdgeEventId,
        event_type: AlertEventType,
        detected_at: str,
        probability: float,
        audit: dict[str, object] | None = None,
        snapshot_bytes: bytes | None = None,
        clip_id: str | None = None,
        on_accepted: Callable[[float], None] | None = None,
    ) -> EventReceipt | DeliveryFailure:
        payload = EventApiPayload(
            camera_id=self.camera_id,
            type=event_type,
            detected_at=detected_at,
            confidence=probability,
            **_audit_payload_fields(audit),
        ).as_dict()
        payload["edge_event_id"] = edge_event_id
        if clip_id is not None and clip_id.strip() != "":
            payload["clip_id"] = clip_id
        accepted_at: float | None = None

        def accepted(timestamp: float) -> None:
            nonlocal accepted_at
            accepted_at = timestamp

        result = BackendEvidenceClient(
            self.events_url,
            self.bearer_token,
            self.timeout_sec,
        ).send_event_payload(payload, edge_event_id, on_accepted=accepted)
        if isinstance(result, EventReceipt):
            if on_accepted is not None and accepted_at is not None:
                on_accepted(accepted_at)
            # An accepted_local receipt has no upstream id, so there is no
            # snapshot resource to PUT to; `{events_url}//snapshot` is not one.
            if snapshot_bytes is not None and result.event_id != "":
                self._put_snapshot(
                    _join_url(self.events_url, f"{result.event_id}/snapshot"), snapshot_bytes
                )
        if isinstance(result, DeliveryFailure):
            self._increment_failure()
        return result

    def for_camera(self, camera_id: str) -> EdgeIngestClient:
        if camera_id == self.camera_id:
            return self
        return EdgeIngestClient(
            events_url=self.events_url,
            camera_id=camera_id,
            timeout_sec=self.timeout_sec,
            bearer_token=self.bearer_token,
        )

    def emit(self, event: EventPayload) -> None:
        event_type = str(event.get("event_type", ""))
        if event_type not in {"fall", "bed-exit"}:
            return
        detected_at = str(event.get("detected_at", ""))
        if detected_at == "":
            detected_at = _utc_iso_timestamp()
        probability = _event_probability(event)
        self.send_alert(
            event_type="bed-exit" if event_type == "bed-exit" else "fall",
            detected_at=detected_at,
            probability=probability,
            clip_id=_event_clip_id(event),
        )

    def publish(self, event: _PublishedEvent) -> None:
        if event.event_type not in {"fall", "bed-exit"}:
            return
        self.send_alert(
            event_type="bed-exit" if event.event_type == "bed-exit" else "fall",
            detected_at=_utc_iso_timestamp(),
            probability=_event_probability(event.evidence),
            clip_id=_event_clip_id(event.evidence),
        )

    def _post(self, url: str, payload: dict[str, str | float]) -> bool:
        return self._post_classified(url, payload).ok

    def _post_classified(self, url: str, payload: dict[str, str | float]) -> IngestSendResult:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            self._increment_failure()
            # Annotated so the ternary keeps the Literal type instead of widening
            # to str, which is what IngestSendResult.error_class requires.
            error_class: Literal["auth", "unreachable"] = (
                "auth" if exc.code in (401, 403) else "unreachable"
            )
            return IngestSendResult(ok=False, error_class=error_class)
        except urllib.error.URLError as exc:
            self._increment_failure()
            if isinstance(exc.reason, TimeoutError):
                return IngestSendResult(ok=False, error_class="timeout")
            return IngestSendResult(ok=False, error_class="unreachable")
        except TimeoutError:
            self._increment_failure()
            return IngestSendResult(ok=False, error_class="timeout")
        except OSError:
            self._increment_failure()
            return IngestSendResult(ok=False, error_class="unreachable")
        return IngestSendResult(ok=True, error_class=None)

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object] | None:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read()
        except (TimeoutError, OSError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        if body == b"":
            return {}
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _put_snapshot(self, url: str, snapshot_bytes: bytes) -> None:
        request = urllib.request.Request(
            url,
            data=snapshot_bytes,
            headers={"Content-Type": "image/jpeg", **self._auth_headers()},
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response.read()
        except (TimeoutError, OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"snapshot upload failed: {exc}", file=sys.stderr)

    def _auth_headers(self) -> dict[str, str]:
        if self.bearer_token is None:
            return {}
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def _increment_failure(self) -> None:
        with self._lock:
            self._failure_count += 1


def _required(value: str, name: str) -> str:
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{name} must be set")
    return stripped


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc == "":
        raise ValueError(f"Event API URL must be absolute HTTP(S): {url}")
    return url


def _join_url(base_url: str, child: str) -> str:
    return f"{base_url.rstrip('/')}/{child}"


def _utc_iso_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _event_probability(event: EventPayload) -> float:
    value = event.get("probability", event.get("confidence", 1.0))
    if isinstance(value, int | float):
        return min(1.0, max(0.0, float(value)))
    return 1.0


def _event_clip_id(event: EventPayload) -> str | None:
    value = event.get("clip_id")
    if isinstance(value, str) and value.strip() != "":
        return value
    return None


class _AuditPayloadFields(TypedDict, total=False):
    """The optional EventApiPayload audit fields, typed so ** unpacking checks.

    A plain dict[str, object] cannot be verified against EventApiPayload's
    int/str/float parameters, which is what mypy was reporting; each value is
    narrowed here instead of being asserted at the call site.
    """

    config_version: int
    model_version: str
    detector_version: str
    operating_threshold: float
    clock_source: str


def _audit_payload_fields(audit: dict[str, object] | None) -> _AuditPayloadFields:
    fields: _AuditPayloadFields = {}
    if audit is None:
        return fields
    config_version = audit.get("config_version")
    if isinstance(config_version, int):
        fields["config_version"] = config_version
    model_version = audit.get("model_version")
    if isinstance(model_version, str):
        fields["model_version"] = model_version
    detector_version = audit.get("detector_version")
    if isinstance(detector_version, str):
        fields["detector_version"] = detector_version
    operating_threshold = audit.get("operating_threshold")
    if isinstance(operating_threshold, int | float):
        fields["operating_threshold"] = float(operating_threshold)
    clock_source = audit.get("clock_source")
    if isinstance(clock_source, str):
        fields["clock_source"] = clock_source
    return fields


__all__ = ["DEFAULT_TIMEOUT_SEC", "EdgeIngestClient", "IngestSendResult", "_utc_iso_timestamp"]
