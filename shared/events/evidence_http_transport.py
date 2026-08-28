"""Bounded HTTP transport and receipt parsing for evidence delivery."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import BinaryIO, Literal, TypeAlias, TypeVar, cast

from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)

MAX_RESPONSE_BYTES = 65536

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
PayloadValue = TypeVar("PayloadValue")
HttpResult: TypeAlias = tuple[int, Mapping[str, str], bytes] | DeliveryFailure


MAX_TRANSPORT_ERROR_CHARS = 200


class EvidenceClientConfigurationError(ValueError):
    """Raised when evidence transport configuration is invalid."""


def _describe_exception(exc: BaseException) -> str:
    """Render a transport exception's class + message for diagnostic logging.

    Bounded in length -- some socket errors embed the full request URL or a
    long OS error string, and this value flows straight into log messages.
    """
    text = f"{type(exc).__name__}: {exc}"
    if len(text) > MAX_TRANSPORT_ERROR_CHARS:
        return text[: MAX_TRANSPORT_ERROR_CHARS - 1] + "…"
    return text


def bounded_request(
    url: str,
    method: str,
    headers: Mapping[str, str],
    data: bytes | None,
    timeout: float,
    on_response: Callable[[int], None] | None = None,
) -> HttpResult:
    request = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if on_response is not None and 200 <= response.status < 300:
                on_response(response.status)
            body = response.read(MAX_RESPONSE_BYTES + 1)
            return response.status, response.headers, body
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read(MAX_RESPONSE_BYTES + 1)
    except (TimeoutError, OSError, urllib.error.URLError) as exc:
        return DeliveryFailure(
            DeliveryDisposition.RETRY, "NETWORK", transport_error=_describe_exception(exc)
        )


def bounded_file_request(
    url: str,
    headers: Mapping[str, str],
    media: BinaryIO,
    timeout: float,
) -> HttpResult:
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname
    if hostname is None:
        raise EvidenceClientConfigurationError("evidence URL must include a host")
    connection: http.client.HTTPConnection
    if parsed.scheme == "https":
        connection = http.client.HTTPSConnection(hostname, parsed.port, timeout=timeout)
    else:
        connection = http.client.HTTPConnection(hostname, parsed.port, timeout=timeout)
    target = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
    try:
        media.seek(0)
        connection.request("PUT", target, body=media, headers=dict(headers))
        response = connection.getresponse()
        return response.status, dict(response.headers), response.read(MAX_RESPONSE_BYTES + 1)
    except (TimeoutError, OSError, http.client.HTTPException) as exc:
        return DeliveryFailure(
            DeliveryDisposition.RETRY, "NETWORK", transport_error=_describe_exception(exc)
        )
    finally:
        connection.close()


def parse_event_result(
    result: HttpResult,
    expected: str,
) -> EventReceipt | DeliveryFailure:
    if isinstance(result, DeliveryFailure):
        return result
    status, headers, body = result
    receipt = _event_receipt(body)
    if (
        receipt is not None
        and receipt.edge_event_id == expected
        and (200 <= status < 300 or status == 409)
    ):
        return receipt
    if 200 <= status < 300:
        return DeliveryFailure(DeliveryDisposition.RETRY, "MALFORMED_RECEIPT")
    return classify_http_failure(status, headers)


def parse_clip_result(
    result: HttpResult,
    expected_clip_id: str,
    expected_version: int,
) -> ClipReceipt | DeliveryFailure:
    if isinstance(result, DeliveryFailure):
        return result
    status, headers, body = result
    receipt = _clip_receipt(body)
    version_matches = receipt is not None and (
        receipt.state_version == expected_version
        or (receipt.state == "EXPIRED" and receipt.state_version >= expected_version)
    )
    if (
        receipt is not None
        and receipt.clip_id == expected_clip_id
        and version_matches
        and (200 <= status < 300 or status == 409)
    ):
        return receipt
    if 200 <= status < 300:
        return DeliveryFailure(DeliveryDisposition.RETRY, "MALFORMED_RECEIPT")
    return classify_http_failure(status, headers)


def parse_capabilities(body: bytes) -> BackendCapabilities | DeliveryFailure:
    payload = parse_json_object(body)
    clip_export = payload.get("clip_export")
    if payload.get("event_idempotency") != 1 or clip_export not in {0, 1}:
        return DeliveryFailure(DeliveryDisposition.RETRY, "MALFORMED_RECEIPT")
    return BackendCapabilities(1, cast(Literal[0, 1], clip_export))


def classify_http_failure(status: int, headers: Mapping[str, str]) -> DeliveryFailure:
    # 401/403 are ambient auth/facility-config state, not a property of this
    # specific payload (unlike 400/413/415/422, which are genuinely permanent
    # -- retrying the exact same bytes cannot change a schema/size rejection).
    # A wrong or not-yet-provisioned relay token/facility binding gets fixed
    # out-of-band, and the event is still perfectly valid once it is -- so
    # dead-lettering it forever turns a recoverable config error into data
    # loss for no reason (see #183, #202). RETRY already has exactly the
    # backoff behavior this needs: retry_after_seconds when the response
    # supplies one, else capped exponential backoff (EvidenceSender._delay),
    # indefinitely -- it never gives up on its own.
    if status in {404, 405}:
        disposition = DeliveryDisposition.COMPATIBILITY
    elif status in {401, 403, 408, 425, 429} or 500 <= status <= 599:
        disposition = DeliveryDisposition.RETRY
    else:
        disposition = DeliveryDisposition.PERMANENT
    return DeliveryFailure(
        disposition,
        f"HTTP_{status}",
        status_code=status,
        retry_after_seconds=_retry_after(_header_value(headers, "Retry-After")),
    )


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.casefold()
    for header_name, value in headers.items():
        if header_name.casefold() == expected:
            return value
    return None


def parse_json_object(value: str | bytes) -> JsonObject:
    if len(value) > MAX_RESPONSE_BYTES:
        return {}
    try:
        payload = cast(JsonValue, json.loads(value))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return cast(JsonObject, payload) if isinstance(payload, dict) else {}


def encode_json(payload: Mapping[str, PayloadValue]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def join_http_url(base: str, suffix: str) -> str:
    return f"{base.rstrip('/')}/{suffix.lstrip('/')}"


def normalize_http_base(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise EvidenceClientConfigurationError("evidence URL must be absolute HTTP(S)")
    return value.rstrip("/")


def _event_receipt(body: bytes) -> EventReceipt | None:
    payload = parse_json_object(body)
    status_value = payload.get("status")
    edge_id = payload.get("edge_event_id")
    if not isinstance(edge_id, str):
        return None
    if status_value == "accepted_local":
        # The edge backend recorded the event durably and decided it will never
        # be pushed upstream (no Hub mapping yet, or no cloud client built), so
        # there is no upstream id to echo and demanding one retries forever.
        # This is terminal ONLY because the backend said so by name: an absent
        # or unrecognized status still falls through to MALFORMED_RECEIPT, which
        # is what a genuinely mangled response deserves.
        return EventReceipt("accepted_local", edge_id, "")
    if status_value != "accepted":
        return None
    event_id = payload.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return None
    return EventReceipt("accepted", edge_id, event_id)


def _clip_receipt(body: bytes) -> ClipReceipt | None:
    payload = parse_json_object(body)
    clip_id, state = payload.get("clip_id"), payload.get("state")
    version = payload.get("state_version")
    if not isinstance(clip_id, str) or state not in {"READY", "UNAVAILABLE", "EXPIRED"}:
        return None
    if not isinstance(version, int):
        return None
    sha256, size_bytes = payload.get("sha256"), payload.get("size_bytes")
    if sha256 is not None and not isinstance(sha256, str):
        return None
    if size_bytes is not None and not isinstance(size_bytes, int):
        return None
    return ClipReceipt(
        clip_id,
        state,
        version,
        sha256,
        size_bytes,
    )


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return min(900.0, max(0.0, float(value)))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return min(900.0, max(0.0, (parsed - datetime.now(UTC)).total_seconds()))
