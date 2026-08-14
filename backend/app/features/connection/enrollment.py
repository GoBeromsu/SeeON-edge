from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Literal, final

from pydantic import JsonValue, TypeAdapter, ValidationError

from backend.app.features.connection.hub_url import hub_url_transport_allowed
from contracts.edge_provisioning_codec import serialize_enrollment_verification
from contracts.edge_provisioning_enrollment import parse_enrollment_verification_result
from contracts.edge_provisioning_models import (
    ContractViolation,
    EnrollmentVerification,
    EnrollmentVerificationResult,
)

EnrollmentErrorClass = Literal["auth", "conflict", "timeout", "unreachable", "invalid_response"]
_JSON_RECORD = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class EnrollmentCredentials:
    facility_code: str
    client_installation_ref: str
    facility_token: str = field(repr=False)


@final
class EnrollmentVerificationFailure(Exception):
    __slots__ = ("error_class", "status_code")
    error_class: EnrollmentErrorClass
    status_code: int

    def __init__(self, error_class: EnrollmentErrorClass, status_code: int) -> None:
        super().__init__(error_class)
        self.error_class = error_class
        self.status_code = status_code


def verify_enrollment(
    events_url: str | None,
    credentials: EnrollmentCredentials,
    *,
    timeout_sec: float,
) -> EnrollmentVerificationResult:
    endpoint = enrollment_endpoint(events_url)
    if endpoint is None:
        # Refuse before building an Authorization header so a rejected cleartext
        # public origin never observes the facility bearer token.
        raise EnrollmentVerificationFailure("unreachable", 503)
    body = serialize_enrollment_verification(
        EnrollmentVerification(
            credentials.facility_code,
            credentials.client_installation_ref,
        )
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {credentials.facility_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        # Default SSL context (normal certificate validation). Never pass an
        # unverified context — Hub enrollment must not skip TLS checks.
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            parsed = _JSON_RECORD.validate_json(response.read())
        return parse_enrollment_verification_result(parsed)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise EnrollmentVerificationFailure("auth", exc.code) from None
        if exc.code == 409:
            raise EnrollmentVerificationFailure("conflict", exc.code) from None
        raise EnrollmentVerificationFailure("unreachable", 503) from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise EnrollmentVerificationFailure("timeout", 504) from None
        raise EnrollmentVerificationFailure("unreachable", 503) from None
    except TimeoutError:
        raise EnrollmentVerificationFailure("timeout", 504) from None
    except (ContractViolation, ValidationError, UnicodeDecodeError):
        raise EnrollmentVerificationFailure("invalid_response", 502) from None


def enrollment_endpoint(events_url: str | None) -> str | None:
    if events_url is None:
        return None
    cleaned = events_url.strip()
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if not hub_url_transport_allowed(cleaned):
        return None
    origin = urllib.parse.urlunsplit(parsed._replace(path="", query="", fragment=""))
    return f"{origin.rstrip('/')}/api/v1/edge/enrollments/verify"


__all__ = [
    "EnrollmentCredentials",
    "EnrollmentErrorClass",
    "EnrollmentVerificationFailure",
    "enrollment_endpoint",
    "verify_enrollment",
]
