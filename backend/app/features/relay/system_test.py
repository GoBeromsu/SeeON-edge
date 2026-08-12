"""Facility-level SYSTEM_TEST relay through the enrolled backend client."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.features.relay.auth import authorize_relay
from backend.app.shared.backend_client_bundle import backend_client_bundle
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure

RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"
SYSTEM_TEST_LABEL = "SYSTEM TEST - NOT A RESIDENT ALERT"
_INVALID_AUTH_DIAGNOSTIC_TOKEN = "eft_v1.system-test.invalid"
_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
_UUID4_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

router = APIRouter(prefix="/system-tests")


class RelaySystemTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_event_id: str = Field(pattern=_UUID4_PATTERN)
    type: Literal["SYSTEM_TEST"]
    source: Literal["SYSTEM_TEST"]
    test_mode: Literal["SYSTEM_TEST"]
    label: Literal["SYSTEM TEST - NOT A RESIDENT ALERT"]
    detected_at: str = Field(min_length=1)
    validation_run_id: str = Field(pattern=_UUID_PATTERN)
    attempt_ordinal: int = Field(ge=1)

    @field_validator("detected_at")
    @classmethod
    def require_aware_timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("detected_at must include a timezone")
        return value


@router.post("/auth-check")
def classify_system_test_invalid_auth(
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> dict[str, str | int]:
    """Classify a payload-free backend auth failure without mutating enrollment."""
    authorize_relay(request, relay_token)
    bundle = backend_client_bundle(request.app)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="enrolled backend client is not configured",
        )
    diagnostic = bundle.evidence_client.with_bearer_token(
        _INVALID_AUTH_DIAGNOSTIC_TOKEN
    )
    result = diagnostic.probe_capabilities("SYSTEM_TEST")
    if (
        not isinstance(result, DeliveryFailure)
        or result.disposition is not DeliveryDisposition.RETRY
        or result.status_code not in {401, 403}
        or result.code != f"HTTP_{result.status_code}"
    ):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="invalid-auth diagnostic did not return 401 or 403",
        )
    return {
        "disposition": result.disposition.value,
        "code": result.code,
        "status_code": result.status_code,
    }


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def relay_system_test(
    payload: RelaySystemTestRequest,
    request: Request,
    relay_token: Annotated[str | None, Header(alias=RELAY_TOKEN_HEADER)] = None,
) -> dict[str, str]:
    authorize_relay(request, relay_token)
    bundle = backend_client_bundle(request.app)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="enrolled backend client is not configured",
        )
    backend_payload = payload.model_dump(
        include={
            "edge_event_id",
            "type",
            "test_mode",
            "detected_at",
            "validation_run_id",
        }
    )
    result = bundle.evidence_client.send_event_payload(
        backend_payload,
        payload.edge_event_id,
    )
    if isinstance(result, DeliveryFailure):
        if result.disposition is DeliveryDisposition.COMPATIBILITY:
            response_status = status.HTTP_404_NOT_FOUND
        elif result.disposition is DeliveryDisposition.RETRY:
            response_status = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            response_status = result.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(
            status_code=response_status,
            detail="backend ingest rejected system test",
        )
    return {
        "status": result.status,
        "edge_event_id": result.edge_event_id,
        "event_id": result.event_id,
    }


__all__ = ["router"]
