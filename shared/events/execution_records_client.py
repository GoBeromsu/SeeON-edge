"""Relay HTTP client for Worker -> Backend execution-record batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.evidence_http_transport import (
    EvidenceClientConfigurationError,
    bounded_request,
    classify_http_failure,
    join_http_url,
    normalize_http_base,
    parse_json_object,
)
from shared.events.execution_records import (
    MAX_EXECUTION_RECORD_BODY_BYTES,
    RELAY_EXECUTION_RECORDS_PATH,
    ExecutionRecordContractError,
    WireBatch,
    WireBatchReceipt,
)

_RELAY_PATH: Final = f"/api/v1{RELAY_EXECUTION_RECORDS_PATH}"
_DEFAULT_TIMEOUT_SEC: Final = 2.0


@dataclass(frozen=True, slots=True)
class ExecutionRecordsClient:
    """POST one ``WireBatch`` to the Backend relay ingest path."""

    base_url: str
    relay_token: str = field(repr=False)
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", normalize_http_base(self.base_url))
        if not self.relay_token:
            raise EvidenceClientConfigurationError("relay token must be set")
        if self.timeout_sec <= 0:
            raise EvidenceClientConfigurationError("timeout_sec must be positive")

    def post_batch(self, batch: WireBatch) -> WireBatchReceipt | DeliveryFailure:
        body = batch.encode()
        if len(body) > MAX_EXECUTION_RECORD_BODY_BYTES:
            return DeliveryFailure(DeliveryDisposition.PERMANENT, "OVERSIZE")
        path = _RELAY_PATH.lstrip("/")
        result = bounded_request(
            join_http_url(self.base_url, path),
            "POST",
            {
                "Content-Type": "application/json",
                "X-Edge-Relay-Token": self.relay_token,
            },
            body,
            self.timeout_sec,
        )
        if isinstance(result, DeliveryFailure):
            return result
        status, headers, response_body = result
        if 200 <= status < 300:
            return _parse_receipt(response_body, batch.batch_id)
        return classify_http_failure(status, headers, response_body)


def _parse_receipt(body: bytes, expected_batch_id: str) -> WireBatchReceipt | DeliveryFailure:
    try:
        receipt = WireBatchReceipt.from_json(parse_json_object(body))
    except ExecutionRecordContractError:
        return DeliveryFailure(DeliveryDisposition.RETRY, "MALFORMED_RECEIPT")
    if receipt.batch_id != expected_batch_id:
        return DeliveryFailure(DeliveryDisposition.RETRY, "MALFORMED_RECEIPT")
    return receipt


__all__ = ["ExecutionRecordsClient"]
