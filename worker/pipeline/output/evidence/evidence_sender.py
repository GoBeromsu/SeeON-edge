"""Durable event-first sender for backend-proven evidence export."""

from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from shared.events.evidence_export_client import RelayEvidenceClient
from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimedClip,
    ClaimLease,
    EvidenceOutbox,
)
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClaimedEvent,
    ClipLocalState,
    EdgeEventId,
)


class SenderStep(StrEnum):
    DISABLED = "DISABLED"  # legacy value; live policy uses CLIP_POLICY_BLOCKED
    CLIP_POLICY_BLOCKED = "CLIP_POLICY_BLOCKED"
    CAPABILITY_BLOCKED = "CAPABILITY_BLOCKED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    PERMANENT_FAILURE = "PERMANENT_FAILURE"
    EVENT_ACKED = "EVENT_ACKED"
    CLIP_ACKED = "CLIP_ACKED"
    IDLE = "IDLE"
    LEASE_LOST = "LEASE_LOST"


class EvidenceTransport(Protocol):
    def probe_capabilities(self, camera_id: str) -> BackendCapabilities | DeliveryFailure: ...

    def send_event(
        self,
        payload_json: str,
        edge_event_id: EdgeEventId,
    ) -> EventReceipt | DeliveryFailure: ...

    def send_clip(self, claim: ClaimedClip) -> ClipReceipt | DeliveryFailure: ...


@dataclass(frozen=True, slots=True)
class SenderConfig:
    relay_url: str
    relay_token: str = field(repr=False)
    probe_camera_id: str
    lease_seconds: float = 30.0
    max_backoff_seconds: float = 300.0
    compatibility_reprobe_seconds: float = 60.0


class EvidenceSender:
    def __init__(
        self,
        database_path: Path,
        config: SenderConfig,
        *,
        transport: EvidenceTransport | None = None,
        clip_export_enabled: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.time,
        random_value: Callable[[], float] = random.random,
        owner: str | None = None,
    ) -> None:
        self.database_path = database_path
        self.config = config
        self._transport = transport or RelayEvidenceClient(config.relay_url, config.relay_token)
        self._clip_export_enabled = clip_export_enabled
        self._clock = clock
        self._random_value = random_value
        self._owner = owner or f"sender-{uuid.uuid4()}"
        self._next_capability_probe_at = 0.0

    def run_once(self) -> SenderStep:
        event_now = self._clock()
        event_lease = ClaimLease(self._owner, event_now, self.config.lease_seconds)
        with EvidenceOutbox.open(self.database_path) as outbox:
            outbox.release_compatibility()
            event = outbox.claim(event_lease)
            if event is not None:
                return self._send_event(outbox, event, event_now)

        if not self._clip_export_enabled():
            return SenderStep.CLIP_POLICY_BLOCKED
        if event_now < self._next_capability_probe_at:
            return SenderStep.CAPABILITY_BLOCKED
        capabilities = self._transport.probe_capabilities(self.config.probe_camera_id)
        if isinstance(capabilities, DeliveryFailure):
            if capabilities.disposition is DeliveryDisposition.COMPATIBILITY:
                self._next_capability_probe_at = (
                    event_now + self.config.compatibility_reprobe_seconds
                )
            return SenderStep.CAPABILITY_BLOCKED
        self._next_capability_probe_at = 0.0
        if capabilities.clip_export != 1:
            return SenderStep.CAPABILITY_BLOCKED
        # The capability probe is network I/O. Re-read the live switch after
        # it returns so an OFF transition cannot race into a durable claim.
        if not self._clip_export_enabled():
            return SenderStep.CLIP_POLICY_BLOCKED

        clip_now = self._clock()
        clip_lease = ClaimLease(self._owner, clip_now, self.config.lease_seconds)
        with EvidenceOutbox.open(self.database_path) as outbox:
            clip = outbox.claim_clip(clip_lease)
            if clip is not None:
                # This is the final dispatch boundary. If OFF wins after the
                # claim, release it with the lease CAS and restore the attempt
                # count because no request was sent.
                if not self._clip_export_enabled():
                    return (
                        SenderStep.CLIP_POLICY_BLOCKED
                        if outbox.release_clip_claim(clip)
                        else SenderStep.LEASE_LOST
                    )
                return self._send_clip(outbox, clip, clip_now)
        return SenderStep.IDLE

    def _send_event(
        self,
        outbox: EvidenceOutbox,
        claim: ClaimedEvent,
        now: float,
    ) -> SenderStep:
        result = self._transport.send_event(
            _payload_with_attempt_ordinal(claim.payload_json, claim.attempt_count),
            claim.edge_event_id,
        )
        if isinstance(result, DeliveryFailure):
            return self._handle_event_failure(outbox, claim, result, now)
        if result.edge_event_id != claim.edge_event_id or result.status != "accepted":
            failure = DeliveryFailure(DeliveryDisposition.RETRY, "RECEIPT_MISMATCH")
            return self._handle_event_failure(outbox, claim, failure, now)
        transitioned = outbox.acknowledge(claim, backend_event_id=result.event_id)
        return SenderStep.EVENT_ACKED if transitioned else SenderStep.LEASE_LOST

    def _send_clip(
        self,
        outbox: EvidenceOutbox,
        claim: ClaimedClip,
        now: float,
    ) -> SenderStep:
        result = self._transport.send_clip(claim)
        if isinstance(result, DeliveryFailure):
            return self._handle_clip_failure(outbox, claim, result, now)
        expected_state = "READY" if claim.local_state == "VERIFIED" else "UNAVAILABLE"
        immutable_facts_match = (
            result.sha256 == claim.sha256 and result.size_bytes == claim.size_bytes
        )
        if result.state == "EXPIRED":
            exact = (
                claim.local_state is ClipLocalState.VERIFIED
                and result.clip_id == claim.clip_id
                and result.state_version >= claim.state_version
                and immutable_facts_match
            )
        else:
            exact = (
                result.clip_id == claim.clip_id
                and result.state_version == claim.state_version
                and result.state == expected_state
                and immutable_facts_match
            )
        if not exact:
            failure = DeliveryFailure(DeliveryDisposition.RETRY, "RECEIPT_MISMATCH")
            return self._handle_clip_failure(outbox, claim, failure, now)
        transitioned = outbox.acknowledge_clip(
            claim,
            acknowledged_at=now,
            remote_state=result.state,
        )
        return SenderStep.CLIP_ACKED if transitioned else SenderStep.LEASE_LOST

    def _handle_event_failure(
        self,
        outbox: EvidenceOutbox,
        claim: ClaimedEvent,
        failure: DeliveryFailure,
        now: float,
    ) -> SenderStep:
        if failure.disposition is DeliveryDisposition.RETRY:
            transitioned = outbox.schedule_retry(
                claim,
                next_attempt_at=now + self._delay(claim.attempt_count, failure),
            )
            if transitioned:
                return SenderStep.RETRY_SCHEDULED
        else:
            transitioned = outbox.mark_event_failure(
                claim,
                state=failure.disposition.value,
                error_code=failure.code,
            )
            if transitioned:
                return SenderStep.PERMANENT_FAILURE
        return SenderStep.LEASE_LOST

    def _handle_clip_failure(
        self,
        outbox: EvidenceOutbox,
        claim: ClaimedClip,
        failure: DeliveryFailure,
        now: float,
    ) -> SenderStep:
        if failure.disposition is DeliveryDisposition.RETRY:
            transitioned = outbox.schedule_clip_retry(
                claim,
                next_attempt_at=now + self._delay(claim.attempt_count, failure),
                error_code=failure.code,
            )
            if transitioned:
                return SenderStep.RETRY_SCHEDULED
        else:
            transitioned = outbox.mark_clip_failure(
                claim,
                state=failure.disposition.value,
                error_code=failure.code,
            )
            if transitioned:
                return SenderStep.PERMANENT_FAILURE
        return SenderStep.LEASE_LOST

    def _delay(self, attempt: int, failure: DeliveryFailure) -> float:
        if failure.retry_after_seconds is not None:
            return min(900.0, max(0.0, failure.retry_after_seconds))
        cap = min(self.config.max_backoff_seconds, float(2 ** max(0, attempt - 1)))
        return cap * min(1.0, max(0.0, self._random_value()))


def _payload_with_attempt_ordinal(payload_json: str, attempt_count: int) -> str:
    payload = json.loads(payload_json)
    payload["attempt_ordinal"] = attempt_count
    return json.dumps(payload, separators=(",", ":"))


__all__ = ["EvidenceSender", "SenderConfig", "SenderStep"]
