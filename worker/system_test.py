"""Explicit, operator-gated facility-level SYSTEM_TEST delivery."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, assert_never
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from shared.events.evidence_export_client import RelayEvidenceClient
from shared.events.evidence_export_contract import DeliveryFailure, EventReceipt
from worker.pipeline.output.evidence.evidence_outbox import (
    EdgeEventId,
    EvidenceOutbox,
    StagedEvent,
)
from worker.pipeline.output.evidence.evidence_outbox_types import EventDeliveryState
from worker.pipeline.output.evidence.evidence_sender import (
    EvidenceSender,
    SenderConfig,
    SenderStep,
)
from worker.runtime.config import WORKER_STATE_DB_FILENAME

LOGGER: Final = logging.getLogger(__name__)
SYSTEM_TEST_GATE_ENV: Final = "ML_WORKER_SYSTEM_TEST_GATE"
SYSTEM_TEST_GATE_VALUE: Final = "SYSTEM_TEST_OPERATOR_ENABLED"
SYSTEM_TEST_LABEL: Final = "SYSTEM TEST - NOT A RESIDENT ALERT"


class SystemTestAction(StrEnum):
    EMIT = "emit"
    RETRY = "retry"
    REPLAY = "replay"
    AUTH_CHECK = "auth-check"


class SystemTestStatus(StrEnum):
    ACKED = "ACKED"
    PREVIOUSLY_ACKED = "PREVIOUSLY_ACKED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    REPLAY_ACKED = "REPLAY_ACKED"
    AUTH_CLASSIFIED = "AUTH_CLASSIFIED"
    FAILED = "FAILED"


class SystemTestDisabledError(RuntimeError):
    """Raised when the one-shot operator gate is absent."""


class SystemTestConfigurationError(RuntimeError):
    """Raised when an operator supplies a malformed gate or invocation."""


class _SystemTestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_event_id: UUID
    type: Literal["SYSTEM_TEST"] = "SYSTEM_TEST"
    source: Literal["SYSTEM_TEST"] = "SYSTEM_TEST"
    test_mode: Literal["SYSTEM_TEST"] = "SYSTEM_TEST"
    label: Literal["SYSTEM TEST - NOT A RESIDENT ALERT"] = SYSTEM_TEST_LABEL
    detected_at: str
    validation_run_id: UUID


@dataclass(frozen=True, slots=True)
class SystemTestConfig:
    database_path: Path
    relay_url: str
    relay_token: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class SystemTestCommand:
    action: SystemTestAction
    confirmation: str | None
    validation_run_id: str | None
    edge_event_id: str | None


@dataclass(frozen=True, slots=True)
class SystemTestOutcome:
    status: SystemTestStatus
    edge_event_id: str | None = None
    correlation_id: str | None = None
    backend_event_id: str | None = None
    error_code: str | None = None


class SystemTestEmitter:
    """Stage and explicitly drive operator-only rows through production egress."""

    def __init__(
        self,
        config: SystemTestConfig,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        event_id_factory: Callable[[], UUID] = uuid4,
        transport: RelayEvidenceClient | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._event_id_factory = event_id_factory
        self._transport = transport or RelayEvidenceClient(
            config.relay_url,
            config.relay_token,
        )

    def emit(self, validation_run_id: UUID) -> SystemTestOutcome:
        with EvidenceOutbox.open(self._config.database_path) as outbox:
            registration = outbox.create_or_load_operator_event(
                str(validation_run_id),
                lambda: self._new_event(validation_run_id),
            )
        if (
            not registration.created
            and registration.delivery_state is EventDeliveryState.ACKED
            and registration.backend_event_id is not None
        ):
            outcome = SystemTestOutcome(
                status=SystemTestStatus.PREVIOUSLY_ACKED,
                edge_event_id=registration.edge_event_id,
                correlation_id=registration.edge_event_id,
                backend_event_id=registration.backend_event_id,
            )
            self._log(outcome)
            return outcome
        return self._attempt(registration.edge_event_id)

    def _new_event(self, validation_run_id: UUID) -> StagedEvent:
        now = self._clock().astimezone(UTC)
        event_id = EdgeEventId(str(self._event_id_factory()))
        payload = _SystemTestPayload(
            edge_event_id=UUID(event_id),
            detected_at=now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            validation_run_id=validation_run_id,
        )
        return StagedEvent(
            edge_event_id=event_id,
            detected_at=payload.detected_at,
            payload_json=json.dumps(
                payload.model_dump(mode="json"),
                separators=(",", ":"),
                sort_keys=True,
            ),
            queued_at=now.timestamp(),
            operator_only=True,
        )

    def retry(self, edge_event_id: EdgeEventId) -> SystemTestOutcome:
        """Explicitly retry one pending operator-only row when its backoff is due."""
        return self._attempt(edge_event_id)

    def replay(self, edge_event_id: EdgeEventId) -> SystemTestOutcome:
        """Send one exact ACKed payload replay without resetting outbox state."""
        result = self._sender().replay_operator(edge_event_id)
        match result:
            case EventReceipt(event_id=backend_event_id):
                outcome = SystemTestOutcome(
                    status=SystemTestStatus.REPLAY_ACKED,
                    edge_event_id=edge_event_id,
                    correlation_id=edge_event_id,
                    backend_event_id=backend_event_id,
                )
            case DeliveryFailure(code=code):
                outcome = SystemTestOutcome(
                    status=SystemTestStatus.FAILED,
                    edge_event_id=edge_event_id,
                    correlation_id=edge_event_id,
                    error_code=code,
                )
        self._log(outcome)
        return outcome

    def classify_invalid_auth(self) -> SystemTestOutcome:
        """Classify ml-api's payload-free in-memory invalid credential probe."""
        result = self._transport.classify_invalid_auth()
        status = (
            SystemTestStatus.AUTH_CLASSIFIED
            if result.code in {"HTTP_401", "HTTP_403"}
            else SystemTestStatus.FAILED
        )
        outcome = SystemTestOutcome(status=status, error_code=result.code)
        self._log(outcome)
        return outcome

    def _attempt(self, edge_event_id: EdgeEventId) -> SystemTestOutcome:
        step = self._sender().run_operator_once(edge_event_id)
        with EvidenceOutbox.open(self._config.database_path) as outbox:
            backend_event_id = outbox.event_backend_event_id(edge_event_id)
            delivery_state = outbox.event_delivery_state(edge_event_id)
        status = (
            SystemTestStatus.ACKED
            if step is SenderStep.EVENT_ACKED
            else SystemTestStatus.PREVIOUSLY_ACKED
            if delivery_state == EventDeliveryState.ACKED.value
            and backend_event_id is not None
            else SystemTestStatus.RETRY_SCHEDULED
            if step is SenderStep.RETRY_SCHEDULED
            or (step is SenderStep.IDLE and delivery_state == EventDeliveryState.PENDING.value)
            else SystemTestStatus.FAILED
        )
        outcome = SystemTestOutcome(
            status=status,
            edge_event_id=edge_event_id,
            correlation_id=edge_event_id,
            backend_event_id=backend_event_id,
        )
        self._log(outcome)
        return outcome

    def _sender(self) -> EvidenceSender:
        return EvidenceSender(
            self._config.database_path,
            SenderConfig(
                relay_url=self._config.relay_url,
                relay_token=self._config.relay_token,
                probe_camera_id="",
                enabled=True,
            ),
            transport=self._transport,
            clock=lambda: self._clock().timestamp(),
        )

    @staticmethod
    def _log(outcome: SystemTestOutcome) -> None:
        LOGGER.info(
            "system_test status=%s edge_event_id=%s correlation_id=%s",
            outcome.status.value,
            outcome.edge_event_id,
            outcome.correlation_id,
            extra={
                "status": outcome.status.value,
                "edge_event_id": outcome.edge_event_id,
                "correlation_id": outcome.correlation_id,
            },
        )


def execute_system_test(
    command: SystemTestCommand,
    environment: Mapping[str, str],
    state_dir: Path,
) -> SystemTestOutcome:
    """Parse one CLI command and invoke no normal worker boot/inference seam."""
    require_system_test_gate(environment)
    if command.confirmation != "SYSTEM_TEST":
        raise SystemTestConfigurationError(
            "--confirm-system-test must equal SYSTEM_TEST"
        )
    relay_url = environment.get("RELAY_URL", "").strip()
    relay_token = environment.get("RELAY_TOKEN", "").strip()
    if not relay_url or not relay_token:
        raise SystemTestConfigurationError("RELAY_URL and RELAY_TOKEN must be set")
    emitter = SystemTestEmitter(
        SystemTestConfig(
            database_path=state_dir / WORKER_STATE_DB_FILENAME,
            relay_url=relay_url,
            relay_token=relay_token,
        )
    )
    match command.action:
        case SystemTestAction.EMIT:
            if command.validation_run_id is None or command.edge_event_id is not None:
                raise SystemTestConfigurationError(
                    "emit requires --system-test-validation-run-id only"
                )
            return emitter.emit(_parse_uuid(command.validation_run_id, "validation run"))
        case SystemTestAction.RETRY:
            return emitter.retry(_required_edge_event_id(command))
        case SystemTestAction.REPLAY:
            return emitter.replay(_required_edge_event_id(command))
        case SystemTestAction.AUTH_CHECK:
            if command.validation_run_id is not None or command.edge_event_id is not None:
                raise SystemTestConfigurationError("auth-check accepts no event identifiers")
            return emitter.classify_invalid_auth()
        case unreachable:
            assert_never(unreachable)


def _required_edge_event_id(command: SystemTestCommand) -> EdgeEventId:
    if command.edge_event_id is None or command.validation_run_id is not None:
        raise SystemTestConfigurationError(
            f"{command.action.value} requires --system-test-edge-event-id only"
        )
    parsed = _parse_uuid(command.edge_event_id, "edge event")
    if parsed.version != 4 or str(parsed) != command.edge_event_id:
        raise SystemTestConfigurationError("edge event ID must be lowercase UUIDv4")
    return EdgeEventId(str(parsed))


def _parse_uuid(raw: str, label: str) -> UUID:
    try:
        parsed = UUID(raw)
    except ValueError:
        raise SystemTestConfigurationError(f"{label} ID must be a UUID") from None
    if str(parsed) != raw:
        raise SystemTestConfigurationError(f"{label} ID must be canonical lowercase UUID")
    return parsed


def require_system_test_gate(environment: Mapping[str, str]) -> None:
    """Require the exact typed one-shot gate; similar truthy strings are invalid."""
    raw = environment.get(SYSTEM_TEST_GATE_ENV)
    if raw is None:
        raise SystemTestDisabledError("SYSTEM_TEST operator gate is disabled")
    if raw != SYSTEM_TEST_GATE_VALUE:
        raise SystemTestConfigurationError(
            f"{SYSTEM_TEST_GATE_ENV} must equal {SYSTEM_TEST_GATE_VALUE}"
        )


__all__ = [
    "SYSTEM_TEST_GATE_ENV",
    "SYSTEM_TEST_GATE_VALUE",
    "SystemTestAction",
    "SystemTestCommand",
    "SystemTestConfig",
    "SystemTestConfigurationError",
    "SystemTestDisabledError",
    "SystemTestEmitter",
    "SystemTestOutcome",
    "SystemTestStatus",
    "execute_system_test",
    "require_system_test_gate",
]
