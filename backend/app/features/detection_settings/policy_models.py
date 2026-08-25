"""Detection policy store values and typed outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shared.detection_policies import EffectivePolicy

ActivationStatus = Literal["pending", "applied", "failed"]


class PolicyActivationRefused(RuntimeError):
    def __init__(self, activation_id: int, reason: str) -> None:
        self.activation_id = activation_id
        self.reason = reason
        super().__init__(f"detection policy activation {activation_id} refused: {reason}")


class PolicyRevisionConflict(RuntimeError):
    pass


class PolicyRollbackUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PolicyCameraIdentity:
    camera_id: str


@dataclass(frozen=True, slots=True)
class PolicyActivation:
    activation_id: int
    facility_id: str
    camera_id: str | None
    module_id: str
    module_version: int
    active_revision_id: int | None
    previous_revision_id: int | None
    activation_generation: int
    status: ActivationStatus
    refusal_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "activation_id": self.activation_id,
            "facility_id": self.facility_id,
            "camera_id": self.camera_id,
            "module_id": self.module_id,
            "module_version": self.module_version,
            "active_revision_id": self.active_revision_id,
            "previous_revision_id": self.previous_revision_id,
            "activation_generation": self.activation_generation,
            "status": self.status,
            "refusal_reason": self.refusal_reason,
        }


@dataclass(frozen=True, slots=True)
class PolicyDiff:
    changed: bool
    current: EffectivePolicy
    proposed: EffectivePolicy
    compared_payload: dict[str, object]
    concurrency_token: int

    def as_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "current": self.current.as_dict(),
            "proposed": self.proposed.as_dict(),
            "compared_payload": dict(self.compared_payload),
            "concurrency_token": self.concurrency_token,
        }


__all__ = [
    "ActivationStatus",
    "PolicyActivation",
    "PolicyActivationRefused",
    "PolicyCameraIdentity",
    "PolicyDiff",
    "PolicyRevisionConflict",
    "PolicyRollbackUnavailable",
]
