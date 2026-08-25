"""Fail-closed facility-rung authorization independent of Docker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from worker.tools.deepstream_canary.models import AuthorizationArtifact, CanaryMode


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    rungs: tuple[int, ...]
    mode: CanaryMode
    appliance_id: str
    worker_image_digest: str
    camera_ids: tuple[str, ...]
    now: datetime
    minimum_projected_slack_mib: float


@dataclass(frozen=True, slots=True)
class CanaryAuthorizationError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


def authorize_rungs(
    request: AuthorizationRequest,
    artifact: AuthorizationArtifact | None,
) -> None:
    """Require an explicit owner artifact bound to every facility input."""
    if not request.rungs:
        return
    if artifact is None:
        raise CanaryAuthorizationError("authorization_required")
    if artifact.expires_at <= request.now:
        raise CanaryAuthorizationError("authorization_expired")
    if artifact.appliance_id != request.appliance_id:
        raise CanaryAuthorizationError("appliance_identity_mismatch")
    if artifact.worker_image != request.worker_image_digest:
        raise CanaryAuthorizationError("worker_image_mismatch")
    if artifact.camera_ids != request.camera_ids:
        raise CanaryAuthorizationError("camera_ids_mismatch")
    if max(request.rungs) != len(request.camera_ids):
        raise CanaryAuthorizationError("camera_count_mismatch")
    if any(rung not in artifact.authorized_rungs for rung in request.rungs):
        raise CanaryAuthorizationError("rung_not_authorized")
    if 13 in request.rungs:
        if artifact.eight_pass_report_sha256 is None:
            raise CanaryAuthorizationError("eight_pass_report_required")
        if artifact.projected_slack_mib < request.minimum_projected_slack_mib:
            raise CanaryAuthorizationError("projected_gpu_slack_below_3_gib")
