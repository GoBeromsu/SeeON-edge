"""Strict canary policy, authorization, raw-receipt, and report boundaries."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CanaryMode(StrEnum):
    COMMISSIONING = "commissioning"
    SHARED_HOST_SMOKE = "shared-host-smoke"


class EnginePreparationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    utilization_gate_active: bool
    required_live_signals: tuple[str, ...]


class PolicyVersion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str
    reason: str


class GatePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    policy_id: str
    provenance: str
    version_history: tuple[PolicyVersion, ...]
    engine_preparation: EnginePreparationPolicy
    fps_p05_min: float = Field(gt=0)
    fps_p50_min: float = Field(gt=0)
    fps_p95_min: float = Field(gt=0)
    latency_p50_max_ms: float = Field(gt=0)
    latency_p95_max_ms: float = Field(gt=0)
    latency_p99_max_ms: float = Field(gt=0)
    latency_absolute_max_ms: float = Field(gt=0)
    metadata_overwrite_fraction_max: float = Field(ge=0, le=1)
    gpu_utilization_p95_max: float = Field(gt=0, le=100)
    gpu_utilization_absolute_max: float = Field(gt=0, le=100)
    minimum_gpu_slack_mib: float = Field(gt=0)
    protected_gpu_process_loss_grace_seconds: float = Field(gt=0)
    gpu_warmup_peak_max_mib: float = Field(gt=0)
    gpu_steady_p95_max_mib: float = Field(gt=0)
    gpu_recovery_max_mib: float = Field(gt=0)
    zero_clean_seconds: int = Field(gt=0)
    loopback_clean_seconds: int = Field(gt=0)
    warmup_seconds: int = Field(gt=0)
    standard_rung_clean_seconds: int = Field(gt=0)
    candidate_rung_clean_seconds: int = Field(gt=0)
    authorization_max_age_seconds: int = Field(gt=0)


class LatencySignals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    p50: float = Field(ge=0)
    p95: float = Field(ge=0)
    p99: float = Field(ge=0)
    max: float = Field(ge=0)


class CameraSignals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    camera_id: str
    fps_windows: tuple[float, ...] = Field(min_length=1)
    latency_ms: LatencySignals
    au_gaps: int = Field(ge=0)
    config_discontinuities: int = Field(ge=0)
    timestamp_discontinuities: int = Field(ge=0)
    metadata_published: int = Field(gt=0)
    metadata_overwritten: int = Field(ge=0)
    event_evidence_parity: bool
    preview_ok: bool
    derivative_ok: bool


class GpuSignals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    child_pid: int = Field(gt=0)
    warmup_peak_mib: float = Field(ge=0)
    steady_p50_mib: float = Field(ge=0)
    steady_p95_mib: float = Field(ge=0)
    recovery_mib: float = Field(ge=0)
    global_used_mib: float = Field(ge=0)
    total_mib: float = Field(gt=0)
    minimum_slack_mib: float = Field(ge=0)
    utilization_p50: float = Field(ge=0, le=100)
    utilization_p95: float = Field(ge=0, le=100)
    utilization_max: float = Field(ge=0, le=100)
    new_xids: tuple[int, ...]


class NvdecSignals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hardware_branches: int = Field(ge=0)
    software_fallbacks: int = Field(ge=0)


class TimelineEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["event", "evidence", "preview", "derivative"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    playable: bool


class LiveProtectionSignals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    container_restarts: int = Field(ge=0)
    camera_stale_transitions: int = Field(ge=0)
    evidence_drop_increase: int = Field(ge=0)
    relay_sentinel_leaks: int = Field(ge=0)
    mount_intersections: int = Field(ge=0)
    kernel_faults: int = Field(ge=0)


class WorkloadFacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    codec: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    gop: int = Field(gt=0)
    camera_phase_offsets_ms: tuple[int, ...]


class ArtifactBindings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_image: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    support_images: tuple[str, ...]
    models_manifest: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine_manifest: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_policy: str = Field(pattern=r"^[0-9a-f]{64}$")
    compose: str = Field(pattern=r"^[0-9a-f]{64}$")


class RungReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    rung: str
    mode: CanaryMode
    camera_count: int = Field(ge=0)
    clean_steady_seconds: int = Field(ge=0)
    cameras: tuple[CameraSignals, ...]
    gpu: GpuSignals
    nvdec: NvdecSignals
    timeline: tuple[TimelineEntry, ...]
    live_protection: LiveProtectionSignals
    fault_windows: tuple[str, ...]
    workload: WorkloadFacts
    artifacts: ArtifactBindings


class GateCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    passed: bool
    actual: str
    required: str


class GateReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    verdict: Literal["PASS", "FAIL"]
    rung: str
    claim_eligible: bool
    gate_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks: tuple[GateCheck, ...]


class AuthorizationArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    appliance_id: str = Field(min_length=1)
    worker_image: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    camera_ids: tuple[str, ...] = Field(min_length=1)
    owner: str = Field(min_length=1)
    issue: str = Field(pattern=r"^https://github\.com/SeniorAILab/SeeON-edge/issues/[0-9]+$")
    expires_at: datetime
    authorized_rungs: tuple[Literal[1, 4, 8, 13], ...]
    eight_pass_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    projected_slack_mib: float = Field(ge=0)
