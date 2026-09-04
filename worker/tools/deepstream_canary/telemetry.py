"""Transform recorded runtime samples into independently verifiable rung receipts."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from worker.tools.deepstream_canary.models import (
    ArtifactBindings,
    CameraSignals,
    CanaryMode,
    GpuSignals,
    LatencySignals,
    LiveProtectionSignals,
    NvdecSignals,
    RungReceipt,
    TimelineEntry,
    WorkloadFacts,
)
from worker.tools.deepstream_canary.report import write_once


class NativeWindowSample(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    camera_id: str
    window_started_ns: int
    window_ended_ns: int
    decision_count: int = Field(ge=0)
    latency_samples_ms: tuple[float, ...]
    metadata_published: int = Field(ge=0)
    metadata_overwritten: int = Field(ge=0)
    timestamp_discontinuities: int = Field(ge=0)


class RuntimeGpuSample(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    child_pid: int = Field(gt=0)
    child_memory_mib: float = Field(ge=0)
    global_used_mib: float = Field(ge=0)
    total_mib: float = Field(gt=0)
    utilization: float = Field(ge=0, le=100)


class RecordedCameraTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    camera_id: str
    decision_window_counts: tuple[int, ...] = Field(min_length=1)
    decision_window_seconds: float = Field(gt=0)
    latency_samples_ms: tuple[float, ...] = Field(min_length=1)
    au_gaps: int = Field(ge=0)
    config_discontinuities: int = Field(ge=0)
    timestamp_discontinuities: int = Field(ge=0)
    metadata_published: int = Field(gt=0)
    metadata_overwritten: int = Field(ge=0)
    event_evidence_parity: bool
    preview_ok: bool
    derivative_ok: bool


class RecordedGpuTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    child_pid: int = Field(gt=0)
    warmup_memory_mib: tuple[float, ...] = Field(min_length=1)
    steady_memory_mib: tuple[float, ...] = Field(min_length=1)
    recovery_memory_mib: tuple[float, ...] = Field(min_length=1)
    global_used_mib: float = Field(ge=0)
    total_mib: float = Field(gt=0)
    slack_samples_mib: tuple[float, ...] = Field(min_length=1)
    utilization_samples: tuple[float, ...] = Field(min_length=1)
    new_xids: tuple[int, ...]


class RecordedRungTelemetry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    rung: str
    mode: CanaryMode
    camera_count: int = Field(ge=0)
    clean_steady_seconds: int = Field(ge=0)
    cameras: tuple[RecordedCameraTelemetry, ...]
    gpu: RecordedGpuTelemetry
    nvdec: NvdecSignals
    timeline: tuple[TimelineEntry, ...]
    live_protection: LiveProtectionSignals
    fault_windows: tuple[str, ...]
    workload: WorkloadFacts


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_rung_receipt(
    recorded: RecordedRungTelemetry,
    artifacts: ArtifactBindings,
) -> RungReceipt:
    """Compute all percentiles from raw samples rather than reported summaries."""
    cameras = tuple(
        CameraSignals(
            camera_id=camera.camera_id,
            fps_windows=tuple(
                count / camera.decision_window_seconds for count in camera.decision_window_counts
            ),
            latency_ms=LatencySignals(
                p50=_percentile(camera.latency_samples_ms, 0.50),
                p95=_percentile(camera.latency_samples_ms, 0.95),
                p99=_percentile(camera.latency_samples_ms, 0.99),
                max=max(camera.latency_samples_ms),
            ),
            au_gaps=camera.au_gaps,
            config_discontinuities=camera.config_discontinuities,
            timestamp_discontinuities=camera.timestamp_discontinuities,
            metadata_published=camera.metadata_published,
            metadata_overwritten=camera.metadata_overwritten,
            event_evidence_parity=camera.event_evidence_parity,
            preview_ok=camera.preview_ok,
            derivative_ok=camera.derivative_ok,
        )
        for camera in recorded.cameras
    )
    gpu = recorded.gpu
    return RungReceipt(
        schema_version=1,
        rung=recorded.rung,
        mode=recorded.mode,
        camera_count=recorded.camera_count,
        clean_steady_seconds=recorded.clean_steady_seconds,
        cameras=cameras,
        gpu=GpuSignals(
            child_pid=gpu.child_pid,
            warmup_peak_mib=max(gpu.warmup_memory_mib),
            steady_p50_mib=_percentile(gpu.steady_memory_mib, 0.50),
            steady_p95_mib=_percentile(gpu.steady_memory_mib, 0.95),
            recovery_mib=gpu.recovery_memory_mib[-1],
            global_used_mib=gpu.global_used_mib,
            total_mib=gpu.total_mib,
            minimum_slack_mib=min(gpu.slack_samples_mib),
            utilization_p50=_percentile(gpu.utilization_samples, 0.50),
            utilization_p95=_percentile(gpu.utilization_samples, 0.95),
            utilization_max=max(gpu.utilization_samples),
            new_xids=gpu.new_xids,
        ),
        nvdec=recorded.nvdec,
        timeline=recorded.timeline,
        live_protection=recorded.live_protection,
        fault_windows=recorded.fault_windows,
        workload=recorded.workload,
        artifacts=artifacts,
    )


def emit_rung_receipt(
    telemetry_path: Path,
    evidence_dir: Path,
    artifacts: ArtifactBindings,
) -> Path:
    recorded = RecordedRungTelemetry.model_validate_json(telemetry_path.read_bytes())
    receipt = build_rung_receipt(recorded, artifacts)
    destination = evidence_dir / "raw" / f"rung-{recorded.rung}.json"
    write_once(destination, (receipt.model_dump_json() + "\n").encode())
    return destination


__all__ = [
    "NativeWindowSample",
    "RecordedCameraTelemetry",
    "RecordedGpuTelemetry",
    "RecordedRungTelemetry",
    "RuntimeGpuSample",
    "build_rung_receipt",
    "emit_rung_receipt",
]
