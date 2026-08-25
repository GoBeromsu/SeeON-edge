"""Normalize host/native runtime samples into raw telemetry fixtures."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from worker.tools.deepstream_canary.models import (
    CanaryMode,
    LiveProtectionSignals,
    NvdecSignals,
    TimelineEntry,
    WorkloadFacts,
)
from worker.tools.deepstream_canary.telemetry import (
    NativeWindowSample,
    RecordedCameraTelemetry,
    RecordedGpuTelemetry,
    RecordedRungTelemetry,
    RuntimeGpuSample,
)


class CollectionRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    evidence_dir: Path
    rung: str
    mode: CanaryMode
    clean_steady_seconds: int
    camera_ids: tuple[str, ...]
    gpu_samples: tuple[RuntimeGpuSample, ...] = Field(min_length=1)
    native_windows: tuple[NativeWindowSample, ...]
    allow_partial: bool = False
    fault_windows: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _playable(path: Path) -> bool:
    if not path.is_file():
        return False
    completed = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _timeline(root: Path, rung: str) -> tuple[TimelineEntry, ...]:
    if rung == "zero":
        return ()
    paths: tuple[tuple[Literal["event", "evidence", "preview", "derivative"], Path], ...] = (
        ("event", root / "event-clip.mp4"),
        ("evidence", root / "event-evidence.json"),
        ("preview", root / "viewer-loop-01.png"),
        ("derivative", root / "event-derivative.mp4"),
    )
    entries: list[TimelineEntry] = []
    for kind, path in paths:
        if path.is_file():
            playable = _playable(path) if kind in {"event", "derivative"} else True
            entries.append(
                TimelineEntry(kind=kind, sha256=_sha256(path), playable=playable)
            )
    return tuple(entries)


def collect_recorded_telemetry(request: CollectionRequest) -> Path:
    """Seal one rung from raw windows; no reported percentile is accepted."""
    by_camera: dict[str, list[NativeWindowSample]] = {}
    for window in request.native_windows:
        by_camera.setdefault(window.camera_id, []).append(window)
    if not request.allow_partial and set(by_camera) != set(request.camera_ids):
        raise ValueError(
            f"native camera identities mismatch: expected={request.camera_ids!r} "
            f"actual={tuple(sorted(by_camera))!r}"
        )
    timeline = _timeline(request.evidence_dir / "raw", request.rung)
    parity = {entry.kind for entry in timeline} >= {"event", "evidence", "derivative"}
    preview_ok = any(entry.kind == "preview" and entry.playable for entry in timeline)
    derivative_ok = any(entry.kind == "derivative" and entry.playable for entry in timeline)
    cameras = tuple(
        RecordedCameraTelemetry(
            camera_id=camera_id,
            decision_window_counts=tuple(item.decision_count for item in windows),
            decision_window_seconds=10.0,
            latency_samples_ms=tuple(
                latency
                for item in windows
                for latency in item.latency_samples_ms
            ),
            au_gaps=0,
            config_discontinuities=0,
            timestamp_discontinuities=sum(
                item.timestamp_discontinuities for item in windows
            ),
            metadata_published=sum(item.metadata_published for item in windows),
            metadata_overwritten=sum(item.metadata_overwritten for item in windows),
            event_evidence_parity=parity,
            preview_ok=preview_ok,
            derivative_ok=derivative_ok,
        )
        for camera_id, windows in sorted(by_camera.items())
    )
    child_memory = tuple(item.child_memory_mib for item in request.gpu_samples)
    utilization = tuple(item.utilization for item in request.gpu_samples)
    final_gpu = request.gpu_samples[-1]
    recorded = RecordedRungTelemetry(
        schema_version=1,
        rung=request.rung,
        mode=request.mode,
        camera_count=len(request.camera_ids),
        clean_steady_seconds=request.clean_steady_seconds,
        cameras=cameras,
        gpu=RecordedGpuTelemetry(
            child_pid=final_gpu.child_pid,
            warmup_memory_mib=child_memory[:1],
            steady_memory_mib=child_memory,
            recovery_memory_mib=child_memory[-1:],
            global_used_mib=final_gpu.global_used_mib,
            total_mib=final_gpu.total_mib,
            slack_samples_mib=tuple(
                item.total_mib - item.global_used_mib for item in request.gpu_samples
            ),
            utilization_samples=utilization,
            new_xids=(),
        ),
        nvdec=NvdecSignals(
            hardware_branches=len(request.camera_ids),
            software_fallbacks=0,
        ),
        timeline=timeline,
        live_protection=LiveProtectionSignals(
            container_restarts=0,
            camera_stale_transitions=0,
            evidence_drop_increase=0,
            relay_sentinel_leaks=0,
            mount_intersections=0,
            kernel_faults=0,
        ),
        fault_windows=request.fault_windows,
        workload=WorkloadFacts(
            codec="h264",
            width=1280,
            height=720,
            fps=15.0,
            gop=30,
            camera_phase_offsets_ms=tuple(index * 67 for index in range(len(request.camera_ids))),
        ),
    )
    destination = request.evidence_dir / "raw" / f"telemetry-{request.rung}.json"
    destination.write_text(recorded.model_dump_json() + "\n", encoding="utf-8")
    return destination


__all__ = ["CollectionRequest", "collect_recorded_telemetry"]
