"""Metric sampling and JSON assembly for the recorded-stream fan-out benchmark.

Everything here reads *observable worker state* -- ``WorkerDiagnostics.snapshot()``
(per-camera bus counters and stage-timing accumulators), ``InferenceWatchdog.in_flight()``
(deadline margin), and ``nvidia-smi`` (GPU/NVDEC utilization). Nothing is
synthesized: a field that has no real observation is emitted as ``null`` and the
run records why, so a bench JSON can never look healthy on defaults.

Pose forward latency percentiles are collected separately, by the harness's
timing proxy around the serving client's pose runner (``fanout_benchmark_harness``),
because ``StageTimingAccumulator`` keeps only samples/total/last/max -- no
distribution -- and this benchmark's #312 gate is p95, not the mean.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import monotonic
from typing import Any, final

_NVIDIA_SMI_QUERY = "utilization.gpu,utilization.decoder,memory.used"
_NVIDIA_SMI_TIMEOUT_SEC = 5.0
_LANES = ("inference", "live", "evidence")


@dataclass(frozen=True, slots=True)
class BusLaneCounters:
    published: int
    taken: int
    dropped: int

    def as_dict(self) -> dict[str, int]:
        return {"published": self.published, "taken": self.taken, "dropped": self.dropped}


@dataclass(slots=True)
class CameraSample:
    """One camera's counters at one sampling instant."""

    camera_id: str
    at_sec: float
    lanes: dict[str, BusLaneCounters]
    inference_queue_age_sec: float
    pose_samples: int
    failure_category: str | None
    pump_failures: int = 0
    pump_processed: int = 0


@dataclass(slots=True)
class RunSample:
    at_sec: float
    cameras: tuple[CameraSample, ...]
    watchdog_margin_sec: float | None
    gpu: dict[str, float] | None


def sample_gpu() -> dict[str, float] | None:
    """One ``nvidia-smi`` reading, or ``None`` when the tool is unavailable."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed local binary, no shell
            [binary, f"--query-gpu={_NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    first = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3:
        return None
    try:
        gpu_util, decoder_util, memory_used = (float(part) for part in parts)
    except ValueError:
        return None
    return {
        "gpu_utilization_pct": gpu_util,
        "nvdec_utilization_pct": decoder_util,
        "memory_used_mib": memory_used,
    }


def watchdog_margin_sec(watchdog: Any) -> float | None:
    """Smallest remaining budget across in-flight forwards; ``None`` when idle.

    ``None`` is a truthful "no forward was in flight at this instant", not a
    placeholder -- the run-level margin below reduces over every sample that
    did observe one.
    """
    if watchdog is None:
        return None
    in_flight = watchdog.in_flight()
    if not in_flight:
        return None
    now = monotonic()
    return min(entry.deadline_at - now for entry in in_flight)


def take_sample(diagnostics: Any, watchdog: Any, pumps: Any = ()) -> RunSample:
    """One instant's reading of every observable counter.

    ``pumps`` are the runtime's per-camera ``CameraPipelinePump`` objects; their
    ``failure_count`` is sampled so a pipeline that raises on every frame cannot
    publish a JSON that only shows healthy bus traffic.
    """
    snapshot = diagnostics.snapshot()
    by_camera = {getattr(pump, "camera_id", ""): pump for pump in pumps}
    cameras = tuple(
        CameraSample(
            camera_id=camera.camera_id,
            at_sec=monotonic(),
            lanes={
                lane.name: BusLaneCounters(lane.published, lane.taken, lane.dropped)
                for lane in camera.bus
                if lane.name in _LANES
            },
            inference_queue_age_sec=next(
                (lane.queue_age_sec for lane in camera.bus if lane.name == "inference"), 0.0
            ),
            pose_samples=next(
                (timing.samples for timing in camera.stage_timings if timing.stage == "pose"), 0
            ),
            failure_category=camera.failure_category,
            pump_failures=getattr(by_camera.get(camera.camera_id), "failure_count", 0),
            pump_processed=getattr(by_camera.get(camera.camera_id), "processed_count", 0),
        )
        for camera in snapshot.cameras
    )
    return RunSample(
        at_sec=monotonic(),
        cameras=cameras,
        watchdog_margin_sec=watchdog_margin_sec(watchdog),
        gpu=sample_gpu(),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _rate(first: int, last: int, elapsed_sec: float) -> float | None:
    if elapsed_sec <= 0:
        return None
    return (last - first) / elapsed_sec


@final
class RunMetrics:
    """Reduce a series of samples into the bench JSON document."""

    def __init__(self) -> None:
        self._samples: list[RunSample] = []
        self.pose_latencies_ms: list[float] = []
        self.errors: list[str] = []
        self.notes: dict[str, Any] = {}

    def add(self, sample: RunSample) -> None:
        self._samples.append(sample)

    @property
    def samples(self) -> tuple[RunSample, ...]:
        return tuple(self._samples)

    def counter_advanced(self) -> bool:
        """True when any lane counter moved between the first and last sample.

        The misleading-success guard: a bench JSON whose numbers came from
        dataclass defaults rather than a live worker cannot satisfy this.
        """
        if len(self._samples) < 2:
            return False
        first, last = self._samples[0], self._samples[-1]
        firsts = {camera.camera_id: camera for camera in first.cameras}
        for camera in last.cameras:
            earlier = firsts.get(camera.camera_id)
            if earlier is None:
                continue
            for lane, counters in camera.lanes.items():
                before = earlier.lanes.get(lane)
                if before is not None and counters.published > before.published:
                    return True
        return False

    def _per_camera(self) -> dict[str, dict[str, Any]]:
        if len(self._samples) < 2:
            return {}
        first, last = self._samples[0], self._samples[-1]
        elapsed = last.at_sec - first.at_sec
        firsts = {camera.camera_id: camera for camera in first.cameras}
        result: dict[str, dict[str, Any]] = {}
        for camera in last.cameras:
            earlier = firsts.get(camera.camera_id)
            if earlier is None:
                continue
            inference_before = earlier.lanes.get("inference")
            inference_after = camera.lanes.get("inference")
            result[camera.camera_id] = {
                "source_fps": (
                    None
                    if inference_before is None or inference_after is None
                    else _rate(inference_before.published, inference_after.published, elapsed)
                ),
                "inference_admitted_fps": (
                    None
                    if inference_before is None or inference_after is None
                    else _rate(inference_before.taken, inference_after.taken, elapsed)
                ),
                "pose_inferences": camera.pose_samples - earlier.pose_samples,
                "bus": {
                    lane: {
                        key: value - getattr(earlier.lanes[lane], key)
                        for key, value in counters.as_dict().items()
                    }
                    for lane, counters in camera.lanes.items()
                    if lane in earlier.lanes
                },
                "bus_cumulative": {
                    lane: counters.as_dict() for lane, counters in camera.lanes.items()
                },
                "max_inference_frame_age_sec": max(
                    (
                        sample_camera.inference_queue_age_sec
                        for sample in self._samples
                        for sample_camera in sample.cameras
                        if sample_camera.camera_id == camera.camera_id
                    ),
                    default=0.0,
                ),
                "failure_category": camera.failure_category,
                "pump_failures": camera.pump_failures,
                "pump_processed": camera.pump_processed,
            }
        return result

    def document(self, *, header: dict[str, Any]) -> dict[str, Any]:
        cameras = self._per_camera()
        gpu_samples = [sample.gpu for sample in self._samples if sample.gpu is not None]
        margins = [
            sample.watchdog_margin_sec
            for sample in self._samples
            if sample.watchdog_margin_sec is not None
        ]
        admitted = [
            value
            for camera in cameras.values()
            if (value := camera["inference_admitted_fps"]) is not None
        ]
        latencies = list(self.pose_latencies_ms)
        return {
            **header,
            "samples": len(self._samples),
            "cameras": cameras,
            "aggregate_inference_fps": sum(admitted) if admitted else None,
            "pose_stage_latency_ms": (
                None
                if not latencies
                else {
                    "count": len(latencies),
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                    "max": max(latencies),
                }
            ),
            "max_inference_frame_age_sec": max(
                (camera["max_inference_frame_age_sec"] for camera in cameras.values()),
                default=None,
            ),
            "watchdog_margin_sec": (min(margins) if margins else None),
            "watchdog_margin_observations": len(margins),
            "gpu": (
                None
                if not gpu_samples
                else {
                    key: {
                        "min": min(sample[key] for sample in gpu_samples),
                        "median": median([sample[key] for sample in gpu_samples]),
                        "max": max(sample[key] for sample in gpu_samples),
                    }
                    for key in sorted(gpu_samples[0])
                }
            ),
            "counters_advanced": self.counter_advanced(),
            "pump_failures": sum(camera["pump_failures"] for camera in cameras.values()),
            "errors": list(self.errors),
            "notes": dict(self.notes),
        }


def write_document(path: Path, document: dict[str, Any]) -> Path:
    """Write one bench JSON, replacing any previous content at ``path``.

    Deliberately a whole-file write (never an append): rerunning the same N
    must publish the new run's numbers, not accumulate stale ones.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "BusLaneCounters",
    "CameraSample",
    "RunMetrics",
    "RunSample",
    "sample_gpu",
    "take_sample",
    "watchdog_margin_sec",
    "write_document",
]
