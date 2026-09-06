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
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from time import monotonic, sleep
from typing import Any, Final, final

_NVIDIA_SMI_QUERY = "utilization.gpu,utilization.decoder,memory.used"
_NVIDIA_SMI_TIMEOUT_SEC = 5.0
_LANES = ("inference", "live", "evidence")
# A silent latest-only drop that stays under this fraction of admitted frames
# is treated as jitter, not saturation. Anything above it cannot be ACHIEVABLE.
_ACHIEVABLE_OVERWRITE_FRACTION: Final = 0.01
_ACHIEVABLE_FPS_FRACTION: Final = 0.90
_MARGINAL_FPS_FRACTION: Final = 0.80


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
    decode_backend: str | None = None
    # Coordinator latest-only overwrite count for this camera. Distinct from
    # the bus ``dropped`` counter: the coordinator reads the same field but
    # this is the value the capacity verdict must cite (todo 13).
    overwritten: int = 0


@dataclass(slots=True)
class RunSample:
    at_sec: float
    cameras: tuple[CameraSample, ...]
    watchdog_margin_sec: float | None
    gpu: dict[str, float] | None
    # Cross-camera coordinator telemetry (worker/pipeline/inference_coordinator.py):
    # cumulative batch-size histogram plus the coordinator's own forward
    # percentiles, both read straight off ``diagnostics.snapshot()``. Empty when
    # no coordinator is registered (i.e. the pre-Wave-3 serialized topology).
    batch_sizes: dict[int, int] = field(default_factory=dict)
    coordinator_forward_p50_sec: float = 0.0
    coordinator_forward_p95_sec: float = 0.0
    loadavg: tuple[float, float, float] | None = None


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


@final
class StallWatcher:
    """Sub-second watcher for gaps in cross-camera inference progress.

    The document's 2s sampling cadence cannot resolve the plan's "zero stalls
    > 2s" gate -- two adjacent samples straddling a 3s freeze look identical to
    two adjacent samples with steady progress. This polls the aggregate
    ``inference.taken`` counter on its own thread at ``interval_sec`` and keeps
    the longest wall-clock gap between two observed advances. It reads the same
    live bus counters as the sampler; nothing here is synthesized.
    """

    def __init__(
        self,
        total_taken: Callable[[], int],
        *,
        interval_sec: float = 0.1,
        stall_threshold_sec: float = 2.0,
    ) -> None:
        self._total_taken = total_taken
        self._interval_sec = interval_sec
        self._stall_threshold_sec = stall_threshold_sec
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._max_gap_sec = 0.0
        self._stalls: list[dict[str, float]] = []
        self._observations = 0
        self._thread = threading.Thread(
            target=self._run, name="fanout-bench-stall-watcher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)

    def _run(self) -> None:
        last_value = self._total_taken()
        last_advance_at = monotonic()
        while not self._stop.is_set():
            sleep(self._interval_sec)
            now = monotonic()
            value = self._total_taken()
            with self._lock:
                self._observations += 1
            if value <= last_value:
                continue
            gap = now - last_advance_at
            with self._lock:
                self._max_gap_sec = max(self._max_gap_sec, gap)
                if gap > self._stall_threshold_sec:
                    self._stalls.append({"at_sec": now, "gap_sec": gap})
            last_value, last_advance_at = value, now
        # The tail: a freeze that never resolved before shutdown is still a stall.
        trailing = monotonic() - last_advance_at
        with self._lock:
            self._max_gap_sec = max(self._max_gap_sec, trailing)
            if trailing > self._stall_threshold_sec:
                self._stalls.append({"at_sec": monotonic(), "gap_sec": trailing})

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "poll_interval_sec": self._interval_sec,
                "threshold_sec": self._stall_threshold_sec,
                "polls": self._observations,
                "max_progress_gap_sec": self._max_gap_sec,
                "stall_count": len(self._stalls),
                "stalls": list(self._stalls),
            }


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
            decode_backend=_decode_backend_name(camera),
            overwritten=_coordinator_overwritten(camera),
        )
        for camera in snapshot.cameras
    )
    return RunSample(
        at_sec=monotonic(),
        cameras=cameras,
        watchdog_margin_sec=watchdog_margin_sec(watchdog),
        gpu=sample_gpu(),
        loadavg=_sample_loadavg(),
        **_coordinator_fields(snapshot),
    )


def _decode_backend_name(camera: Any) -> str | None:
    """``requested -> resolved (adapter class)`` for one camera, or ``None``.

    Requested and resolved are both kept: the plan's ADR-0002 guardrail is a
    silent downgrade, which only shows up as a mismatch between the two.
    """
    backend = getattr(camera, "decode_backend", None)
    if backend is None:
        return None
    requested = getattr(backend, "requested_profile_decode", None)
    resolved = getattr(backend, "resolved_backend", None)
    adapter = getattr(backend, "actual_adapter_class", None)
    if requested is None and resolved is None:
        return str(backend)
    return f"{requested} -> {resolved} ({adapter})"


def _coordinator_overwritten(camera: Any) -> int:
    """Read ``CameraInferenceTelemetry.overwritten`` off one diagnostics camera.

    The coordinator publishes this as the latest-only drop count. A missing
    ``inference`` object (pre-coordinator topology) is reported as 0, never as
    a fabricated healthy-looking None that a verdict could misread.
    """
    inference = getattr(camera, "inference", None)
    if inference is None:
        return 0
    return int(getattr(inference, "overwritten", 0))


def _sample_loadavg() -> tuple[float, float, float] | None:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return None
    return (float(one), float(five), float(fifteen))


def _coordinator_fields(snapshot: Any) -> dict[str, Any]:
    """Read the coordinator's cumulative batch histogram off any camera view.

    ``RuntimeDiagnostics.snapshot()`` copies the single coordinator's telemetry
    onto every camera entry, so the first camera carrying a non-empty histogram
    is the coordinator's own state, not a per-camera value.
    """
    for camera in getattr(snapshot, "cameras", ()):
        sizes = getattr(camera, "batch_sizes", ())
        if sizes:
            return {
                "batch_sizes": {int(size): int(count) for size, count in sizes},
                "coordinator_forward_p50_sec": float(getattr(camera, "forward_p50_sec", 0.0)),
                "coordinator_forward_p95_sec": float(getattr(camera, "forward_p95_sec", 0.0)),
            }
    return {}


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
        self.stalls: dict[str, Any] = {}

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
                "decode_backend": camera.decode_backend,
                "overwritten": camera.overwritten - earlier.overwritten,
                "overwritten_cumulative": camera.overwritten,
            }
        return result

    def _batch_size_histogram(self) -> dict[str, int]:
        """Batch sizes issued inside the measurement window (last minus first).

        The coordinator's counter is cumulative from process start, so warmup
        forwards (batch size 1, before every camera is publishing) would
        otherwise be charged to the steady-state window.
        """
        if len(self._samples) < 2:
            return {}
        first, last = self._samples[0].batch_sizes, self._samples[-1].batch_sizes
        delta = {
            size: count - first.get(size, 0)
            for size, count in last.items()
            if count - first.get(size, 0) > 0
        }
        return {str(size): delta[size] for size in sorted(delta)}

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
        histogram = self._batch_size_histogram()
        forwards = sum(histogram.values())
        frames_in_batches = sum(int(size) * count for size, count in histogram.items())
        last_sample = self._samples[-1] if self._samples else None
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
            "batch": {
                "histogram": histogram,
                "forwards": forwards,
                "frames": frames_in_batches,
                "mean_batch_size": (None if forwards == 0 else frames_in_batches / forwards),
                "max_batch_size": (None if not histogram else max(int(size) for size in histogram)),
                "coordinator_forward_p50_ms": (
                    None
                    if last_sample is None
                    else last_sample.coordinator_forward_p50_sec * 1000.0
                ),
                "coordinator_forward_p95_ms": (
                    None
                    if last_sample is None
                    else last_sample.coordinator_forward_p95_sec * 1000.0
                ),
            },
            "live_lane": {
                "published": sum(
                    camera["bus"].get("live", {}).get("published", 0) for camera in cameras.values()
                ),
                "taken": sum(
                    camera["bus"].get("live", {}).get("taken", 0) for camera in cameras.values()
                ),
                "dropped": sum(
                    camera["bus"].get("live", {}).get("dropped", 0) for camera in cameras.values()
                ),
            },
            "evidence_lane": {
                "published": sum(
                    camera["bus"].get("evidence", {}).get("published", 0)
                    for camera in cameras.values()
                ),
                "taken": sum(
                    camera["bus"].get("evidence", {}).get("taken", 0) for camera in cameras.values()
                ),
                "dropped": sum(
                    camera["bus"].get("evidence", {}).get("dropped", 0)
                    for camera in cameras.values()
                ),
            },
            "inference_dropped": sum(
                camera["bus"].get("inference", {}).get("dropped", 0) for camera in cameras.values()
            ),
            "overwritten": sum(int(camera.get("overwritten", 0)) for camera in cameras.values()),
            "overwritten_by_camera": {
                camera_id: int(camera.get("overwritten", 0))
                for camera_id, camera in cameras.items()
            },
            "loadavg": (
                None
                if last_sample is None or last_sample.loadavg is None
                else {
                    "1m": last_sample.loadavg[0],
                    "5m": last_sample.loadavg[1],
                    "15m": last_sample.loadavg[2],
                }
            ),
            "capacity": capacity_verdict(
                cameras,
                target_fps=_offered_fps(header),
            ),
            "stalls": dict(self.stalls) if self.stalls else None,
            "counters_advanced": self.counter_advanced(),
            "pump_failures": sum(camera["pump_failures"] for camera in cameras.values()),
            "errors": list(self.errors),
            "notes": dict(self.notes),
        }


def _offered_fps(header: dict[str, Any]) -> float | None:
    """Prefer the operator-declared offered rate over the recorded-clip fps."""
    for key in ("offered_fps", "camera_fps"):
        value = header.get(key)
        if value is not None:
            return float(value)
    return None


def capacity_verdict(
    cameras: dict[str, dict[str, Any]],
    *,
    target_fps: float | None,
    overwrite_fraction_limit: float = _ACHIEVABLE_OVERWRITE_FRACTION,
    achievable_fps_fraction: float = _ACHIEVABLE_FPS_FRACTION,
    marginal_fps_fraction: float = _MARGINAL_FPS_FRACTION,
) -> dict[str, Any]:
    """Classify a measured fan-out as ACHIEVABLE, MARGINAL, or NOT.

    The gate that prevents a silent latest-only drop from looking healthy is
    ``overwritten``: a camera can keep publishing at the offered rate while
    the coordinator overwrites unread frames. Bus ``dropped`` alone is not
    enough -- that is why this function reads the coordinator field.
    """
    if not cameras or target_fps is None or float(target_fps) <= 0:
        return {
            "verdict": "NOT",
            "reason": "no cameras or no target fps to judge",
            "min_admitted_fps": None,
            "max_admitted_fps": None,
            "overwritten_total": 0,
            "overwrite_fraction": None,
        }
    admitted = [
        float(value)
        for camera in cameras.values()
        if (value := camera.get("inference_admitted_fps")) is not None
    ]
    overwritten_total = sum(int(camera.get("overwritten", 0)) for camera in cameras.values())
    published_total = sum(
        int(camera.get("bus", {}).get("inference", {}).get("published", 0))
        for camera in cameras.values()
    )
    # Coordinator overwritten is the latest-only drop count. Denominator is
    # frames the ingest path offered (published), not taken+overwritten: the
    # two counters can overlap and would understate the drop rate.
    offered = published_total if published_total > 0 else overwritten_total
    overwrite_fraction = None if offered <= 0 else overwritten_total / offered
    min_fps = min(admitted) if admitted else 0.0
    max_fps = max(admitted) if admitted else 0.0
    target = float(target_fps)
    if overwrite_fraction is not None and overwrite_fraction > overwrite_fraction_limit:
        verdict = "NOT"
        reason = (
            f"coordinator overwritten {overwritten_total} frames "
            f"({overwrite_fraction:.1%} of offered) exceeds "
            f"{overwrite_fraction_limit:.0%} drop budget"
        )
    elif min_fps >= target * achievable_fps_fraction:
        verdict = "ACHIEVABLE"
        reason = (
            f"every camera admitted >= {achievable_fps_fraction:.0%} of "
            f"{target:g} fps (min {min_fps:.2f}) with overwritten={overwritten_total}"
        )
    elif min_fps >= target * marginal_fps_fraction:
        verdict = "MARGINAL"
        reason = (
            f"slowest camera admitted {min_fps:.2f} fps "
            f"({min_fps / target:.0%} of {target:g}); overwritten={overwritten_total}"
        )
    else:
        verdict = "NOT"
        reason = (
            f"slowest camera admitted {min_fps:.2f} fps "
            f"({0.0 if target == 0 else min_fps / target:.0%} of {target:g}); "
            f"overwritten={overwritten_total}"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "min_admitted_fps": min_fps,
        "max_admitted_fps": max_fps,
        "overwritten_total": overwritten_total,
        "overwrite_fraction": overwrite_fraction,
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
    "StallWatcher",
    "capacity_verdict",
    "sample_gpu",
    "take_sample",
    "watchdog_margin_sec",
    "write_document",
]
