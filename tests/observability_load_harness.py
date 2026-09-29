"""Measurement-only observability load harness (Gate M/V).

Numbers written here are the only source for deployment budgets. Product
code must not invent thresholds from this harness.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from observability_stack_fixtures import (
    deepstream_available,
    ffmpeg_available,
    mediamtx_available,
    serve_backend,
    wait_until,
)

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_sandbox import ProductSandbox

_RELAY_TOKEN: Final = "obs-load-relay-token"
_BUDGET_BYTES: Final = 32 * 1024 * 1024
_SAMPLE_HZ: Final = 1.0


def _free_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class ObservabilityLoadSkip(RuntimeError):
    """Operator-gated tools or the recorded stream path are missing."""


@dataclass
class _Sample:
    at_sec: float
    queued: int
    overflow_pending: int
    receipts: int
    failures: int
    cpu_user_sec: float
    cpu_system_sec: float
    accepted_records: int
    gap_rows: int
    used_bytes: int | None
    queryable_min_ns: int | None
    queryable_max_ns: int | None
    exporter_exception: str | None


@dataclass
class _TimedClient:
    inner: Any
    latencies_sec: list[float] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)

    def post_batch(self, batch: object) -> object:
        started = time.monotonic()
        try:
            return self.inner.post_batch(batch)
        except Exception as error:  # noqa: BLE001 - measurement must not abort sampling
            self.exceptions.append(f"{type(error).__name__}: {error}")
            raise
        finally:
            self.latencies_sec.append(time.monotonic() - started)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _slope(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denom


def _overflow_pending(lanes: Any) -> int:
    with lanes._lock:  # noqa: SLF001
        total = 0
        for lane in lanes._lanes.values():  # noqa: SLF001
            dropped = lane.overflow
            if dropped:
                total += len(dropped)
        return total


def _cpu_times() -> tuple[float, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime), float(usage.ru_stime)


def _query_stats(
    backend: Any, camera_ids: Sequence[str]
) -> tuple[int, int, int | None, int | None, int | None]:
    accepted = 0
    gaps = 0
    logical = 0
    mins: list[int] = []
    maxs: list[int] = []
    for camera_id in camera_ids:
        body = backend.query(camera_id, 0, (1 << 62) - 1, limit=500)
        accepted += len(body["records"])
        gaps += sum(int(row["record_count"]) for row in body["coverage"])
        queryable = body["queryable_range"]
        if queryable["min_observed_at_ns"] is not None:
            mins.append(int(queryable["min_observed_at_ns"]))
        if queryable["max_observed_at_ns"] is not None:
            maxs.append(int(queryable["max_observed_at_ns"]))
        span = queryable["max_observed_at_ns"]
        start = queryable["min_observed_at_ns"]
        if span is not None and start is not None:
            logical += max(0, int(span) - int(start) + 1)
    return (
        accepted,
        gaps,
        logical if mins else None,
        min(mins) if mins else None,
        max(maxs) if maxs else None,
    )


def _mediamtx_yml(port: int, api_port: int) -> str:
    return (
        "logLevel: warn\n"
        "rtsp: true\n"
        f"rtspAddress: :{port}\n"
        "protocols: [tcp]\n"
        "hls: false\n"
        "webrtc: false\n"
        "srt: false\n"
        "api: true\n"
        f"apiAddress: :{api_port}\n"
        "pathDefaults:\n"
        "  source: publisher\n"
        "  overridePublisher: false\n"
        "paths:\n"
        "  all_others:\n"
    )


def _start_mediamtx(work_dir: Path, rtsp_port: int, api_port: int) -> subprocess.Popen[bytes]:
    config = work_dir / "mediamtx.yml"
    config.write_text(_mediamtx_yml(rtsp_port, api_port), encoding="utf-8")
    binary = shutil.which("mediamtx")
    if binary is None:
        raise ObservabilityLoadSkip("mediamtx is not on PATH")
    return subprocess.Popen(  # noqa: S603 - local operator binary
        [binary, str(config)],
        cwd=work_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _start_looping_publishers(
    stream_path: Path, rtsp_port: int, streams: int, camera_fps: float
) -> list[subprocess.Popen[bytes]]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ObservabilityLoadSkip("ffmpeg is not on PATH")
    publishers: list[subprocess.Popen[bytes]] = []
    for index in range(streams):
        url = f"rtsp://127.0.0.1:{rtsp_port}/cam-{index + 1}"
        process = subprocess.Popen(  # noqa: S603 - local operator binary
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-stream_loop",
                "-1",
                "-i",
                str(stream_path),
                "-c",
                "copy",
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        publishers.append(process)
    del camera_fps
    return publishers


def _stop_processes(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        process.terminate()
    deadline = time.monotonic() + 5.0
    for process in processes:
        remaining = max(0.05, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _worker_config(relay_url: str, streams: int, rtsp_port: int) -> Any:
    from worker.runtime.config import WorkerConfig

    cameras = [
        {
            "camera_id": f"cam-{index + 1}",
            "facility_id": "facility-obs",
            "rtsp_url": f"rtsp://127.0.0.1:{rtsp_port}/cam-{index + 1}",
        }
        for index in range(streams)
    ]
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": relay_url, "token": _RELAY_TOKEN},
            "cameras": cameras,
            "clip": {"enabled": False},
        }
    )


def _worker_env(relay_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "ML_WORKER_PROFILE": "flow",
            "ML_WORKER_EXECUTION_RECORDS_ENABLED": "1",
            "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY": "4096",
            "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX": "32",
            "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS": "50",
            "ML_RTSP_ALLOW_LOCAL_DESTINATIONS": "1",
            "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS": "1",
        }
    )
    env.setdefault("ML_WORKER_BUILD_REVISION", "obs-load-rev")
    del relay_url
    return env


def _start_worker(
    config: Any, env: Mapping[str, str], state_dir: Path
) -> tuple[Any, threading.Thread]:
    from worker.adapters.model.in_process import InProcessServingClient
    from worker.adapters.model.registry import flow_registry
    from worker.runtime.lease import GpuLease
    from worker.runtime.worker import WorkerRuntime

    runtime = WorkerRuntime(
        config,
        serving_client=InProcessServingClient(flow_registry()),
        env=env,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        state_dir=state_dir,
        clip_store_dir=state_dir / "clips",
        build_revision=env.get("ML_WORKER_BUILD_REVISION", "obs-load-rev"),
    )
    thread = threading.Thread(target=runtime.run, daemon=True, name="observability-worker")
    thread.start()
    return runtime, thread


def _document(
    *,
    streams: int,
    duration_sec: float,
    camera_fps: float,
    samples: Sequence[_Sample],
    latencies_sec: Sequence[float],
    exceptions: Sequence[str],
) -> dict[str, Any]:
    first = samples[0] if samples else None
    last = samples[-1] if samples else None
    elapsed = 0.0 if first is None or last is None else max(last.at_sec - first.at_sec, 1e-9)
    accepted_delta = (
        0 if first is None or last is None else last.accepted_records - first.accepted_records
    )
    gap_delta = 0 if first is None or last is None else last.gap_rows - first.gap_rows
    half = samples[len(samples) // 2 :]
    backlog_points = [(sample.at_sec, float(sample.queued)) for sample in half]
    cpu_delta = None
    if first is not None and last is not None:
        cpu_delta = (last.cpu_user_sec + last.cpu_system_sec) - (
            first.cpu_user_sec + first.cpu_system_sec
        )
    return {
        "streams": streams,
        "duration_sec": duration_sec,
        "offered_fps": camera_fps,
        "records_per_sec_accepted": accepted_delta / elapsed,
        "gap_rows_per_sec": gap_delta / elapsed,
        "lane_high_water": max((sample.queued for sample in samples), default=0),
        "backlog_slope": _slope(backlog_points),
        "p50_exporter_batch_latency_sec": _percentile(latencies_sec, 0.50),
        "p95_exporter_batch_latency_sec": _percentile(latencies_sec, 0.95),
        "cpu_delta_sec": cpu_delta,
        "overflow_high_water": max((sample.overflow_pending for sample in samples), default=0),
        "exporter_receipts": 0 if last is None else last.receipts,
        "exporter_failures": 0 if last is None else last.failures,
        "exporter_exceptions": list(exceptions),
        "used_bytes_last": None if last is None else last.used_bytes,
        "queryable_range": {
            "min_observed_at_ns": None if last is None else last.queryable_min_ns,
            "max_observed_at_ns": None if last is None else last.queryable_max_ns,
        },
        "samples": [
            {
                "at_sec": sample.at_sec,
                "queued": sample.queued,
                "overflow_pending": sample.overflow_pending,
                "receipts": sample.receipts,
                "failures": sample.failures,
                "accepted_records": sample.accepted_records,
                "gap_rows": sample.gap_rows,
                "used_bytes": sample.used_bytes,
            }
            for sample in samples
        ],
    }


def run_measurement(
    *,
    streams: int,
    duration_sec: float,
    camera_fps: float,
    output_dir: Path,
    sandbox: ProductSandbox,
    audit_runtime: PostgresAuditRuntime,
    diagnostics_schema: str,
) -> Path:
    """Run one N-stream measurement and write ``obs-<N>.json`` under ``output_dir``.

    Missing operator tools skip via ``ObservabilityLoadSkip``. The document
    records measurements only; callers must not assert numeric thresholds.
    The backend serves on the caller's PostgreSQL sandbox root.
    """
    if streams < 1:
        raise ValueError("streams must be a positive integer")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if camera_fps <= 0:
        raise ValueError("camera_fps must be positive")
    if not mediamtx_available():
        raise ObservabilityLoadSkip("mediamtx is not on PATH")
    if not ffmpeg_available():
        raise ObservabilityLoadSkip("ffmpeg is not on PATH")
    if not deepstream_available():
        raise ObservabilityLoadSkip("pyservicemaker is not importable")
    stream_raw = os.environ.get("OBS_STREAM_PATH", "").strip()
    if not stream_raw:
        raise ObservabilityLoadSkip("OBS_STREAM_PATH is unset")
    stream_path = Path(stream_raw)
    if not stream_path.is_file():
        raise ObservabilityLoadSkip(f"OBS_STREAM_PATH is not a file: {stream_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    document_path = output_dir / f"obs-{streams}.json"
    with tempfile.TemporaryDirectory(prefix="obs-load-") as raw_tmp:
        tmp_path = Path(raw_tmp)
        rtsp_port = _free_tcp_port()
        api_port = _free_tcp_port()
        mediamtx = _start_mediamtx(tmp_path, rtsp_port, api_port)
        publishers: list[subprocess.Popen[bytes]] = []
        runtime = None
        worker_thread: threading.Thread | None = None
        try:
            wait_until(
                lambda: mediamtx.poll() is None,
                timeout=5.0,
                what="mediamtx stay alive",
            )
            publishers = _start_looping_publishers(stream_path, rtsp_port, streams, camera_fps)
            with serve_backend(
                tmp_path / "backend",
                budget_bytes=_BUDGET_BYTES,
                relay_token=_RELAY_TOKEN,
                sandbox=sandbox,
                audit_runtime=audit_runtime,
                diagnostics_schema=diagnostics_schema,
            ) as backend:
                config = _worker_config(backend.base_url, streams, rtsp_port)
                env = _worker_env(backend.base_url)
                runtime, worker_thread = _start_worker(config, env, tmp_path / "worker")
                wait_until(
                    lambda: runtime._execution_record_exporter is not None,  # noqa: SLF001
                    timeout=60.0,
                    what="worker execution-record exporter composition",
                )
                exporter = runtime._execution_record_exporter  # noqa: SLF001
                lanes = runtime._execution_record_lanes  # noqa: SLF001
                timed = _TimedClient(exporter._client)  # noqa: SLF001
                exporter._client = timed  # noqa: SLF001
                camera_ids = [f"cam-{index + 1}" for index in range(streams)]
                samples: list[_Sample] = []
                started = time.monotonic()
                while time.monotonic() - started < duration_sec:
                    accepted, gaps, logical, qmin, qmax = _query_stats(backend, camera_ids)
                    user_sec, system_sec = _cpu_times()
                    samples.append(
                        _Sample(
                            at_sec=time.monotonic() - started,
                            queued=int(lanes.queued()),
                            overflow_pending=_overflow_pending(lanes),
                            receipts=len(exporter.receipts()),
                            failures=len(exporter.failures()),
                            cpu_user_sec=user_sec,
                            cpu_system_sec=system_sec,
                            accepted_records=accepted,
                            gap_rows=gaps,
                            used_bytes=logical,
                            queryable_min_ns=qmin,
                            queryable_max_ns=qmax,
                            exporter_exception=timed.exceptions[-1] if timed.exceptions else None,
                        )
                    )
                    remaining = duration_sec - (time.monotonic() - started)
                    time.sleep(min(1.0 / _SAMPLE_HZ, max(0.0, remaining)))
                document = _document(
                    streams=streams,
                    duration_sec=duration_sec,
                    camera_fps=camera_fps,
                    samples=samples,
                    latencies_sec=timed.latencies_sec,
                    exceptions=timed.exceptions,
                )
                document_path.write_text(
                    json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        finally:
            if runtime is not None:
                runtime.stop()
            if worker_thread is not None:
                worker_thread.join(timeout=30.0)
            _stop_processes(publishers)
            mediamtx.terminate()
            try:
                mediamtx.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                mediamtx.kill()
                mediamtx.wait(timeout=2.0)
    return document_path


def skip_reason() -> str | None:
    if not mediamtx_available():
        return "mediamtx is not on PATH"
    if not ffmpeg_available():
        return "ffmpeg is not on PATH"
    if not deepstream_available():
        return "pyservicemaker is not importable"
    if not os.environ.get("OBS_STREAM_PATH", "").strip():
        return "OBS_STREAM_PATH is unset"
    return None


__all__ = ["ObservabilityLoadSkip", "run_measurement", "skip_reason"]
