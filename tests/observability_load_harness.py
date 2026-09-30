"""Measurement-only observability load harness (Gate M/V).

Numbers written here are the only source for deployment budgets. Product
code must not invent thresholds from this harness.
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
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
# Upper bound for one worker boot (model backend init, Flow warmup, camera
# activation). A boot failure ends the worker thread and fails fast through
# the liveness check; this bound only catches a boot that hangs.
_BOOTSTRAP_TIMEOUT_SEC: Final = 120.0
# The restart check is polled once per second, and the Flow stop joins its
# own pipeline within 10 s; a thread still alive after this join failed to stop.
_WORKER_JOIN_SEC: Final = 30.0


def _free_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class WorkerTerminated(RuntimeError):
    """The in-process worker stopped, requested a hard exit, or did not stop cleanly."""


class PublisherFailed(RuntimeError):
    """A fixture RTSP publisher exited or never registered its path on mediamtx."""


class _HardExitRequested(BaseException):
    """Raised in place of ``os._exit`` so no worker code runs after a hard exit.

    It derives from ``BaseException`` so the worker's ``except Exception``
    handlers cannot swallow it on the way out of the thread.
    """

    def __init__(self, code: int) -> None:
        super().__init__(f"hard exit requested with code {code}")
        self.code = code


@dataclass
class _WorkerObserver:
    """Stands in for ``os._exit`` and records how the worker thread ended.

    A hard exit records its code and raises ``_HardExitRequested``, so the
    code after the ``hard_exit`` call does not run, as with ``os._exit``.
    ``os._exit`` also skips ``finally`` blocks, which an exception cannot
    mimic. A hard exit from a supervisor or watchdog thread ends that thread
    the same way, and the recorded code still fails the measurement.
    ``run`` records the worker thread's own error (bootstrap stage, Flow
    shutdown) instead of re-raising it, and the harness reports it after
    the join.
    """

    codes: list[int] = field(default_factory=list)
    error: BaseException | None = None

    def __call__(self, code: int) -> None:
        self.codes.append(code)
        raise _HardExitRequested(code)

    def run(self, target: Callable[[], object]) -> None:
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - reported by the harness after the join
            self.error = exc


def _worker_failure_reason(observer: _WorkerObserver) -> str:
    chain: list[str] = []
    error = observer.error
    while error is not None and len(chain) < 4:
        chain.append(f"{type(error).__name__}: {error}")
        error = error.__context__
    return " <- ".join(chain) if chain else "no thread error recorded"


def _ensure_worker_running(observer: _WorkerObserver, thread: threading.Thread) -> None:
    if observer.codes:
        raise WorkerTerminated(
            f"worker requested hard exit with code {observer.codes[0]}; "
            f"thread: {_worker_failure_reason(observer)}"
        )
    if not thread.is_alive():
        raise WorkerTerminated(
            "worker thread stopped before the measurement finished; "
            f"thread: {_worker_failure_reason(observer)}"
        )


_RTSP_FIXTURE_ALLOWANCE_KEYS: Final = (
    "ML_RTSP_ALLOW_LOCAL_DESTINATIONS",
    "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS",
)


@contextlib.contextmanager
def _rtsp_fixture_allowance() -> Iterator[None]:
    """Admit the loopback mediamtx fixture cameras for the measurement only.

    ``shared.rtsp_url_policy`` reads the process environment, not the mapping
    handed to ``WorkerRuntime``; the previous values are restored on exit.
    """
    previous = {key: os.environ.get(key) for key in _RTSP_FIXTURE_ALLOWANCE_KEYS}
    for key in _RTSP_FIXTURE_ALLOWANCE_KEYS:
        os.environ[key] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def _ready_mediamtx_paths(api_port: int) -> set[str] | None:
    """Names mediamtx itself reports as ready, or ``None`` while its API is down."""
    try:
        response = httpx.get(f"http://127.0.0.1:{api_port}/v3/paths/list", timeout=1.0)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return {str(item["name"]) for item in response.json().get("items", []) if item.get("ready")}


def _wait_for_mediamtx(mediamtx: subprocess.Popen[bytes], api_port: int) -> None:
    def api_answers() -> bool:
        if mediamtx.poll() is not None:
            raise PublisherFailed(f"mediamtx exited with code {mediamtx.returncode} during start")
        return _ready_mediamtx_paths(api_port) is not None

    wait_until(api_answers, timeout=10.0, what="mediamtx API to answer")


def _publisher_log(log_dir: Path, index: int) -> Path:
    return log_dir / f"publisher-cam-{index + 1}.log"


def _start_looping_publishers(
    stream_path: Path, rtsp_port: int, streams: int, camera_fps: float, log_dir: Path
) -> list[subprocess.Popen[bytes]]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ObservabilityLoadSkip("ffmpeg is not on PATH")
    publishers: list[subprocess.Popen[bytes]] = []
    for index in range(streams):
        url = f"rtsp://127.0.0.1:{rtsp_port}/cam-{index + 1}"
        with _publisher_log(log_dir, index).open("wb") as stderr:
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
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
        publishers.append(process)
    del camera_fps
    return publishers


def _wait_for_publishers(
    publishers: Sequence[subprocess.Popen[bytes]], api_port: int, log_dir: Path
) -> None:
    """Block until mediamtx reports every ``cam-N`` path ready; fail on a dead publisher."""
    expected = {f"cam-{index + 1}" for index in range(len(publishers))}

    def all_ready() -> bool:
        for index, process in enumerate(publishers):
            if process.poll() is not None:
                tail = _publisher_log(log_dir, index).read_text(errors="replace")[-2000:]
                raise PublisherFailed(
                    f"ffmpeg publisher cam-{index + 1} exited with code "
                    f"{process.returncode} before its path was ready:\n{tail}"
                )
        ready = _ready_mediamtx_paths(api_port)
        return ready is not None and expected <= ready

    wait_until(all_ready, timeout=20.0, what=f"mediamtx paths {sorted(expected)} ready")


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


def _worker_config(relay_url: str, streams: int, rtsp_port: int, env: Mapping[str, str]) -> Any:
    """Build the worker config the way the production boot path settles it.

    ``resolve_startup_config`` merges ``models`` from the process environment
    before the runtime boots; the runtime itself refuses to boot when
    ``config.models`` is ``None``. The harness hands its config straight to
    ``WorkerRuntime``, so it has to resolve ``models`` from ``env`` the same
    way, or the ``model_backend_init`` stage fails before any stream starts.
    """
    from worker.runtime.config import WorkerConfig, worker_models_config_from_environment

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
            "models": worker_models_config_from_environment(env),
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
            # Graceful stop needs EOS propagation, and nvurisrcbin's reconnect
            # logic swallows the EOS as a stream loss. The load harness measures
            # a stable looping fixture where reconnect is idle; production keeps
            # the default interval.
            "ML_WORKER_FLOW_RTSP_RECONNECT_INTERVAL_SEC": "0",
        }
    )
    env.setdefault("ML_WORKER_BUILD_REVISION", "obs-load-rev")
    del relay_url
    return env


def _start_worker(
    config: Any, env: Mapping[str, str], state_dir: Path, *, observer: _WorkerObserver
) -> tuple[Any, threading.Thread, threading.Event]:
    """Start ``WorkerRuntime.run`` on a thread; setting the event makes ``run`` return.

    The event is the worker's ``restart_check``: ``run`` leaves its loop and
    its own ``finally`` stops the Flow once, the production restart-directive
    path.
    """
    from worker.adapters.model.in_process import InProcessServingClient
    from worker.adapters.model.registry import flow_registry
    from worker.runtime.lease import GpuLease
    from worker.runtime.worker import WorkerRuntime

    stop_requested = threading.Event()
    runtime = WorkerRuntime(
        config,
        serving_client=InProcessServingClient(flow_registry()),
        env=env,
        hard_exit=observer,
        restart_check=stop_requested.is_set,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        state_dir=state_dir,
        clip_store_dir=state_dir / "clips",
        build_revision=env.get("ML_WORKER_BUILD_REVISION", "obs-load-rev"),
    )
    thread = threading.Thread(
        target=observer.run, args=(runtime.run,), daemon=True, name="observability-worker"
    )
    thread.start()
    return runtime, thread, stop_requested


def _document(
    *,
    streams: int,
    duration_sec: float,
    camera_fps: float,
    samples: Sequence[_Sample],
    latencies_sec: Sequence[float],
    exceptions: Sequence[str],
    worker_alive_at_unix_sec: float | None,
    measurement_started_at_unix_sec: float,
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
        "worker_alive_at_unix_sec": worker_alive_at_unix_sec,
        "measurement_started_at_unix_sec": measurement_started_at_unix_sec,
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
        observer = _WorkerObserver()
        try:
            _wait_for_mediamtx(mediamtx, api_port)
            publishers = _start_looping_publishers(
                stream_path, rtsp_port, streams, camera_fps, tmp_path
            )
            _wait_for_publishers(publishers, api_port, tmp_path)
            with (
                _rtsp_fixture_allowance(),
                serve_backend(
                    tmp_path / "backend",
                    budget_bytes=_BUDGET_BYTES,
                    relay_token=_RELAY_TOKEN,
                    sandbox=sandbox,
                    audit_runtime=audit_runtime,
                    diagnostics_schema=diagnostics_schema,
                ) as backend,
            ):
                env = _worker_env(backend.base_url)
                config = _worker_config(backend.base_url, streams, rtsp_port, env)
                runtime, worker_thread, stop_requested = _start_worker(
                    config, env, tmp_path / "worker", observer=observer
                )
                # The worker must stop while its relay backend still answers:
                # tearing the backend down first turns every in-flight export
                # into RETRY NETWORK noise and hides the real stop outcome.
                try:
                    document = _measure(
                        runtime,
                        worker_thread,
                        observer,
                        backend,
                        streams=streams,
                        duration_sec=duration_sec,
                        camera_fps=camera_fps,
                    )
                except BaseException as measure_error:
                    try:
                        _stop_worker(worker_thread, observer, stop_requested)
                    except WorkerTerminated as stop_error:
                        measure_error.add_note(f"worker stop also failed: {stop_error}")
                    raise
                document["worker_stop"] = _stop_worker(worker_thread, observer, stop_requested)
                document_path.write_text(
                    json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        finally:
            try:
                _stop_processes(publishers)
            finally:
                _stop_processes([mediamtx])
    return document_path


def _measure(
    runtime: Any,
    worker_thread: threading.Thread,
    observer: _WorkerObserver,
    backend: Any,
    *,
    streams: int,
    duration_sec: float,
    camera_fps: float,
) -> dict[str, Any]:
    """Sample one booted, running worker for ``duration_sec`` and return its document.

    The clock starts only once bootstrap completed (the worker published an
    alive status), so a worker that never boots cannot fill the window with
    idle samples. Liveness is checked before every sample and once more
    after the last one.
    """

    def exporter_composed() -> bool:
        _ensure_worker_running(observer, worker_thread)
        return runtime._execution_record_exporter is not None  # noqa: SLF001

    wait_until(
        exporter_composed,
        timeout=60.0,
        what="worker execution-record exporter composition",
    )
    exporter = runtime._execution_record_exporter  # noqa: SLF001
    lanes = runtime._execution_record_lanes  # noqa: SLF001
    timed = _TimedClient(exporter._client)  # noqa: SLF001
    exporter._client = timed  # noqa: SLF001

    def bootstrap_complete() -> bool:
        _ensure_worker_running(observer, worker_thread)
        worker = runtime.diagnostics.to_payload("facility-obs", None, 0).get("worker")
        return worker is not None and worker["alive"] is True

    wait_until(
        bootstrap_complete,
        timeout=_BOOTSTRAP_TIMEOUT_SEC,
        what="worker bootstrap complete (worker status alive)",
    )
    camera_ids = [f"cam-{index + 1}" for index in range(streams)]
    samples: list[_Sample] = []
    measurement_started_at_unix_sec = time.time()
    started = time.monotonic()
    while time.monotonic() - started < duration_sec:
        _ensure_worker_running(observer, worker_thread)
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
    _ensure_worker_running(observer, worker_thread)
    # The worker stamps its own alive status; the document carries it so the
    # real-stack test can check the window opened after bootstrap.
    worker = runtime.diagnostics.to_payload("facility-obs", None, 0).get("worker")
    return _document(
        streams=streams,
        duration_sec=duration_sec,
        camera_fps=camera_fps,
        samples=samples,
        latencies_sec=timed.latencies_sec,
        exceptions=timed.exceptions,
        worker_alive_at_unix_sec=None if worker is None else worker["started_at_sec"],
        measurement_started_at_unix_sec=measurement_started_at_unix_sec,
    )


def _stop_worker(
    thread: threading.Thread,
    observer: _WorkerObserver,
    stop_requested: threading.Event,
    *,
    join_timeout_sec: float = _WORKER_JOIN_SEC,
) -> dict[str, Any]:
    """Ask ``run`` to return, join the thread and fail unless it ended cleanly.

    Setting the restart-check event is the only stop request: ``run``'s own
    ``finally`` stops the Flow once. The harness never calls ``stop`` itself,
    because ``stop`` is not safe to run twice concurrently. Raises
    ``WorkerTerminated`` when the thread outlives the join, recorded a hard
    exit, or ended with an error; otherwise returns the join outcome.
    """
    stop_requested.set()
    started = time.monotonic()
    thread.join(timeout=join_timeout_sec)
    join_sec = time.monotonic() - started
    if thread.is_alive():
        raise WorkerTerminated(
            f"worker thread did not exit within the {join_timeout_sec:g} s join "
            "after the stop request"
        )
    if observer.codes:
        raise WorkerTerminated(
            f"worker requested hard exit with code {observer.codes[0]}; "
            f"thread: {_worker_failure_reason(observer)}"
        )
    if observer.error is not None:
        raise WorkerTerminated(
            f"worker thread ended with an error: {_worker_failure_reason(observer)}"
        )
    return {
        "join_sec": join_sec,
        "thread_alive_after_join": False,
        "hard_exit_codes": [],
        "thread_error": None,
    }


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


__all__ = [
    "ObservabilityLoadSkip",
    "PublisherFailed",
    "WorkerTerminated",
    "run_measurement",
    "skip_reason",
]
