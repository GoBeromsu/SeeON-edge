"""Recorded-stream fan-out benchmark: N in {1,2,4,8,13} (plan todo 1, issue #312).

Real stack, local only: one ``mediamtx`` serving N looping recorded streams, one
real ``WorkerRuntime`` with the real ``models/`` artifacts, and one
``bench-<N>.json`` per run under ``BENCH_OUTPUT_DIR`` (default
``.omo/evidence/bench``).

Which N run is selected is an operator decision, never a default that quietly
burns minutes in someone's suite: the parametrization is filtered by
``BENCH_STREAMS`` (comma-separated), and everything not listed is skipped with a
reason. The default is ``1,2`` -- the baseline the plan asks for -- so a bare
``uv run pytest -m real_stack -k fanout_benchmark`` runs the low-N baseline and
nothing more.

Environment knobs (all bounded, all recorded into the JSON):
  BENCH_STREAMS        stream counts to run, e.g. "1,2,4"      (default "1,2")
  BENCH_DURATION_SEC   measurement window per run              (default 60)
  BENCH_PROFILE        ML_WORKER_PROFILE for the run           (default nvidia-host-bridge)
  BENCH_OUTPUT_DIR     bench JSON destination directory
  BENCH_LABEL          filename suffix, e.g. "soak" -> bench-13-soak.json
  BENCH_VIEWERS        cameras to attach a real MJPEG viewer to (default 1, 0 disables)
  BENCH_CAMERA_FPS     per-camera offered fps (default: the product's own 5.0)

Todo 12 additions (all read from already-published worker telemetry, nothing
new instrumented in product code): the coordinator's cross-camera batch-size
histogram and forward percentiles (``diagnostics.snapshot()``), a sub-second
stall watcher on aggregate inference progress (the 2s sampling cadence cannot
resolve the plan's "no stall > 2s" gate), and a real HTTP viewer on the
worker's own ``/stream/{camera}`` surface so "live lane consumed when a viewer
is attached" is measured through the shipped path rather than asserted.
"""

from __future__ import annotations

import os
import shutil
import socket
from collections.abc import Iterator
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Final

import pytest
from e2e_worker_relay_fixtures import free_tcp_port
from fanout_benchmark_harness import (
    LiveViewViewer,
    StubRelay,
    TimingServingClient,
    WorkerRun,
    build_config,
    build_recorded_clip,
    recorded_stream_fanout,
)
from fanout_benchmark_metrics import (
    RunMetrics,
    StallWatcher,
    take_sample,
    write_document,
)

from shared.rtsp_url_policy import ALLOW_LOCAL_RTSP_ENV

pytestmark = pytest.mark.real_stack

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
STREAM_COUNTS: Final = (1, 2, 4, 8, 13)
DEFAULT_STREAMS: Final = "1,2"
DEFAULT_DURATION_SEC: Final = 60.0
DEFAULT_PROFILE: Final = "nvidia-host-bridge"
SAMPLE_INTERVAL_SEC: Final = 2.0
# Bounded: a run that never takes a frame off an inference lane within this
# window is the #312 stall signature, recorded as such -- never waited out.
FIRST_INFERENCE_TIMEOUT_SEC: Final = 120.0


def _selected_counts() -> frozenset[int]:
    raw = os.environ.get("BENCH_STREAMS", DEFAULT_STREAMS)
    return frozenset(int(part) for part in raw.replace(" ", "").split(",") if part)


def _output_dir() -> Path:
    configured = os.environ.get("BENCH_OUTPUT_DIR")
    return Path(configured) if configured else REPO_ROOT / ".omo" / "evidence" / "bench"


@pytest.fixture(scope="session")
def recorded_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH; the fan-out benchmark runs locally only")
    return build_recorded_clip(tmp_path_factory.mktemp("bench-clip") / "recorded.mp4")


@pytest.fixture
def relay() -> Iterator[StubRelay]:
    stub = StubRelay()
    try:
        yield stub
    finally:
        stub.stop()


@pytest.fixture
def allow_loopback_rtsp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt this process into loopback RTSP destinations for the run's lifetime.

    ``shared.rtsp_url_policy`` reads ``ML_RTSP_ALLOW_LOCAL_DESTINATIONS`` from
    the real environment (not the runtime's injected ``env`` mapping) and
    otherwise refuses ``rtsp://127.0.0.1`` outright. ``scripts/edge-preflight/
    check-env.sh`` calls that flag fixture-only and fails production preflight
    when it is set -- which is exactly what this is: a local benchmark serving
    its own recorded streams from loopback, scoped to one test by monkeypatch.
    """
    monkeypatch.setenv(ALLOW_LOCAL_RTSP_ENV, "1")


@pytest.mark.parametrize("stream_count", STREAM_COUNTS)
def test_fanout_benchmark(
    stream_count: int,
    recorded_clip: Path,
    relay: StubRelay,
    allow_loopback_rtsp: None,
    tmp_path: Path,
) -> None:
    """Serve ``stream_count`` recorded streams and emit ``bench-<N>.json``."""
    if stream_count not in _selected_counts():
        pytest.skip(f"N={stream_count} not selected; set BENCH_STREAMS to include it")
    if shutil.which("mediamtx") is None:
        pytest.skip("mediamtx not on PATH; real-stack benchmark runs locally only")

    duration_sec = float(os.environ.get("BENCH_DURATION_SEC", DEFAULT_DURATION_SEC))
    profile = os.environ.get("BENCH_PROFILE", DEFAULT_PROFILE)
    viewer_count = int(os.environ.get("BENCH_VIEWERS", "1"))
    label = os.environ.get("BENCH_LABEL", "")
    camera_fps_raw = os.environ.get("BENCH_CAMERA_FPS")
    camera_fps = None if camera_fps_raw is None else float(camera_fps_raw)
    metrics = RunMetrics()
    latencies: list[float] = []
    viewers: list[LiveViewViewer] = []

    with recorded_stream_fanout(
        stream_count=stream_count, clip=recorded_clip, tmp_path=tmp_path
    ) as (_server, rtsp_urls):
        config = build_config(
            relay_url=relay.base_url,
            rtsp_urls=rtsp_urls,
            models_dir=REPO_ROOT / "models",
            live_view_port=(free_tcp_port() if viewer_count > 0 else None),
            camera_fps=camera_fps,
        )
        serving = TimingServingClient(latencies)
        run = WorkerRun(
            config,
            serving=serving,
            profile=profile,
            state_dir=tmp_path / "state",
            clip_store_dir=tmp_path / "clips",
        )
        try:
            run.wait_for_cameras(stream_count)
            viewers = _attach_viewers(run, min(viewer_count, stream_count))
            _collect(run, metrics, duration_sec=duration_sec)
        finally:
            for viewer in viewers:
                viewer.stop()
            run.stop()

    metrics.pose_latencies_ms = latencies
    metrics.notes["viewers"] = [viewer.report() for viewer in viewers]
    document = metrics.document(
        header={
            "stream_count": stream_count,
            "profile": profile,
            "duration_sec": duration_sec,
            "sample_interval_sec": SAMPLE_INTERVAL_SEC,
            "source_nominal_fps": 15.0,
            "git_revision": _git_revision(),
            "label": label or None,
            "camera_fps": camera_fps,
        }
    )
    suffix = f"-{label}" if label else ""
    path = write_document(_output_dir() / f"bench-{stream_count}{suffix}.json", document)

    assert document["cameras"], f"no camera diagnostics recorded; see {path}"
    assert document["counters_advanced"], (
        "bus counters never advanced between samples -- the JSON would report "
        f"defaults, not measurements; see {path}"
    )
    assert document["errors"] == [], f"run recorded failures: {document['errors']}; see {path}"
    assert document["pump_failures"] == 0, (
        f"{document['pump_failures']} per-frame pipeline failures were swallowed by the "
        f"pump's camera boundary; the run's latency numbers are not trustworthy; see {path}"
    )
    assert document["pose_stage_latency_ms"] is not None, (
        f"no pose forward was timed; the benchmark measured nothing; see {path}"
    )


def _attach_viewers(run: WorkerRun, count: int) -> list[LiveViewViewer]:
    """Open ``count`` real MJPEG viewers, or none when the live view is off."""
    if count <= 0:
        return []
    port = run.live_view_port()
    if port is None:
        return []
    viewers = [
        LiveViewViewer(port=port, camera_id=camera.scene_state.camera_id)
        for camera in run.runtime.cameras[:count]
    ]
    for viewer in viewers:
        viewer.start()
    return viewers


def _collect(run: WorkerRun, metrics: RunMetrics, *, duration_sec: float) -> None:
    """Sample counters for ``duration_sec``, recording a stall instead of hanging."""
    try:
        run.wait_for_first_inference(timeout=FIRST_INFERENCE_TIMEOUT_SEC)
    except TimeoutError as error:
        metrics.errors.append(f"inference_stall: {error}")
        metrics.notes["failure_signature"] = "no inference lane take within bounded window (#312)"
    pumps = tuple(camera.pump for camera in run.runtime.cameras)
    watcher = StallWatcher(run.total_inference_taken)
    watcher.start()
    deadline = monotonic() + duration_sec
    metrics.add(take_sample(run.runtime.diagnostics, run.runtime.watchdog, pumps))
    try:
        while monotonic() < deadline:
            sleep(min(SAMPLE_INTERVAL_SEC, max(0.0, deadline - monotonic())))
            metrics.add(take_sample(run.runtime.diagnostics, run.runtime.watchdog, pumps))
    finally:
        watcher.stop()
    metrics.stalls = watcher.report()
    metrics.notes["cameras_activated"] = len(run.runtime.cameras)
    metrics.notes["watchdog_tripped"] = bool(
        run.runtime.watchdog is not None and run.runtime.watchdog.tripped
    )


def _git_revision() -> str | None:
    head = REPO_ROOT / ".git"
    try:
        if head.is_file():
            gitdir = Path(head.read_text(encoding="utf-8").split(":", 1)[1].strip())
            head = gitdir if gitdir.is_absolute() else REPO_ROOT / gitdir
        return (head / "HEAD").read_text(encoding="utf-8").strip()[:80]
    except OSError:
        return None


def test_fanout_benchmark_fails_fast_on_dead_rtsp_port(
    tmp_path: Path, relay: StubRelay, allow_loopback_rtsp: None
) -> None:
    """A dead RTSP port must produce a bounded diagnostic, never a hang.

    Points a single camera at a closed loopback port and asserts the harness's
    own readiness gate gives up inside its deadline with a diagnostic naming
    what it waited for. Nothing here waits on the worker's own reconnect
    backoff, which is unbounded by design.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]

    config = build_config(
        relay_url=relay.base_url,
        rtsp_urls=(f"rtsp://127.0.0.1:{dead_port}/dead",),
        models_dir=REPO_ROOT / "models",
    )
    serving = TimingServingClient([])
    run = WorkerRun(
        config,
        serving=serving,
        profile=os.environ.get("BENCH_PROFILE", DEFAULT_PROFILE),
        state_dir=tmp_path / "state",
        clip_store_dir=tmp_path / "clips",
    )
    try:
        run.wait_for_cameras(1)
        started = monotonic()
        with pytest.raises(TimeoutError) as failure:
            run.wait_for_first_inference(timeout=10.0)
        elapsed = monotonic() - started
    finally:
        run.stop()

    assert elapsed < 30.0, f"dead-port wait was not bounded: {elapsed:.1f}s"
    assert "inference lane" in str(failure.value)


def test_bench_document_schema_is_complete() -> None:
    """Pin the machine-readable field set the plan's todo 12 diffs against."""
    metrics = RunMetrics()
    document: dict[str, Any] = metrics.document(header={"stream_count": 0, "profile": "cpu"})
    assert set(document) >= {
        "aggregate_inference_fps",
        "cameras",
        "counters_advanced",
        "errors",
        "gpu",
        "max_inference_frame_age_sec",
        "notes",
        "pose_stage_latency_ms",
        "profile",
        "pump_failures",
        "samples",
        "stream_count",
        "watchdog_margin_sec",
    }
