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
  BENCH_GEOMETRIES     optional per-camera WxH plan, comma-separated
  BENCH_DURATION_SEC   measurement window per run              (default 60)
  BENCH_PROFILE        ML_WORKER_PROFILE for the run           (default nvidia-host-bridge)
  BENCH_OUTPUT_DIR     bench JSON destination directory
  BENCH_LABEL          filename suffix, e.g. "soak" -> bench-13-soak.json
  BENCH_VIEWERS        cameras to attach a real MJPEG viewer to (default 1, 0 disables)
  BENCH_CAMERA_FPS     per-camera offered fps (default: the product's own 5.0).
                       TemporalProfile is not on origin/main yet (PR #356); this
                       is the fps owner for a 15fps capacity run until that
                       contract lands. 13 cameras at 15fps:
                       BENCH_STREAMS=13 BENCH_CAMERA_FPS=15 BENCH_DURATION_SEC=300
                       BENCH_VIEWERS=0 uv run pytest -m real_stack -k 'test_fanout_benchmark['

Todo 12 additions (all read from already-published worker telemetry, nothing
new instrumented in product code): the coordinator's cross-camera batch-size
histogram and forward percentiles (``diagnostics.snapshot()``), a sub-second
stall watcher on aggregate inference progress (the 2s sampling cadence cannot
resolve the plan's "no stall > 2s" gate), and a real HTTP viewer on the
worker's own ``/stream/{camera}`` surface so "live lane consumed when a viewer
is attached" is measured through the shipped path rather than asserted.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
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
    BenchHarnessConfigError,
    LiveViewViewer,
    StubRelay,
    TimingServingClient,
    WorkerRun,
    build_config,
    build_recorded_clip,
    parse_geometry_plan,
    recorded_clips_for_plan,
    recorded_stream_fanout,
    recorded_streams_for_clips,
)
from fanout_benchmark_metrics import (
    BusLaneCounters,
    CameraSample,
    RunMetrics,
    RunSample,
    StallWatcher,
    capacity_verdict,
    take_sample,
    write_document,
)

from shared.rtsp_url_policy import ALLOW_LOCAL_RTSP_ENV

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


@pytest.mark.real_stack
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
    plan = parse_geometry_plan(os.environ.get("BENCH_GEOMETRIES"), stream_count)
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
    fanout = (
        recorded_stream_fanout(
            stream_count=stream_count, clip=recorded_clip, tmp_path=tmp_path
        )
        if plan is None
        else recorded_streams_for_clips(
            clips=recorded_clips_for_plan(tmp_path / "geometry-clips", plan),
            tmp_path=tmp_path,
        )
    )
    work_item_log = _CoordinatorErrorLog()
    coordinator_logger = logging.getLogger("worker.pipeline.inference_coordinator")
    if plan is not None:
        coordinator_logger.addHandler(work_item_log)

    try:
        with fanout as (_server, rtsp_urls):
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
                if plan is not None:
                    metrics.notes["compatibility_keys"] = list(
                        observed_compatibility_keys(run.runtime.diagnostics.snapshot())
                    )
                    metrics.notes["geometries"] = [geometry.token for geometry in plan]
            finally:
                for viewer in viewers:
                    viewer.stop()
                run.stop()
    finally:
        coordinator_logger.removeHandler(work_item_log)

    metrics.pose_latencies_ms = latencies
    metrics.notes["viewers"] = [viewer.report() for viewer in viewers]
    if plan is not None:
        metrics.notes["work_item_errors"] = list(work_item_log.messages)
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
            # Product default is 5.0 when the operator does not override.
            # Do not fall through to source_nominal_fps (the clip rate):
            # that would judge a 5fps ingest run against a 15fps target.
            "offered_fps": 5.0 if camera_fps is None else camera_fps,
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
    if plan is not None and len({geometry.token for geometry in plan}) > 1:
        assert_mixed_facility(
            document,
            frozenset(geometry.token for geometry in plan),
            stream_count,
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


@pytest.mark.real_stack
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


@pytest.mark.real_stack
def test_bench_document_schema_is_complete() -> None:
    """Pin the machine-readable field set the plan's todo 12 diffs against."""
    metrics = RunMetrics()
    document: dict[str, Any] = metrics.document(header={"stream_count": 0, "profile": "cpu"})
    assert set(document) >= {
        "aggregate_inference_fps",
        "cameras",
        "capacity",
        "counters_advanced",
        "errors",
        "gpu",
        "max_inference_frame_age_sec",
        "notes",
        "overwritten",
        "overwritten_by_camera",
        "pose_stage_latency_ms",
        "profile",
        "pump_failures",
        "samples",
        "stream_count",
        "watchdog_margin_sec",
    }


def test_geometry_plan_absent_keeps_default_none() -> None:
    # Given / When / Then
    assert parse_geometry_plan(None, 13) is None
    assert parse_geometry_plan("", 13) is None
    assert parse_geometry_plan("   ", 13) is None


def test_geometry_plan_parses_live_incident_topology() -> None:
    # Given
    raw = ",".join(["640x360"] * 12 + ["1920x1080"])
    # When
    plan = parse_geometry_plan(raw, 13)
    # Then
    assert plan is not None
    assert len(plan) == 13
    assert [geometry.token for geometry in plan] == ["640x360"] * 12 + ["1920x1080"]


@pytest.mark.parametrize(
    ("raw", "stream_count"),
    (
        ("640x360,not-a-size", 2),
        ("640x0", 1),
        ("0x360", 1),
        ("-1x360", 1),
        ("640x-360", 1),
        ("640X360", 1),
        ("640*360", 1),
        ("640 x 360", 1),
        ("640x360,,1920x1080", 3),
        ("640x360,1920x1080", 13),
    ),
)
def test_geometry_plan_rejects_malformed_input(raw: str, stream_count: int) -> None:
    # Given a closed geometry-plan parse
    # When the token is malformed or the count disagrees with BENCH_STREAMS
    # Then configuration fails before any publisher or worker exists
    with pytest.raises(BenchHarnessConfigError):
        parse_geometry_plan(raw, stream_count)


def test_malformed_geometry_plan_does_not_spawn_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[str] = []

    def _forbid(*_args: object, **_kwargs: object) -> None:
        spawned.append("spawn")
        raise AssertionError("geometry-plan parse must not start a process")

    monkeypatch.setattr("subprocess.Popen", _forbid)
    monkeypatch.setattr("subprocess.run", _forbid)
    monkeypatch.setattr("fanout_benchmark_harness.subprocess.Popen", _forbid)
    monkeypatch.setattr("fanout_benchmark_harness.subprocess.run", _forbid)
    with pytest.raises(BenchHarnessConfigError):
        parse_geometry_plan("640x360,bad-token", 2)
    assert spawned == []


def test_recorded_stream_fanout_signature_keeps_homogeneous_clip_parameter() -> None:
    parameters = inspect.signature(recorded_stream_fanout).parameters
    assert "clip" in parameters
    assert list(parameters) == ["stream_count", "clip", "tmp_path"]


def test_bench_document_omits_geometry_fields_when_plan_absent() -> None:
    document = RunMetrics().document(header={"stream_count": 1, "profile": "cpu"})
    assert "geometries" not in document
    assert "compatibility_keys" not in document["notes"]
    assert "work_item_errors" not in document["notes"]


def test_mixed_facility_assertion_fails_when_only_one_geometry_is_present() -> None:
    document = _healthy_mixed_document(keys=("640x360",))
    with pytest.raises(AssertionError, match="compatibility keys"):
        assert_mixed_facility(document, frozenset({"640x360", "1920x1080"}), 13)


def test_mixed_facility_assertion_fails_when_a_camera_does_not_advance_pose() -> None:
    document = _healthy_mixed_document(keys=("640x360", "1920x1080"))
    document["cameras"]["bench-cam-13"]["pose_inferences"] = 0
    with pytest.raises(AssertionError, match="did not advance pose"):
        assert_mixed_facility(document, frozenset({"640x360", "1920x1080"}), 13)


def test_capacity_verdict_catches_silent_overwrites() -> None:
    """A healthy-looking fps with coordinator drops is NOT achievable.

    The classic misleading-success failure: latest-only slots overwrite unread
    frames, admitted fps stays near the target, and a naive fps-only report
    would call the run a success. The verdict must fail on ``overwritten``.
    """
    # Given 13 cameras admitting ~15 fps while the coordinator overwrote 20%
    cameras = {
        f"bench-cam-{index:02d}": {
            "inference_admitted_fps": 14.8,
            "overwritten": 300,
            "bus": {"inference": {"published": 1500, "taken": 1200, "dropped": 300}},
        }
        for index in range(1, 14)
    }
    # When the capacity gate reads coordinator overwritten, not just fps
    result = capacity_verdict(cameras, target_fps=15.0)
    # Then the run cannot be reported as ACHIEVABLE
    assert result["verdict"] == "NOT"
    assert result["overwritten_total"] == 3900
    assert result["overwrite_fraction"] == pytest.approx(0.2)
    assert "overwritten" in result["reason"]


def test_capacity_verdict_is_achievable_when_overwrites_stay_near_zero() -> None:
    # Given 13 cameras at the target with a handful of latest-only jitter drops
    cameras = {
        f"bench-cam-{index:02d}": {
            "inference_admitted_fps": 14.7,
            "overwritten": 2,
            "bus": {"inference": {"published": 1472, "taken": 1470, "dropped": 2}},
        }
        for index in range(1, 14)
    }
    # When / Then
    result = capacity_verdict(cameras, target_fps=15.0)
    assert result["verdict"] == "ACHIEVABLE"
    assert result["overwritten_total"] == 26


def test_take_sample_reads_coordinator_overwritten() -> None:
    """Sampling must surface coordinator overwritten, not invent a zero."""
    from types import SimpleNamespace

    # Given a diagnostics snapshot whose coordinator already recorded drops
    snapshot = SimpleNamespace(
        cameras=(
            SimpleNamespace(
                camera_id="bench-cam-01",
                bus=(
                    SimpleNamespace(
                        name="inference",
                        published=10,
                        taken=8,
                        dropped=2,
                        queue_age_sec=0.01,
                    ),
                ),
                stage_timings=(SimpleNamespace(stage="pose", samples=8),),
                failure_category=None,
                decode_backend=None,
                inference=SimpleNamespace(overwritten=7, admitted=8, inferred=8),
                batch_sizes=((13, 4),),
                forward_p50_sec=0.01,
                forward_p95_sec=0.02,
            ),
        )
    )
    # When the harness samples that snapshot
    sample = take_sample(SimpleNamespace(snapshot=lambda: snapshot), watchdog=None)
    # Then overwritten is the coordinator field, not the bus dropped count alone
    assert sample.cameras[0].overwritten == 7
    assert sample.coordinator_forward_p95_sec == pytest.approx(0.02)
    assert sample.batch_sizes == {13: 4}


def test_document_reports_overwritten_so_a_drop_cannot_look_healthy() -> None:
    metrics = RunMetrics()
    first = _sample_with_overwritten(overwritten=10, published=100, taken=90)
    last = _sample_with_overwritten(overwritten=40, published=250, taken=220)
    last.at_sec = first.at_sec + 10.0
    metrics.add(first)
    metrics.add(last)
    document = metrics.document(header={"stream_count": 1, "profile": "cpu", "camera_fps": 15.0})
    assert document["overwritten"] == 30
    assert document["overwritten_by_camera"] == {"bench-cam-01": 30}
    assert document["cameras"]["bench-cam-01"]["overwritten"] == 30
    assert document["capacity"]["overwritten_total"] == 30
    assert document["capacity"]["verdict"] in {"ACHIEVABLE", "MARGINAL", "NOT"}
    # 30 overwrites on 130 taken is 18.75% -- the drop catch must fire
    assert document["capacity"]["verdict"] == "NOT"


def _sample_with_overwritten(*, overwritten: int, published: int, taken: int) -> RunSample:
    return RunSample(
        at_sec=1000.0,
        cameras=(
            CameraSample(
                camera_id="bench-cam-01",
                at_sec=1000.0,
                lanes={
                    "inference": BusLaneCounters(
                        published=published, taken=taken, dropped=overwritten
                    )
                },
                inference_queue_age_sec=0.0,
                pose_samples=taken,
                failure_category=None,
                overwritten=overwritten,
            ),
        ),
        watchdog_margin_sec=None,
        gpu=None,
        batch_sizes={13: 1},
        coordinator_forward_p95_sec=0.02,
    )


def test_bench_document_overwrite_replaces_previous_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bench-13-issue328.json"
    first = write_document(path, {"stream_count": 1, "samples": 1, "label": "old"})
    first_digest = hashlib.md5(first.read_bytes(), usedforsecurity=False).hexdigest()
    first_mtime_ns = first.stat().st_mtime_ns
    second = write_document(path, {"stream_count": 13, "samples": 9, "label": "issue328"})
    second_bytes = second.read_bytes()
    second_digest = hashlib.md5(second_bytes, usedforsecurity=False).hexdigest()
    assert first == second
    assert first_digest != second_digest
    assert second.stat().st_mtime_ns >= first_mtime_ns
    assert second_bytes.count(b"stream_count") == 1
    assert b'"stream_count": 13' in second_bytes
    assert b'"label": "old"' not in second_bytes


def _healthy_mixed_document(*, keys: tuple[str, ...]) -> dict[str, Any]:
    cameras = {
        f"bench-cam-{index:02d}": {
            "pose_inferences": 4,
            "pump_failures": 0,
            "failure_category": None,
        }
        for index in range(1, 14)
    }
    return {
        "cameras": cameras,
        "errors": [],
        "pump_failures": 0,
        "watchdog_margin_sec": 12.5,
        "notes": {
            "compatibility_keys": list(keys),
            "work_item_errors": [],
            "watchdog_tripped": False,
        },
    }


def assert_mixed_facility(
    document: dict[str, Any],
    expected_keys: frozenset[str],
    expected_camera_count: int,
) -> None:
    cameras = document["cameras"]
    assert isinstance(cameras, dict)
    assert len(cameras) == expected_camera_count, (
        f"expected {expected_camera_count} cameras, found {sorted(cameras)}"
    )
    idle = [
        camera_id
        for camera_id, camera in cameras.items()
        if int(camera["pose_inferences"]) <= 0
    ]
    assert idle == [], f"cameras did not advance pose inference: {idle}"
    notes = document["notes"]
    observed = frozenset(notes.get("compatibility_keys", ()))
    missing = expected_keys - observed
    assert not missing, (
        f"missing compatibility keys {sorted(missing)}; observed {sorted(observed)}"
    )
    assert document["errors"] == []
    assert document["pump_failures"] == 0
    assert notes.get("work_item_errors", []) == []
    assert notes.get("watchdog_tripped") is False
    margin = document["watchdog_margin_sec"]
    assert margin is not None and margin > 0, f"watchdog margin was not positive: {margin}"
    failed = [
        camera_id
        for camera_id, camera in cameras.items()
        if camera.get("failure_category") is not None
    ]
    assert failed == [], f"cameras reported failure_category: {failed}"


def observed_compatibility_keys(snapshot: Any) -> tuple[str, ...]:
    keys: set[str] = set()
    for camera in snapshot.cameras:
        for histogram in getattr(camera, "geometry_batch_sizes", ()):
            width, height = histogram.geometry
            keys.add(f"{width}x{height}")
        inference = getattr(camera, "inference", None)
        geometry = None if inference is None else getattr(inference, "observed_geometry", None)
        if geometry is not None:
            width, height = geometry
            keys.add(f"{width}x{height}")
    return tuple(sorted(keys))


class _CoordinatorErrorLog(logging.Handler):
    """Collect coordinator ERROR records; mutation is the documented purpose."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())
