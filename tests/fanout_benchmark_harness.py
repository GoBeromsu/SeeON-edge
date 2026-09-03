"""Recorded-stream fan-out benchmark harness (plan todo 1, issue #312).

Serves ``N`` recorded RTSP streams through one local ``mediamtx`` (the
``docs/runbooks/local-e2e-rtsp-source.md`` topology, with the pinned clip
replaced by a locally generated recorded file so the harness owns its input),
boots one real ``WorkerRuntime`` against them with the REAL model artifacts in
``models/``, samples observable counters while it runs, and emits one
machine-readable ``bench-<N>.json``.

Two seams are wrapped, both test-side and both non-mutating:

* the serving client's ``pose`` runner, to collect the forward-latency
  distribution ``StageTimingAccumulator`` does not keep (samples/total/last/max
  only); and
* the relay endpoint, replaced by a loopback stub that answers every POST 200 --
  the benchmark measures the decode/inference fan-out, not relay behavior, and a
  dead relay port would otherwise charge each camera thread a connect timeout.

No worker product code is modified or monkeypatched.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Final, final

from e2e_worker_relay_fixtures import MediaMtxProcess, free_tcp_port, wait_until

from worker.adapters.model.in_process import InProcessServingClient
from worker.pipeline.output.evidence.clip_config import CLIP_STORE_DIR_ENV
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.worker import WorkerRuntime

RELAY_TOKEN: Final = "fanout-bench-token"  # noqa: S105 - harness-scoped constant
RECORDED_CLIP_SECONDS: Final = 20
RECORDED_CLIP_FPS: Final = 15
RECORDED_CLIP_SIZE: Final = "640x480"
BOOT_TIMEOUT_SEC: Final = 180.0
_GEOMETRY_TOKEN: Final = re.compile(r"^([1-9][0-9]*)x([1-9][0-9]*)$")


@dataclass(frozen=True, slots=True)
class BenchGeometry:
    """One camera's requested publisher geometry, encoded as ``WIDTHxHEIGHT``."""

    width: int
    height: int

    @property
    def token(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True, slots=True)
class BenchHarnessConfigError(Exception):
    """Raised when a benchmark environment knob is malformed."""

    message: str

    def __str__(self) -> str:
        return self.message


def parse_geometry_plan(raw: str | None, stream_count: int) -> tuple[BenchGeometry, ...] | None:
    """Parse ``BENCH_GEOMETRIES`` or return ``None`` when the knob is absent.

    Absent or blank input keeps the historical single-clip default. A present
    value is a closed parse: every token must be a positive ``WxH`` pair and
    the token count must equal ``stream_count``. This function never starts a
    process.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if stripped == "":
        return None
    tokens = [token.strip() for token in stripped.split(",")]
    if len(tokens) != stream_count:
        raise BenchHarnessConfigError(
            f"BENCH_GEOMETRIES has {len(tokens)} token(s) but BENCH_STREAMS is {stream_count}"
        )
    parsed: list[BenchGeometry] = []
    for token in tokens:
        parsed_token = _GEOMETRY_TOKEN.fullmatch(token)
        if parsed_token is None:
            raise BenchHarnessConfigError(f"invalid BENCH_GEOMETRIES token: {token!r}")
        parsed.append(
            BenchGeometry(width=int(parsed_token.group(1)), height=int(parsed_token.group(2)))
        )
    return tuple(parsed)


def require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(
            f"{name!r} is required on PATH for the fan-out benchmark harness "
            "(external RTSP tooling, not a production runtime dependency)"
        )
    return resolved


def build_recorded_clip(path: Path) -> Path:
    """Render one deterministic H.264 clip at the harness default geometry."""
    return render_recorded_clip(path, RECORDED_CLIP_SIZE)


def render_recorded_clip(path: Path, size: str) -> Path:
    """Render one deterministic H.264 clip the publishers loop over.

    A generated file rather than a pinned release clip: the benchmark must be
    reproducible on any host, and no restricted media may enter this repo or
    its temp dirs (docs/runbooks/local-e2e-rtsp-source.md keeps the real clip
    strictly operator-side).
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        require_tool("ffmpeg"),
        "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={size}:rate={RECORDED_CLIP_FPS}",
        "-t", str(RECORDED_CLIP_SECONDS),
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-g", str(RECORDED_CLIP_FPS), "-keyint_min", str(RECORDED_CLIP_FPS),
        "-sc_threshold", "0", "-pix_fmt", "yuv420p",
        str(path),
    ]  # fmt: skip
    completed = subprocess.run(  # noqa: S603 - fixed local binary, no shell
        command, capture_output=True, text=True, timeout=120.0, check=False
    )
    if completed.returncode != 0 or not path.exists():
        raise RuntimeError(f"recorded clip render failed: {completed.stderr.strip()[:400]}")
    return path


def recorded_clips_for_plan(directory: Path, plan: tuple[BenchGeometry, ...]) -> tuple[Path, ...]:
    """Render one clip per distinct geometry and return the per-camera sequence."""
    rendered: dict[str, Path] = {}
    clips: list[Path] = []
    for geometry in plan:
        token = geometry.token
        existing = rendered.get(token)
        if existing is None:
            existing = render_recorded_clip(directory / f"recorded-{token}.mp4", token)
            rendered[token] = existing
        clips.append(existing)
    return tuple(clips)


@final
class RecordedStreamPublisher:
    """One looping stream-copy publisher: decode cost stays with the worker."""

    def __init__(self, *, clip: Path, url: str) -> None:
        command = [
            require_tool("ffmpeg"),
            "-nostdin", "-hide_banner", "-loglevel", "error",
            "-re", "-stream_loop", "-1", "-i", str(clip),
            "-map", "0:v:0", "-an", "-c:v", "copy",
            "-f", "rtsp", "-rtsp_transport", "tcp", url,
        ]  # fmt: skip
        self._process = subprocess.Popen(  # noqa: S603 - fixed local binary, no shell
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )

    def assert_alive(self) -> None:
        if self._process.poll() is None:
            return
        stderr = "" if self._process.stderr is None else self._process.stderr.read()
        raise RuntimeError(
            f"rtsp publisher exited early with code {self._process.returncode}: {stderr[:400]}"
        )

    def stop(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        if self._process.stderr is not None:
            self._process.stderr.close()


class _RelayHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length") or 0)
        _ = self.rfile.read(length)
        body = json.dumps({"status": "accepted", "event_id": "bench"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return None


@final
class StubRelay:
    """Loopback relay that accepts every heartbeat/alert/status POST."""

    def __init__(self) -> None:
        self.port = free_tcp_port()
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _RelayHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="fanout-bench-relay"
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10.0)


@final
class TimingServingClient:
    """Real ``InProcessServingClient`` with a stopwatch around pose forwards.

    Exposes ``batch_serving_client`` because the composition root's
    ``_batch_client_for`` (worker/runtime/model_composition.py) gates the whole
    capability coordinator on the injected client being a ``BatchServingClient``
    or a ``BatchServingProvider``. A wrapper without it silently downgrades
    every camera to "camera pipeline requires the batched pose coordinator" --
    i.e. the benchmark would measure the pre-fix topology while claiming to
    measure the fix.
    """

    def __init__(self, latencies_ms: list[float]) -> None:
        self._inner = InProcessServingClient()
        self._latencies_ms = latencies_ms
        self._lock = threading.Lock()
        self._batch: _TimedBatchServingClient | None = None

    def create(self, task: str, **kwargs: Any) -> Any:
        runner = self._inner.create(task, **kwargs)
        if task != "pose":
            return runner
        return _TimedRunner(runner, self._record)

    @property
    def batch_serving_client(self) -> _TimedBatchServingClient:
        with self._lock:
            if self._batch is None:
                self._batch = _TimedBatchServingClient(
                    self._inner.batch_serving_client, self._record
                )
            return self._batch

    def _record(self, elapsed_ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(elapsed_ms)


@final
class _TimedBatchServingClient:
    """Times each real batched pose forward, which is where the work now is.

    One sample per ``infer_batch`` call, not per frame: the plan's p95 gate is
    on the forward pass the watchdog guards, and a batch of 13 frames is one
    forward. Per-frame attribution stays available through the coordinator's
    own stage timings in the bench JSON.
    """

    def __init__(self, inner: Any, record: Callable[[float], None]) -> None:
        self._inner = inner
        self._record = record

    def create(self, task: str, **kwargs: Any) -> Any:
        return self._inner.create(task, **kwargs)

    def infer_batch(self, task: str, frames: Sequence[Any], **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return self._inner.infer_batch(task, frames, **kwargs)
        finally:
            self._record((perf_counter() - started) * 1000.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@final
class _TimedRunner:
    """Delegating runner proxy that times the real forward pass.

    Deliberately exposes ``run`` and NOT ``__call__``: ``NamedExtractor`` binds
    its call target through ``_runner_call`` (``worker/pipeline/analytics/
    models.py``), which prefers a callable runner over ``run``. A proxy that is
    itself callable would therefore be invoked as ``proxy(image)`` and would
    then have to guess how the wrapped runner wants to be called -- the real
    ``YoloPoseRunner`` is not callable, so that guess is wrong and every pose
    forward fails inside the pump's per-frame boundary while the harness still
    records a (meaningless, sub-microsecond) latency sample.

    ``warmup`` is likewise declared on the class rather than left to
    ``__getattr__``: ``WorkerRuntime._warm_one`` gates on
    ``isinstance(model, _Warmable)``, and a ``runtime_checkable`` Protocol
    resolves members against the class, not the instance.
    """

    def __init__(self, inner: Any, record: Callable[[float], None]) -> None:
        self._inner = inner
        self._record = record
        self._call: Callable[[Any], Any] = inner if callable(inner) else inner.run

    def run(self, image: Any) -> Any:
        started = perf_counter()
        try:
            return self._call(image)
        finally:
            self._record((perf_counter() - started) * 1000.0)

    def warmup(self) -> None:
        self._inner.warmup()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_config(
    *,
    relay_url: str,
    rtsp_urls: Sequence[str],
    models_dir: Path,
    live_view_port: int | None = None,
    camera_fps: float | None = None,
) -> WorkerConfig:
    """N cameras, fall+bed_exit enabled, real fall artifact, clips off.

    Clip recording stays off (the product default): this benchmark measures the
    decode/inference fan-out, and evidence-clip behavior under the new decode
    boundary is plan todo 11's subject, not this one's.
    """
    dev_mjpeg = (
        {}
        if live_view_port is None
        else {"dev_mjpeg": {"enabled": True, "host": "127.0.0.1", "port": live_view_port}}
    )
    return WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": relay_url, "token": RELAY_TOKEN},
            "clip": {"enabled": False},
            **dev_mjpeg,
            "models": {"fall": _fall_model_config(models_dir)},
            "domains": {"fall": {"enabled": True}, "bed_exit": {"enabled": True}},
            "cameras": [
                {
                    "camera_id": f"bench-cam-{index + 1:02d}",
                    "facility_id": "bench-facility",
                    "rtsp_url": url,
                    "heartbeat_interval_sec": 30.0,
                    "frame_stride": 1,
                    # Product default 5.0 unless the operator asks otherwise.
                    # The only sanctioned use of an override is the headroom
                    # check: ingest paces at `1/fps` measured AFTER each yield
                    # (worker/pipeline/ingest/rtsp.py), so the admitted rate is
                    # structurally just under `fps` and 13x5.0 is an unreachable
                    # ceiling, not a serving limit. Raising the offered rate is
                    # how that distinction is measured rather than argued.
                    **({} if camera_fps is None else {"fps": camera_fps}),
                }
                for index, url in enumerate(rtsp_urls)
            ],
        }
    )


def _fall_model_config(models_dir: Path) -> dict[str, Any]:
    artifact_dir = models_dir / "fall" / "lstm"
    if not (artifact_dir / "model.pt").is_file():
        raise RuntimeError(
            f"fall model weights missing under {artifact_dir}; the benchmark boots the "
            "real model composition and refuses to fabricate one"
        )
    return {
        "type": "lstm",
        "framework": "pytorch",
        "mode": "sequence",
        "artifact_dir": str(artifact_dir),
        "weights": "model.pt",
        "architecture": "arch.json",
        "metadata": "metadata.yaml",
        "window": 30,
        "stride": 5,
        "input_shape": [30, 51],
        "operating_threshold": 0.5,
        "schema_version": 1,
        "preprocessing_identity": "legacy-coco17-xyc-frame-normalized-zero-fill-v1",
    }


@final
class WorkerRun:
    """A booted ``WorkerRuntime`` on its own thread, with bounded readiness."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        serving: TimingServingClient,
        profile: str,
        state_dir: Path,
        clip_store_dir: Path,
    ) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        clip_store_dir.mkdir(parents=True, exist_ok=True)
        self._cuda_visible_devices_present = "CUDA_VISIBLE_DEVICES" in os.environ
        self._cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.runtime = WorkerRuntime(
            config,
            env={"ML_WORKER_PROFILE": profile, CLIP_STORE_DIR_ENV: str(clip_store_dir)},
            serving_client=serving,
            acquire_lease=lambda: GpuLease.acquire(state_dir),
            hard_exit=lambda _code: None,
            state_dir=state_dir,
            # Explicit constructor seam, not the CLIP_STORE_DIR env: the
            # runtime resolves its store from this argument, and the env value
            # above only reaches collaborators that read the real environment.
            # Left unset it defaults to /var/lib/clip-store, which a local
            # benchmark process cannot lock.
            clip_store_dir=clip_store_dir,
        )
        self.thread = threading.Thread(
            target=self.runtime.run, daemon=True, name="fanout-bench-worker"
        )
        self.thread.start()

    def wait_for_cameras(self, expected: int, *, timeout: float = BOOT_TIMEOUT_SEC) -> None:
        wait_until(
            lambda: len(self.runtime.cameras) >= expected,
            timeout=timeout,
            interval=0.25,
            what=f"{expected} camera(s) to activate (worker boot + model load)",
        )

    def wait_for_first_inference(self, *, timeout: float) -> None:
        wait_until(
            self._any_inference_taken,
            timeout=timeout,
            interval=0.25,
            what="the first frame to be taken off an inference lane",
        )

    def _any_inference_taken(self) -> bool:
        return any(camera.bus.metrics("inference").taken > 0 for camera in self.runtime.cameras)

    def total_inference_taken(self) -> int:
        """Aggregate frames pulled off every camera's inference lane.

        The stall watcher's progress signal: this is the one counter that only
        moves when the coordinator actually drains and forwards work.
        """
        return sum(camera.bus.metrics("inference").taken for camera in self.runtime.cameras)

    def live_view_port(self) -> int | None:
        server = getattr(self.runtime, "_mjpeg_server", None)
        return None if server is None else int(server.port)

    def stop(self, *, timeout: float = 60.0) -> None:
        self.runtime.stop()
        self.thread.join(timeout=timeout)
        # Ultralytics' CPU device selection writes CUDA_VISIBLE_DEVICES="".
        # That process-global mutation belongs to this local benchmark run and
        # must not hide the GPU from later real-stack contracts in the suite.
        if self._cuda_visible_devices_present:
            assert self._cuda_visible_devices is not None
            os.environ["CUDA_VISIBLE_DEVICES"] = self._cuda_visible_devices
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)


@final
class LiveViewViewer:
    """One real HTTP client on ``/stream/{camera_id}``, i.e. an attached viewer.

    Not a call to ``LatestFrameStore.mark_viewer_connected``: the plan's target
    is "live lane consumed when a viewer is attached", and the only honest
    evidence for that is the shipped viewer surface -- an open MJPEG connection
    the worker's own HTTP handler counts. Frames read are counted so the
    benchmark can show the viewer really received JPEGs, and a read timeout
    keeps the reader thread bounded.
    """

    def __init__(self, *, port: int, camera_id: str, read_timeout_sec: float = 10.0) -> None:
        self._url = f"http://127.0.0.1:{port}/stream/{camera_id}"
        self._read_timeout_sec = read_timeout_sec
        self.camera_id = camera_id
        self.boundaries_seen = 0
        self.bytes_read = 0
        self.connects = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"fanout-bench-viewer-{camera_id}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        # Reconnecting loop, not a single GET: `_handle_stream` answers 503 when
        # no frame is cached within STREAM_FIRST_FRAME_TIMEOUT_SECONDS (0.5s) of
        # the connect, and viewer gating means the first encode only happens
        # *because* this connection exists -- so the first attempt legitimately
        # loses that race. Reconnect until frames flow, then stream.
        while not self._stop.is_set():
            self._stream_once()
            if not self._stop.is_set():
                self._stop.wait(0.2)

    def _stream_once(self) -> None:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(  # noqa: S310 - fixed loopback http URL
            self._url, headers={"X-Edge-Relay-Token": RELAY_TOKEN}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._read_timeout_sec) as response:
                self.connects += 1
                # A later successful connect retires the earlier 503 the
                # viewer-gating race produces; a real failure re-sets it below.
                self.error = None
                while not self._stop.is_set():
                    chunk = response.read(65536)
                    if not chunk:
                        return
                    self.bytes_read += len(chunk)
                    self.boundaries_seen += chunk.count(b"--frame")
        except (urllib.error.URLError, OSError, ValueError) as error:
            if not self._stop.is_set():
                self.error = f"{type(error).__name__}: {error}"

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def report(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "url": self._url,
            "bytes_read": self.bytes_read,
            "mjpeg_parts_seen": self.boundaries_seen,
            "connects": self.connects,
            "error": self.error,
        }


@contextmanager
def recorded_stream_fanout(
    *, stream_count: int, clip: Path, tmp_path: Path
) -> Iterator[tuple[MediaMtxProcess, tuple[str, ...]]]:
    """One mediamtx serving ``stream_count`` looping copies of one recorded clip."""
    with _publish_recorded_streams((clip,) * stream_count, tmp_path) as ready:
        yield ready


@contextmanager
def recorded_streams_for_clips(
    *, clips: Sequence[Path], tmp_path: Path
) -> Iterator[tuple[MediaMtxProcess, tuple[str, ...]]]:
    """One mediamtx serving one looping recorded clip per camera."""
    with _publish_recorded_streams(tuple(clips), tmp_path) as ready:
        yield ready


@contextmanager
def _publish_recorded_streams(
    clips: Sequence[Path], tmp_path: Path
) -> Iterator[tuple[MediaMtxProcess, tuple[str, ...]]]:
    del tmp_path
    path_names = tuple(f"bench{index + 1:02d}" for index in range(len(clips)))
    server = MediaMtxProcess(rtsp_port=free_tcp_port(), path_names=path_names)
    publishers: list[RecordedStreamPublisher] = []
    try:
        for name, clip in zip(path_names, clips, strict=True):
            publishers.append(RecordedStreamPublisher(clip=clip, url=server.rtsp_url(name)))
        deadline = monotonic() + 30.0
        while monotonic() < deadline:
            for publisher in publishers:
                publisher.assert_alive()
            if _streams_ready(server, path_names):
                break
        else:
            raise TimeoutError(
                f"recorded publishers did not become readable within 30s for {path_names}"
            )
        yield server, tuple(server.rtsp_url(name) for name in path_names)
    finally:
        for publisher in publishers:
            publisher.stop()
        server.stop()


def _streams_ready(server: MediaMtxProcess, path_names: Sequence[str]) -> bool:
    """Gate on ffprobe seeing real video metadata, per the runbook's live gate."""
    probe = require_tool("ffprobe")
    for name in path_names:
        completed = subprocess.run(  # noqa: S603 - fixed local binary, no shell
            [
                probe,
                "-v",
                "error",
                "-rw_timeout",
                "5000000",
                "-rtsp_transport",
                "tcp",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                server.rtsp_url(name),
            ],
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
        if completed.returncode != 0 or '"video"' not in completed.stdout:
            return False
    return True


__all__ = [
    "BOOT_TIMEOUT_SEC",
    "RELAY_TOKEN",
    "BenchGeometry",
    "BenchHarnessConfigError",
    "LiveViewViewer",
    "RecordedStreamPublisher",
    "StubRelay",
    "TimingServingClient",
    "WorkerRun",
    "build_config",
    "build_recorded_clip",
    "parse_geometry_plan",
    "recorded_clips_for_plan",
    "recorded_stream_fanout",
    "recorded_streams_for_clips",
    "render_recorded_clip",
    "require_tool",
]
