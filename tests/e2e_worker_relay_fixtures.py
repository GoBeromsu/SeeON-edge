"""Shared fixtures for the real-stack night/fall/day E2E relay test.

Everything here is test-scoped composition glue: a synthetic RTSP source
(MediaMTX + an ffmpeg lavfi publisher), a real FastAPI relay backend running
in-process over uvicorn, deterministic scripted model runners that satisfy
the worker's serving-client seams, and small helpers to boot/tear down a
real ``WorkerRuntime`` against them. Production code is monkeypatched in two
narrow, restored-in-``finally`` spots: ``worker.runtime.worker.ClipRecorderConfig``
(a bare-name module lookup the composition root itself resolves at call
time), swapped only to shrink pre/post-event padding so clip finalize is
fast enough for a test process instead of production's default 30s/30s
window; and ``WorkerRuntime._create_fall_model``, pinned back to sourcing
the fall runner from the injected ``ScriptedServingClient`` (this harness
has no real LSTM artifact on disk, so it cannot use the fail-closed
explicit-config path -- see ``WorkerRuntime._create_fall_model``).

``WorkerRuntime.__init__`` also unconditionally builds a ``SnapshotStore()``
with no DI seam of its own -- it always resolves its root via
``worker.pipeline.output.evidence.clip_config.configured_store_dir()``,
which defaults to ``/var/lib/clip-store`` when ``CLIP_STORE_DIR`` is unset.
That path is neither owned by nor writable by a local macOS test process, so
``start_worker_runtime`` temporarily points ``CLIP_STORE_DIR`` at the test's
own clip store directory around the ``WorkerRuntime(...)`` construction
call, restoring the prior value afterward.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, Literal, final
from zoneinfo import ZoneInfo

import numpy as np
import uvicorn
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.status.runtime_status_store import (
    RuntimeStatusStore,
    get_runtime_status_store,
)
from backend.app.main import create_app, no_lifespan
from contracts.relay import AlertEventType
from contracts.runner import (
    Image,
    RunnerResult,
    bed_result,
    person_result,
    pose_result,
)
from shared.events.evidence_export_contract import EventReceipt
from worker.pipeline.output.evidence.clip_config import CLIP_STORE_DIR_ENV
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.worker import CameraRuntimeContext, WorkerRuntime

# --------------------------------------------------------------------------
# generic polling / networking helpers
# --------------------------------------------------------------------------


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(
    predicate: Callable[[], bool], *, timeout: float, interval: float = 0.05, what: str
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")


# --------------------------------------------------------------------------
# synthetic RTSP fixture: MediaMTX server + ffmpeg lavfi publisher
# --------------------------------------------------------------------------


@final
class MediaMtxProcess:
    """A disposable MediaMTX RTSP server, configured entirely through env vars.

    MediaMTX v1.19.3 with no config file has zero configured paths by
    default, so publishing to any path is rejected with RTSP 400 unless the
    path is explicitly declared (``MTX_PATHS_<NAME>_SOURCE=publisher``). All
    other listener types (API/metrics/playback/rtmp/hls/webrtc/srt/moq) are
    disabled since this fixture only needs raw RTSP.
    """

    def __init__(self, *, rtsp_port: int, path_names: Sequence[str]) -> None:
        binary = _require_tool("mediamtx")
        env: dict[str, str] = {
            "MTX_RTSPADDRESS": f":{rtsp_port}",
            # TCP-only: the default UDP RTP/RTCP listeners bind fixed ports
            # (:8000/:8001) regardless of MTX_RTSPADDRESS and collide with
            # unrelated services on shared hosts.
            "MTX_RTSPTRANSPORTS": "tcp",
            "MTX_LOGLEVEL": "error",
            "MTX_API": "no",
            "MTX_METRICS": "no",
            "MTX_PLAYBACK": "no",
            "MTX_RTMP": "no",
            "MTX_HLS": "no",
            "MTX_WEBRTC": "no",
            "MTX_SRT": "no",
            "MTX_MOQ": "no",
        }
        for name in path_names:
            env[f"MTX_PATHS_{name.upper()}_SOURCE"] = "publisher"
        self._rtsp_port = rtsp_port
        # mediamtx auto-generates a self-signed auto.crt/auto.key in its cwd
        # when it starts (even with TLS listeners disabled here). Without an
        # explicit cwd, that cwd is whatever the test process happened to be
        # run from -- the repo root for a local `pytest`/demo invocation --
        # littering the working tree with untracked cert files. Give it a
        # disposable directory of its own instead.
        self._cwd = tempfile.mkdtemp(prefix="mediamtx-e2e-")
        self._process = subprocess.Popen(  # noqa: S603 - fixed local test binary, no shell
            [binary],
            env={**_inherited_env(), **env},
            cwd=self._cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self._wait_ready(timeout=10.0)
        except Exception:
            self.stop()
            raise

    def rtsp_url(self, path_name: str) -> str:
        return f"rtsp://127.0.0.1:{self._rtsp_port}/{path_name}"

    def _wait_ready(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"mediamtx exited early with code {self._process.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", self._rtsp_port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("mediamtx did not open its RTSP listener in time")

    def stop(self) -> None:
        try:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
        finally:
            shutil.rmtree(self._cwd, ignore_errors=True)


@final
class FfmpegPublisher:
    """A synthetic lavfi test pattern, published over RTSP/TCP via libx264.

    Matches the worker's own CPU decode adapter's transport choice
    (``rtsp_transport;tcp`` in ``worker/adapters/decode/cpu_av/adapter.py``)
    so the fixture and the code under test agree on how the stream is
    carried.

    The nominal stream matches the supported operating contract: 30 fps with
    one keyframe per second (``-g``/``-keyint_min`` at the stream's own fps,
    ``-sc_threshold 0`` to stop scenecut-triggered keyframes from stretching
    that interval). That is what real cameras in this deployment emit, so it
    is what the reconnect/liveness contract is tuned against.

    ``gop_frames`` exists only so a test can *deliberately* build an
    unsupported long-GOP stream. Without an explicit override, libx264's
    default ~250-frame GOP means the worker's CPU decode adapter
    (``cv2.VideoCapture`` with a read timeout, see
    ``worker/adapters/decode/cpu_av/adapter.py``) can join mid-GOP and wait
    many seconds for the next IDR before it can decode a single frame. That
    condition is exercised as an explicit negative contract, never as the
    nominal path.
    """

    def __init__(
        self,
        *,
        url: str,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
        gop_frames: int | None = None,
    ) -> None:
        keyframe_interval = fps if gop_frames is None else gop_frames
        binary = _require_tool("ffmpeg")
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._process = subprocess.Popen(  # noqa: S603 - fixed local test binary, no shell
            [
                binary,
                "-nostdin",
                "-loglevel",
                "warning",
                "-re",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={width}x{height}:rate={fps}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                str(keyframe_interval),
                "-keyint_min",
                str(keyframe_interval),
                "-sc_threshold",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                url,
            ],
            env=_inherited_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._drain_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="e2e-ffmpeg-stderr-drain"
        )
        self._drain_thread.start()

    def _drain_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        for line in stderr:
            self._stderr_tail.append(line.rstrip())

    def ensure_alive(self, *, settle_sec: float = 1.0) -> None:
        time.sleep(settle_sec)
        if self._process.poll() is not None:
            tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                f"ffmpeg publisher exited early with code {self._process.returncode}:\n{tail}"
            )

    def stop(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        self._drain_thread.join(timeout=5.0)
        if self._process.stderr is not None:
            self._process.stderr.close()


def _inherited_env() -> dict[str, str]:
    return dict(os.environ)


def _require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(
            f"{name!r} is required on PATH for the real-stack E2E fixture "
            "(external RTSP tooling, not a production runtime dependency)"
        )
    return resolved


# --------------------------------------------------------------------------
# real relay backend: FastAPI app over uvicorn, recording ingest client
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordedAlert:
    event_type: AlertEventType
    detected_at: str
    probability: float
    edge_event_id: str | None
    audit: dict[str, object] | None
    snapshot_bytes: bytes | None
    clip_id: str | None


@final
class RecordingBackendIngestClient:
    """Implements ``BackendIngestClient`` and records everything it receives."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alerts: list[RecordedAlert] = []
        self._heartbeats = 0
        self._next_event_id = 0

    def send_alert(
        self,
        *,
        event_type: AlertEventType,
        detected_at: str,
        probability: float,
        audit: dict[str, object] | None = None,
        snapshot_bytes: bytes | None = None,
        clip_id: str | None = None,
    ) -> bool:
        with self._lock:
            self._alerts.append(
                RecordedAlert(
                    event_type, detected_at, probability, None, audit, snapshot_bytes, clip_id
                )
            )
        return True

    def send_alert_receipt(
        self,
        *,
        edge_event_id: str,
        event_type: AlertEventType,
        detected_at: str,
        probability: float,
        audit: dict[str, object] | None = None,
        snapshot_bytes: bytes | None = None,
        clip_id: str | None = None,
        on_accepted: Callable[[float], None] | None = None,
    ) -> EventReceipt:
        with self._lock:
            self._alerts.append(
                RecordedAlert(
                    event_type,
                    detected_at,
                    probability,
                    edge_event_id,
                    audit,
                    snapshot_bytes,
                    clip_id,
                )
            )
            self._next_event_id += 1
            event_id = f"backend-event-{self._next_event_id}"
        if on_accepted is not None:
            on_accepted(time.time())
        return EventReceipt(status="accepted", edge_event_id=edge_event_id, event_id=event_id)

    def send_heartbeat(self) -> bool:
        with self._lock:
            self._heartbeats += 1
        return True

    def snapshot_alerts(self) -> tuple[RecordedAlert, ...]:
        with self._lock:
            return tuple(self._alerts)

    @property
    def heartbeats(self) -> int:
        with self._lock:
            return self._heartbeats


@final
class LiveBackend:
    """A real ``backend.app`` FastAPI instance served by uvicorn in a thread.

    Uses ``create_app(lifespan=no_lifespan)`` plus manual ``app.state``
    seeding -- ``get_heartbeat_store``/``get_runtime_status_store`` document
    this exact pattern as the intended way to exercise relay/status routes
    without the real backend-config-polling lifespan.

    That same lazy-app.state pattern is also how ``get_runtime_status_store``
    and ``get_catalog_store`` resolve their storage paths: absent a
    pre-seeded ``app.state`` entry, both fall back to hardcoded
    ``/var/lib/ml-api/...`` defaults that a local macOS test process cannot
    write to (``RuntimeStatusStore``'s latency persist raises; the catalog
    degrades to "unavailable" instead). Pre-seeding both here, under the
    test's own ``state_dir``, avoids relying on either default.
    """

    def __init__(
        self,
        *,
        relay_token: str,
        camera_inventory: dict[str, dict[str, str | None]],
        state_dir: Path,
    ):
        self.ingest_client = RecordingBackendIngestClient()
        self.app = create_app(lifespan=no_lifespan)
        self.app.state.edge_relay_token = relay_token
        state_dir.mkdir(parents=True, exist_ok=True)
        # Camera binding is registry-only now (no camera_inventory fallback --
        # see _camera_binding_from_registry in relay/router.py), so each
        # inventory entry must be seeded into a CameraRegistryStore instead.
        # Shares state_dir/"catalog.sqlite3" with runtime_status_store/
        # catalog_store below, matching CameraRegistryStore.from_env()'s own
        # resolve_state_dir("ml-api")/"catalog.sqlite3" convention.
        registry = CameraRegistryStore(state_dir / "catalog.sqlite3")
        for camera_id, entry in camera_inventory.items():
            facility_id = entry.get("facility_id")
            registry.create(
                camera_id=camera_id,
                label=camera_id,
                rtsp_url=f"rtsp://camera/{camera_id}",
                space_id=facility_id if isinstance(facility_id, str) else None,
                status="online",
            )
        self.app.state.camera_registry = registry
        self.app.state.backend_ingest_client = self.ingest_client
        self.app.state.runtime_status_store = RuntimeStatusStore(
            latency_state_path=state_dir / "catalog.sqlite3"
        )
        self.app.state.catalog_store = CatalogStore.open(state_dir / "catalog.sqlite3")
        self.port = free_tcp_port()
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="off"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, daemon=True, name="e2e-live-backend"
        )
        self._thread.start()
        wait_until(lambda: self._server.started, timeout=10.0, what="live backend uvicorn startup")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def runtime_status_snapshot(self, facility_id: str) -> dict[str, object] | None:
        facilities = get_runtime_status_store(self.app).snapshot()["facilities"]
        return facilities.get(facility_id)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


# --------------------------------------------------------------------------
# scripted, deterministic model runners
# --------------------------------------------------------------------------


def _flat_pose(cx: float, cy: float) -> tuple[float, ...]:
    values: list[float] = []
    for offset in range(17):
        values.extend((cx + (offset % 3), cy + (offset % 5), 0.9))
    return tuple(values)


# Empirically-derived transition against the bed=(10,10,90,80) box and the
# module defaults (min_containment=0.35, hold_frames=2, grace_frames=3):
# steps 1-2 fully contain the person (ratio=1.0, establishes bed assignment
# by step 2); steps 3-4 slide partly out (ratio 0.71, 0.36 -- still >=0.35,
# resets grace); step 5 onward has zero containment (grace increments each
# frame, firing once grace_frames > 3, i.e. on the 4th consecutive frame
# outside). Each 25px step against a 70px-wide box keeps IoU with the prior
# frame at ~0.47, safely above GreedyIouTracker's default min_iou=0.3, so
# the same track_id survives the whole walk.
_BED_EXIT_XS: Final = (15.0, 15.0, 40.0, 65.0, 90.0, 90.0, 90.0, 90.0)
_BED_BOX: Final = (10.0, 10.0, 90.0, 80.0, 0.9)


@final
class _HoldingIndex:
    """Thread-safe monotonically-advancing index, clamped to a max value."""

    def __init__(self, *, maximum: int) -> None:
        self._maximum = maximum
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            value = min(self._value, self._maximum)
            self._value += 1
            return value


@final
class ScriptedBedExitPoseRunner:
    """Walks a person out of the bed region, holding the final position.

    Emits the walk through ``boxes`` alone, exactly the shape the real
    ``box_source="pose"`` production path produces: ``YoloPoseRunner``
    (``worker/adapters/model/yolo_pose.py``) reads both keypoints and boxes
    off the *same* Ultralytics forward pass -- ``_extract_boxes`` returns
    the model's own detection boxes, not something a separate "person"
    extractor has to supply. ``poses`` stays empty here because bed-exit
    tracking runs on person geometry alone and doesn't need keypoint
    signal; an empty-pose camera also can never fill the fall classifier's
    per-track buffer even if the fall domain were enabled for this camera.

    Issue #209: an earlier version of this fixture scripted the walk through
    a dedicated ``person`` runner and set ``box_source="person"`` to route to
    it. That made the test green through a configuration production can
    never reach -- nothing in this repo ever sets ``box_source`` away from
    its "pose" default (no env var, compose file, or example YAML does), so
    a test exercising "person" covers a path the shipped worker never runs.
    Scripting the walk through the pose runner's own ``boxes`` field instead
    means this scenario runs the exact box_source="pose" default production
    ships, and the scripted double now matches the real adapter's shape
    (boxes conditioned on detection, not hardcoded empty).
    """

    def __init__(self) -> None:
        self._index = _HoldingIndex(maximum=len(_BED_EXIT_XS) - 1)

    def __call__(self, _image: Image) -> RunnerResult:
        x1 = _BED_EXIT_XS[self._index.next()]
        return pose_result(poses=(), boxes=((x1, 15.0, x1 + 70.0, 75.0, 0.95),))

    def warmup(self) -> None:
        return None


@final
class ScriptedBedRunner:
    def __init__(self, *, box: tuple[float, float, float, float, float] | None) -> None:
        self._box = box

    def __call__(self, _image: Image) -> RunnerResult:
        return bed_result(boxes=() if self._box is None else (self._box,))

    def warmup(self) -> None:
        return None


@final
class ScriptedFallPersonRunner:
    """A static box: IoU=1.0 across frames keeps one stable track_id."""

    def __call__(self, _image: Image) -> RunnerResult:
        return person_result(boxes=((140.0, 90.0, 180.0, 150.0, 0.95),))

    def warmup(self) -> None:
        return None


@final
class ScriptedFallPoseRunner:
    def __call__(self, _image: Image) -> RunnerResult:
        return pose_result(
            poses=(_flat_pose(160.0, 120.0),),
            boxes=((140.0, 90.0, 180.0, 150.0, 0.95),),
        )

    def warmup(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class ScriptedFallModel:
    """Two rising edges over 8 steps, so the caller can assert cooldown /
    duplicate suppression collapses them into exactly one relayed alert."""

    def __init__(
        self,
        probabilities: Sequence[float] = (0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1),
        *,
        operating_threshold: float = 0.5,
    ) -> None:
        self.metadata = _FallMetadata()
        self.operating_threshold = operating_threshold
        self.warmup_count = 0
        self._probabilities = tuple(probabilities)
        self._index = _HoldingIndex(maximum=len(self._probabilities) - 1)

    def predict(self, _features: NDArray[np.float32]) -> float:
        return self._probabilities[self._index.next()]

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class ScriptedServingClient:
    """Dispatches by task name; one instance is scoped to a single camera run."""

    def __init__(
        self,
        *,
        pose: object,
        bed: object,
        fall: ScriptedFallModel,
        person: object | None = None,
    ) -> None:
        self._runners: dict[str, object] = {"pose": pose, "bed": bed, "fall": fall}
        if person is not None:
            self._runners["person"] = person

    def create(self, task: str, **_options: str | int | float | bool | None) -> object:
        try:
            return self._runners[task]
        except KeyError as error:
            raise AssertionError(f"unexpected serving task requested: {task!r}") from error


def bed_exit_serving_client() -> ScriptedServingClient:
    """No ``person`` runner: box_source stays at its production "pose"
    default (see ``ScriptedBedExitPoseRunner``'s docstring), so the "person"
    task is never requested here. Omitting it rather than wiring in an inert
    stand-in means a future regression that starts scheduling "person"
    unexpectedly fails loudly (``ScriptedServingClient.create`` raises)
    instead of silently returning a runner nobody scripted."""
    return ScriptedServingClient(
        pose=ScriptedBedExitPoseRunner(),
        bed=ScriptedBedRunner(box=_BED_BOX),
        fall=ScriptedFallModel(),
    )


def fall_serving_client() -> ScriptedServingClient:
    return ScriptedServingClient(
        pose=ScriptedFallPoseRunner(),
        person=ScriptedFallPersonRunner(),
        bed=ScriptedBedRunner(box=None),
        fall=ScriptedFallModel(),
    )


# --------------------------------------------------------------------------
# night-window helpers (BedExitConfig.night_window is checked against the
# real wall clock -- worker.py hardcodes clock=lambda: datetime.now(UTC))
# --------------------------------------------------------------------------


def night_window_including_now(*, tz: str = "UTC") -> dict[str, str]:
    now = datetime.now(ZoneInfo(tz))
    start = (now - timedelta(hours=1)).strftime("%H:%M")
    end = (now + timedelta(hours=1)).strftime("%H:%M")
    return {"start": start, "end": end, "tz": tz}


def night_window_excluding_now(*, tz: str = "UTC") -> dict[str, str]:
    now = datetime.now(ZoneInfo(tz))
    start = (now + timedelta(hours=3)).strftime("%H:%M")
    end = (now + timedelta(hours=4)).strftime("%H:%M")
    return {"start": start, "end": end, "tz": tz}


# --------------------------------------------------------------------------
# WorkerConfig / WorkerRuntime composition helpers
# --------------------------------------------------------------------------


def build_worker_config(
    *,
    relay_url: str,
    relay_token: str,
    camera_id: str,
    facility_id: str,
    rtsp_url: str,
    domains: dict[str, object],
    clip_enabled: bool = False,
    models: dict[str, object] | None = None,
) -> WorkerConfig:
    """Worker config for the real-stack e2e.

    ``clip_enabled`` mirrors the product default (off). Only the test that
    asserts on a finalized clip opts in, which keeps clip recording an explicit
    operating decision here exactly as it is in production.

    ``models`` defaults to ``None`` (``WorkerModelsConfig``'s own default,
    ``box_source="pose"`` -- the only value production ever resolves to; see
    ``ScriptedBedExitPoseRunner``'s docstring for why). Kept as an explicit
    override seam for scenarios that need something other than the
    production default, rather than hardcoding it away entirely.
    """
    return WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": relay_url, "token": relay_token},
            "clip": {"enabled": clip_enabled},
            "domains": domains,
            **({"models": models} if models is not None else {}),
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": facility_id,
                    "rtsp_url": rtsp_url,
                    "heartbeat_interval_sec": 5.0,
                    "frame_stride": 1,
                }
            ],
        }
    )


def fast_clip_recorder_config_factory(store_dir: Path) -> Callable[[], ClipRecorderConfig]:
    """Bare-name factory for ``worker_module.ClipRecorderConfig`` (module-attribute
    monkeypatch target) that shrinks pre/post-event padding + finalize grace
    so a bed-exit clip finalizes within the test's run window instead of
    production's default 30s/30s.

    ``disk_high_watermark`` is pinned near 1.0 rather than left at production's
    default 0.80: ``EvidenceRetention._over_watermark`` (evidence_retention.py)
    checks real ``shutil.disk_usage(store_dir)`` against that fraction on every
    rotate, and a dev laptop's boot volume can easily sit above 80% used for
    reasons that have nothing to do with this test's clip-store directory.
    Correctly, production then fails closed (``ClipAdmission.accept_event``
    refuses every new clip reservation once ``recording_suspended`` is set) --
    this is not a bug to route around, just host disk pressure this test's
    wiring proof does not intend to exercise.
    """

    def factory(**_kwargs: object) -> ClipRecorderConfig:
        # WorkerRuntime now calls ClipRecorderConfig(store_dir=...) itself
        # (see _resolved_clip_store_dir); accept and ignore that kwarg here
        # rather than the closed-over store_dir it already resolves to
        # (the runtime's injected CLIP_STORE_DIR is pinned to store_dir, and
        # this harness's WorkerConfig sets no clip.store_subdir), so this fixture
        # never has to track that call signature.
        return ClipRecorderConfig(
            store_dir=store_dir,
            segment_seconds=1.0,
            pre_event_seconds=0.5,
            post_event_seconds=0.5,
            fps=10.0,
            finalize_grace_seconds=0.5,
            stale_staging_seconds=60.0,
            rotate_min_interval_seconds=1.0,
            disk_high_watermark=0.999,
        )

    return factory


@dataclass(slots=True)
class WorkerRunHandle:
    runtime: WorkerRuntime
    camera: CameraRuntimeContext
    thread: threading.Thread


def _fall_model_via_serving_client(runtime: WorkerRuntime, _device: str) -> object:
    """Pin the pre-fail-closed fall model source for this harness.

    ``WorkerRuntime._create_fall_model`` now refuses to boot without an
    explicitly configured LSTM artifact directory (see its docstring). This
    harness has no real artifact on disk -- it exercises wiring/relay
    behavior with a scripted fall model, not fall-model accuracy -- so
    ``start_worker_runtime`` monkeypatches the method back to sourcing the
    fall runner from the injected ``ScriptedServingClient``, scoped to the
    lifetime of one worker run.
    """
    return runtime._serving.create("fall")  # noqa: SLF001


def start_worker_runtime(
    config: WorkerConfig,
    serving: ScriptedServingClient,
    *,
    state_dir: Path,
    clip_store_dir: Path,
) -> WorkerRunHandle:
    original_clip_recorder_config = worker_module.ClipRecorderConfig
    worker_module.ClipRecorderConfig = fast_clip_recorder_config_factory(clip_store_dir)
    original_create_fall_model = WorkerRuntime._create_fall_model
    WorkerRuntime._create_fall_model = _fall_model_via_serving_client  # type: ignore[method-assign]
    try:
        # Snapshot and evidence-delivery composition share one injected store.
        runtime = WorkerRuntime(
            config,
            env={
                "ML_WORKER_PROFILE": "cpu",
                CLIP_STORE_DIR_ENV: str(clip_store_dir),
            },
            serving_client=serving,
            acquire_lease=lambda: GpuLease.acquire(state_dir),
            hard_exit=lambda _code: None,
            state_dir=state_dir,
        )
        thread = threading.Thread(target=runtime.run, daemon=True, name="e2e-worker-runtime")
        thread.start()
        wait_until(lambda: len(runtime.cameras) > 0, timeout=20.0, what="camera activation")
    finally:
        worker_module.ClipRecorderConfig = original_clip_recorder_config
        WorkerRuntime._create_fall_model = original_create_fall_model  # type: ignore[method-assign]

    camera = runtime.cameras[0]
    return WorkerRunHandle(runtime, camera, thread)


def stop_worker_runtime(handle: WorkerRunHandle, *, timeout: float = 15.0) -> None:
    handle.runtime.stop()
    handle.thread.join(timeout=timeout)


__all__ = [
    "FfmpegPublisher",
    "LiveBackend",
    "MediaMtxProcess",
    "RecordedAlert",
    "RecordingBackendIngestClient",
    "ScriptedBedExitPoseRunner",
    "ScriptedBedRunner",
    "ScriptedFallModel",
    "ScriptedFallPersonRunner",
    "ScriptedFallPoseRunner",
    "ScriptedServingClient",
    "WorkerRunHandle",
    "bed_exit_serving_client",
    "build_worker_config",
    "fall_serving_client",
    "fast_clip_recorder_config_factory",
    "free_tcp_port",
    "night_window_excluding_now",
    "night_window_including_now",
    "start_worker_runtime",
    "stop_worker_runtime",
    "wait_until",
]
