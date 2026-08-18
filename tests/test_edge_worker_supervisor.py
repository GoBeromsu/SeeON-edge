"""EdgeWorkerSupervisor -> IngestSupervisor migration.

Every edge assertion this file used to carry has one of four dispositions
(reported in full to the orchestrator alongside this change); this file keeps
only the one guarantee that has no existing worker-side test:

- one-offline-camera isolation, RTSP degraded->ready recovery, and the
  "runner never touches the capture thread" guarantee are all superseded by
  tests/test_worker_ingest_lifecycle.py and tests/test_worker_camera_pipeline_pump.py
  (IngestSupervisor wires ingest and pump loops as physically separate threads,
  so extraction structurally cannot run on the capture thread -- a stronger
  guarantee than edge's callback-based `test_runner_called_by_scheduler_only`).
- the two-bed-boxes-not-misread-as-pose-pair guarantee is now structural:
  worker/pipeline/analytics/merge.py dispatches by `module_name` through a
  typed `_RESULT_MERGERS` table (singledispatch registered against each
  concrete RunnerResult), so edge's isinstance-based branch-guessing bug class
  cannot recur.
- heartbeat gating (only a READY transition can reach the relay) is ported
  below directly against `HeartbeatReporter`, which now owns that guarantee.
- `test_supervisor_processing_updates_measured_fps` is ported below against
  `CameraPipelinePump._record_measured_fps` (worker/pipeline/camera_pipeline.py):
  a production gap closed under todo 20 -- nothing called
  `WorkerDiagnostics.update_measured_fps` per frame until the pump was wired
  to do so, mirroring edge's `_record_measured_fps` 10s sliding-window
  algorithm verbatim.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import final

import numpy as np
import pytest

import worker.runtime.worker as worker_module
from contracts.frame import Frame
from contracts.runner import pose_result
from shared.events.evidence_http_transport import HttpResult
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.worker import HeartbeatReporter
from worker.types import BusinessEvent, FramePacket, ModuleResult


def _config() -> tuple[WorkerConfig, CameraRuntimeConfig]:
    worker_config = WorkerConfig.model_validate(
        {
            "version": 3,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://example.test/camera-1",
                    "heartbeat_interval_sec": 30.0,
                }
            ],
        }
    )
    return worker_config, worker_config.cameras[0]


def test_heartbeat_reporter_sends_only_on_the_ready_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors edge's `_send_heartbeats` READY-only gate
    (edge/runtime/edge_worker_supervisor.py:147-154). Worker moved the same
    guarantee onto `HeartbeatReporter` itself: `mark_starting`/`mark_degraded`
    are no-ops (worker/runtime/worker.py:114-116,144-145), so only `mark_ready`
    can ever reach the relay.
    """
    requests: list[str] = []

    def bounded_request(
        url: str,
        _method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        requests.append(url)
        return 204, {}, b""

    monkeypatch.setattr(worker_module, "bounded_request", bounded_request)
    worker_config, camera = _config()
    reporter = HeartbeatReporter(worker_config, camera)

    reporter.mark_starting("camera-1")
    reporter.mark_degraded("camera-1", category="rtsp_reconnecting")
    assert requests == []

    reporter.mark_ready("camera-1")
    assert requests == ["http://relay.test/api/v1/relay/heartbeat"]


def test_heartbeat_reporter_logs_status_code_endpoint_and_hint_on_403(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #184: before this fix, a failed heartbeat only logged the bare
    string "relay heartbeat failed" -- no status code, no endpoint, and no
    way to tell a config error (#183's 403) from a transient one.
    """

    def bounded_request(
        _url: str,
        _method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        return 403, {}, b'{"detail":"facility mismatch"}'

    monkeypatch.setattr(worker_module, "bounded_request", bounded_request)
    worker_config, camera = _config()
    reporter = HeartbeatReporter(worker_config, camera)

    with caplog.at_level("WARNING", logger="worker.runtime.worker"):
        reporter.mark_ready("camera-1")

    assert reporter.failure_count == 1
    heartbeat_records = [r for r in caplog.records if "api/v1/relay/heartbeat" in r.message]
    assert len(heartbeat_records) == 1
    assert "403" in heartbeat_records[0].message
    assert "check runtime enrollment / auth" in heartbeat_records[0].message
    assert heartbeat_records[0].levelno == logging.ERROR
    # Response-body content never leaks into the log.
    assert "facility mismatch" not in heartbeat_records[0].message


@final
class _NoOpSink:
    def emit(self, event: BusinessEvent) -> None:
        del event


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    frame = Frame(index=seq, time_sec=float(seq), image=image)
    return FramePacket(camera_id, frame, float(seq), seq, 1, 1, 0.0)


def test_camera_pipeline_pump_updates_measured_fps_after_two_processed_frames() -> None:
    """Mirrors edge's `test_supervisor_processing_updates_measured_fps`
    (edge/runtime/edge_worker_supervisor.py sliding-fps window): worker moved
    the same guarantee onto `CameraPipelinePump._record_measured_fps`, called
    once per pumped frame ahead of the decision stage.
    """
    diagnostics = WorkerDiagnostics()
    diagnostics.register_decode("camera-1", "auto")
    bus = BoundedFrameBus()
    analytics = CompositeExtractor(
        extractors=(),
        scheduler=Scheduler(task_intervals={}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id="camera-1"),
    )
    decision = EventAggregator(deciders=(), incidents=IncidentManager())
    pump = CameraPipelinePump(
        "camera-1", bus.inference, analytics, decision, _NoOpSink(),
        diagnostics=diagnostics,
    )

    pose = ModuleResult("pose", pose_result((), ()), 0.0, "pose")
    pump._pump_one(_packet("camera-1", 0), pose)
    pump._pump_one(_packet("camera-1", 1), pose)

    camera = diagnostics.to_payload("facility-1", None, 1)["cameras"][0]
    assert camera["measured_fps"] is not None
