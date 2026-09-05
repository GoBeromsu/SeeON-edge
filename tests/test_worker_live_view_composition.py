from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from contracts.runner import Image, bed_result
from worker.adapters.deepstream.service_maker import DeepStreamFlowStopTimeout
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.worker import WorkerRuntime

_JPEG = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))[1].tobytes()


class _FlowPlane:
    def __init__(self, stop_error: Exception | None = None) -> None:
        self.stop_error = stop_error
        self.snapshot_calls: list[str] = []

    def clean_snapshot(self, camera_id: str) -> bytes:
        self.snapshot_calls.append(camera_id)
        return _JPEG

    def stop(self) -> None:
        if self.stop_error is not None:
            raise self.stop_error


class _BedRunner:
    def __call__(self, _image: Image) -> object:
        return bed_result([(1, 2, 12, 14, 0.9, [[1, 2], [12, 2], [12, 14], [1, 14]])])


class _ServingClient:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, str | None]] = []

    def create(self, task: str, **options: object) -> object:
        device = options.get("device")
        self.create_calls.append((task, device if isinstance(device, str) else None))
        if task != "bed" or device != "cpu":
            raise AssertionError(f"unexpected model request: task={task!r}, device={device!r}")
        return _BedRunner()


class _FallModel:
    def predict(self, _window: object) -> tuple[float, float, float]:
        return (0.1, 0.2, 0.7)


def _config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://example.test/camera-a",
                }
            ],
        }
    )


def test_flow_live_view_injects_bed_recognizer_and_recognize_request_reaches_it(
    tmp_path: Path, monkeypatch: object
) -> None:
    serving = _ServingClient()
    runtime = WorkerRuntime(
        _config(),
        env={"ML_WORKER_PROFILE": "flow"},
        serving_client=serving,
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
        state_dir=tmp_path,
    )
    runtime._boot = SimpleNamespace(profile=SimpleNamespace(name="flow"))  # noqa: SLF001
    runtime._mjpeg_config = MjpegServerConfig(  # noqa: SLF001
        enabled=True, host="127.0.0.1", port=0, probe_token="relay-token"
    )
    fall_model = _FallModel()
    runtime.fall_model = fall_model
    runtime._live_frames = LatestFrameStore()  # noqa: SLF001
    runtime._live_frames.publish_jpeg("camera-a", _JPEG, frame_index=1)  # noqa: SLF001
    plane = _FlowPlane()
    runtime._flow_media_plane = plane  # noqa: SLF001
    captured: dict[str, object] = {}

    def start_server(
        store: LatestFrameStore,
        config: MjpegServerConfig,
        *,
        probe: object = None,
        bed_zone_recognizer: object = None,
        replay_fall_model: object = None,
        bed_zone_snapshot: object = None,
    ) -> MjpegServer:
        captured["bed_zone_recognizer"] = bed_zone_recognizer
        captured["replay_fall_model"] = replay_fall_model
        server = MjpegServer(
            store,
            config,
            probe=probe,
            bed_zone_recognizer=bed_zone_recognizer,
            replay_fall_model=replay_fall_model,
            bed_zone_snapshot=bed_zone_snapshot,
        )
        server.start()
        return server

    monkeypatch.setattr("worker.runtime.worker.start_optional_mjpeg_server", start_server)

    runtime._start_live_view_server()  # noqa: SLF001
    assert captured["bed_zone_recognizer"] is not None
    assert captured["replay_fall_model"] is fall_model
    server = runtime._mjpeg_server  # noqa: SLF001
    assert server is not None
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/overlay/camera-a/bed-zone/recognize",
            data=b"{}",
            headers={"X-Edge-Relay-Token": "relay-token"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        assert response_payload == {
            "polygon": [[1, 2], [12, 2], [12, 14], [1, 14]],
            "image_width": 16,
            "image_height": 16,
        }
        assert serving.create_calls == [("bed", "cpu")]
        assert plane.snapshot_calls == ["camera-a"]
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    ("stop_error", "exit_codes"),
    [
        (None, []),
        (DeepStreamFlowStopTimeout("deadline exceeded"), [4]),
        (RuntimeError("unrelated stop error"), []),
    ],
)
def test_only_flow_stop_deadline_uses_terminal_exit(
    tmp_path: Path, stop_error: Exception | None, exit_codes: list[int]
) -> None:
    actual_exits: list[int] = []
    runtime = WorkerRuntime(
        _config(),
        env={"ML_WORKER_PROFILE": "flow"},
        serving_client=_ServingClient(),
        state_dir=tmp_path,
        hard_exit=actual_exits.append,
    )
    plane = _FlowPlane(stop_error)
    runtime._flow_media_plane = plane  # noqa: SLF001
    if stop_error is None:
        runtime._stop_flow_media_plane()  # noqa: SLF001
        assert runtime._flow_media_plane is None  # noqa: SLF001
    else:
        with pytest.raises(type(stop_error)) as error:
            runtime._stop_flow_media_plane()  # noqa: SLF001
        assert error.value is stop_error
        assert runtime._flow_media_plane is plane  # noqa: SLF001
    assert actual_exits == exit_codes
