from __future__ import annotations

import math
from collections.abc import Callable
from typing import final

import cv2
import pytest

from worker.adapters.decode.nvdec_cuvid import (
    NvdecCuvidAdapter,
    NvdecCuvidConfig,
    NvdecUnavailableError,
    UnsupportedCodecError,
)
from worker.interfaces.decode import DecodeAdapter, DecodeSession
from worker.pipeline.ingest.probe import RTSPProbeError, probe_first_frame
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.runtime.profile import (
    BootDependencies,
    ProfileVerifyError,
    VerifyResult,
    resolve_boot_context,
)


@final
class _DecoderProcess:
    def __init__(self, payloads: list[bytes | None], returncode: int = 0) -> None:
        self._payloads = payloads
        self._returncode = returncode
        self.read_timeouts: list[float] = []
        self.reap_calls = 0

    def read_frame(self, timeout_sec: float) -> bytes | None:
        self.read_timeouts.append(timeout_sec)
        return self._payloads.pop(0) if self._payloads else None

    def reap(self, timeout_sec: float) -> int | None:
        del timeout_sec
        self.reap_calls += 1
        return self._returncode


def _probe_runner(codec_name: str = "h264") -> Callable[[tuple[str, ...], float], str]:
    def run(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        return (
            '{"streams":[{"width":2,"height":1,"codec_name":"'
            f'{codec_name}"}}]}}'
        )

    return run


def test_adapter_emits_writable_rgb_packets_with_decode_metrics_and_one_reap() -> None:
    # Given
    red_green = bytes((255, 0, 0, 0, 255, 0))
    blue_white = bytes((0, 0, 255, 255, 255, 255))
    process = _DecoderProcess([red_green, blue_white])
    spawn_calls: list[tuple[tuple[str, ...], int]] = []

    def spawn(args: tuple[str, ...], frame_size: int) -> _DecoderProcess:
        spawn_calls.append((args, frame_size))
        return process

    clock_values = iter((10.0, 10.004, 10.1, 10.106))
    adapter = NvdecCuvidAdapter(
        probe_runner=_probe_runner(),
        process_spawner=spawn,
        clock=lambda: next(clock_values),
    )
    config = NvdecCuvidConfig(
        camera_id="camera-a",
        url="rtsp://camera.local/live",
        read_timeout_ms=250,
    )

    # When
    session = adapter.open(config)
    first = session.read()
    second = session.read()
    session.close()
    session.close()

    # Then
    assert isinstance(adapter, DecodeAdapter)
    assert isinstance(session, DecodeSession)
    assert first is not None
    assert second is not None
    assert first.frame.image.tolist() == [[[255, 0, 0], [0, 255, 0]]]
    assert first.frame.image.flags.writeable
    first.frame.image[0, 0, 0] = 1
    assert (first.camera_id, first.seq, first.width, first.height) == (
        "camera-a",
        0,
        2,
        1,
    )
    assert first.pts == first.frame.time_sec == 0.0
    assert math.isclose(first.decode_time_ms, 4.0)
    assert second.seq == 1
    assert second.pts is not None
    assert math.isclose(second.pts, 0.102)
    assert second.frame.time_sec == second.pts
    assert math.isclose(second.decode_time_ms, 6.0)
    assert process.read_timeouts == [0.25, 0.25]
    assert process.reap_calls == 1
    args, frame_size = spawn_calls[0]
    hwaccel_index = args.index("-hwaccel")
    decoder_index = args.index("-c:v")
    assert frame_size == 6
    assert args[hwaccel_index : hwaccel_index + 2] == ("-hwaccel", "cuda")
    assert args[decoder_index : decoder_index + 2] == ("-c:v", "h264_cuvid")
    assert args[-3:] == ("-pix_fmt", "rgb24", "pipe:1")


def test_unsupported_codec_never_spawns_or_constructs_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    opencv_calls: list[str] = []
    spawn_calls: list[tuple[str, ...]] = []

    def video_capture(url: str) -> None:
        opencv_calls.append(url)

    def spawn(args: tuple[str, ...], frame_size: int) -> _DecoderProcess:
        del frame_size
        spawn_calls.append(args)
        return _DecoderProcess([])

    monkeypatch.setattr(cv2, "VideoCapture", video_capture)
    adapter = NvdecCuvidAdapter(
        probe_runner=_probe_runner("mpeg4"),
        process_spawner=spawn,
    )

    # When / Then
    with pytest.raises(UnsupportedCodecError):
        _ = adapter.open(NvdecCuvidConfig("camera-a", "rtsp://camera.local/live"))

    assert spawn_calls == []
    assert opencv_calls == []


def test_spawn_failure_is_sanitized_and_never_constructs_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    raw_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"
    opencv_calls: list[str] = []

    def video_capture(url: str) -> None:
        opencv_calls.append(url)

    def fail_spawn(args: tuple[str, ...], frame_size: int) -> _DecoderProcess:
        del args, frame_size
        raise OSError(raw_url)

    monkeypatch.setattr(cv2, "VideoCapture", video_capture)
    adapter = NvdecCuvidAdapter(
        probe_runner=_probe_runner(),
        process_spawner=fail_spawn,
    )

    # When
    with pytest.raises(NvdecUnavailableError) as error:
        _ = adapter.open(NvdecCuvidConfig("camera-a", raw_url))

    # Then
    assert str(error.value) == "ffmpeg spawn failed: OSError (returncode=None)"
    assert "operator" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert "plain" not in str(error.value)
    assert opencv_calls == []


def test_first_frame_failure_is_masked_camera_local_and_has_no_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    raw_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"
    processes = [_DecoderProcess([None]), _DecoderProcess([None])]
    opencv_calls: list[str] = []
    degraded: list[tuple[str, str]] = []

    def video_capture(url: str) -> None:
        opencv_calls.append(url)

    def spawn(args: tuple[str, ...], frame_size: int) -> _DecoderProcess:
        del args, frame_size
        return processes.pop(0)

    monkeypatch.setattr(cv2, "VideoCapture", video_capture)
    adapter = NvdecCuvidAdapter(
        probe_runner=_probe_runner(),
        process_spawner=spawn,
    )
    config = NvdecCuvidConfig("camera-bad", raw_url)
    source = RTSPSource(config, adapter, max_failures=1, max_total_reconnects=0)
    source.set_liveness_callbacks(
        on_reconnecting=lambda reason: degraded.append((config.camera_id, reason))
    )

    # When
    assert list(source) == []
    clock_values = iter((0.0, 0.0, 0.1))
    with pytest.raises(RTSPProbeError) as error:
        _ = probe_first_frame(
            raw_url,
            decoder=adapter,
            config=config,
            timeout_ms=50,
            monotonic=lambda: next(clock_values),
        )

    # Then
    assert degraded == [("camera-bad", "read_failure")]
    assert error.value.masked_url == (
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A"
    )
    assert "operator" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert "plain" not in str(error.value)
    assert opencv_calls == []


def test_cuda_profile_still_runs_decode_preflight_when_device_check_fails() -> None:
    """Issue #79 (track 2): device verification, decode preflight, and the
    legacy-conflict check are independent gates over the same resolved
    profile -- none depends on another's outcome, so a failed device check no
    longer skips the decode preflight. All gate failures are collected into
    one raised ``ProfileVerifyError`` instead of surfacing only the first."""
    # Given
    decode_calls: list[str] = []
    dependencies = BootDependencies(
        {
            "nvidia": lambda: VerifyResult(
                ok=False,
                profile="nvidia",
                stage="device",
                reason="CUDA unavailable",
            )
        }
    )

    def decode_probe(decode: str) -> VerifyResult:
        decode_calls.append(decode)
        return VerifyResult(True, "cuda", "decode", "unexpected")

    # When / Then
    with pytest.raises(ProfileVerifyError, match="CUDA unavailable"):
        _ = resolve_boot_context(
            {"ML_WORKER_PROFILE": "nvidia"},
            dependencies,
            decode_probe,
        )

    assert decode_calls == ["nvdec"]
