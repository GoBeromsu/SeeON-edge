from __future__ import annotations

import math
from collections.abc import Callable
from typing import final

import pytest

from worker.adapters.decode.vaapi import (
    VaapiAdapter,
    VaapiConfig,
    VaapiUnavailableError,
)
from worker.interfaces.decode import DecodeAdapter, DecodeSession
from worker.pipeline.ingest.probe import RTSPProbeError, probe_first_frame
from worker.pipeline.ingest.rtsp import RTSPSource


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
    adapter = VaapiAdapter(
        probe_runner=_probe_runner(),
        process_spawner=spawn,
        clock=lambda: next(clock_values),
    )
    config = VaapiConfig(
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
    device_index = args.index("-hwaccel_device")
    assert frame_size == 6
    assert args[hwaccel_index : hwaccel_index + 2] == ("-hwaccel", "vaapi")
    assert args[device_index : device_index + 2] == (
        "-hwaccel_device",
        "/dev/dri/renderD128",
    )
    assert args[-3:] == ("-pix_fmt", "rgb24", "pipe:1")
    # No explicit `-c:v <codec>_vaapi` decoder selection -- ffmpeg
    # auto-negotiates the hardware decoder for whatever codec the stream
    # uses, unlike nvdec_cuvid's explicit `cuvid_decoder_for` mapping.
    assert "-c:v" not in args


def test_custom_render_device_is_threaded_into_the_ffmpeg_args() -> None:
    process = _DecoderProcess([])
    spawn_calls: list[tuple[str, ...]] = []

    def spawn(args: tuple[str, ...], frame_size: int) -> _DecoderProcess:
        del frame_size
        spawn_calls.append(args)
        return process

    adapter = VaapiAdapter(probe_runner=_probe_runner(), process_spawner=spawn)
    config = VaapiConfig(
        camera_id="camera-a",
        url="rtsp://camera.local/live",
        render_device="/dev/dri/renderD129",
    )

    _ = adapter.open(config)

    args = spawn_calls[0]
    device_index = args.index("-hwaccel_device")
    assert args[device_index : device_index + 2] == (
        "-hwaccel_device",
        "/dev/dri/renderD129",
    )


def test_spawn_failure_is_sanitized() -> None:
    # Given
    raw_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"

    def fail_spawn(args: tuple[str, ...], frame_size: int) -> _DecoderProcess:
        del args, frame_size
        raise OSError(raw_url)

    adapter = VaapiAdapter(
        probe_runner=_probe_runner(),
        process_spawner=fail_spawn,
    )

    # When
    with pytest.raises(VaapiUnavailableError) as error:
        _ = adapter.open(VaapiConfig("camera-a", raw_url))

    # Then
    assert str(error.value) == "ffmpeg spawn failed: OSError (returncode=None)"
    assert "operator" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert "plain" not in str(error.value)


def test_first_frame_failure_is_masked_camera_local_and_has_no_further_fallback(
) -> None:
    # Given -- once the boot-level VAAPI-vs-opencv decision has been made
    # (worker/runtime/profile/boot.py:resolve_decode_or_fallback), the
    # adapter itself never probes its way to a different backend
    # (worker/adapters/AGENTS.md): a per-camera read failure degrades that
    # camera only, exactly like nvdec_cuvid.
    raw_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"
    processes = [_DecoderProcess([None]), _DecoderProcess([None])]
    degraded: list[tuple[str, str]] = []

    def spawn(args: tuple[str, ...], frame_size: int) -> _DecoderProcess:
        del args, frame_size
        return processes.pop(0)

    adapter = VaapiAdapter(
        probe_runner=_probe_runner(),
        process_spawner=spawn,
    )
    config = VaapiConfig("camera-bad", raw_url)
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
