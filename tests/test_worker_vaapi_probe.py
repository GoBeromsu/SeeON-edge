from __future__ import annotations

import subprocess

import pytest

from worker.adapters.decode.vaapi.errors import VaapiUnavailableError
from worker.adapters.decode.vaapi.models import VaapiConfig
from worker.adapters.decode.vaapi.probe import (
    ffprobe_args,
    ffprobe_binary,
    probe_stream_dimensions,
    probe_vaapi_capability,
)


def test_ffprobe_binary_swaps_the_ffmpeg_suffix() -> None:
    assert ffprobe_binary("/opt/ffmpeg/bin/ffmpeg") == "/opt/ffmpeg/bin/ffprobe"


def test_ffprobe_binary_falls_back_to_bare_ffprobe_for_a_nonstandard_name() -> None:
    assert ffprobe_binary("my-custom-decoder") == "ffprobe"


def test_ffprobe_args_forces_rtsp_over_tcp() -> None:
    config = VaapiConfig("camera-a", "rtsp://camera.local/live")

    args = ffprobe_args(config)

    assert "-rtsp_transport" in args
    assert args[args.index("-rtsp_transport") + 1] == "tcp"
    assert args[-1] == config.url


def test_probe_stream_dimensions_parses_the_first_video_stream() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        return '{"streams":[{"width":1920,"height":1080,"codec_name":"hevc"}]}'

    dimensions = probe_stream_dimensions(
        VaapiConfig("camera-a", "rtsp://camera.local/live"), runner=runner
    )

    assert dimensions.width == 1920
    assert dimensions.height == 1080
    assert dimensions.codec_name == "hevc"


def test_probe_stream_dimensions_sanitizes_a_raw_credentialed_url_on_failure() -> None:
    raw_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"

    def failing_runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=5.0)

    with pytest.raises(VaapiUnavailableError) as error:
        probe_stream_dimensions(VaapiConfig("camera-a", raw_url), runner=failing_runner)

    assert "operator" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert "plain" not in str(error.value)


def test_probe_stream_dimensions_rejects_unusable_metadata() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        return '{"streams":[]}'

    with pytest.raises(VaapiUnavailableError):
        probe_stream_dimensions(
            VaapiConfig("camera-a", "rtsp://camera.local/live"), runner=runner
        )


# -- probe_vaapi_capability: the boot-level VAAPI host-capability check ------


def test_probe_vaapi_capability_fails_closed_without_a_render_device() -> None:
    def unreachable_runner(args: tuple[str, ...], timeout_sec: float) -> str:
        raise AssertionError(f"ffmpeg must not run without a render device: {args}")

    capability = probe_vaapi_capability(
        render_device="/dev/dri/renderD128",
        runner=unreachable_runner,
        device_exists=lambda _path: False,
    )

    assert capability.available is False
    assert "renderD128" in capability.reason


def test_probe_vaapi_capability_true_when_device_init_succeeds() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        calls.append(args)
        return ""

    capability = probe_vaapi_capability(
        render_device="/dev/dri/renderD128",
        runner=runner,
        device_exists=lambda _path: True,
    )

    assert capability.available is True
    assert calls, "expected the VAAPI device-init command to actually run"
    (args,) = calls
    assert "-init_hw_device" in args
    assert args[args.index("-init_hw_device") + 1] == "vaapi=va:/dev/dri/renderD128"


def test_probe_vaapi_capability_catches_the_driver_missing_case_that_hwaccels_would_miss() -> None:
    """This is the specific trap the target edge node hit: ffmpeg's build has
    the vaapi hwaccel compiled in (so `-hwaccels` output would list it), but
    the iHD driver package is not installed, so the render device exists yet
    real device init still fails. A probe that only greps `-hwaccels` text
    would report a false positive here; this one must not."""

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        assert "-init_hw_device" in args
        raise subprocess.CalledProcessError(returncode=1, cmd=args)

    capability = probe_vaapi_capability(
        render_device="/dev/dri/renderD128",
        runner=runner,
        device_exists=lambda _path: True,
    )

    assert capability.available is False
    assert "driver" in capability.reason.lower()


def test_probe_vaapi_capability_false_when_ffmpeg_binary_is_missing() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        raise FileNotFoundError("no such file: ffmpeg")

    capability = probe_vaapi_capability(runner=runner, device_exists=lambda _path: True)

    assert capability.available is False
    assert capability.reason == "ffmpeg missing"


def test_probe_vaapi_capability_false_on_timeout() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout_sec)

    capability = probe_vaapi_capability(runner=runner, device_exists=lambda _path: True)

    assert capability.available is False
    assert "timed out" in capability.reason


def test_probe_vaapi_capability_false_on_unexpected_exception() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        raise ValueError("unexpected")

    capability = probe_vaapi_capability(runner=runner, device_exists=lambda _path: True)

    assert capability.available is False
    assert "ValueError" in capability.reason
