from __future__ import annotations

import subprocess

from worker.adapters.device.cuda.probe import NvencCapability, probe_nvenc_capability


def test_nvenc_capability_true_when_encoder_token_present() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        calls.append(args)
        return "V..... h264_nvenc            NVIDIA NVENC H.264 encoder\n"

    capability = probe_nvenc_capability(runner=runner)

    assert capability == NvencCapability(True, "ffmpeg h264_nvenc encoder is available")
    assert calls == [("ffmpeg", "-hide_banner", "-encoders")]


def test_nvenc_capability_false_when_encoder_token_absent() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        return "V..... libx264               libx264 H.264 / AVC encoder\n"

    # Fail closed: this repo's macOS dev machines hit exactly this path (real
    # `ffmpeg -encoders` output on this host has no h264_nvenc).
    capability = probe_nvenc_capability(runner=runner)

    assert capability == NvencCapability(False, "ffmpeg has no h264_nvenc encoder")


def test_nvenc_capability_matches_encoder_by_word_boundary_not_substring() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        return "This build was compiled without h264_nvenc_extra support tokens.\n"

    capability = probe_nvenc_capability(runner=runner)

    assert capability.available is False


def test_nvenc_capability_false_when_ffmpeg_binary_missing() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        raise FileNotFoundError("no such file: ffmpeg")

    capability = probe_nvenc_capability(runner=runner)

    assert capability == NvencCapability(False, "ffmpeg missing")


def test_nvenc_capability_false_when_ffmpeg_query_times_out() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        raise subprocess.TimeoutExpired(args, timeout_sec)

    capability = probe_nvenc_capability(runner=runner)

    assert capability == NvencCapability(False, "ffmpeg encoder probe timed out")


def test_nvenc_capability_false_when_ffmpeg_query_exits_non_zero() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del timeout_sec
        raise subprocess.CalledProcessError(1, args)

    capability = probe_nvenc_capability(runner=runner)

    assert capability == NvencCapability(False, "ffmpeg encoder probe failed with exit code 1")


def test_nvenc_capability_false_when_ffmpeg_query_raises_unexpected_error() -> None:
    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        raise ValueError("unexpected")

    # Encode probe must never break startup, even on an unmodeled error.
    capability = probe_nvenc_capability(runner=runner)

    assert capability == NvencCapability(False, "ffmpeg encoder probe failed: ValueError")
