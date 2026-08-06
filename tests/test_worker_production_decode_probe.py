"""Tests for the composition root's real decode_probe wiring.

Before this wiring existed, ``WorkerRuntime`` threaded ``decode_probe=None``
straight through to bootstrap, which fails closed with "decode capability
probe is not configured" on every profile -- the worker refused to start
under cuda, mps, *and* cpu. ``production_decode_probe`` (worker/runtime/
worker.py) is the real, hardware-checking default that fixes that while
preserving fail-closed behavior for backends that really are unsupported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import worker.runtime.worker as worker_module
from worker.adapters.decode.cpu_av.probe import OpenCvCapability
from worker.adapters.decode.nvdec_cuvid.probe import NvdecCapability
from worker.adapters.decode.vaapi.probe import VaapiCapability
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime, production_decode_probe


class _FakeServingClient:
    def create(self, task: str) -> object:
        raise AssertionError(f"unexpected serving client call for task {task!r}")


def _config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://example.test/camera-a",
                    "heartbeat_interval_sec": 30.0,
                }
            ],
        }
    )


def test_production_decode_probe_opencv_true_when_capability_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(
        worker_module,
        "probe_opencv_ffmpeg_capability",
        lambda: OpenCvCapability(True, "OpenCV is available"),
    )

    # When
    result = production_decode_probe("opencv")

    # Then
    assert result == VerifyResult(True, "cpu", "decode", "OpenCV is available")


def test_production_decode_probe_opencv_false_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(
        worker_module,
        "probe_opencv_ffmpeg_capability",
        lambda: OpenCvCapability(False, "no module named cv2"),
    )

    # When
    result = production_decode_probe("opencv")

    # Then
    assert result.ok is False
    assert result.reason == "no module named cv2"


def test_production_decode_probe_nvdec_true_when_capability_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(
        worker_module,
        "probe_nvdec_cuvid_capability",
        lambda: NvdecCapability(True, "ffmpeg CUDA/CUVID hwaccel is available"),
    )

    # When
    result = production_decode_probe("nvdec")

    # Then
    assert result == VerifyResult(True, "cuda", "decode", "ffmpeg CUDA/CUVID hwaccel is available")


def test_production_decode_probe_nvdec_false_fails_closed_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given -- the real signal on this repo's macOS dev machines: no CUDA/CUVID
    # hwaccel and no *_cuvid decoder (confirmed via real `ffmpeg -hwaccels`/
    # `-decoders` on this host).
    monkeypatch.setattr(
        worker_module,
        "probe_nvdec_cuvid_capability",
        lambda: NvdecCapability(
            False, "ffmpeg has no CUDA/CUVID/NVDEC hwaccel or *_cuvid decoder"
        ),
    )

    # When
    result = production_decode_probe("nvdec")

    # Then
    assert result.ok is False
    assert result.profile == "cuda"
    assert result.reason == "ffmpeg has no CUDA/CUVID/NVDEC hwaccel or *_cuvid decoder"


def test_production_decode_probe_vaapi_true_when_capability_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(
        worker_module,
        "probe_vaapi_capability",
        lambda: VaapiCapability(True, "VAAPI device init succeeded on /dev/dri/renderD128"),
    )

    # When
    result = production_decode_probe("vaapi")

    # Then
    assert result == VerifyResult(
        True, "igpu", "decode", "VAAPI device init succeeded on /dev/dri/renderD128"
    )


def test_production_decode_probe_vaapi_false_fails_closed_without_render_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given -- the target edge node's documented pre-Dockerfile-fix state:
    # ffmpeg's VAAPI hwaccel API is compiled in, but the iHD driver package is
    # not installed, so device init fails even though `-hwaccels` would list
    # "vaapi". This is exactly what production_decode_probe must catch.
    monkeypatch.setattr(
        worker_module,
        "probe_vaapi_capability",
        lambda: VaapiCapability(
            False,
            "VAAPI device init failed with exit code 1 (iHD/i965 driver likely missing)",
        ),
    )

    # When
    result = production_decode_probe("vaapi")

    # Then
    assert result.ok is False
    assert result.profile == "igpu"
    assert result.reason == (
        "VAAPI device init failed with exit code 1 (iHD/i965 driver likely missing)"
    )


def test_production_decode_probe_unknown_decode_fails_closed() -> None:
    # When
    result = production_decode_probe("unknown-backend")

    # Then -- default_decode_probe's existing "not configured" fail-closed path
    assert result.ok is False
    assert result.reason == "unknown-backend capability probe is not configured"


def test_worker_runtime_defaults_decode_probe_to_production_probe(tmp_path: Path) -> None:
    """WorkerRuntime must wire the real probe itself when none is injected.

    Every test elsewhere in this suite injects a stub ``decode_probe`` on
    purpose; this test is the one place that must *not*, to prove the
    composition root's own default is the real one and not ``None``
    threaded through to bootstrap's fail-closed fallback.
    """
    runtime = WorkerRuntime(
        _config(),
        serving_client=_FakeServingClient(),
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
    )

    assert runtime._decode_probe is production_decode_probe  # noqa: SLF001
