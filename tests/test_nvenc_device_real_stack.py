"""Real-stack: device-input NVENC capability probe on real NVIDIA hardware.

Unlike ``test_nvenc_device_encoder.py`` and ``test_cuda_overlay_renderer.py``
(fakes only, deterministic, run everywhere including this repo's Apple
Silicon dev/CI hosts), this exercises the real combined
``probe_device_input_nvenc_capability`` path end to end -- the same two
checks (device-resident CUDA/stream/DLPack capability, plus an ffmpeg build
with ``h264_nvenc``) a real edge deployment would need before this
experimental profile's device-input NVENC seam could open a session at all.

Marked ``real_stack`` (see ``tests/AGENTS.md``) and deselected in CI via
``-m "not real_stack"``: the assertion (``capability.available is True``) is
only true on a real NVIDIA host with a working driver/CUDA/torch/ffmpeg
pairing, so it cannot run deterministically on every dev/CI machine the way
the fake-backed tests do. Skipped (not asserted false) when no NVIDIA GPU is
enumerable, since a negative result there is an environment fact, not a
regression -- mirrors ``test_nvidia_device_resident_real_stack.py``'s
skip-not-assert convention for the equivalent decode-side case.
"""

from __future__ import annotations

import pytest

from worker.adapters.device.nvml.probe import probe_nvml_gpu_status
from worker.adapters.encode.nvenc_device.capability import probe_device_input_nvenc_capability

pytestmark = pytest.mark.real_stack


def test_device_input_nvenc_capability_probe_on_real_nvidia_hardware() -> None:
    pytest.importorskip("torch")
    nvml_status = probe_nvml_gpu_status()
    if not nvml_status.nvml_available:
        pytest.skip(f"requires real NVIDIA hardware (NVML unavailable: {nvml_status.reason})")

    capability = probe_device_input_nvenc_capability()

    assert capability.available is True, capability.reason
    assert capability.device_resident.available is True
    assert capability.nvenc.available is True
