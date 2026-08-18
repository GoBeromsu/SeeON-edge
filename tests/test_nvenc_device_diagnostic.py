"""Todo 18: explicit diagnostic-CLI proof for the device-input NVENC seam.

Exercises ``worker.adapters.encode.nvenc_device.diagnostic`` exactly the way
an operator would: ``--help``, the truthful unavailable-host probe path (this
repo's dev/CI hosts have no NVIDIA hardware), the deterministic fake happy
path, and the fake invalid/bad-pressure path. No real hardware required.
"""

from __future__ import annotations

import pytest

from worker.adapters.decode.nvdec_device.capability import DeviceResidentCapability
from worker.adapters.device.cuda.probe import CudaCapability, NvencCapability
from worker.adapters.device.nvml.probe import NvmlGpuStatus
from worker.adapters.encode.nvenc_device import diagnostic
from worker.adapters.encode.nvenc_device.capability import DeviceInputNvencCapability
from worker.adapters.encode.nvenc_device.diagnostic import (
    BAD_PRESSURE_DID_NOT_FAIL_EXIT_CODE,
    PROBE_UNAVAILABLE_EXIT_CODE,
    _build_parser,
    main,
    run_fake_bad_pressure,
    run_fake_smoke_test,
    run_probe,
)


def test_help_exits_zero_and_documents_every_action(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--probe" in out
    assert "--fake-smoke-test" in out
    assert "--fake-bad-pressure" in out


def test_mutually_exclusive_flags_are_rejected() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--probe", "--fake-smoke-test"])


def test_default_action_and_explicit_probe_flag_agree() -> None:
    assert main([]) == main(["--probe"])


def _capability(*, available: bool, reason: str) -> DeviceInputNvencCapability:
    return DeviceInputNvencCapability(
        available=available,
        reason=reason,
        device_resident=DeviceResidentCapability(
            available=available,
            reason=reason,
            cuda=CudaCapability(available=available, reason=reason),
            nvml=NvmlGpuStatus(nvml_available=available, reason=reason),
            stream_event_supported=available,
            dlpack_supported=available,
        ),
        nvenc=NvencCapability(available, reason),
    )


def test_probe_reports_unavailable_capability_with_the_probe_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The negative path, proven by injection rather than by the host.

    This previously asserted that *this machine* has no NVIDIA hardware, which
    the module docstring even stated as a premise. That premise stopped being
    true and the test inverted (tests/AGENTS.md, Local Hero). Injecting the
    capability keeps the coverage and works on any host.
    """
    reason = "probe-negative-sentinel"
    monkeypatch.setattr(
        diagnostic,
        "probe_device_input_nvenc_capability",
        lambda: _capability(available=False, reason=reason),
    )

    exit_code = run_probe()

    out = capsys.readouterr().out
    assert exit_code == PROBE_UNAVAILABLE_EXIT_CODE
    assert "available: False" in out
    # The exact reason must survive to the operator, not merely the label.
    assert f"reason: {reason}" in out
    assert f"device_resident.available: False ({reason})" in out
    assert f"nvenc.available: False ({reason})" in out


def test_probe_reports_available_capability_with_a_zero_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The positive path, which nothing covered before.

    The old host-dependent test could only ever exercise one branch, and on the
    repo's assumed hardware that branch was always the negative one.
    """
    reason = "probe-positive-sentinel"
    monkeypatch.setattr(
        diagnostic,
        "probe_device_input_nvenc_capability",
        lambda: _capability(available=True, reason=reason),
    )

    exit_code = run_probe()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "available: True" in out
    assert f"reason: {reason}" in out


def test_cli_probe_flag_dispatches_to_run_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """--probe routes to run_probe and returns its value verbatim.

    Comparing main(["--probe"]) against run_probe() would pass even if both were
    wrong in the same way, and both depended on the host. A sentinel proves the
    routing itself.
    """
    sentinel = 77
    monkeypatch.setattr(diagnostic, "run_probe", lambda: sentinel)

    assert main(["--probe"]) == sentinel


def test_fake_smoke_test_happy_path_passes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_fake_smoke_test()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "fake smoke test: PASS" in out
    assert "selected: h264/mp4" in out
    assert "device_resident: True" in out
    assert "submissions_accepted=2" in out
    assert "d2h_transfers=1" in out


def test_cli_fake_smoke_test_flag_matches_direct_call() -> None:
    assert main(["--fake-smoke-test"]) == 0


def test_fake_bad_pressure_path_passes_and_proves_both_rejections(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_fake_bad_pressure()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "fake bad-pressure test: PASS" in out
    assert "host-input rejection (expected)" in out
    assert "backpressure rejection (expected)" in out


def test_cli_fake_bad_pressure_flag_matches_direct_call() -> None:
    assert main(["--fake-bad-pressure"]) == 0


def test_bad_pressure_exit_code_constant_is_distinct_from_probe_unavailable() -> None:
    assert BAD_PRESSURE_DID_NOT_FAIL_EXIT_CODE != PROBE_UNAVAILABLE_EXIT_CODE
