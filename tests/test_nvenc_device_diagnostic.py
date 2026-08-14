"""Todo 18: explicit diagnostic-CLI proof for the device-input NVENC seam.

Exercises ``worker.adapters.encode.nvenc_device.diagnostic`` exactly the way
an operator would: ``--help``, the truthful unavailable-host probe path (this
repo's dev/CI hosts have no NVIDIA hardware), the deterministic fake happy
path, and the fake invalid/bad-pressure path. No real hardware required.
"""

from __future__ import annotations

import pytest

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


def test_probe_path_is_truthful_and_unavailable_on_this_non_nvidia_host(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_probe()

    out = capsys.readouterr().out
    assert exit_code == PROBE_UNAVAILABLE_EXIT_CODE
    assert "available: False" in out
    assert "reason:" in out
    assert "device_resident.available" in out
    assert "nvenc.available" in out


def test_cli_probe_flag_matches_direct_call_exit_code() -> None:
    assert main(["--probe"]) == PROBE_UNAVAILABLE_EXIT_CODE


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
