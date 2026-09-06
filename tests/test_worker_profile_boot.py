from __future__ import annotations

from types import SimpleNamespace

import pytest

from worker.runtime.profile.registry import (
    DEFAULT_PROFILE_NAME,
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    ProfileError,
    requested_profile_name,
    select_profile,
)


def test_ml_worker_profile_env_name() -> None:
    assert ML_WORKER_PROFILE_ENV == "ML_WORKER_PROFILE"


def test_flow_is_the_only_production_profile() -> None:
    assert DEFAULT_PROFILE_NAME == "flow"
    assert tuple(PROFILE_REGISTRY) == ("flow",)
    assert requested_profile_name({}) == "flow"
    assert select_profile({}).spec is PROFILE_REGISTRY["flow"]


def test_retired_profile_refuses_to_start_with_adr_0002_diagnostic() -> None:
    try:
        select_profile({ML_WORKER_PROFILE_ENV: "cpu"})
    except ProfileError as error:
        assert str(error) == "ADR-0002: unsupported ML_WORKER_PROFILE 'cpu'; set flow"
    else:
        raise AssertionError("retired profile must refuse rather than fall back to flow")


def test_flow_device_residency_is_established_from_nvml_and_refuses_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retired nvidia profile proved residency by opening an NVDEC device.

    Under flow the SDK owns decode, so the parent's evidence is NVML naming a
    driver and a device. Losing that is a refusal to start, not a warning.
    """
    from worker.runtime import worker as worker_module

    monkeypatch.setattr(
        worker_module,
        "probe_nvml_gpu_status",
        lambda: SimpleNamespace(
            nvml_available=True,
            driver_version="580.65",
            device_name="NVIDIA RTX 5080",
            reason="",
        ),
    )
    verified = worker_module._production_device_resident_source()
    assert verified.ok
    assert "NVIDIA RTX 5080" in verified.reason
    assert "580.65" in verified.reason

    monkeypatch.setattr(
        worker_module,
        "probe_nvml_gpu_status",
        lambda: SimpleNamespace(
            nvml_available=False,
            driver_version=None,
            device_name=None,
            reason="NVML library not present",
        ),
    )
    refused = worker_module._production_device_resident_source()
    assert not refused.ok
    assert refused.reason == "NVML library not present"
