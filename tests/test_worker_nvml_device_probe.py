from __future__ import annotations

from typing import Any

import pytest

from worker.adapters.device.nvml.probe import NvmlGpuStatus, probe_nvml_gpu_status


class _FakePynvml:
    def __init__(
        self,
        *,
        init_error: Exception | None = None,
        device_count: int | Exception = 1,
        driver_version: str | Exception = "550.90.07",
        device_name: str | Exception = "Tesla T4",
    ) -> None:
        self._init_error = init_error
        self._device_count = device_count
        self._driver_version = driver_version
        self._device_name = device_name
        self.shutdown_called = False

    def nvmlInit(self) -> None:  # noqa: N802 - mirrors the real pynvml API name
        if self._init_error is not None:
            raise self._init_error

    def nvmlShutdown(self) -> None:  # noqa: N802 - mirrors the real pynvml API name
        self.shutdown_called = True

    def nvmlDeviceGetCount(self) -> int:  # noqa: N802 - mirrors the real pynvml API name
        if isinstance(self._device_count, Exception):
            raise self._device_count
        return self._device_count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:  # noqa: N802
        del index
        return object()

    def nvmlDeviceGetName(self, handle: object) -> str:  # noqa: N802
        del handle
        if isinstance(self._device_name, Exception):
            raise self._device_name
        return self._device_name

    def nvmlSystemGetDriverVersion(self) -> str:  # noqa: N802
        if isinstance(self._driver_version, Exception):
            raise self._driver_version
        return self._driver_version


def test_nvml_gpu_status_true_when_available() -> None:
    # Given -- a healthy NVML install: driver loads, one GPU enumerable
    fake = _FakePynvml(device_count=1, driver_version="550.90.07", device_name="Tesla T4")

    # When
    status = probe_nvml_gpu_status(importer=lambda: fake)

    # Then
    assert status == NvmlGpuStatus(
        nvml_available=True,
        reason="NVML reports a usable GPU device",
        driver_version="550.90.07",
        device_name="Tesla T4",
    )
    assert fake.shutdown_called is True


def test_nvml_gpu_status_false_when_pynvml_import_fails() -> None:
    # Given -- the real signal on this repo's macOS dev/CI machines: no
    # `pynvml` binding installed at all.
    def failing_importer() -> Any:
        raise ImportError("no module named pynvml")

    # When
    status = probe_nvml_gpu_status(importer=failing_importer)

    # Then -- fail closed, never raises past the probe boundary
    assert status.nvml_available is False
    assert "pynvml import failed" in status.reason
    assert status.driver_version is None
    assert status.device_name is None


def test_nvml_gpu_status_false_when_nvml_init_fails() -> None:
    # Given -- `pynvml` imports but the NVML shared library / driver isn't
    # present (e.g. `NVMLError_LibraryNotFound` on macOS).
    fake = _FakePynvml(init_error=RuntimeError("NVML Shared Library Not Found"))

    # When
    status = probe_nvml_gpu_status(importer=lambda: fake)

    # Then
    assert status.nvml_available is False
    assert "nvmlInit failed" in status.reason
    assert "NVML Shared Library Not Found" in status.reason
    # `nvmlShutdown` must never be called for an NVML handle that was never
    # successfully initialized.
    assert fake.shutdown_called is False


def test_nvml_gpu_status_false_when_no_devices_visible() -> None:
    # Given -- NVML initializes (driver-only host) but enumerates zero GPUs
    fake = _FakePynvml(device_count=0)

    # When
    status = probe_nvml_gpu_status(importer=lambda: fake)

    # Then
    assert status.nvml_available is False
    assert "no GPU devices are visible" in status.reason
    assert status.device_name is None
    # Driver version is still readable and carried through for diagnosis
    assert status.driver_version == "550.90.07"
    assert fake.shutdown_called is True


def test_nvml_gpu_status_false_when_device_count_query_raises() -> None:
    fake = _FakePynvml(device_count=RuntimeError("driver error"))

    # When
    status = probe_nvml_gpu_status(importer=lambda: fake)

    # Then
    assert status.nvml_available is False
    assert "nvmlDeviceGetCount failed" in status.reason
    assert fake.shutdown_called is True


def test_nvml_gpu_status_available_but_device_name_none_when_name_query_fails() -> None:
    # Given -- a device is visible and NVML is otherwise healthy, but the
    # name query itself fails; that must not mask the overall availability.
    fake = _FakePynvml(device_count=1, device_name=RuntimeError("name query failed"))

    # When
    status = probe_nvml_gpu_status(importer=lambda: fake)

    # Then
    assert status.nvml_available is True
    assert status.device_name is None
    assert status.driver_version == "550.90.07"


def test_nvml_gpu_status_driver_version_none_when_driver_version_query_fails() -> None:
    fake = _FakePynvml(device_count=1, driver_version=RuntimeError("driver query failed"))

    # When
    status = probe_nvml_gpu_status(importer=lambda: fake)

    # Then -- a failing driver-version read must not break the whole probe
    assert status.nvml_available is True
    assert status.driver_version is None
    assert status.device_name == "Tesla T4"


def test_probe_nvml_gpu_status_against_real_pynvml_is_honest_on_this_dev_machine() -> None:
    """No fakes: exercises the real ``import pynvml`` default on this host.

    This repo's macOS CI/dev machines have no NVML shared library installed,
    so the real probe must report ``nvml_available=False`` here -- a ``True``
    result on this host would be a false positive. Real-hardware verification
    on a Linux+NVIDIA host is deferred to the #128 hardware session.
    """
    pytest.importorskip("pynvml")

    status = probe_nvml_gpu_status()

    assert status.nvml_available is False
    assert status.reason
