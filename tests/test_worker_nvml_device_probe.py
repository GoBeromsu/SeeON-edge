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


def test_probe_nvml_gpu_status_against_real_pynvml_always_states_a_reason() -> None:
    """실제 ``import pynvml`` 기본 경로를 이 호스트에 대고 그대로 실행한다.

    호스트에 GPU 가 있는지는 단언하지 않는다. 그건 코드가 아니라 실행 머신의
    성질이고, 그렇게 쓴 테스트가 nvidia 프로파일 전환만으로 무더기로 뒤집혔다
    (tests/AGENTS.md 의 Local Hero 항목 참조). 검증하는 것은 어느 호스트에서든
    성립하는 계약 하나다: 프로브는 자기 판단의 사유를 반드시 댄다.

    available 값과 metadata 필드는 단언하지 않는다. available=True 여도
    metadata 가 None 일 수 있고(같은 파일의 가짜 주입 테스트가 그 계약을 고정한다),
    device_count/arch_list 는 게이트가 아니라 프로브가 보고하려는 진단값이다.
    음성 경로는 같은 파일의 가짜 주입 테스트가 덮는다.
    """
    pytest.importorskip("pynvml")

    status = probe_nvml_gpu_status()

    assert status.reason
