"""Real NVML GPU-telemetry probe for the `/status` `runtime.device` producer.

`WorkerDiagnostics.set_gpu_status` (`worker/runtime/telemetry/runtime_diagnostics.py`)
and its wire shape `RelayGpuPayload` (`worker/runtime/telemetry/wire.py:56-64`)
have existed since the relay wire schema was defined, but nothing in
production ever called `set_gpu_status` -- issue #132, the same failure class
as #124 (`runtime.cameras[*].decode.selected` staying permanently null because
no production code ever called `update_decode`). This probe is half of the fix:
it answers, using `nvidia-ml-py`'s `pynvml` binding (already declared in
`pyproject.toml`, imported nowhere before this), whether NVML can enumerate a
real NVIDIA GPU in this process. The composition root
(`worker/runtime/worker.py`) combines this probe's result with the existing
`probe_cuda_capability` (`worker/adapters/device/cuda/probe.py`) to build the
full `RelayGpuPayload` and calls `set_gpu_status` once at boot.

`probe_nvml_gpu_status` returns `nvml_available=True` only when:

1. `pynvml` imports without error. A missing NVML Python binding (this repo's
   macOS dev/CI machines never have it) makes every subsequent NVML call
   impossible regardless of the host's actual GPU.
2. `pynvml.nvmlInit()` succeeds. This is NVML's own runtime check that the
   shared library is present and the driver is loaded -- on macOS this always
   raises `NVMLError_LibraryNotFound` ("NVML Shared Library Not Found"),
   which is exactly the clear, never-raised-out-of-the-probe `nvml_error`
   this module reports.
3. At least one GPU device is enumerable via `nvmlDeviceGetCount()`. NVML can
   initialize on a driver-only host with zero attached devices; that is not
   the "a GPU is really here" signal the wire payload's `nvml_available`
   promises, so it is reported as unavailable with a diagnostic reason.

`driver_version`/`device_name` are carried through for diagnosis whenever they
can be read, independent of the boolean gates above -- e.g. a driver version
is still meaningful even if the specific device-name query fails.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True, slots=True)
class NvmlGpuStatus:
    nvml_available: bool
    reason: str
    driver_version: str | None = None
    device_name: str | None = None


# The probed ``pynvml`` module, typed loosely: only ``nvmlInit``,
# ``nvmlShutdown``, ``nvmlDeviceGetCount``, ``nvmlDeviceGetHandleByIndex``,
# ``nvmlDeviceGetName``, and ``nvmlSystemGetDriverVersion`` are actually read,
# and pinning a narrower Protocol here would not make the real ``pynvml``
# package conform to it any more than duck typing already does.
NvmlImporter: TypeAlias = Callable[[], Any]


def _import_pynvml() -> Any:
    import pynvml

    return pynvml


def probe_nvml_gpu_status(*, importer: NvmlImporter = _import_pynvml) -> NvmlGpuStatus:
    """Real signal for whether NVML can enumerate a usable NVIDIA GPU here.

    ``importer`` defaults to the real ``import pynvml`` and is injectable only
    so tests can exercise every branch (import failure, init failure,
    device-visible-but-unqueryable) without needing real NVIDIA hardware or an
    installed NVML shared library -- mirrors the ``TorchImporter`` injection
    pattern already used by ``probe_cuda_capability``/``probe_mps_capability``
    (``worker/adapters/device/cuda/probe.py``,
    ``worker/adapters/device/mps/probe.py``). Production callers never pass
    ``importer``.

    Never raises: every ``pynvml.*`` call is individually guarded and
    ``nvmlShutdown`` always runs (best-effort) once ``nvmlInit`` succeeds, so
    one query failing (e.g. a driver that initializes but cannot name a
    device) cannot mask the others or leak an unclosed NVML handle.
    """
    try:
        pynvml = importer()
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency boundary
        return NvmlGpuStatus(
            nvml_available=False,
            reason=f"pynvml import failed: {type(exc).__name__}: {exc}",
        )

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - NVML init must not break startup
        return NvmlGpuStatus(
            nvml_available=False,
            reason=f"nvmlInit failed: {type(exc).__name__}: {exc}",
        )

    try:
        return _read_gpu_status(pynvml)
    finally:
        # Shutdown failure must not mask a successful read.
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _read_gpu_status(pynvml: Any) -> NvmlGpuStatus:
    driver_version = _read_driver_version(pynvml)

    try:
        device_count = int(pynvml.nvmlDeviceGetCount())
    except Exception as exc:  # noqa: BLE001 - device-count query must not break startup
        return NvmlGpuStatus(
            nvml_available=False,
            reason=f"nvmlDeviceGetCount failed: {type(exc).__name__}: {exc}",
            driver_version=driver_version,
        )

    if device_count <= 0:
        return NvmlGpuStatus(
            nvml_available=False,
            reason="NVML initialized but no GPU devices are visible",
            driver_version=driver_version,
        )

    device_name = _read_first_device_name(pynvml)
    return NvmlGpuStatus(
        nvml_available=True,
        reason="NVML reports a usable GPU device",
        driver_version=driver_version,
        device_name=device_name,
    )


def _read_driver_version(pynvml: Any) -> str | None:
    try:
        return str(pynvml.nvmlSystemGetDriverVersion())
    except Exception:  # noqa: BLE001,S110 - driver-version query must not break startup
        return None


def _read_first_device_name(pynvml: Any) -> str | None:
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return str(pynvml.nvmlDeviceGetName(handle))
    except Exception:  # noqa: BLE001,S110 - device-name query must not break startup
        return None


__all__ = ["NvmlGpuStatus", "NvmlImporter", "probe_nvml_gpu_status"]
