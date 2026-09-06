"""Real torch/CUDA capability probe for the ``cuda`` device policy.

``WorkerRuntime._create_fall_model`` and registry-based shared composition
(``worker/runtime/worker.py``) construct every model backend with
``device=boot.device`` -- when the resolved profile is ``cuda`` that is the
literal string ``"cuda"`` passed straight into ``torch``-backed adapters
(``yolo_*.py``). This probe
answers, before any lease is held or model is constructed, whether that
``device="cuda"`` call can possibly succeed in this process -- it is the
composition root's real signal for the ``profile_device`` bootstrap stage
(``worker/runtime/bootstrap.py``) on the ``cuda`` profile
(``worker/runtime/profile/registry.py``: ``PROFILE_REGISTRY``).

``probe_cuda_capability`` returns ``available=True`` only when ``torch``
reports a usable CUDA runtime in *this* process (never a hypothetical target
install):

1. ``torch`` imports without error. A missing or CPU-only torch wheel makes
   every subsequent ``.to("cuda")`` call fail regardless of the host's actual
   GPU, so import failure alone is disqualifying.
2. ``torch.cuda.is_available()`` reports ``True``. This is torch's own
   runtime check (driver present, CUDA initializes, at least one device
   visible) -- without it, model construction with ``device="cuda"`` fails
   or silently runs incorrect ops regardless of what ``torch.cuda.device_count()``
   or ``get_arch_list()`` separately report.

``device_count`` and ``arch_list`` are not gates on their own -- they are
carried through for the caller's diagnosis. In particular, ``device_count > 0``
with an empty ``arch_list`` is the documented failure mode of a torch wheel
built without the host GPU's compiled architecture (e.g. a pre-Blackwell
wheel on an sm_120 card): NVML sees the device, but no kernel can run on it,
so ``is_available()`` is ``False`` even though hardware is present.

A ``True`` result means "this process's torch build can initialize and use a
CUDA device"; it does not, and cannot without constructing a real model,
prove that a specific model artifact loads or that inference is numerically
correct on that device -- that remains the warmup stage's concern
(``worker/runtime/bootstrap.py``: ``warmup_stage``).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class CudaCapability:
    available: bool
    reason: str
    device_count: int = 0
    arch_list: tuple[str, ...] = ()


# The probed ``torch`` module, typed loosely: only ``cuda.get_arch_list``,
# ``cuda.device_count``, and ``cuda.is_available`` are actually read, and
# pinning a narrower Protocol here would not make the real ``torch`` package
# conform to it any more than duck typing already does.
TorchImporter: TypeAlias = Callable[[], Any]


def _import_torch() -> Any:
    import torch

    return torch


def probe_cuda_capability(*, importer: TorchImporter = _import_torch) -> CudaCapability:
    """Real signal for whether this process can construct a ``device="cuda"`` model.

    ``importer`` defaults to the real ``import torch`` and is injectable only
    so tests can exercise every branch (import failure, backend-probe
    exceptions, device-visible-but-unusable) without needing a real GPU or a
    broken torch install -- mirrors the ``Cv2Importer``/``ProbeRunner``
    injection pattern already used elsewhere in this package
    (``Flow media plane/cpu_av/probe.py``,
    ``Flow media plane/nvdec_cuvid/probe.py``). Production callers never
    pass ``importer``.

    Never raises: every ``torch.cuda.*`` call is individually guarded so one
    backend probe failing (e.g. a driver that answers ``get_arch_list`` but
    not ``device_count``) cannot mask or crash out of the others.
    """
    try:
        torch = importer()
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency boundary
        return CudaCapability(available=False, reason=f"torch import failed: {type(exc).__name__}")

    try:
        arch_list = tuple(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001,S110 - arch probe must not break startup
        arch_list = ()

    try:
        device_count = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001,S110 - device-count probe must not break startup
        device_count = 0

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - backend probe must not break startup
        return CudaCapability(
            available=False,
            reason=f"torch.cuda.is_available() raised {type(exc).__name__}",
            device_count=device_count,
            arch_list=arch_list,
        )

    if available:
        return CudaCapability(
            available=True,
            reason="cuda available",
            device_count=device_count,
            arch_list=arch_list,
        )

    # GPU visible via NVML (device_count>0) but not usable, and no compiled
    # kernels: this is the documented root cause (broken/CPU-only torch wheel)
    # -- see the module docstring.
    if device_count > 0 and not arch_list:
        reason = (
            f"CUDA not usable: {device_count} device(s) visible but torch build has no "
            "compiled arch kernels (empty arch_list) -- likely a torch wheel without the "
            "required GPU architecture (e.g. Blackwell sm_120)"
        )
    elif device_count == 0:
        reason = "torch.cuda.is_available() is False and no CUDA devices are visible"
    else:
        reason = (
            f"torch.cuda.is_available() is False despite {device_count} device(s) visible "
            f"(arch_list={list(arch_list)})"
        )
    return CudaCapability(
        available=False, reason=reason, device_count=device_count, arch_list=arch_list
    )


# Matches the `h264_nvenc` encoder token in `ffmpeg -encoders` output without
# false-matching unrelated substrings (mirrors `_CUVID_DECODER_PATTERN` in
# Flow media plane/nvdec_cuvid/probe.py).
_NVENC_ENCODER_PATTERN: Final = re.compile(r"\bh264_nvenc\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class NvencCapability:
    available: bool
    reason: str


class FfmpegQueryRunner(Protocol):
    def __call__(self, args: tuple[str, ...], timeout_sec: float, /) -> str: ...


def run_ffmpeg_encoders_query(args: tuple[str, ...], timeout_sec: float) -> str:
    """Run a bounded `ffmpeg -encoders` query, returning combined stdout+stderr.

    Same rationale as `run_ffmpeg_query` in
    Flow media plane/nvdec_cuvid/probe.py: some ffmpeg builds report
    capability listings on stderr rather than stdout, so both streams are
    combined rather than relying on just one.
    """
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
    return f"{completed.stdout}\n{completed.stderr}"


def probe_nvenc_capability(
    ffmpeg_bin: str = "ffmpeg",
    *,
    timeout_sec: float = 3.0,
    runner: FfmpegQueryRunner = run_ffmpeg_encoders_query,
) -> NvencCapability:
    """Never-raise ffmpeg-build capability probe for the `h264_nvenc` encoder.

    This is the encode-side counterpart to `probe_nvdec_cuvid_capability`
    (Flow media plane/nvdec_cuvid/probe.py), and answers a question that
    `probe_cuda_capability` above deliberately does not: `torch.cuda.is_available()`
    only proves a CUDA *device* is usable for model inference, it says nothing
    about whether the `ffmpeg` binary on `ffmpeg_bin` was built with NVENC
    support -- driver/build mismatches are a documented, independent failure
    mode (#53), so a host can have working CUDA and still have no `h264_nvenc`
    encoder available.

    Checks `ffmpeg -hide_banner -encoders` for an `h264_nvenc` token (matched
    with a word-boundary regex, not a bare substring). Never touches a camera,
    a GPU device, or spawns a real encode -- this is purely "does this ffmpeg
    build claim to support h264_nvenc at all"; whether a session can actually
    be opened (e.g. consumer-GPU concurrent-session limits) is a separate,
    per-camera runtime concern handled by `FFmpegSegmentEncoder`
    (Flow media plane/ffmpeg_segment_encoder.py).

    Returns `available=False` -- never a guess, never raises -- when the
    binary is missing, the query times out, exits non-zero, or the encoder
    token is absent (e.g. this repo's macOS dev machines, whose ffmpeg builds
    have no NVENC support), so the boot-time preflight
    (`worker/runtime/profile/boot.py:resolve_encode_or_fallback`) can fail
    open to `libx264` instead of aborting boot.
    """
    try:
        encoders = runner((ffmpeg_bin, "-hide_banner", "-encoders"), timeout_sec)
    except FileNotFoundError:
        return NvencCapability(False, "ffmpeg missing")
    except subprocess.TimeoutExpired:
        return NvencCapability(False, "ffmpeg encoder probe timed out")
    except subprocess.CalledProcessError as error:
        return NvencCapability(
            False, f"ffmpeg encoder probe failed with exit code {error.returncode}"
        )
    except Exception as error:  # noqa: BLE001 - encode probe must never break startup
        return NvencCapability(False, f"ffmpeg encoder probe failed: {type(error).__name__}")

    if _NVENC_ENCODER_PATTERN.search(encoders):
        return NvencCapability(True, "ffmpeg h264_nvenc encoder is available")
    return NvencCapability(False, "ffmpeg has no h264_nvenc encoder")


__all__ = [
    "CudaCapability",
    "FfmpegQueryRunner",
    "NvencCapability",
    "TorchImporter",
    "probe_cuda_capability",
    "probe_nvenc_capability",
    "run_ffmpeg_encoders_query",
]
