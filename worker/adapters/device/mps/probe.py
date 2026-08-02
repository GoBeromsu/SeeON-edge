"""Real torch/MPS capability probe for the ``mps`` device policy.

``WorkerRuntime._create_fall_model`` and ``compose_yolo_extractors``
(``worker/runtime/worker.py``) construct every model backend with
``device=boot.device`` -- when the resolved profile is ``mps`` that is the
literal string ``"mps"`` passed straight into ``torch``-backed adapters
(``worker/adapters/model/torch_lstm_fall.py``, ``yolo_*.py``). This probe
answers, before any lease is held or model is constructed, whether that
``device="mps"`` call can possibly succeed in this process -- it is the
composition root's real signal for the ``profile_device`` bootstrap stage
(``worker/runtime/bootstrap.py``) on the ``mps`` profile
(``worker/runtime/profile/registry.py``: ``PROFILE_REGISTRY``).

``probe_mps_capability`` returns ``available=True`` only when ``torch``
reports a usable MPS backend in *this* process (never a hypothetical target
install):

1. ``torch`` imports without error. A missing or non-macOS torch wheel makes
   every subsequent ``.to("mps")`` call fail regardless of the host's actual
   Apple Silicon GPU, so import failure alone is disqualifying.
2. ``torch.backends.mps.is_available()`` reports ``True``. This is torch's own
   runtime check (Metal device visible, macOS version supported) -- without
   it, model construction with ``device="mps"`` fails regardless of what
   ``torch.backends.mps.is_built()`` separately reports.

``is_built`` is not a gate on its own -- it is carried through for the
caller's diagnosis. In particular, a torch wheel built without MPS support
(``is_built()`` is ``False``, e.g. a Linux/CUDA wheel) makes ``is_available()``
``False`` for a different reason than a build that supports MPS but finds no
usable Metal device on this host (e.g. an unsupported macOS version).

A ``True`` result means "this process's torch build can initialize and use an
MPS device"; it does not, and cannot without constructing a real model, prove
that a specific model artifact loads or that inference is numerically correct
on that device -- that remains the warmup stage's concern
(``worker/runtime/bootstrap.py``: ``warmup_stage``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True, slots=True)
class MpsCapability:
    available: bool
    reason: str
    is_built: bool = False


# The probed ``torch`` module, typed loosely: only ``backends.mps.is_built``
# and ``backends.mps.is_available`` are actually read, and pinning a narrower
# Protocol here would not make the real ``torch`` package conform to it any
# more than duck typing already does.
TorchImporter: TypeAlias = Callable[[], Any]


def _import_torch() -> Any:
    import torch

    return torch


def probe_mps_capability(*, importer: TorchImporter = _import_torch) -> MpsCapability:
    """Real signal for whether this process can construct a ``device="mps"`` model.

    ``importer`` defaults to the real ``import torch`` and is injectable only
    so tests can exercise every branch (import failure, backend-probe
    exceptions, backend-visible-but-unusable) without needing real Apple
    Silicon hardware or a broken torch install -- mirrors the
    ``Cv2Importer``/``ProbeRunner`` injection pattern already used elsewhere
    in this package (``worker/adapters/decode/cpu_av/probe.py``,
    ``worker/adapters/decode/nvdec_cuvid/probe.py``,
    ``worker/adapters/device/cuda/probe.py``). Production callers never pass
    ``importer``.

    Never raises: every ``torch.backends.mps.*`` call is individually guarded
    so one backend probe failing (e.g. a build that answers ``is_built`` but
    not ``is_available``) cannot mask or crash out of the others.
    """
    try:
        torch = importer()
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency boundary
        return MpsCapability(available=False, reason=f"torch import failed: {type(exc).__name__}")

    try:
        is_built = bool(torch.backends.mps.is_built())
    except Exception:  # noqa: BLE001,S110 - build-flag probe must not break startup
        is_built = False

    try:
        available = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001 - backend probe must not break startup
        return MpsCapability(
            available=False,
            reason=f"torch.backends.mps.is_available() raised {type(exc).__name__}",
            is_built=is_built,
        )

    if available:
        return MpsCapability(available=True, reason="mps available", is_built=is_built)

    # Distinguish "torch wasn't built with MPS support at all" (e.g. a
    # Linux/CUDA wheel) from "MPS is built into this torch install but no
    # usable Metal device was found" (e.g. an unsupported macOS version) --
    # see the module docstring.
    if not is_built:
        reason = (
            "MPS not usable: torch.backends.mps.is_built() is False -- this torch "
            "install was not built with Metal Performance Shaders support"
        )
    else:
        reason = (
            "torch.backends.mps.is_available() is False despite MPS being built into "
            "this torch install -- no usable Metal device found on this host"
        )
    return MpsCapability(available=False, reason=reason, is_built=is_built)


__all__ = ["MpsCapability", "TorchImporter", "probe_mps_capability"]
