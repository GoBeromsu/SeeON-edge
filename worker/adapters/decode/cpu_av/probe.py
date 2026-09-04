"""Real OpenCV decode-capability probe for the ``opencv`` decode backend.

This probe answers, before any camera is allowed to start, whether this
process can actually decode RTSP through the backend ``CpuAvAdapter`` demands.
It is the composition root's real signal for the ``decode_capability``
bootstrap stage (``worker/runtime/bootstrap.py``) on the ``cpu`` and ``mps``
profiles, both of which resolve to the ``opencv`` decode policy
(``worker/runtime/profile/registry.py``: ``PROFILE_REGISTRY``).

``CpuAvAdapter`` (``worker/adapters/decode/cpu_av/adapter.py``) always opens
an RTSP source via ``cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)``. The
adapter therefore requires two things, and this probe checks both:

1. ``cv2`` imports in this process at all. A missing or broken native OpenCV
   install makes every subsequent ``cv2.VideoCapture`` call fail regardless of
   build flags.
2. That ``cv2`` build actually carries the FFMPEG video I/O backend, queried
   through OpenCV's own registry: ``cv2.videoio_registry.hasBackend(
   cv2.CAP_FFMPEG)``.

The second check is deliberate and required by ADR-0003 (explicit fallback
only). Earlier this probe checked import alone, mirroring the pre-migration
``edge`` reference. That was a false positive: an OpenCV build without FFMPEG
passed the global decode preflight and then failed on every single camera
open, which is exactly the "degraded path selected silently" failure this
repository has already shipped twice. Judging on the backend registry keeps
preflight honest.

Fail-closed contract: ``available=True`` is returned only when the import
succeeds **and** the registry positively reports the FFMPEG backend. A
``False`` answer, a missing ``videoio_registry`` or ``CAP_FFMPEG`` attribute,
and any exception raised while querying are all reported as
``available=False`` with an explicit, sanitized reason. The probe never falls
back to another backend and never treats an unanswerable query as available.

A ``True`` result means "this process can decode through OpenCV's FFMPEG
backend"; it does not, and cannot without a live URL, prove any particular
camera stream is reachable -- that remains a per-camera concern handled when
the ingest loop opens its capture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True, slots=True)
class OpenCvCapability:
    available: bool
    reason: str


# The probed ``cv2`` module, typed loosely: only the import and the registry
# query are exercised, and pinning a narrower Protocol here would not make the
# real ``cv2`` extension module conform to it any more than duck typing does.
Cv2Importer: TypeAlias = Callable[[], Any]


def _import_cv2() -> Any:
    import cv2

    return cv2


def probe_opencv_ffmpeg_capability(*, importer: Cv2Importer = _import_cv2) -> OpenCvCapability:
    """Real signal for whether this process can decode via OpenCV + FFMPEG.

    ``importer`` defaults to the real ``import cv2`` and is injectable only so
    tests can exercise the failure branches without needing a broken OpenCV
    install -- mirrors the ``ProbeRunner``/``CaptureFactory`` injection pattern
    already used elsewhere in this package
    (``worker/adapters/decode/cpu_av/adapter.py``,
    ``worker/adapters/decode/nvdec_cuvid/probe.py``). Production callers never
    pass ``importer``.

    Catches ``Exception`` broadly (not just ``ImportError``): native extension
    imports can fail with platform-specific errors (e.g. ``OSError`` from a
    broken shared library), and this probe must never itself crash the
    bootstrap sequence. Per ADR-0003 every such failure is reported as an
    explicit unavailable reason rather than degraded into a weaker check.
    """
    try:
        cv2 = importer()
    except Exception as exc:  # noqa: BLE001 - decode probe must never break startup
        return OpenCvCapability(False, f"OpenCV import failed: {exc}")

    registry = getattr(cv2, "videoio_registry", None)
    if registry is None:
        return OpenCvCapability(
            False,
            "OpenCV build exposes no videoio_registry, so the FFMPEG backend cannot be verified",
        )

    has_backend = getattr(registry, "hasBackend", None)
    if has_backend is None:
        return OpenCvCapability(
            False,
            "OpenCV videoio_registry exposes no hasBackend, so the FFMPEG backend "
            "cannot be verified",
        )

    backend_id = getattr(cv2, "CAP_FFMPEG", None)
    if backend_id is None:
        return OpenCvCapability(
            False,
            "OpenCV build exposes no CAP_FFMPEG constant, so the FFMPEG backend cannot be verified",
        )

    try:
        available = bool(has_backend(backend_id))
    except Exception as exc:  # noqa: BLE001 - an unanswerable query is not availability
        return OpenCvCapability(False, f"OpenCV FFMPEG backend query failed: {exc}")

    if not available:
        return OpenCvCapability(False, "OpenCV build has no FFMPEG video I/O backend")
    return OpenCvCapability(True, "OpenCV FFMPEG backend is available")


__all__ = ["Cv2Importer", "OpenCvCapability", "probe_opencv_ffmpeg_capability"]
