from __future__ import annotations

from typing import Any

import pytest

from worker.adapters.decode.cpu_av.probe import OpenCvCapability, probe_opencv_ffmpeg_capability


class _Registry:
    def __init__(self, answer: bool | BaseException) -> None:
        self._answer = answer
        self.queried_with: list[int] = []

    def hasBackend(self, backend_id: int) -> bool:  # noqa: N802 - mirrors the cv2 API
        self.queried_with.append(backend_id)
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._answer


class _FakeCv2:
    CAP_FFMPEG = 1900

    def __init__(self, answer: bool | BaseException) -> None:
        self.videoio_registry = _Registry(answer)


def test_opencv_capability_true_only_when_ffmpeg_backend_is_present() -> None:
    # Given -- cv2 imports and its own registry reports the FFMPEG backend.
    fake_cv2 = _FakeCv2(True)

    # When
    capability = probe_opencv_ffmpeg_capability(importer=lambda: fake_cv2)

    # Then -- ADR-0003: availability is judged on the backend the adapter
    # actually demands, not on import success alone.
    assert capability == OpenCvCapability(True, "OpenCV FFMPEG backend is available")
    assert fake_cv2.videoio_registry.queried_with == [_FakeCv2.CAP_FFMPEG]


def test_opencv_capability_false_when_ffmpeg_backend_is_absent() -> None:
    # Given -- an OpenCV build without the FFMPEG video I/O backend. This used
    # to pass the global decode preflight and then fail on every camera open.
    fake_cv2 = _FakeCv2(False)

    # When
    capability = probe_opencv_ffmpeg_capability(importer=lambda: fake_cv2)

    # Then -- fail closed with an explicit reason, never a silent downgrade.
    assert capability.available is False
    assert capability.reason == "OpenCV build has no FFMPEG video I/O backend"


def test_opencv_capability_false_when_backend_query_raises() -> None:
    # Given -- an unanswerable registry query is not evidence of availability.
    fake_cv2 = _FakeCv2(RuntimeError("videoio registry unavailable"))

    # When
    capability = probe_opencv_ffmpeg_capability(importer=lambda: fake_cv2)

    # Then
    assert capability.available is False
    assert "videoio registry unavailable" in capability.reason


def test_opencv_capability_false_when_registry_is_missing() -> None:
    # Given -- a cv2 build that exposes no videoio_registry at all.
    capability = probe_opencv_ffmpeg_capability(importer=object)

    # Then -- the FFMPEG backend cannot be verified, so it is not available.
    assert capability.available is False
    assert "videoio_registry" in capability.reason


def test_opencv_capability_false_when_cap_ffmpeg_constant_is_missing() -> None:
    # Given -- registry present, but the backend constant is not exposed.
    class _NoConstant:
        def __init__(self) -> None:
            self.videoio_registry = _Registry(True)

    capability = probe_opencv_ffmpeg_capability(importer=_NoConstant)

    # Then
    assert capability.available is False
    assert "CAP_FFMPEG" in capability.reason


def test_opencv_capability_false_when_registry_has_no_has_backend() -> None:
    """A registry that cannot answer the question is not a yes.

    This is the one guard branch in ``probe_opencv_ffmpeg_capability`` that had
    no test. The others -- absent ``videoio_registry``, absent ``CAP_FFMPEG``,
    failed import, raising query, backend reported absent -- were all covered.

    It matters because it is the branch that most resembles a place to be
    lenient: the attribute is simply not there, so "assume it works" is the
    tempting reading. Under ADR-0003 that is the forbidden shape -- an implicit
    fallback that turns an unverifiable claim into an affirmative one. Making
    this branch default to available leaves the whole decode and probe suite
    green, which is how it stayed untested.
    """

    class _RegistryWithoutHasBackend:
        """Present, but cannot answer whether a backend exists."""

    class _NoHasBackend:
        def __init__(self) -> None:
            self.videoio_registry = _RegistryWithoutHasBackend()
            self.CAP_FFMPEG = 1900  # noqa: N815 - mirrors the cv2 constant name

    capability = probe_opencv_ffmpeg_capability(importer=_NoHasBackend)

    assert capability.available is False
    assert "hasBackend" in capability.reason

def test_opencv_capability_false_when_cv2_import_fails() -> None:
    def failing_importer() -> Any:
        raise ImportError("no module named cv2")

    # When
    capability = probe_opencv_ffmpeg_capability(importer=failing_importer)

    # Then -- fail closed, never raises past the probe boundary
    assert capability.available is False
    assert "no module named cv2" in capability.reason


def test_opencv_capability_false_when_cv2_import_raises_non_import_error() -> None:
    # Given -- a broken native shared library can raise OSError rather than
    # ImportError; the probe catches Exception broadly so a broken install
    # cannot crash the bootstrap sequence.
    def failing_importer() -> Any:
        raise OSError("dlopen failed: libopencv_videoio.so.4 not found")

    # When
    capability = probe_opencv_ffmpeg_capability(importer=failing_importer)

    # Then
    assert capability.available is False
    assert "dlopen failed" in capability.reason


def test_probe_opencv_ffmpeg_capability_against_real_cv2() -> None:
    """No fakes: exercises the real ``import cv2`` default on this host.

    This is the same signal the worker actually boots against, so it is worth
    running for real (not just via the injected fake) wherever cv2 is
    installed -- which is always, in this repo's test environment. Ground
    truth for this host: cv2 imports cleanly (4.13.0) and its own
    ``videoio_registry`` reports the FFMPEG backend, and separately-verified
    real ``cv2.VideoCapture`` calls against a live RTSP stream do open and
    read real frames, so this probe must report True here.
    """
    cv2 = pytest.importorskip("cv2")
    if not cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG):
        pytest.skip("this OpenCV build has no FFMPEG backend")

    capability = probe_opencv_ffmpeg_capability()

    assert capability == OpenCvCapability(True, "OpenCV FFMPEG backend is available")
