"""Deterministic fake device-input NVENC adapters for hosts without NVIDIA hardware.

Same rationale as ``worker.adapters.decode.nvdec_device.fake``: this repo's
CI/dev machines are Apple Silicon, so every lifecycle/backpressure/
zero-host-transfer/failure claim this prototype makes is proven against a
deterministic double, never against a real GPU. ``FakeDeviceSceneDrawer``
mutates a host-backed ``numpy`` buffer tagged with a non-host ``MemoryKind``
(the same faithful-double technique as ``FakeDeviceAllocator``), and
``FakeDeviceInputNvencEncoder`` stands in for an NVENC session: it accepts
only device-resident leases (never silently reads a host frame), models a
real async encoder's in-flight submission queue (a caller may submit frame
N+1 before frame N's bitstream callback fires; ``capacity`` bounds exactly
that outstanding, not-yet-retired queue, refusing new submissions rather
than growing past it), and emits a tiny deterministic byte stream per
retired frame so tests can assert artifact provenance without a real
bitstream.

Neither type is ever selected by a production profile: constructed only by
tests and by ``worker.adapters.encode.nvenc_device.diagnostic``'s explicit
``--fake-smoke-test`` dry run.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

import numpy as np

from worker.adapters.encode.nvenc_device.errors import (
    DeviceEncoderPoolExhaustedError,
    DeviceEncoderRejectedInputError,
)
from worker.adapters.encode.nvenc_device.models import (
    DeviceEncoderPoolConfig,
    DeviceEncoderSelection,
)
from worker.adapters.encode.nvenc_device.telemetry import DeviceEncoderTelemetry
from worker.types import FrameDescriptor, FrameLease, MemoryKind, PixelFormat
from worker.types.overlay_scene import OverlayScene

_RGB_CHANNELS = 3


@final
class FakeDeviceSceneDrawer:
    """Deterministic ``DeviceSceneDrawer``: paints scene person-count as a corner marker."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, handle: object, scene: OverlayScene) -> object:
        assert isinstance(handle, np.ndarray)  # noqa: S101 - fake drawer invariant
        self.calls += 1
        annotated = handle.copy()
        marker = min(255, len(scene.persons) * 17 + len(scene.decisions) * 5)
        annotated[0, 0, :] = marker
        return annotated


def fake_device_resident_scene(
    *, width: int, height: int, fill: int = 0
) -> tuple[object, FrameDescriptor]:
    """Build one fake device-resident surface, mirroring ``FakeDeviceAllocator.__call__``."""
    buffer = np.full((height, width, _RGB_CHANNELS), fill, dtype=np.uint8)
    descriptor = FrameDescriptor(
        width=width,
        height=height,
        memory_kind=MemoryKind.CUDA_DEVICE,
        pixel_format=PixelFormat.RGB24,
        plane_strides=(int(buffer.strides[0]),),
        size_bytes=int(buffer.nbytes),
    )
    return buffer, descriptor


def fake_device_resident_lease(*, width: int, height: int, fill: int = 0) -> FrameLease:
    handle, descriptor = fake_device_resident_scene(width=width, height=height, fill=fill)
    return FrameLease.from_device(handle, descriptor)


@dataclass(frozen=True, slots=True)
class FakeArtifactResult:
    path: Path
    sha256: str
    size_bytes: int
    selection: DeviceEncoderSelection


@final
class FakeDeviceInputNvencEncoder:
    """Deterministic double for a bounded, backpressured device-input NVENC session pool.

    Accepts only device-resident ``FrameLease`` submissions -- a host-memory
    lease is rejected with ``DeviceEncoderRejectedInputError``, never silently
    read back and encoded, matching the real seam's "no host readback"
    contract exactly. Frames are "encoded" by hashing their device buffer
    bytes (a deterministic stand-in bitstream); the one real host transfer
    this fake performs is the final artifact-byte materialization, which the
    real NVENC session would also have to do to persist a clip file, and is
    counted via ``telemetry.record_d2h``.
    """

    def __init__(
        self,
        config: DeviceEncoderPoolConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        telemetry: DeviceEncoderTelemetry | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self.telemetry = telemetry or DeviceEncoderTelemetry(pool_capacity=config.capacity)
        self._outstanding = 0
        self._pending: list[tuple[FrameLease, float]] = []
        self._frames: list[bytes] = []
        self._closed = False
        self._cancelled = False
        self.selection: DeviceEncoderSelection | None = None

    @property
    def capacity(self) -> int:
        return self._config.capacity

    @property
    def outstanding(self) -> int:
        return self._outstanding

    def open(self) -> DeviceEncoderSelection:
        if self.selection is not None:
            return self.selection
        codec = self._config.codec_candidates[0]
        container = self._config.container_candidates[0]
        profile = self._config.profile_candidates[0]
        self.selection = DeviceEncoderSelection(
            requested_codec=codec,
            requested_container=container,
            requested_profile=profile,
            selected_codec=codec,
            selected_container=container,
            selected_profile=profile,
            device_resident=True,
            reason="fake device-input NVENC session opened",
        )
        self.telemetry.record_session_opened()
        return self.selection

    def submit(self, lease: FrameLease) -> None:
        """Enqueue one device-resident frame; stays in-flight until ``retire_one``.

        Mirrors a real async NVENC submission: the call returns once the
        frame is accepted into the encoder's own bounded queue, not once its
        bitstream is produced. The submitted ``lease`` is retained (via
        ``FrameLease.retain``) for the duration it sits in-flight, so a
        caller that releases its own reference immediately after ``submit``
        never invalidates a still-queued encode -- ownership genuinely
        transfers to the encoder until retirement, exactly like the real
        seam's "explicit device surfaces/ownership" contract.
        """
        if self._closed:
            raise DeviceEncoderRejectedInputError("device-input NVENC session is closed")
        if lease.descriptor.memory_kind is MemoryKind.HOST:
            self.telemetry.record_submission_rejected_host_input("host-memory lease rejected")
            raise DeviceEncoderRejectedInputError(
                "device-input NVENC seam rejects host-memory leases; no readback fallback"
            )
        if self._outstanding >= self._config.capacity:
            self.telemetry.record_pool_exhausted()
            raise DeviceEncoderPoolExhaustedError(self._config.capacity, self._outstanding)
        retained = lease.retain()
        self._outstanding += 1
        self.telemetry.record_acquire(self._outstanding)
        self.telemetry.record_submission_accepted()
        self._pending.append((retained, self._clock()))

    def retire_one(self) -> None:
        """Complete the oldest in-flight submission, releasing its pool slot.

        Stands in for the real seam's CUDA-event/bitstream-ready callback:
        production code calls this from that completion signal, never from a
        blind poll or a fixed sleep.
        """
        if not self._pending:
            raise DeviceEncoderRejectedInputError("no in-flight submission to retire")
        lease, started = self._pending.pop(0)
        handle = lease.device_handle
        assert isinstance(handle, np.ndarray)  # noqa: S101 - fake encoder invariant
        digest = hashlib.sha256(handle.tobytes()).digest()[:16]
        self._frames.append(digest)
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
        self.telemetry.record_frame_encoded(elapsed_ms)
        lease.release()
        self._outstanding -= 1
        self.telemetry.record_release(self._outstanding)

    def retire_all(self) -> None:
        while self._pending:
            self.retire_one()

    def cancel_pending(self) -> int:
        """Drop every in-flight submission without encoding it; releases their slots.

        The device-input seam's cancellation path: unlike ``retire_one``,
        this never appends to the output bitstream, so a cancelled submission
        leaves no trace in the eventual artifact.
        """
        cancelled = len(self._pending)
        for lease, _started in self._pending:
            lease.release()
        self._pending.clear()
        self._outstanding = 0
        self.telemetry.record_release(self._outstanding)
        self._cancelled = True
        return cancelled

    def finalize(self, destination: Path) -> FakeArtifactResult:
        if self.selection is None:
            raise DeviceEncoderRejectedInputError("finalize called before a session was opened")
        if self._pending:
            raise DeviceEncoderRejectedInputError(
                "finalize called with in-flight submissions still pending retirement"
            )
        if not self._frames:
            raise DeviceEncoderRejectedInputError("finalize called with zero retired frames")
        payload = b"".join(self._frames)
        self.telemetry.record_d2h(len(payload))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        self.telemetry.record_artifact_finalized(len(payload))
        return FakeArtifactResult(
            destination,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            self.selection,
        )

    def close(self) -> None:
        if self._pending:
            _ = self.cancel_pending()
        self._closed = True


def fake_device_input_nvenc_encoder(
    *, camera_id: str, capacity: int, width: int, height: int
) -> FakeDeviceInputNvencEncoder:
    config = DeviceEncoderPoolConfig(
        camera_id=camera_id, capacity=capacity, width=width, height=height
    )
    return FakeDeviceInputNvencEncoder(config)


__all__ = [
    "FakeArtifactResult",
    "FakeDeviceInputNvencEncoder",
    "FakeDeviceSceneDrawer",
    "fake_device_input_nvenc_encoder",
    "fake_device_resident_lease",
    "fake_device_resident_scene",
]
