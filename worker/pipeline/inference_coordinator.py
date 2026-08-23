"""Single-owner cross-camera pose inference with latest-only draining."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from time import monotonic
from typing import Final, Protocol, final

from contracts.runner import RunnerResult
from worker.adapters.model.errors import FatalAcceleratorError
from worker.interfaces.serving import BatchServingClient
from worker.pipeline.inference_telemetry import (
    CameraInferenceTelemetry,
    InferenceGeometryTelemetry,
    InferenceTelemetrySnapshot,
)
from worker.types import FramePacket, ModuleResult

LOGGER: Final = logging.getLogger(__name__)
DEFAULT_IDLE_SLEEP_SEC: Final = 0.005
DEFAULT_MAX_BATCH_SIZE: Final = 16
INFERENCE_DEADLINE_SEC: Final = 30.0


class InferenceGuard(Protocol):
    def guard(
        self,
        *,
        camera_id: str,
        task: str,
        frame_index: int | None = None,
        deadline_sec: float | None = None,
        model_artifact_digest: str | None = None,
    ) -> AbstractContextManager[int]: ...


class StageTimingRecorder(Protocol):
    def record_stage_timing(self, camera_id: str, stage: str, elapsed_sec: float) -> None: ...


class InferenceSubscription(Protocol):
    def take(self, *, timeout_sec: float | None = None) -> FramePacket | None: ...

    def metrics(self) -> object: ...


@dataclass(frozen=True, slots=True)
class CoordinatedInference:
    packet: FramePacket
    pose: ModuleResult


@final
class InferenceResultSlot:
    """Capacity-one latest result handoff; the slot owns queued packet leases."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: CoordinatedInference | None = None
        self._closed = False

    def publish(self, value: CoordinatedInference) -> None:
        with self._condition:
            if self._closed:
                value.packet.release()
                return
            replaced, self._value = self._value, None
            if replaced is not None:
                replaced.packet.release()
            self._value = value
            self._condition.notify()

    def take(self, *, timeout_sec: float | None = None) -> CoordinatedInference | None:
        with self._condition:
            if self._value is None and not self._closed:
                self._condition.wait_for(
                    lambda: self._value is not None or self._closed,
                    timeout=timeout_sec,
                )
            value, self._value = self._value, None
            return value

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            value, self._value = self._value, None
            self._condition.notify_all()
        if value is not None:
            value.packet.release()


@dataclass(slots=True)
class _CameraLane:
    camera_id: str
    subscription: InferenceSubscription
    results: InferenceResultSlot
    admitted: int = 0
    inferred: int = 0
    last_queue_age_sec: float = 0.0


@final
class CapabilityInferenceCoordinator:
    """Drain one latest frame per ready camera and own every pose forward."""

    camera_id = "inference-coordinator"

    def __init__(
        self,
        client: BatchServingClient,
        watchdog: InferenceGuard,
        *,
        clock: Callable[[], float] = monotonic,
        idle_sleep_sec: float = DEFAULT_IDLE_SLEEP_SEC,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        stage_timing_recorder: StageTimingRecorder | None = None,
        idle_wait: Callable[[float], None] | None = None,
        pose_output_adapter: str | None = "pose",
        pose_device: str | None = None,
    ) -> None:
        if not 0.005 <= idle_sleep_sec <= 0.010:
            raise ValueError("idle sleep must be between 5ms and 10ms")
        if max_batch_size <= 0:
            raise ValueError("max batch size must be positive")
        self._client, self._watchdog, self._clock = client, watchdog, clock
        self._idle_sleep_sec, self._max_batch_size = idle_sleep_sec, max_batch_size
        self._stage_timing_recorder = stage_timing_recorder
        self._idle_wait = idle_wait or self._wait
        self._pose_output_adapter = pose_output_adapter
        self._pose_device = pose_device
        self._lanes: list[_CameraLane] = []
        self._cursor = 0
        self._telemetry = InferenceGeometryTelemetry()
        self._stop_event = threading.Event()

    def register(
        self, camera_id: str, subscription: InferenceSubscription, results: InferenceResultSlot
    ) -> None:
        if any(lane.camera_id == camera_id for lane in self._lanes):
            raise ValueError(f"camera already registered: {camera_id}")
        self._lanes.append(_CameraLane(camera_id, subscription, results))

    def run_cycle(self) -> int:
        lanes = tuple(self._lanes)
        selected: list[tuple[_CameraLane, FramePacket]] = []
        start = self._cursor % len(lanes) if lanes else 0
        scanned = 0
        while scanned < len(lanes) and len(selected) < self._max_batch_size:
            lane = lanes[(start + scanned) % len(lanes)]
            scanned += 1
            metrics = lane.subscription.metrics()
            packet = lane.subscription.take(timeout_sec=0)
            if packet is None:
                continue
            lane.last_queue_age_sec = float(getattr(metrics, "queue_age_sec", 0.0))
            lane.admitted += 1
            selected.append((lane, packet))
        if not selected:
            return 0
        self._cursor = (start + scanned) % len(lanes)
        buckets: dict[tuple[int, int], list[tuple[_CameraLane, FramePacket]]] = {}
        for lane, packet in selected:
            self._telemetry.observe_geometry(
                camera_id=lane.camera_id, geometry=(packet.width, packet.height)
            )
            buckets.setdefault((packet.height, packet.width), []).append((lane, packet))
        pending = {id(packet): packet for _lane, packet in selected}
        published = 0
        try:
            for geometry, items in buckets.items():
                started_at = self._clock()
                try:
                    outputs = self._forward(tuple(packet for _lane, packet in items))
                    elapsed = max(0.0, self._clock() - started_at)
                    for (lane, packet), output in zip(items, outputs, strict=True):
                        pose = ModuleResult(
                            "pose", output, elapsed * 1000.0, self._pose_output_adapter
                        )
                        lane.results.publish(CoordinatedInference(packet, pose))
                        del pending[id(packet)]
                        lane.inferred += 1
                        if (recorder := self._stage_timing_recorder) is not None:
                            try:
                                recorder.record_stage_timing(lane.camera_id, "pose", elapsed)
                            except Exception:  # noqa: BLE001 - never blocks a lane
                                LOGGER.warning(
                                    "pose stage-timing recorder failed for camera %s; "
                                    "the batch lane continues",
                                    lane.camera_id,
                                    exc_info=True,
                                )
                        published += 1
                    self._telemetry.record_physical_batch(
                        geometry=(items[0][1].width, items[0][1].height),
                        batch_size=len(items),
                        elapsed_sec=elapsed,
                    )
                except FatalAcceleratorError:
                    raise
                except Exception as error:  # noqa: BLE001 - work-item boundary
                    for _lane, packet in items:
                        leftover = pending.pop(id(packet), None)
                        leftover.release() if leftover is not None else None
                    cameras = ",".join(lane.camera_id for lane, _packet in items)
                    LOGGER.exception(
                        "pose work item failed: error=%s geometry=%sx%s cameras=%s",
                        type(error).__name__, *geometry, cameras,
                    )
        except FatalAcceleratorError:
            for packet in pending.values():
                packet.release()
            raise
        return published

    def _forward(self, frames: Sequence[FramePacket]) -> tuple[RunnerResult, ...]:
        options = {} if self._pose_device is None else {"device": self._pose_device}
        with self._watchdog.guard(
            camera_id=",".join(frame.camera_id for frame in frames),
            task="pose",
            frame_index=max(packet.frame.index for packet in frames),
            deadline_sec=INFERENCE_DEADLINE_SEC,
        ):
            outputs = self._client.infer_batch("pose", frames, **options)
        if len(outputs) != len(frames):
            raise ValueError(f"pose batch returned {len(outputs)} results for {len(frames)} frames")
        return outputs

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                ready = self.run_cycle()
            except FatalAcceleratorError:
                raise
            except Exception as error:  # noqa: BLE001 - shared loop boundary
                LOGGER.exception("batched pose inference failed: error=%s", type(error).__name__)
                continue
            if ready == 0:
                self._idle_wait(self._idle_sleep_sec)

    def stop(self) -> None:
        self._stop_event.set()
        for lane in self._lanes:
            while (packet := lane.subscription.take(timeout_sec=0)) is not None:
                packet.release()
            lane.results.close()

    def snapshot(self) -> InferenceTelemetrySnapshot:
        counters = self._telemetry.counters()
        cameras = {
            lane.camera_id: CameraInferenceTelemetry(
                admitted=lane.admitted,
                overwritten=int(getattr(lane.subscription.metrics(), "dropped", 0)),
                inferred=lane.inferred,
                queue_age_sec=lane.last_queue_age_sec,
                observed_geometry=counters.observed_geometries.get(lane.camera_id),
            )
            for lane in self._lanes
        }
        return InferenceTelemetrySnapshot(
            cameras, counters.batch_sizes, counters.forward_p50_sec,
            counters.forward_p95_sec, counters.geometry_batch_sizes,
        )

    def _wait(self, timeout_sec: float) -> None:
        self._stop_event.wait(timeout_sec)


__all__ = [
    "CapabilityInferenceCoordinator",
    "CameraInferenceTelemetry",
    "CoordinatedInference",
    "InferenceResultSlot",
    "InferenceTelemetrySnapshot",
]
