"""Single-owner cross-camera pose inference with latest-only draining."""

from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from time import monotonic
from typing import Final, Protocol, final

from contracts.runner import RunnerResult
from worker.adapters.model.errors import FatalAcceleratorError
from worker.interfaces.serving import BatchServingClient
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
        replaced: CoordinatedInference | None = None
        with self._condition:
            if self._closed:
                replaced = value
            else:
                replaced, self._value = self._value, value
                self._condition.notify()
        if replaced is not None:
            replaced.packet.release()

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


@dataclass(frozen=True, slots=True)
class CameraInferenceTelemetry:
    admitted: int
    overwritten: int
    inferred: int
    queue_age_sec: float


@dataclass(frozen=True, slots=True)
class InferenceTelemetrySnapshot:
    cameras: dict[str, CameraInferenceTelemetry]
    batch_sizes: dict[int, int]
    forward_p50_sec: float
    forward_p95_sec: float


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
        self._batch_sizes: Counter[int] = Counter()
        self._forward_times: deque[float] = deque(maxlen=1024)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def register(
        self, camera_id: str, subscription: InferenceSubscription, results: InferenceResultSlot
    ) -> None:
        if any(lane.camera_id == camera_id for lane in self._lanes):
            raise ValueError(f"camera already registered: {camera_id}")
        self._lanes.append(_CameraLane(camera_id, subscription, results))

    def run_cycle(self) -> int:
        selected: list[tuple[_CameraLane, FramePacket]] = []
        for lane in self._lanes:
            if len(selected) >= self._max_batch_size:
                break
            metrics = lane.subscription.metrics()
            packet = lane.subscription.take(timeout_sec=0)
            if packet is None:
                continue
            lane.last_queue_age_sec = float(getattr(metrics, "queue_age_sec", 0.0))
            lane.admitted += 1
            selected.append((lane, packet))
        if not selected:
            return 0
        frames = tuple(packet for _lane, packet in selected)
        started_at = self._clock()
        try:
            outputs = self._forward(selected, frames)
        except Exception:
            for _lane, packet in selected:
                packet.release()
            raise
        elapsed = max(0.0, self._clock() - started_at)
        with self._lock:
            self._batch_sizes[len(selected)] += 1
            self._forward_times.append(elapsed)
        for (lane, packet), output in zip(selected, outputs, strict=True):
            lane.inferred += 1
            if self._stage_timing_recorder is not None:
                self._stage_timing_recorder.record_stage_timing(
                    lane.camera_id, "pose", elapsed
                )
            lane.results.publish(
                CoordinatedInference(
                    packet,
                    ModuleResult(
                        module_name="pose",
                        result=output,
                        elapsed_ms=elapsed * 1000.0,
                        output_adapter=self._pose_output_adapter,
                    ),
                )
            )
        return len(selected)

    def _forward(
        self,
        selected: Sequence[tuple[_CameraLane, FramePacket]],
        frames: Sequence[FramePacket],
    ) -> tuple[RunnerResult, ...]:
        options = {} if self._pose_device is None else {"device": self._pose_device}
        with self._watchdog.guard(
            camera_id=",".join(lane.camera_id for lane, _packet in selected),
            task="pose",
            frame_index=max(packet.frame.index for packet in frames),
            deadline_sec=INFERENCE_DEADLINE_SEC,
        ):
            outputs = self._client.infer_batch("pose", frames, **options)
        if len(outputs) != len(frames):
            raise ValueError(
                f"pose batch returned {len(outputs)} results for {len(frames)} frames"
            )
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
        with self._lock:
            times = tuple(self._forward_times)
            sizes = dict(sorted(self._batch_sizes.items()))
        cameras = {
            lane.camera_id: CameraInferenceTelemetry(
                admitted=lane.admitted,
                overwritten=int(getattr(lane.subscription.metrics(), "dropped", 0)),
                inferred=lane.inferred,
                queue_age_sec=lane.last_queue_age_sec,
            )
            for lane in self._lanes
        }
        return InferenceTelemetrySnapshot(
            cameras=cameras,
            batch_sizes=sizes,
            forward_p50_sec=_percentile(times, 0.50),
            forward_p95_sec=_percentile(times, 0.95),
        )

    def _wait(self, timeout_sec: float) -> None:
        self._stop_event.wait(timeout_sec)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, ceil(len(ordered) * quantile) - 1)]


__all__ = [
    "CapabilityInferenceCoordinator",
    "CameraInferenceTelemetry",
    "CoordinatedInference",
    "InferenceResultSlot",
    "InferenceTelemetrySnapshot",
]
