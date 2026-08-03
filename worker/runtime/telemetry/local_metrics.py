"""Local aggregation and structured logging helpers."""

from __future__ import annotations

import logging
from typing import final

from worker.runtime.telemetry.models import (
    BusMetricsSource,
    BusSubscriptionSnapshot,
    RuntimeDiagnosticsSnapshot,
    StageTimingSnapshot,
)

LOGGER = logging.getLogger(__name__)


@final
class StageTimingAccumulator:
    __slots__ = ("last_sec", "max_sec", "samples", "total_sec")

    def __init__(self) -> None:
        self.samples = 0
        self.total_sec = 0.0
        self.last_sec = 0.0
        self.max_sec = 0.0

    def add(self, elapsed_sec: float) -> None:
        self.samples += 1
        self.total_sec += elapsed_sec
        self.last_sec = elapsed_sec
        self.max_sec = max(self.max_sec, elapsed_sec)

    def snapshot(self, stage: str) -> StageTimingSnapshot:
        return StageTimingSnapshot(
            stage=stage,
            samples=self.samples,
            total_sec=self.total_sec,
            last_sec=self.last_sec,
            max_sec=self.max_sec,
        )


def bus_snapshot(
    source: tuple[BusMetricsSource, tuple[str, ...]] | None,
) -> tuple[BusSubscriptionSnapshot, ...]:
    if source is None:
        return ()
    bus, names = source
    return tuple(
        BusSubscriptionSnapshot(
            name=name,
            published=(metrics := bus.metrics(name)).published,
            taken=metrics.taken,
            dropped=metrics.dropped,
            queue_age_sec=metrics.queue_age_sec,
        )
        for name in names
    )


def log_snapshot(snapshot: RuntimeDiagnosticsSnapshot) -> None:
    for camera in snapshot.cameras:
        LOGGER.info(
            "worker.runtime.telemetry",
            extra={
                "camera_id": camera.camera_id,
                "failure_category": camera.failure_category,
                "stage_timings": {
                    timing.stage: {
                        "samples": timing.samples,
                        "total_sec": timing.total_sec,
                        "last_sec": timing.last_sec,
                        "max_sec": timing.max_sec,
                    }
                    for timing in camera.stage_timings
                },
                "bus": {
                    metrics.name: {
                        "published": metrics.published,
                        "taken": metrics.taken,
                        "dropped": metrics.dropped,
                        "queue_age_sec": metrics.queue_age_sec,
                    }
                    for metrics in camera.bus
                },
                "encoder": snapshot.encoder,
                "encode": (
                    None
                    if camera.encode is None
                    else {
                        "requested": camera.encode.requested,
                        "selected": camera.encode.selected,
                        "fallback_count": camera.encode.fallback_count,
                        "last_reason": camera.encode.last_reason,
                    }
                ),
            },
        )


__all__ = ["StageTimingAccumulator", "bus_snapshot", "log_snapshot"]
