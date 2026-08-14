from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from worker.adapters.decode.nvdec_device.fake import (
    FakeDeviceResidentBatcher,
    fake_device_resident_pool,
)
from worker.domains.fall import FallEventLatch
from worker.pipeline.perception import build_frame_observation
from worker.types import DecisionInput, FallModelInput


@dataclass(frozen=True, slots=True)
class _Metadata:
    window: int = 1
    stride: int = 1
    mode: Literal["features"] = "features"


class _ProbabilityModel:
    metadata = _Metadata()
    operating_threshold = 0.5

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.inputs: list[FallModelInput] = []

    def predict(self, features: FallModelInput) -> float:
        self.inputs.append(features)
        return self.probability


def _normalized_observation(mean_rgb: tuple[float, float, float]) -> FrameObservation:
    x_offset = int(round(mean_rgb[0])) % 5
    y_offset = int(round(mean_rgb[1])) % 5
    pose = tuple((20 + x_offset + index, 30 + y_offset + index, 0.9) for index in range(17))
    return build_frame_observation(
        raw_boxes=((10, 10, 80, 100, 0.95),),
        poses=(pose,),
        track_ids=(17,),
    )


def _decision(observation: FrameObservation) -> DecisionInput:
    return DecisionInput(
        observation=observation,
        frame_width=100,
        frame_height=120,
        live_track_ids=(17,),
        time_sec=4.0,
        frame_index=4,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_fake_cpu_cuda_observation_and_domain_outputs_match_without_hidden_readback() -> None:
    host = np.tile(np.array([220, 30, 10], dtype=np.uint8), (4, 5, 1))
    cpu_mean = tuple(float(value) for value in host.reshape(-1, 3).mean(axis=0))
    pool, allocator = fake_device_resident_pool(
        camera_id="camera-parity",
        capacity=1,
        width=5,
        height=4,
    )
    lease = pool.acquire()
    allocator.upload(lease, host)
    batcher = FakeDeviceResidentBatcher(max_batch_size=1, allocator=allocator)

    before = pool.telemetry.snapshot()
    device_mean = batcher.infer_mean_rgb(batcher.form_batch((lease,)))[0]
    cpu_observation = _normalized_observation(cpu_mean)
    device_observation = _normalized_observation(device_mean)

    assert device_mean == cpu_mean
    assert device_observation == cpu_observation
    after_observation = pool.telemetry.snapshot()
    assert after_observation.h2d_transfers == 1
    assert after_observation.d2h_transfers == before.d2h_transfers == 0
    assert after_observation.d2h_bytes == 0

    cpu_model = _ProbabilityModel(cpu_mean[0] / 255.0)
    device_model = _ProbabilityModel(device_mean[0] / 255.0)
    cpu_domain = FallEventLatch(
        cpu_model,
        camera_id="camera-parity",
        facility_id="facility-parity",
        operating_threshold=0.5,
    )
    device_domain = FallEventLatch(
        device_model,
        camera_id="camera-parity",
        facility_id="facility-parity",
        operating_threshold=0.5,
    )

    cpu_events = cpu_domain.update(_decision(cpu_observation))
    device_events = device_domain.update(_decision(device_observation))
    assert device_events == cpu_events
    assert device_domain.last_trace_snapshots == cpu_domain.last_trace_snapshots
    assert len(device_events) == 1
    assert len(cpu_model.inputs) == len(device_model.inputs) == 1
    assert device_model.inputs[0] == cpu_model.inputs[0]
    assert pool.telemetry.snapshot().d2h_transfers == 0

    lease.release()
    assert pool.outstanding == 0
