from __future__ import annotations

import numpy as np

from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from worker.adapters.decode.nvdec_device.fake import (
    FakeDeviceResidentBatcher,
    fake_device_resident_pool,
)
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallWindowClassifierV2
from worker.interfaces.fall_model import FallV2Probabilities
from worker.pipeline.perception import build_frame_observation
from worker.types import DecisionInput, FallModelInput

_FRAME_SEC = 1 / 15
# 30-row window plus predictions at stride 5: three votes confirm at row 40.
_FRAMES = 41


class _ProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.inputs: list[FallModelInput] = []

    def predict(self, features: FallModelInput) -> FallV2Probabilities:
        self.inputs.append(tuple(tuple(row) for row in features))
        return FallV2Probabilities(
            background=1.0 - self.probability, fall_transition=self.probability, fallen=0.0
        )


def _domain(model: _ProbabilityModel) -> FallV2DomainDecider:
    return FallV2DomainDecider(
        classifier=FallWindowClassifierV2(model),
        policy=FallPolicyDeciderV2(
            camera_id="camera-parity",
            facility_id="facility-parity",
            boot_id="boot",
            stream_epoch="1",
            source_generation=0,
        ),
    )


def _normalized_observation(mean_rgb: tuple[float, float, float]) -> FrameObservation:
    x_offset = round(mean_rgb[0]) % 5
    y_offset = round(mean_rgb[1]) % 5
    pose = tuple((20 + x_offset + index, 30 + y_offset + index, 0.9) for index in range(17))
    return build_frame_observation(
        raw_boxes=((10, 10, 80, 100, 0.95),),
        poses=(pose,),
        track_ids=(17,),
    )


def _decision(observation: FrameObservation, frame_index: int) -> DecisionInput:
    return DecisionInput(
        observation=observation,
        frame_width=100,
        frame_height=120,
        live_track_ids=(17,),
        time_sec=frame_index * _FRAME_SEC,
        frame_index=frame_index,
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
    cpu_domain = _domain(cpu_model)
    device_domain = _domain(device_model)

    cpu_events: list[object] = []
    device_events: list[object] = []
    for frame_index in range(_FRAMES):
        cpu_events.extend(cpu_domain.update(_decision(cpu_observation, frame_index)))
        device_events.extend(device_domain.update(_decision(device_observation, frame_index)))
        assert device_domain.last_trace_snapshots == cpu_domain.last_trace_snapshots
    assert device_events == cpu_events
    assert len(device_events) == 1
    assert len(cpu_model.inputs) == len(device_model.inputs) == 3
    assert device_model.inputs == cpu_model.inputs
    assert pool.telemetry.snapshot().d2h_transfers == 0

    lease.release()
    assert pool.outstanding == 0
