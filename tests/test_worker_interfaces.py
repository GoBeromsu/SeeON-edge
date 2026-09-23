from __future__ import annotations

from collections.abc import Sequence
from typing import get_type_hints

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from contracts.runner import Image, RunnerResult, person_result
from worker.interfaces import (
    BatchServingClient,
    ClipEncoder,
    ClipFinalizer,
    Decider,
    DecodeAdapter,
    DecodeSession,
    EncoderSession,
    EventSink,
    ExecutionRecordSink,
    Extractor,
    FrameBus,
    FrameSubscription,
    ServingClient,
)
from worker.types import BusinessEvent, DecisionInput, FramePacket, ModuleResult


def _packet(seq: int = 1) -> FramePacket:
    frame = Frame(index=seq, time_sec=float(seq), image=np.zeros((2, 3, 3), dtype=np.uint8))
    return FramePacket("camera-a", frame, float(seq), seq, 3, 2, 0.5)


def _decision_input() -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(),
        frame_width=3,
        frame_height=2,
        live_track_ids=(),
        time_sec=1.0,
        frame_index=1,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.EMPTY),
    )


def _event() -> BusinessEvent:
    return BusinessEvent("fall", "fall", 1, "camera-a", "facility-a", 1.0, 0.9)


class _DecodeSession:
    def __init__(self, packet: FramePacket) -> None:
        self._packet: FramePacket | None = packet

    def read(self) -> FramePacket | None:
        packet = self._packet
        self._packet = None
        return packet

    def close(self) -> None:
        self._packet = None


class _DecodeAdapter:
    def __init__(self, packet: FramePacket) -> None:
        self._packet: FramePacket = packet

    def open(self, config: str) -> DecodeSession:
        assert config
        return _DecodeSession(self._packet)


class _FrameSubscription:
    def __init__(self) -> None:
        self._packets: list[FramePacket] = []

    def take(self, *, timeout_sec: float | None = None) -> FramePacket | None:
        assert timeout_sec is None or timeout_sec >= 0
        return None if not self._packets else self._packets.pop(0)

    def close(self) -> None:
        self._packets.clear()

    def publish(self, packet: FramePacket) -> None:
        self._packets.append(packet)


class _FrameBus:
    def __init__(self) -> None:
        self.subscription: _FrameSubscription = _FrameSubscription()

    def subscribe(
        self,
        name: str,
        *,
        capacity: int,
        latest_only: bool = False,
    ) -> FrameSubscription:
        assert name and capacity > 0
        assert isinstance(latest_only, bool)
        return self.subscription

    def publish(self, packet: FramePacket) -> None:
        self.subscription.publish(packet)


class _Extractor:
    def __init__(self, name: str) -> None:
        self._name: str = name

    def extract(self, packet: FramePacket) -> ModuleResult:
        return ModuleResult(f"{self._name}-{packet.seq}", person_result(()), 0.25)


class _WrongExtractor:
    def extract(self, packet: FramePacket) -> str:
        return f"wrong-{packet.seq}"


class _Decider:
    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        assert input_value.frame_index >= 0
        return (_event(),)


class _EncoderSession:
    def __init__(self) -> None:
        self.packets: list[FramePacket] = []

    def write(self, packet: FramePacket) -> None:
        self.packets.append(packet)

    def close(self) -> None:
        self.packets.clear()


class _ClipEncoder:
    def open(self, camera: str, profile: str, geometry: tuple[int, int, float]) -> EncoderSession:
        assert camera and profile and geometry[0] > 0 and geometry[1] > 0 and geometry[2] > 0
        return _EncoderSession()


class _ClipFinalizer:
    def finalize(self, segments: Sequence[str], event: BusinessEvent) -> str:
        return f"{event.identity}:{','.join(segments)}"


class _EventSink:
    def __init__(self) -> None:
        self.events: list[BusinessEvent] = []

    def emit(self, event: BusinessEvent) -> None:
        self.events.append(event)


class _Runner:
    def run(self, image: Image) -> RunnerResult:
        assert image.ndim == 3
        return person_result(())


class _ServingClient:
    def create(self, task: str, **kwargs: str | int | float | bool | None) -> _Runner:
        assert task and not kwargs
        return _Runner()


class _BatchServingClient(_ServingClient):
    def infer_batch(
        self,
        task: str,
        frames: Sequence[FramePacket],
        **kwargs: str | int | float | bool | None,
    ) -> tuple[RunnerResult, ...]:
        runner = self.create(task, **kwargs)
        return tuple(runner.run(packet.frame.image) for packet in frames)


_PortFake = (
    _DecodeSession
    | _DecodeAdapter
    | _FrameSubscription
    | _FrameBus
    | _Extractor
    | _Decider
    | _EncoderSession
    | _ClipEncoder
    | _ClipFinalizer
    | _EventSink
    | _ServingClient
    | _BatchServingClient
)


@pytest.mark.parametrize(
    ("candidate", "port"),
    [
        (_DecodeSession(_packet()), DecodeSession),
        (_DecodeAdapter(_packet()), DecodeAdapter),
        (_FrameSubscription(), FrameSubscription),
        (_FrameBus(), FrameBus),
        (_Extractor("pose"), Extractor),
        (_Decider(), Decider),
        (_EncoderSession(), EncoderSession),
        (_ClipEncoder(), ClipEncoder),
        (_ClipFinalizer(), ClipFinalizer),
        (_EventSink(), EventSink),
        (_ServingClient(), ServingClient),
        (_BatchServingClient(), BatchServingClient),
    ],
)
def test_runtime_checkable_ports_accept_conforming_fakes(candidate: _PortFake, port: type) -> None:
    assert isinstance(candidate, port)


@pytest.mark.parametrize(
    "port",
    [
        DecodeSession,
        DecodeAdapter,
        FrameSubscription,
        FrameBus,
        Extractor,
        Decider,
        EncoderSession,
        ClipEncoder,
        ClipFinalizer,
        EventSink,
        ServingClient,
        BatchServingClient,
    ],
)
def test_runtime_checkable_ports_reject_missing_methods(port: type) -> None:
    assert not isinstance(object(), port)


def test_decode_and_extract_implementations_are_swappable_at_one_caller() -> None:
    def run(adapter: DecodeAdapter[str], extractor: Extractor) -> ModuleResult:
        session = adapter.open("source-config")
        packet = session.read()
        assert packet is not None
        return extractor.extract(packet)

    first = run(_DecodeAdapter(_packet(1)), _Extractor("pose"))
    second = run(_DecodeAdapter(_packet(2)), _Extractor("mediapipe-like"))

    assert first.module_name == "pose-1"
    assert second.module_name == "mediapipe-like-2"


def test_wrong_extractor_output_is_rejected_before_decision_call() -> None:
    class CountingDecider:
        def __init__(self) -> None:
            self.calls: int = 0

        def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
            self.calls += 1
            return _Decider().update(input_value)

    def dispatch(extractor: Extractor, decider: CountingDecider) -> None:
        module_result = extractor.extract(_packet())
        assert isinstance(module_result, ModuleResult)
        _ = decider.update(_decision_input())

    candidate = _WrongExtractor()
    decider = CountingDecider()
    assert isinstance(candidate, Extractor)

    with pytest.raises(AssertionError):
        dispatch(candidate, decider)

    assert decider.calls == 0


def test_ports_carry_only_worker_envelopes_across_stage_boundaries() -> None:
    assert get_type_hints(DecodeSession.read)["return"] == FramePacket | None
    assert get_type_hints(Extractor.extract)["return"] is ModuleResult
    assert get_type_hints(Decider.update)["return"] == tuple[BusinessEvent, ...]
    assert get_type_hints(EventSink.emit)["event"] is BusinessEvent
    assert get_type_hints(BatchServingClient.infer_batch)["return"] == tuple[RunnerResult, ...]


def test_session_and_output_fakes_execute_the_declared_contracts() -> None:
    bus = _FrameBus()
    subscription = bus.subscribe("inference", capacity=1, latest_only=True)
    packet = _packet()
    bus.publish(packet)
    assert subscription.take() is packet

    encoder = _ClipEncoder().open("camera-a", "cpu", (3, 2, 5.0))
    encoder.write(packet)
    assert isinstance(encoder, EncoderSession)

    event = _Decider().update(_decision_input())[0]
    sink = _EventSink()
    sink.emit(event)
    assert sink.events == [event]
    assert _ClipFinalizer().finalize(("a.mp4", "b.mp4"), event) == "1:a.mp4,b.mp4"


def test_single_frame_serving_client_does_not_claim_deferred_batching() -> None:
    assert isinstance(_ServingClient(), ServingClient)
    assert not isinstance(_ServingClient(), BatchServingClient)
    assert isinstance(_BatchServingClient(), BatchServingClient)


def test_execution_record_sink_is_runtime_checkable_and_requires_try_emit() -> None:
    class _Ok:
        def try_emit(self, record: object) -> bool:
            return bool(record)

    class _Missing:
        def emit(self, record: object) -> bool:
            return bool(record)

    assert ExecutionRecordSink.__module__ == "worker.interfaces.execution_records"
    assert isinstance(_Ok(), ExecutionRecordSink)
    assert not isinstance(_Missing(), ExecutionRecordSink)
    assert not isinstance(object(), ExecutionRecordSink)
