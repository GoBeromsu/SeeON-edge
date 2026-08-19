"""Unit coverage for ``ClipFrameFeeder``: ``bus.evidence`` -> ``ClipRecorder``."""

from __future__ import annotations

import logging
import threading
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.output.evidence.clip_frame_feeder import ClipFrameFeeder
from worker.types import FrameLeaseReleasedError, FramePacket


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((2, 3, 3), seq, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 3, 2, 0.25)


@final
class _SignalingSubscription:
    def __init__(self, subscription: object) -> None:
        self._subscription = subscription
        self.poll_returned = threading.Event()

    def take(self, *, timeout_sec: float | None = None) -> FramePacket | None:
        packet = self._subscription.take(timeout_sec=timeout_sec)  # type: ignore[attr-defined]
        if packet is None:
            self.poll_returned.set()
        return packet


@final
class _RecordingRecorder:
    def __init__(self) -> None:
        self.calls: list[FramePacket] = []
        self.frames: list[Frame] = []
        self.called = threading.Event()

    def on_frame(self, packet: FramePacket) -> bool:
        self.calls.append(packet)
        self.frames.append(packet.borrow_host_frame())
        self.called.set()
        return True


@final
class _RaisingOnFirstCallRecorder:
    def __init__(self) -> None:
        self.calls: list[FramePacket] = []
        self.raised = threading.Event()
        self.called = threading.Event()

    def on_frame(self, packet: FramePacket) -> bool:
        if not self.raised.is_set():
            self.raised.set()
            raise RuntimeError("recorder admission exploded")
        self.calls.append(packet)
        self.called.set()
        return True


def test_run_forwards_the_same_packet_without_reconstruction() -> None:
    bus = BoundedFrameBus()
    recorder = _RecordingRecorder()
    feeder = ClipFrameFeeder("cam-a", bus.evidence, recorder, poll_timeout_sec=0.02)
    thread = threading.Thread(target=feeder.run, daemon=True)
    thread.start()

    packet = _packet("cam-a", 1)
    source_frame = packet.frame
    bus.publish(packet)
    assert recorder.called.wait(2.0)
    feeder.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert feeder.fed_count == 1
    assert recorder.calls == [packet]
    assert recorder.calls[0] is not packet
    assert recorder.frames == [source_frame]
    assert recorder.calls[0].lease is not packet.lease
    with pytest.raises(FrameLeaseReleasedError):
        recorder.calls[0].borrow_host_frame()


def test_poll_timeout_does_not_count_as_a_feed_or_stop_the_loop() -> None:
    bus = BoundedFrameBus()
    recorder = _RecordingRecorder()
    subscription = _SignalingSubscription(bus.evidence)
    feeder = ClipFrameFeeder("cam-a", subscription, recorder, poll_timeout_sec=0.02)
    thread = threading.Thread(target=feeder.run, daemon=True)
    thread.start()

    assert subscription.poll_returned.wait(2.0)
    assert thread.is_alive()
    assert feeder.fed_count == 0
    assert recorder.calls == []

    feeder.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_a_raising_recorder_is_isolated_and_the_loop_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = BoundedFrameBus()
    recorder = _RaisingOnFirstCallRecorder()
    feeder = ClipFrameFeeder("cam-a", bus.evidence, recorder, poll_timeout_sec=0.02)
    thread = threading.Thread(target=feeder.run, daemon=True)
    thread.start()

    first = _packet("cam-a", 1)
    with caplog.at_level(
        logging.WARNING, logger="worker.pipeline.output.evidence.clip_frame_feeder"
    ):
        bus.publish(first)
        assert recorder.raised.wait(2.0)
        second = _packet("cam-a", 2)
        bus.publish(second)
        assert recorder.called.wait(2.0)

    feeder.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert feeder.failure_count == 1
    assert feeder.fed_count == 1
    assert recorder.calls == [second]
    assert first.released
    record = next(
        item
        for item in caplog.records
        if "clip frame feeder failed to admit a frame" in item.getMessage()
    )
    message = record.getMessage()
    assert "camera_id=cam-a" in message
    assert "error=RuntimeError" in message
    assert "recorder admission exploded" not in message


def test_stop_terminates_the_loop_and_run_returns() -> None:
    bus = BoundedFrameBus()
    recorder = _RecordingRecorder()
    feeder = ClipFrameFeeder("cam-a", bus.evidence, recorder, poll_timeout_sec=0.02)
    thread = threading.Thread(target=feeder.run, daemon=True)
    thread.start()

    feeder.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
