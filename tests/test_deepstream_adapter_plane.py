from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from worker.adapters.deepstream.service_maker import (
    DeepStreamMediaPlane,
    DeepStreamMediaPlaneConfig,
    _FlowHandle,
)
from worker.interfaces.media_plane import MediaPlane, RecordingRefused
from worker.native.deepstream.metadata import LatestMetadataSlot


class _Element:
    def __init__(self) -> None:
        self.emitted: list[tuple[object, ...]] = []

    def emit(self, *args: object) -> int:
        self.emitted.append(args)
        return 4


class _Pipeline:
    def __init__(self) -> None:
        self.elements: dict[str, _Element] = {}

    def __getitem__(self, name: str) -> _Element:
        return self.elements.setdefault(name, _Element())


class _Flow:
    def add_source(self, uri: str) -> None:
        del uri

    def remove_source(self, name: str) -> None:
        del name


def test_plane_binding_and_smart_record_commands() -> None:
    pipeline = _Pipeline()
    config = DeepStreamMediaPlaneConfig("infer", "tracker", "lib", Path("/tmp"), 5, 640, 360)
    plane = DeepStreamMediaPlane(
        config,
        metadata_slot=LatestMetadataSlot(),
        flow_factory=lambda _: _FlowHandle(_Flow(), pipeline),
        worker_boot_id="boot",
        child_instance_id="child",
    )
    assert isinstance(plane, MediaPlane)
    first = plane.add_source("camera", "rtsp://one")
    replacement = plane.source_failure("camera", "timeout")
    assert replacement.source_generation == first.source_generation + 1
    with pytest.raises(RecordingRefused):
        plane.start_recording("camera", lookback_sec=1, duration_sec=2, on_sealed=lambda _: None)
    plane._live.add("camera")
    sealed: list[object] = []
    session = plane.start_recording(
        "camera", lookback_sec=1, duration_sec=2, on_sealed=sealed.append
    )
    assert (
        plane.start_recording("camera", lookback_sec=1, duration_sec=2, on_sealed=sealed.append)
        == session
    )
    plane.stop_recording("camera", session)
    assert pipeline["batch_capture-source-0_0"].emitted[-1] == ("stop-sr", session)
    plane._recording_done(
        "camera",
        SimpleNamespace(
            session_id=session,
            dirpath="/tmp",
            filename="clip.mp4",
            duration=20,
            width=640,
            height=360,
        ),
    )
    plane._recording_done(
        "camera",
        SimpleNamespace(
            session_id=session,
            dirpath="/tmp",
            filename="clip.mp4",
            duration=20,
            width=640,
            height=360,
        ),
    )
    assert len(sealed) == 1
    assert plane.status().nvenc_sessions_active == 0
