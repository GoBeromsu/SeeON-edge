"""Focused contracts for ordered runtime-status delivery."""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

from contracts.decode_diagnostics import DecodeSelection
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RelayRuntimeStatusTransport,
    RuntimeStatusSender,
)
from worker.runtime.telemetry.wire import RelayRuntimeStatusPayload


@final
class _FrozenWireTransport:
    __slots__ = ("generations", "payloads")

    def __init__(self, generations: list[int | None]) -> None:
        self.generations = generations
        self.payloads: list[RelayRuntimeStatusPayload] = []

    def send(self, payload: RelayRuntimeStatusPayload) -> int | None:
        assert set(payload) <= {
            "facility_id",
            "generation",
            "seq",
            "cameras",
            "clip_recorder",
            "clip_export",
            "gpu",
            "worker",
        }
        assert all(
            set(camera) <= {"camera_id", "decode", "measured_fps", "detection"}
            for camera in payload["cameras"]
        )
        assert all(
            camera.get("detection")
            == {
                "expected": False,
                "inference_admitted": 0,
                "inference_succeeded": 0,
                "inference_overwritten": 0,
                "decision_completed": 0,
            }
            for camera in payload["cameras"]
        )
        self.payloads.append(payload)
        return self.generations.pop(0)


def _diagnostics() -> WorkerDiagnostics:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode(
        "camera-1",
        DecodeSelection(
            requested="opencv",
            selected="opencv",
            fallback_count=0,
            last_reason=None,
            updated_at_sec=10.0,
        ),
    )
    return diagnostics


def test_payload_reports_worker_applied_clip_export_value_and_version() -> None:
    diagnostics = _diagnostics()
    diagnostics.set_clip_export_applied(enabled=True, version=7)
    transport = _FrozenWireTransport([3])

    assert RuntimeStatusSender(diagnostics, "facility-1", transport).publish_once() is True

    assert transport.payloads[0]["clip_export"] == {"enabled": True, "version": 7}


def test_local_metric_is_caught_before_transport_and_never_leaks() -> None:
    # Given
    diagnostics = _diagnostics()
    diagnostics.record_stage_timing("camera-1", "local_only_metric", 0.5)
    transport = _FrozenWireTransport([3])
    sender = RuntimeStatusSender(diagnostics, "facility-1", transport)

    # When
    accepted = sender.publish_once()

    # Then
    assert accepted is True
    assert "local_only_metric" not in repr(transport.payloads[0])


def test_sender_preserves_generation_and_monotonic_sequence() -> None:
    # Given
    transport = _FrozenWireTransport([3, 3])
    sender = RuntimeStatusSender(_diagnostics(), "facility-1", transport)

    # When
    first = sender.publish_once()
    second = sender.publish_once()

    # Then
    assert first is True
    assert second is True
    assert [(payload["generation"], payload["seq"]) for payload in transport.payloads] == [
        (None, 1),
        (3, 2),
    ]
    assert sender.generation == 3


def test_sender_partitions_cameras_by_facility() -> None:
    # Given
    diagnostics = _diagnostics()
    diagnostics.update_decode(
        "camera-2",
        DecodeSelection("opencv", "opencv", 0, None, 11.0),
    )
    transport = _FrozenWireTransport([1, 1])
    sender = RuntimeStatusSender(
        diagnostics,
        {"camera-1": "facility-1", "camera-2": "facility-2"},
        transport,
    )

    # When
    accepted = sender.publish_once()

    # Then
    assert accepted is True
    assert [payload["facility_id"] for payload in transport.payloads] == [
        "facility-1",
        "facility-2",
    ]
    camera_ids = [
        [camera["camera_id"] for camera in payload["cameras"]] for payload in transport.payloads
    ]
    assert camera_ids == [
        ["camera-1"],
        ["camera-2"],
    ]


def test_relay_transport_uses_bounded_status_endpoint_and_parses_receipt() -> None:
    # Given
    calls: list[tuple[str, str, Mapping[str, str], bytes | None, float]] = []

    def request(
        url: str,
        method: str,
        headers: Mapping[str, str],
        data: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        calls.append((url, method, headers, data, timeout))
        return 200, {}, b'{"accepted":true,"generation":7}'

    transport = RelayRuntimeStatusTransport(
        "http://relay.internal",
        "relay-token",
        request=request,
    )
    payload = _diagnostics().to_payload("facility-1", None, 1)

    # When
    generation = transport.send(payload)

    # Then
    assert generation == 7
    assert calls[0][0] == "http://relay.internal/api/v1/relay/runtime-status"
    assert calls[0][1] == "POST"
    assert calls[0][2]["Authorization"] == "Bearer relay-token"
    assert calls[0][4] == 2.0
