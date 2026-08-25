"""C5 binary IPC and latest-only metadata behavior."""

from __future__ import annotations

import socket
import uuid
from dataclasses import replace

import pytest

import worker.native.deepstream.ipc as ipc
import worker.native.deepstream.metadata as metadata


def _frame(*, camera: str = "camera-a", generation: int = 3, epoch: int = 7, seq: int = 1):
    return ipc.MetadataFrame.empty(
        ipc.ControlMessage(
            kind=ipc.MessageKind.METADATA,
            worker_boot_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
            child_instance_id=uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            camera_id=camera,
            source_generation=generation,
            stream_epoch=epoch,
            source_pts=seq * 1_000,
            source_sequence=seq,
            native_publish_sequence=seq,
            request_id=0,
            transform_id="seeon-perception-v1",
        )
    )


def _binding(*, camera: str = "camera-a", generation: int = 3, epoch: int = 7):
    return metadata.SourceBinding(
        worker_boot_id="12345678-1234-5678-1234-567812345678",
        child_instance_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        camera_id=camera,
        source_generation=generation,
        stream_epoch=epoch,
        transform_id="seeon-perception-v1",
    )


def test_binary_control_round_trip_when_message_has_full_identity() -> None:
    # Given
    message = ipc.ControlMessage(
        kind=ipc.MessageKind.ADD_SOURCE,
        worker_boot_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        child_instance_id=uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        camera_id="camera-a",
        source_generation=3,
        stream_epoch=7,
        source_pts=11,
        source_sequence=12,
        native_publish_sequence=13,
        request_id=14,
        transform_id="seeon-perception-v1",
        payload=b"loopback://camera-a",
    )

    # When
    encoded = ipc.encode_message(message)
    decoded = ipc.decode_control_message(encoded)

    # Then
    assert decoded == message
    assert encoded[:4] == b"SDS1"
    assert not encoded.startswith(b"{")


def test_oversized_control_frame_is_refused_before_socket_send() -> None:
    # Given
    message = ipc.ControlMessage(
        kind=ipc.MessageKind.ADD_SOURCE,
        worker_boot_id=uuid.uuid4(),
        child_instance_id=uuid.uuid4(),
        camera_id="camera-a",
        source_generation=1,
        stream_epoch=0,
        source_pts=0,
        source_sequence=0,
        native_publish_sequence=0,
        request_id=1,
        transform_id="seeon-perception-v1",
        payload=b"x" * 65_536,
    )

    # When / Then
    with pytest.raises(ipc.IpcProtocolError) as failed:
        _ = ipc.encode_message(message)
    assert failed.value.code == "frame_too_large"


def test_latest_frame_wins_when_consumer_pauses() -> None:
    # Given
    slot = metadata.LatestMetadataSlot()
    _ = slot.register_source(_binding())

    # When
    assert slot.publish(_frame(seq=1))
    assert slot.publish(_frame(seq=2))
    latest = slot.take("camera-a")

    # Then
    assert latest is not None
    assert latest.identity.seq == 2
    assert slot.counters().overwritten == 1


def test_stale_unknown_and_epoch_mismatch_metadata_are_dropped() -> None:
    # Given
    slot = metadata.LatestMetadataSlot()
    _ = slot.register_source(_binding())
    assert slot.publish(_frame(seq=4))
    assert slot.take("camera-a") is not None

    # When
    outcomes = (
        slot.publish(_frame(seq=3)),
        slot.publish(_frame(camera="unknown", seq=5)),
        slot.publish(_frame(generation=2, seq=5)),
        slot.publish(_frame(epoch=6, seq=5)),
    )

    # Then
    assert outcomes == (False, False, False, False)
    counters = slot.counters()
    assert counters.late == 1
    assert counters.unknown_source == 1
    assert counters.generation_mismatch == 1
    assert counters.epoch_mismatch == 1


def test_boot_child_transform_and_each_high_water_are_enforced() -> None:
    # Given
    slot = metadata.LatestMetadataSlot()
    _ = slot.register_source(_binding())
    accepted = _frame(seq=10)
    assert slot.publish(accepted)
    assert slot.take("camera-a") is not None

    # When
    wrong_boot = replace(
        accepted,
        frame=replace(
            accepted.frame,
            identity=replace(accepted.identity, worker_boot_id=str(uuid.uuid4())),
        ),
    )
    wrong_child = replace(accepted, child_instance_id=uuid.uuid4())
    wrong_transform = replace(accepted, transform_id="other-transform")
    newer_pts_old_seq = replace(
        accepted,
        frame=replace(
            accepted.frame,
            identity=replace(accepted.identity, source_pts=20_000, seq=9),
        ),
        native_publish_sequence=11,
    )
    outcomes = tuple(
        slot.publish(frame)
        for frame in (wrong_boot, wrong_child, wrong_transform, newer_pts_old_seq)
    )

    # Then
    assert outcomes == (False, False, False, False)
    counters = slot.counters()
    assert counters.boot_mismatch == 1
    assert counters.child_mismatch == 1
    assert counters.transform_mismatch == 1
    assert counters.late == 1


def test_metadata_datagram_receiver_uses_inherited_socketpair() -> None:
    # Given
    sender, wake_receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    slot = metadata.LatestMetadataSlot()
    _ = slot.register_source(_binding())

    class Puller:
        def pull_latest(self, camera_id: str) -> ipc.MetadataFrame:
            return _frame(camera=camera_id, seq=8)

    token = slot.subscribe(_binding())

    # When
    with metadata.MetadataReceiver(wake_receiver, slot, Puller()):
        try:
            _ = sender.send(b"camera-a")
            received = slot.wait_accepted(token, timeout_sec=1.0)
        finally:
            sender.close()

    # Then
    assert received is not None
    assert received.identity.seq == 8
    assert wake_receiver.family == socket.AF_UNIX
