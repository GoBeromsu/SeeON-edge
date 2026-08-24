"""Complete bounded PerceptionFrameV1 binary wire regression tests."""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from worker.native.deepstream.ipc import (
    ControlMessage,
    IpcProtocolError,
    MessageKind,
    MetadataFrame,
    decode_metadata,
    encode_message,
)
from worker.native.deepstream.perception_wire import PerceptionWireError, encode_perception_frame
from worker.types.perception_frame import (
    LEGACY_ASSOCIATION_STRATEGY,
    AssociationResult,
    BedRegion,
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
)

_BOOT = uuid.UUID("12345678-1234-5678-1234-567812345678")
_CHILD = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _nonempty_metadata() -> MetadataFrame:
    identity = PerceptionFrameIdentity(str(_BOOT), "camera-a", 7, 11, 123_456)
    return MetadataFrame(
        frame=PerceptionFrameV1(
            identity=identity,
            person_box=PersonBoxChannel(
                ChannelState.INFERRED,
                (PersonBox(1, 2, 30, 40, 0.75), PersonBox(5, 6, 50, 60, 0.5)),
            ),
            human_pose=HumanPoseChannel(
                ChannelState.INFERRED,
                (
                    (Keypoint(3, 4, 0.9), Keypoint(7, 8, 0.8)),
                    (Keypoint(10, 11, 0.7),),
                ),
            ),
            bed_region=BedRegionChannel(
                ChannelState.INFERRED,
                (BedRegion(0, 0, 100, 80, 0.95, ((0, 0), (100, 0), (100, 80))),),
            ),
            association=AssociationResult(
                strategy=LEGACY_ASSOCIATION_STRATEGY,
                track_ids=(41, 42),
                selected_cue_indexes=(0, 1),
                identity=identity,
            ),
        ),
        source_generation=3,
        child_instance_id=_CHILD,
        native_publish_sequence=99,
        transform_id="seeon-perception-v1",
    )


def _encode_for_test(metadata: MetadataFrame) -> bytes:
    frame = metadata.frame
    return encode_message(
        ControlMessage(
            MessageKind.METADATA,
            uuid.UUID(frame.identity.worker_boot_id),
            metadata.child_instance_id,
            frame.identity.camera_id,
            metadata.source_generation,
            frame.identity.stream_epoch,
            frame.identity.source_pts or 0,
            frame.identity.seq,
            metadata.native_publish_sequence,
            0,
            metadata.transform_id,
            encode_perception_frame(frame),
        )
    )


def test_empty_payload_matches_cross_language_golden_vector() -> None:
    # Given
    message = ControlMessage(
        MessageKind.METADATA,
        _BOOT,
        _CHILD,
        "camera-a",
        3,
        7,
        123_456,
        11,
        99,
        0,
        "seeon-perception-v1",
    )

    # When
    payload = encode_perception_frame(MetadataFrame.empty(message).frame)

    # Then
    assert payload.hex() == (
        "5046563112345678123456781234567812345678080063616d6572612d61"
        "070000000000000040e20100000000000b0000000000000002020200000000000000"
    )


def test_complete_nonempty_perception_frame_round_trips() -> None:
    # Given
    metadata = _nonempty_metadata()

    # When
    encoded = _encode_for_test(metadata)
    decoded = decode_metadata(encoded)

    # Then
    assert decoded == metadata
    assert len(encoded) < 4_096


def test_wire_rejects_invalid_association_index() -> None:
    # Given
    metadata = _nonempty_metadata()
    association = metadata.frame.association
    assert association is not None
    invalid = replace(
        metadata,
        frame=replace(
            metadata.frame,
            association=replace(association, selected_cue_indexes=(0, 9)),
        ),
    )

    # When / Then
    with pytest.raises(PerceptionWireError) as failed:
        _ = _encode_for_test(invalid)
    assert failed.value.code == "invalid_perception_frame"


def test_wire_rejects_malformed_payload_lengths_and_counts() -> None:
    # Given
    encoded = bytearray(_encode_for_test(_nonempty_metadata()))
    encoded[-1] = 0xFF

    # When / Then
    with pytest.raises(IpcProtocolError):
        _ = decode_metadata(bytes(encoded))


def test_wire_rejects_inner_identity_mismatch() -> None:
    # Given
    encoded = bytearray(_encode_for_test(_nonempty_metadata()))
    payload_offset = 92 + len("camera-a") + len("seeon-perception-v1")
    encoded[payload_offset + 4] ^= 0x01

    # When / Then
    with pytest.raises(IpcProtocolError) as failed:
        _ = decode_metadata(bytes(encoded))
    assert failed.value.code == "inner_identity_mismatch"
