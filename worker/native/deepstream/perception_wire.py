"""Bounded compact encoding for complete worker-internal PerceptionFrameV1 values.

Wire v2 (magic ``PFV2``) additionally carries the decoded source geometry and
the capture wall-clock right after the frame identity: the Python-side
DecisionInput and evidence trigger need both, and neither exists in
``PerceptionFrameV1`` itself (C1 contract stays untouched).
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass
from typing import Final

from worker.native.deepstream.perception_wire_primitives import (
    MAGIC as _MAGIC,
)
from worker.native.deepstream.perception_wire_primitives import (
    PerceptionWireError,
)
from worker.native.deepstream.perception_wire_primitives import (
    Reader as _Reader,
)
from worker.native.deepstream.perception_wire_primitives import (
    Writer as _Writer,
)
from worker.types.perception_frame import (
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
    assemble_perception_frame,
)

_MAX_KEYPOINTS: Final = 64
_GEOMETRY: Final = struct.Struct("<HHQ")
_I64_U16: Final = struct.Struct("<qH")
_BOX: Final = struct.Struct("<iiiid")
_KEYPOINT: Final = struct.Struct("<iid")
_POINT: Final = struct.Struct("<ii")


_STATE_CODES: Final = {
    ChannelState.INFERRED: 1,
    ChannelState.INFERRED_EMPTY: 2,
    ChannelState.SKIPPED: 3,
}
_STATES_BY_CODE: Final = {code: state for state, code in _STATE_CODES.items()}


def _state_code(state: ChannelState) -> int:
    return _STATE_CODES[state]


def _state(value: int) -> ChannelState:
    try:
        return _STATES_BY_CODE[value]
    except KeyError as error:
        raise PerceptionWireError("channel_state", str(value)) from error


def _write_identity(writer: _Writer, identity: PerceptionFrameIdentity) -> None:
    try:
        writer.raw(uuid.UUID(identity.worker_boot_id).bytes)
    except ValueError as error:
        raise PerceptionWireError("identity_uuid", identity.worker_boot_id) from error
    writer.text(identity.camera_id)
    writer.raw(struct.pack("<QQQ", identity.stream_epoch, identity.source_pts or 0, identity.seq))


def _read_identity(reader: _Reader) -> PerceptionFrameIdentity:
    boot = uuid.UUID(bytes=reader.raw(16))
    camera = reader.text()
    epoch, pts, sequence = reader.u64(), reader.u64(), reader.u64()
    return PerceptionFrameIdentity(str(boot), camera, epoch, sequence, pts)


@dataclass(frozen=True, slots=True)
class DecodedPerception:
    """One decoded wire frame plus its geometry/source-time carrier."""

    frame: PerceptionFrameV1
    source_width: int
    source_height: int
    source_time_ns: int


def encode_perception_frame(
    frame: PerceptionFrameV1,
    *,
    source_width: int = 0,
    source_height: int = 0,
    source_time_ns: int = 0,
) -> bytes:
    outcome = assemble_perception_frame(
        identity=frame.identity,
        person_box=frame.person_box,
        human_pose=frame.human_pose,
        bed_region=frame.bed_region,
        association=frame.association,
    )
    if outcome != frame:
        raise PerceptionWireError("invalid_perception_frame", repr(outcome))
    writer = _Writer()
    _write_identity(writer, frame.identity)
    if not (0 <= source_width <= 65_535 and 0 <= source_height <= 65_535):
        raise PerceptionWireError("geometry_bounds", f"{source_width}x{source_height}")
    if source_time_ns < 0:
        raise PerceptionWireError("source_time_bounds", str(source_time_ns))
    writer.raw(_GEOMETRY.pack(source_width, source_height, source_time_ns))
    writer.raw(
        bytes(
            (
                _state_code(frame.person_box.state),
                _state_code(frame.human_pose.state),
                _state_code(frame.bed_region.state),
                int(frame.association is not None),
            )
        )
    )
    writer.u16(len(frame.person_box.boxes))
    for box in frame.person_box.boxes:
        writer.raw(_BOX.pack(box.x1, box.y1, box.x2, box.y2, box.confidence))
    writer.u16(len(frame.human_pose.poses))
    for pose in frame.human_pose.poses:
        writer.u16(len(pose))
        if len(pose) > _MAX_KEYPOINTS:
            raise PerceptionWireError("keypoint_bounds", str(len(pose)))
        for point in pose:
            writer.raw(_KEYPOINT.pack(point.x, point.y, point.score))
    writer.u16(len(frame.bed_region.regions))
    for region in frame.bed_region.regions:
        polygon = region.polygon or ()
        writer.raw(_BOX.pack(region.x1, region.y1, region.x2, region.y2, region.confidence))
        writer.u16(len(polygon))
        for x, y in polygon:
            writer.raw(_POINT.pack(x, y))
    if frame.association is not None:
        association = frame.association
        _write_identity(writer, association.identity)
        writer.text(association.strategy)
        writer.text(association.cue_source)
        writer.u16(len(association.track_ids))
        for track_id, cue_index in zip(
            association.track_ids,
            association.selected_cue_indexes,
            strict=True,
        ):
            writer.raw(_I64_U16.pack(track_id, cue_index))
        writer.u16(len(association.live_track_ids))
        for live_id in association.live_track_ids:
            writer.raw(struct.pack("<q", live_id))
    return bytes(writer.value)


def decode_perception_wire(payload: bytes, expected: PerceptionFrameIdentity) -> DecodedPerception:
    reader = _Reader(payload)
    if reader.raw(4) != _MAGIC:
        raise PerceptionWireError("payload_magic", repr(payload[:4]))
    identity = _read_identity(reader)
    if identity != expected:
        raise PerceptionWireError("inner_identity_mismatch", repr(identity.durable_key))
    source_width, source_height, source_time_ns = _GEOMETRY.unpack(reader.raw(_GEOMETRY.size))
    person_state, pose_state, bed_state, association_present = (reader.u8() for _ in range(4))
    boxes = tuple(
        PersonBox(reader.i32(), reader.i32(), reader.i32(), reader.i32(), reader.f64())
        for _ in range(reader.u16())
    )
    poses = tuple(
        tuple(
            Keypoint(reader.i32(), reader.i32(), reader.f64())
            for _ in range(reader.u16(maximum=_MAX_KEYPOINTS))
        )
        for _ in range(reader.u16())
    )
    regions: list[BedRegion] = []
    for _ in range(reader.u16()):
        x1, y1, x2, y2, confidence = (
            reader.i32(),
            reader.i32(),
            reader.i32(),
            reader.i32(),
            reader.f64(),
        )
        polygon = tuple((reader.i32(), reader.i32()) for _ in range(reader.u16()))
        regions.append(BedRegion(x1, y1, x2, y2, confidence, polygon or None))
    association = None
    if association_present == 1:
        association_identity = _read_identity(reader)
        if association_identity != identity:
            raise PerceptionWireError("association_identity_mismatch", repr(association_identity))
        strategy, cue_source = reader.text(), reader.text()
        pairs = tuple((reader.i64(), reader.u16()) for _ in range(reader.u16()))
        live_ids = tuple(reader.i64() for _ in range(reader.u16()))
        association = AssociationResult(
            strategy,
            tuple(pair[0] for pair in pairs),
            tuple(pair[1] for pair in pairs),
            identity,
            cue_source,
            live_ids,
        )
    elif association_present != 0:
        raise PerceptionWireError("association_flag", str(association_present))
    if reader.offset != len(payload):
        raise PerceptionWireError("payload_trailing", str(len(payload) - reader.offset))
    outcome = assemble_perception_frame(
        identity=identity,
        person_box=PersonBoxChannel(_state(person_state), boxes),
        human_pose=HumanPoseChannel(_state(pose_state), poses),
        bed_region=BedRegionChannel(_state(bed_state), tuple(regions)),
        association=association,
    )
    if not isinstance(outcome, PerceptionFrameV1):
        raise PerceptionWireError("invalid_perception_frame", repr(outcome))
    return DecodedPerception(outcome, source_width, source_height, source_time_ns)


def decode_perception_frame(payload: bytes, expected: PerceptionFrameIdentity) -> PerceptionFrameV1:
    return decode_perception_wire(payload, expected).frame


__all__ = [
    "DecodedPerception",
    "PerceptionWireError",
    "decode_perception_frame",
    "decode_perception_wire",
    "encode_perception_frame",
]
