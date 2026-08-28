"""Worker-internal PerceptionFrameV1 envelopes.

These types stay inside the worker process boundary. They are not a public
contract and must not be imported from ``contracts``, ``backend``, or ``front``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class ChannelState(StrEnum):
    INFERRED = "inferred"
    INFERRED_EMPTY = "inferred_empty"
    SKIPPED = "skipped"


class PerceptionFrameFailureCode(StrEnum):
    MALFORMED_IDENTITY = "malformed_identity"
    EPOCH_MISMATCH = "epoch_mismatch"
    STALE_EPOCH = "stale_epoch"
    BED_IDENTITY_CUE = "bed_identity_cue"
    INVALID_CHANNEL_STATE = "invalid_channel_state"
    INVALID_CUE_INDEX = "invalid_cue_index"


LEGACY_ASSOCIATION_STRATEGY = "legacy-greedy-bbox-iou.v1"
PERSON_BOX_CUE_SOURCE = "person_box"


@dataclass(frozen=True, slots=True)
class PerceptionFrameIdentity:
    worker_boot_id: str
    camera_id: str
    stream_epoch: int
    seq: int
    source_pts: int | None = None

    @property
    def durable_key(self) -> tuple[str, str, int, int]:
        token = self.seq if self.source_pts is None else self.source_pts
        return (self.worker_boot_id, self.camera_id, self.stream_epoch, token)


@dataclass(frozen=True, slots=True)
class PersonBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(frozen=True, slots=True)
class BedRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    polygon: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True, slots=True)
class Keypoint:
    x: int
    y: int
    score: float


@dataclass(frozen=True, slots=True)
class PersonBoxChannel:
    state: ChannelState
    boxes: tuple[PersonBox, ...] = ()


@dataclass(frozen=True, slots=True)
class HumanPoseChannel:
    state: ChannelState
    poses: tuple[tuple[Keypoint, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class BedRegionChannel:
    state: ChannelState
    regions: tuple[BedRegion, ...] = ()


@dataclass(frozen=True, slots=True)
class AssociationResult:
    strategy: str
    track_ids: tuple[int, ...]
    selected_cue_indexes: tuple[int, ...]
    identity: PerceptionFrameIdentity
    cue_source: str = PERSON_BOX_CUE_SOURCE
    # Every currently-live durable track id (matched + coasting), in ascending
    # order. DecisionInput.live_track_ids consumes this; per-cue track_ids
    # alone cannot reconstruct coasting tracks.
    live_track_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PerceptionFrameV1:
    identity: PerceptionFrameIdentity
    person_box: PersonBoxChannel
    human_pose: HumanPoseChannel
    bed_region: BedRegionChannel
    association: AssociationResult | None = None


def _empty_details() -> dict[str, object]:
    return {}


@dataclass(frozen=True, slots=True)
class PerceptionFrameFailure:
    code: PerceptionFrameFailureCode
    message: str
    details: Mapping[str, object] = field(default_factory=_empty_details, hash=False)


def _failure(
    code: PerceptionFrameFailureCode,
    message: str,
    **details: object,
) -> PerceptionFrameFailure:
    payload: dict[str, object] = dict(details)
    return PerceptionFrameFailure(code=code, message=message, details=payload)


def identity_failure(identity: PerceptionFrameIdentity) -> PerceptionFrameFailure | None:
    if identity.worker_boot_id == "":
        return _failure(
            PerceptionFrameFailureCode.MALFORMED_IDENTITY,
            "perception identity worker_boot_id must be a non-empty string",
        )
    if identity.camera_id == "":
        return _failure(
            PerceptionFrameFailureCode.MALFORMED_IDENTITY,
            "perception identity camera_id must be a non-empty string",
        )
    if type(identity.stream_epoch) is not int or identity.stream_epoch < 0:
        return _failure(
            PerceptionFrameFailureCode.MALFORMED_IDENTITY,
            "perception identity stream_epoch must be an integer >= 0",
        )
    if type(identity.seq) is not int or identity.seq < 0:
        return _failure(
            PerceptionFrameFailureCode.MALFORMED_IDENTITY,
            "perception identity seq must be an integer >= 0",
        )
    if identity.source_pts is not None and (
        type(identity.source_pts) is not int or identity.source_pts < 0
    ):
        return _failure(
            PerceptionFrameFailureCode.MALFORMED_IDENTITY,
            "perception identity source_pts must be an integer >= 0 when set",
        )
    return None


def _channel_item_count(channel: PersonBoxChannel | HumanPoseChannel | BedRegionChannel) -> int:
    if isinstance(channel, PersonBoxChannel):
        return len(channel.boxes)
    if isinstance(channel, HumanPoseChannel):
        return len(channel.poses)
    return len(channel.regions)


def channel_state_failure(
    name: str,
    channel: PersonBoxChannel | HumanPoseChannel | BedRegionChannel,
) -> PerceptionFrameFailure | None:
    count = _channel_item_count(channel)
    if channel.state is ChannelState.INFERRED and count == 0:
        return _failure(
            PerceptionFrameFailureCode.INVALID_CHANNEL_STATE,
            f"{name} marked inferred but contains no items",
        )
    if channel.state is ChannelState.INFERRED_EMPTY and count != 0:
        return _failure(
            PerceptionFrameFailureCode.INVALID_CHANNEL_STATE,
            f"{name} marked inferred_empty but contains items",
        )
    if channel.state is ChannelState.SKIPPED and count != 0:
        return _failure(
            PerceptionFrameFailureCode.INVALID_CHANNEL_STATE,
            f"{name} marked skipped but contains items",
        )
    return None


def association_failure(
    identity: PerceptionFrameIdentity,
    association: AssociationResult | None,
    *,
    person_box_count: int,
) -> PerceptionFrameFailure | None:
    if association is None:
        return None
    if association.cue_source != PERSON_BOX_CUE_SOURCE:
        return _failure(
            PerceptionFrameFailureCode.BED_IDENTITY_CUE,
            "bed regions cannot create, update, or evict a person track",
            cue_source=association.cue_source,
        )
    if association.identity.stream_epoch < identity.stream_epoch:
        return _failure(
            PerceptionFrameFailureCode.STALE_EPOCH,
            "association stream_epoch is stale relative to the frame identity",
            frame_epoch=identity.stream_epoch,
            association_epoch=association.identity.stream_epoch,
        )
    if association.identity != identity:
        return _failure(
            PerceptionFrameFailureCode.EPOCH_MISMATCH,
            "association epoch does not match the frame durable identity",
            frame_epoch=identity.stream_epoch,
            association_epoch=association.identity.stream_epoch,
        )
    if len(association.track_ids) != len(association.selected_cue_indexes):
        return _failure(
            PerceptionFrameFailureCode.INVALID_CUE_INDEX,
            "association track_ids and selected_cue_indexes must be the same length",
        )
    if any(
        type(index) is not int or index < 0 or index >= person_box_count
        for index in association.selected_cue_indexes
    ):
        return _failure(
            PerceptionFrameFailureCode.INVALID_CUE_INDEX,
            "association selected_cue_indexes must address person_box cues",
            person_box_count=person_box_count,
            selected_cue_indexes=association.selected_cue_indexes,
        )
    return None


def assemble_perception_frame(
    *,
    identity: PerceptionFrameIdentity,
    person_box: PersonBoxChannel,
    human_pose: HumanPoseChannel,
    bed_region: BedRegionChannel,
    association: AssociationResult | None,
) -> PerceptionFrameV1 | PerceptionFrameFailure:
    failed = identity_failure(identity)
    if failed is not None:
        return failed
    if association is not None:
        failed = association_failure(
            identity,
            association,
            person_box_count=len(person_box.boxes),
        )
        if failed is not None:
            return failed
    for name, channel in (
        ("person_box", person_box),
        ("human_pose", human_pose),
        ("bed_region", bed_region),
    ):
        failed = channel_state_failure(name, channel)
        if failed is not None:
            return failed
    return PerceptionFrameV1(
        identity=identity,
        person_box=person_box,
        human_pose=human_pose,
        bed_region=bed_region,
        association=association,
    )


__all__ = [
    "LEGACY_ASSOCIATION_STRATEGY",
    "PERSON_BOX_CUE_SOURCE",
    "AssociationResult",
    "BedRegion",
    "BedRegionChannel",
    "ChannelState",
    "HumanPoseChannel",
    "Keypoint",
    "PerceptionFrameFailure",
    "PerceptionFrameFailureCode",
    "PerceptionFrameIdentity",
    "PerceptionFrameV1",
    "PersonBox",
    "PersonBoxChannel",
    "assemble_perception_frame",
    "association_failure",
    "channel_state_failure",
    "identity_failure",
]
