"""Capability protocol for worker-internal PerceptionFrame adaptation.

Channel payloads (`PersonBoxChannel`, `HumanPoseChannel`, `BedRegionChannel`,
`AssociationResult`) live only as frozen dataclasses in
``worker.types.perception_frame``. They are not re-declared as Protocols here:
a second object with the same name would make imports ambiguous for C4, and
``@runtime_checkable`` only checks attribute presence, not types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from contracts.runner import BedRunnerResult, PersonRunnerResult, PoseRunnerResult
from worker.types.perception_frame import (
    LEGACY_ASSOCIATION_STRATEGY,
    PERSON_BOX_CUE_SOURCE,
    PerceptionFrameFailure,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
)


@runtime_checkable
class PerceptionFrameAdapter(Protocol):
    """Map inference outputs onto PerceptionFrameV1 or a typed failure."""

    def adapt(
        self,
        *,
        identity: PerceptionFrameIdentity,
        pose: PoseRunnerResult | None = None,
        person: PersonRunnerResult | None = None,
        bed: BedRunnerResult | None = None,
        track_ids: tuple[int, ...] | None = None,
        selected_cue_indexes: tuple[int, ...] | None = None,
        association_identity: PerceptionFrameIdentity | None = None,
        association_strategy: str = LEGACY_ASSOCIATION_STRATEGY,
        association_cue_source: str = PERSON_BOX_CUE_SOURCE,
    ) -> PerceptionFrameV1 | PerceptionFrameFailure: ...

    def parse(
        self,
        payload: Mapping[str, object],
    ) -> PerceptionFrameV1 | PerceptionFrameFailure: ...

    def diagnostic(self, frame: PerceptionFrameV1) -> Mapping[str, object]: ...


__all__ = [
    "PerceptionFrameAdapter",
]
