"""``AssociationStrategy`` port: the seam every native strategy implements.

The native `GstBaseTransform` calls exactly one strategy per camera through
this narrow surface. `observe` takes the caller's durable
`PerceptionFrameIdentity` plus a typed `PersonBoxChannel` -- never an
undifferentiated box tuple and never a `BedRegionChannel` -- and returns the
real C1 `AssociationResult` (`worker/types/perception_frame.py`), bound to
that identity, with `cue_source` fixed at `"person_box"` and
`selected_cue_indexes` in person-box input order. Bed masks are structurally
unrepresentable at this boundary: the port has no parameter a `BedRegionChannel`
can satisfy, so bed regions cannot create, update, or evict a person track
(Task 4 guardrail).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types.perception_frame import (
    AssociationResult,
    PerceptionFrameIdentity,
    PersonBoxChannel,
)


@runtime_checkable
class AssociationStrategy(Protocol):
    """Stateful per-camera person-identity assignment across frames."""

    #: Explicit version identity. Config/registry proves only one strategy's
    #: identity is active at cutover; never inferred from the class name.
    identity: str

    @property
    def live_ids(self) -> frozenset[int]:
        """Every track id this strategy currently considers live."""
        ...

    def observe(
        self,
        identity: PerceptionFrameIdentity,
        person_box: PersonBoxChannel,
    ) -> AssociationResult:
        """Apply one actual inference observation, including an empty one."""
        ...

    def coast(self) -> None:
        """Preserve every track when this frame carried no inference result."""
        ...

    def reset(self) -> None:
        """Drop all track state. Called on reconnect / stream-epoch rollover.

        A fresh boot id or a rolled stream epoch must never let the next
        frame observe a track id minted before the reset.
        """
        ...


__all__ = ["AssociationStrategy"]
