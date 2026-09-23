from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from worker.interfaces.decision import (
    Decider,
    FreshnessProvider,
    ShadowTraceProvider,
    TraceSnapshotProvider,
)
from worker.pipeline.decision.incident_manager import IncidentManager
from worker.types import (
    AttributedSnapshot,
    BusinessEvent,
    DecisionIdentity,
    DecisionInput,
    DecisionTraceSnapshot,
)

#: Producers retained for late releases. Bounded so memory cannot grow.
_MAX_TRACKED_PRODUCERS: Final = 64


_MAX_WRAPPER_DEPTH = 8


def unwrap_decider(decider: object) -> object:
    """Reach through window/policy wrappers to the inner decider.

    Bounded like ``EventAggregator.release``'s walk: a cyclic wrapper chain
    stops at the depth cap instead of spinning. ``release`` keeps its own
    walk because it stops at the first object exposing ``release_onset``,
    not at the innermost object.
    """
    target: object = decider
    for _ in range(_MAX_WRAPPER_DEPTH):
        inner = getattr(target, "decider", None)
        if inner is None:
            return target
        target = inner
    return target


@dataclass(frozen=True, slots=True)
class EventAggregator:
    deciders: tuple[Decider, ...]
    incidents: IncidentManager
    monotonic: Callable[[], float] = time.monotonic
    #: Parallel to ``deciders``: the module/policy identity composition gave
    #: each decider, or None when it gave none. Empty means "no identities".
    identities: tuple[DecisionIdentity | None, ...] = ()
    _producers: dict[str, tuple[Decider, BusinessEvent]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.identities and len(self.identities) != len(self.deciders):
            raise ValueError("identities must be empty or parallel to deciders")

    @property
    def last_trace_snapshots(self) -> tuple[DecisionTraceSnapshot, ...]:
        return tuple(item.snapshot for item in self.attributed_trace_snapshots())

    def attributed_trace_snapshots(self) -> tuple[AttributedSnapshot, ...]:
        """Every decider's snapshots, each tagged with its own identity and role.

        A decider that also exposes ``last_shadow_trace_count`` marks that many
        trailing snapshots as ``shadow``; everything else is ``authoritative``.
        Nothing here ever borrows another decider's identity.
        """
        identities = self.identities or (None,) * len(self.deciders)
        out: list[AttributedSnapshot] = []
        for index, (decider, identity) in enumerate(zip(self.deciders, identities, strict=True)):
            if not isinstance(decider, TraceSnapshotProvider):
                continue
            snapshots = decider.last_trace_snapshots
            # The object whose last_trace_snapshots we read is the one whose
            # freshness and shadow facts apply. A wrapper (window gate) that
            # exposes its own snapshots mirrors those facts; otherwise read
            # them from the innermost decider.
            source: object = (
                decider
                if isinstance(decider, (FreshnessProvider, ShadowTraceProvider))
                else unwrap_decider(decider)
            )
            shadow = (
                source.last_shadow_trace_count if isinstance(source, ShadowTraceProvider) else 0
            )
            fresh = source.last_update_evaluated if isinstance(source, FreshnessProvider) else True
            cut = len(snapshots) - max(0, min(shadow, len(snapshots)))
            for position, snapshot in enumerate(snapshots):
                out.append(
                    AttributedSnapshot(
                        snapshot=snapshot,
                        identity=identity,
                        authority="shadow" if position >= cut else "authoritative",
                        producer_index=index,
                        fresh=fresh,
                    )
                )
        return tuple(out)

    def index_of(self, decider: Decider) -> int | None:
        """Position of ``decider`` (or the wrapper holding it) in ``deciders``."""
        target = unwrap_decider(decider)
        for index, candidate in enumerate(self.deciders):
            if candidate is decider or unwrap_decider(candidate) is target:
                return index
        return None

    def producer_for(self, event_id: str) -> Decider | None:
        """The decider that produced ``event_id`` in this frame, if tracked."""
        pair = self._producers.get(event_id)
        return None if pair is None else pair[0]

    def identity_for(self, decider: Decider) -> DecisionIdentity | None:
        """Identity of ``decider`` (or of the wrapper that holds it), else None."""
        index = self.index_of(decider)
        return None if index is None else (self.identities or (None,) * len(self.deciders))[index]

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        produced: list[tuple[BusinessEvent, Decider]] = [
            (event, decider) for decider in self.deciders for event in decider.update(input_value)
        ]
        produced.sort(key=lambda pair: _event_order(pair[0]))
        now_sec = self.monotonic()
        # Which decider produced each admitted event. A release must reach that
        # decider and no other: matching on a `domain` attribute does not work
        # because the real latches do not declare one, so the check silently
        # passed for every decider and a bed-exit failure re-opened the fall.
        # Deliberately NOT cleared each frame. The pump releases inside the
        # same iteration that produced the events, so clearing would be safe
        # today -- but a release arriving one frame later would then find no
        # producer and silently do nothing, leaving the fall destroyed. Bounded
        # instead, so a slightly late release still lands and memory cannot
        # grow.
        while len(self._producers) >= _MAX_TRACKED_PRODUCERS:
            self._producers.pop(next(iter(self._producers)))
        emitted: list[BusinessEvent] = []
        for event, decider in produced:
            admitted = self.incidents.admit(event, now_sec=now_sec)
            if admitted is None:
                continue
            # Admission replaces the source episode identity with the durable
            # edge identity. Keep both: IncidentManager.release needs the
            # latter to clear its cooldown, while the producer's lifecycle
            # authority recognizes only the former.
            self._producers[str(admitted.identity)] = (decider, event)
            emitted.append(admitted)
        return tuple(emitted)

    def release(self, event: BusinessEvent) -> None:
        """Re-open a decision whose envelope never reached durable storage.

        Two separate pieces of state consumed that decision, and undoing only
        one is not enough. The incident manager recorded a cooldown, and the
        decider consumed its own rising edge. Releasing just the cooldown leaves
        the latch saying "still falling", so the next positive frame is not an
        onset and emits nothing -- the fall is still destroyed, only less
        obviously.
        """
        self.incidents.release(event)
        producer = self._producers.pop(str(event.identity), None)
        for decider, source_event in () if producer is None else (producer,):
            # Reach through wrappers. Production wraps deciders (see
            # _WindowGatedDecider), and a plain getattr on the wrapper returns
            # None, so the release would silently never arrive.
            target: object | None = decider
            seen = 0
            while target is not None and not hasattr(target, "release_onset"):
                seen += 1
                if seen > 8:  # pragma: no cover - cycle guard
                    target = None
                    break
                target = getattr(target, "decider", None)
            if target is None:
                continue
            target.release_onset(source_event)


def _event_order(event: BusinessEvent) -> tuple[str, ...]:
    identity_kind, identity_value = _identity_order(event.identity)
    return (
        event.camera_id,
        event.facility_id,
        event.domain,
        event.event_type,
        f"{event.time_sec:.17g}",
        "" if event.bed_id is None else str(event.bed_id),
        "" if event.person_id is None else str(event.person_id),
        identity_kind,
        identity_value,
    )


def _identity_order(identity: str | int) -> tuple[str, str]:
    return (type(identity).__name__, str(identity))


__all__ = ["EventAggregator", "unwrap_decider"]
