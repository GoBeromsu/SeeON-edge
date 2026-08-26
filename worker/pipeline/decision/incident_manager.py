from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, TypeAlias, override
from uuid import uuid4

from worker.pipeline.decision.event_identity import EventIdentityStore
from worker.types import BusinessEvent

_LOGGER: Final = logging.getLogger(__name__)

CooldownKey: TypeAlias = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class IncidentAuditSnapshot:
    edge_event_id: str
    source_identity: str | int
    cooldown_key: CooldownKey
    domain: str
    event_type: str
    camera_id: str
    facility_id: str
    time_sec: float
    probability: float
    person_id: int | None
    bed_id: int | None


@dataclass(slots=True)
class IncidentConfigurationError(ValueError):
    cooldown_sec: float

    @override
    def __str__(self) -> str:
        return f"cooldown_sec must be non-negative, received {self.cooldown_sec}"


@dataclass(slots=True)  # policy: MUTABLE_OK - owns per-camera cooldown state
class IncidentManager:
    cooldown_sec: float = 30.0
    identity_path: Path | None = None
    _last_seen: dict[CooldownKey, float] = field(default_factory=dict, init=False)
    #: Cooldown key each admitted event was recorded under, so a release can
    #: undo the exact record. Recomputing the key from the ADMITTED event would
    #: silently miss for the identity-keyed branch, because admit() replaces the
    #: identity -- a no-op release that looks like it worked.
    _admitted_keys: dict[str, CooldownKey] = field(default_factory=dict, init=False)
    _identities: EventIdentityStore = field(init=False, repr=False)
    #: Alerts admitted with a fresh identity because the journal failed.
    identity_journal_failures: int = field(default=0, init=False)
    last_audit_snapshot: IncidentAuditSnapshot | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.cooldown_sec < 0.0:
            raise IncidentConfigurationError(self.cooldown_sec)
        try:
            self._identities = EventIdentityStore(self.identity_path)
        except Exception:  # noqa: BLE001 - durability never suppresses detection
            # Same principle as a resolve failure, one step earlier. A journal
            # left malformed by an earlier crash made construction raise, so the
            # camera never activated at all and detected nothing until someone
            # noticed and deleted the file by hand. Losing the stored identities
            # costs deduplication across this restart; losing the camera costs
            # every fall it would have seen.
            # In-memory, with no path at all. An earlier version of this
            # fallback created a scratch journal in a temporary directory, which
            # reintroduced the exact defect it was fixing: if the disk is full
            # -- a very likely reason the real journal failed in the first place
            # -- mkdtemp raises too and the camera still never activates.
            # EventIdentityStore(None) performs no I/O on construction or on
            # resolve, so it cannot fail for the same reason. The unusable file
            # is left exactly where it is for an operator to inspect.
            self._identities = EventIdentityStore(None)
            self.identity_journal_failures += 1
            _LOGGER.error(
                "event identity journal at %s is unusable; continuing without "
                "persisted identities so the camera still detects. A restart may "
                "produce duplicates the backend will deduplicate",
                self.identity_path,
                exc_info=True,
            )

    def admit(
        self,
        event: BusinessEvent,
        *,
        now_sec: float | None = None,
    ) -> BusinessEvent | None:
        event_time = event.time_sec if now_sec is None else now_sec
        key = self.idempotency_key(event, event_time)
        last_seen = self._last_seen.get(key)
        if last_seen is not None and event_time - last_seen < self.cooldown_sec:
            return None

        source_identity = event.identity
        try:
            edge_event_id = self._identities.resolve(_source_key(event))
        except Exception:  # noqa: BLE001 - durability never suppresses an alert
            # The journal exists so a restart reuses the same edge event id and
            # the backend can deduplicate. It is a durability aid, not the
            # decision. Unguarded, any journal I/O failure -- a full disk, a
            # permission change, an fsync error -- propagated out of admit() and
            # the resident's event was never admitted, never queued and never
            # delivered. A fresh identity risks a duplicate alert after a
            # restart, which the backend already deduplicates; a missing alert
            # is the accident this system exists to prevent.
            edge_event_id = str(uuid4())
            self.identity_journal_failures += 1
            _LOGGER.error(
                "event identity journal failed for camera %s; admitting %s with a "
                "fresh identity so the alert is not lost. A restart may produce a "
                "duplicate the backend will deduplicate (failures=%d)",
                event.camera_id,
                event.event_type,
                self.identity_journal_failures,
                exc_info=True,
            )
        admitted = replace(event, identity=edge_event_id)
        self._last_seen[key] = event_time
        self._admitted_keys[edge_event_id] = key
        self.last_audit_snapshot = IncidentAuditSnapshot(
            edge_event_id=edge_event_id,
            source_identity=source_identity,
            cooldown_key=key,
            domain=event.domain,
            event_type=event.event_type,
            camera_id=event.camera_id,
            facility_id=event.facility_id,
            time_sec=event.time_sec,
            probability=event.probability,
            person_id=event.person_id,
            bed_id=event.bed_id,
        )
        return admitted

    def register(
        self,
        event: BusinessEvent,
        *,
        now_sec: float | None = None,
    ) -> BusinessEvent | None:
        return self.admit(event, now_sec=now_sec)


    def release(self, event: BusinessEvent, *, now_sec: float | None = None) -> None:
        """Undo the consumption of a decision whose envelope was never admitted.

        `admit` records a cooldown entry, which is what stops the same fall
        being reported once per frame. If the envelope then fails to reach the
        durable queue, that record is the reason the NEXT frame produces
        nothing: the rising edge has been spent and the fall is lost for good,
        even though staging would have succeeded a frame later.

        A decision must not stay consumed unless its envelope is durable.
        """
        del now_sec
        key = self._admitted_keys.pop(str(event.identity), None)
        if key is not None:
            self._last_seen.pop(key, None)

    def idempotency_key(
        self,
        event: BusinessEvent,
        event_time: float | None = None,
    ) -> CooldownKey:
        del event_time
        if event.domain == "fall":
            return (event.camera_id, event.domain, event.event_type)
        if event.domain == "bed_exit" and event.bed_id is not None:
            return (event.camera_id, event.domain, event.event_type, event.bed_id)
        return (event.camera_id, event.domain, event.event_type, event.identity)

    def reset(self) -> None:
        self._last_seen.clear()
        self.last_audit_snapshot = None


def _source_key(event: BusinessEvent) -> str:
    return json.dumps(
        [
            event.facility_id,
            event.camera_id,
            event.domain,
            event.event_type,
            event.identity,
            event.time_sec,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )


__all__ = [
    "CooldownKey",
    "IncidentAuditSnapshot",
    "IncidentConfigurationError",
    "IncidentManager",
]
