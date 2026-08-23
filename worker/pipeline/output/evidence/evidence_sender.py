"""Delivery-queue sender for evidence envelopes."""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

from shared.events.delivery_queue import DeliveryQueue
from shared.events.evidence_export_client import RelayEvidenceClient
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)

_LOGGER = logging.getLogger(__name__)

#: Delivery attempts before an entry is retained for an operator instead of
#: being retried forever. Retrying forever halts every entry behind it.
_MAX_ENTRY_ATTEMPTS: Final = 10

#: Recorded on a dead-lettered entry that exhausted its attempts rather
#: than being explicitly refused by the backend.
_EXHAUSTED_STATUS: Final = 599
_SHED_DETAIL_WARNING_INTERVAL_SECONDS = 60.0


class SenderStep(StrEnum):
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    EVENT_ACKED = "EVENT_ACKED"
    CLIP_ACKED = "CLIP_ACKED"
    IDLE = "IDLE"


class EvidenceTransport(Protocol):
    def send_event(
        self, payload_json: str, edge_event_id: str
    ) -> EventReceipt | DeliveryFailure: ...

    def send_snapshot_attachment(
        self, payload: dict[str, object]
    ) -> None | DeliveryFailure: ...

    def send_snapshot_disposition(
        self, payload: dict[str, object]
    ) -> None | DeliveryFailure: ...


@dataclass(frozen=True, slots=True)
class SenderConfig:
    relay_url: str
    relay_token: str = field(repr=False)
    probe_camera_id: str


class EvidenceSender:
    """Send one immutable queue entry and acknowledge only that entry."""

    def __init__(
        self,
        queue_directory: Path,
        config: SenderConfig,
        *,
        transport: EvidenceTransport | None = None,
        clip_export_enabled: Callable[[], bool] | None = None,
        **_ignored: object,
    ) -> None:
        del clip_export_enabled
        self.queue_directory = queue_directory
        self.config = config
        self._transport = transport or RelayEvidenceClient(config.relay_url, config.relay_token)
        self._last_shed_detail_warning_at = float("-inf")
        self._attempts: dict[str, int] = {}
        self._deferred: set[str] = set()

    def _retain(self, queue: DeliveryQueue, entry_id: str, status_code: int) -> bool:
        """Retain a refused entry, treating an I/O failure as a refusal.

        `dead_letter` writes to the filesystem, so it can raise for exactly the
        reason the entry could not be delivered in the first place -- a full or
        failing disk. Unguarded, that exception escaped this branch entirely and
        the outer sender loop caught it silently without deferring, so the same
        head was reselected on every iteration and every newer resident event
        behind it was blocked indefinitely.

        A guard that depends on the resource it is guarding against is not a
        guard. Returning False here routes the failure into the same path as a
        full retention area: the entry stays live, gets deferred, and the truth
        is logged.
        """
        try:
            return queue.dead_letter(entry_id, status_code)
        except Exception:  # noqa: BLE001 - retention I/O never stalls the queue
            _LOGGER.exception(
                "could not retain evidence entry %s; the retention area is "
                "unwritable, so the entry stays queued and undelivered",
                entry_id,
            )
            return False

    def _select(self, entries: tuple[dict[str, object], ...]) -> dict[str, object]:
        """Pick the next entry to send, skipping ones that keep failing.

        Selection used to be a fixed preference for the first EVENT, and
        `entries()` returns a sorted, deterministic order. So an entry failing
        with a retryable status was re-selected on every single call and every
        other entry behind it -- including newer fall evidence -- never got sent
        at all. One bad entry silently halted the whole delivery queue.

        Entries that have exhausted their attempt budget are passed over so the
        rest of the queue drains; they are dead-lettered separately rather than
        being retried forever or deleted.
        """
        live = tuple(
            item
            for item in entries
            if self._attempts.get(str(item["entry_id"]), 0) < _MAX_ENTRY_ATTEMPTS
        )
        candidates = live or entries
        # Rotate past entries that failed on a recent pass. A transient failure
        # never exhausts the attempt budget -- correctly, because a relay outage
        # is what the durable queue exists to survive -- so the budget alone
        # cannot keep a failing head from blocking everything behind it. When
        # every candidate has been deferred the cycle restarts, so a queue whose
        # entries all fail still retries them all.
        undeferred = tuple(
            item for item in candidates if str(item["entry_id"]) not in self._deferred
        )
        if not undeferred:
            self._deferred.clear()
            undeferred = candidates
        return next(
            (item for item in undeferred if item["kind"] == "EVENT"), undeferred[0]
        )

    def run_once(self) -> SenderStep:
        queue = DeliveryQueue(self.queue_directory)
        entries = tuple(queue.entries())
        if not entries:
            return SenderStep.IDLE
        entry = self._select(entries)
        entry_id = str(entry["entry_id"])
        attempts = self._attempts.get(entry_id, 0)
        if attempts >= _MAX_ENTRY_ATTEMPTS:
            # Exhausted. Retain it for an operator instead of retrying forever
            # or deleting it, so the queue behind it can drain.
            if self._retain(queue, entry_id, _EXHAUSTED_STATUS):
                self._attempts.pop(entry_id, None)
                self._deferred.discard(entry_id)
                _LOGGER.error(
                    "evidence entry %s exhausted %d delivery attempts; retained in %s "
                    "for operator review so the queue can drain, NOT delivered",
                    entry_id,
                    _MAX_ENTRY_ATTEMPTS,
                    queue.dead_letter_directory,
                )
                return SenderStep.RETRY_SCHEDULED
            # Retention is full. The entry stays in the live queue, so it MUST be
            # deferred: without that it is reselected on every call forever, the
            # queue never drains, and new alerts are eventually refused
            # admission. Reporting it retained here would also be a lie.
            self._deferred.add(entry_id)
            _LOGGER.error(
                "evidence entry %s exhausted %d delivery attempts and the retention "
                "area at %s is FULL; the entry is still queued and undelivered. Run "
                "scripts/ops/review-refused-evidence.py to drain retention",
                entry_id,
                _MAX_ENTRY_ATTEMPTS,
                queue.dead_letter_directory,
            )
            return SenderStep.RETRY_SCHEDULED
        try:
            result = self._send(entry)
        except Exception:  # noqa: BLE001 - one bad entry never starves the queue
            # The outer loop catches this too, but silently and without moving
            # on, so a corrupt or unserialisable entry was reselected on every
            # iteration and nothing behind it was ever delivered. Defer it and
            # say so; the entry stays durable and is retried on the next cycle.
            self._deferred.add(entry_id)
            self._attempts[entry_id] = attempts + 1
            _LOGGER.exception(
                "evidence entry %s could not be sent; deferring it so newer "
                "evidence still drains (attempt %d of %d)",
                entry_id,
                attempts + 1,
                _MAX_ENTRY_ATTEMPTS,
            )
            return SenderStep.RETRY_SCHEDULED
        if isinstance(result, DeliveryFailure):
            if result.status_code == 422:
                # Refused, not delivered. Retain it for an operator rather than
                # deleting it and reporting success; that deletion is how 41
                # real bed-exit events were destroyed here.
                if self._retain(queue, entry_id, result.status_code):
                    self._deferred.discard(entry_id)
                    _LOGGER.error(
                        "backend refused evidence entry %s with HTTP 422; retained in "
                        "%s for operator review, NOT delivered",
                        entry_id,
                        queue.dead_letter_directory,
                    )
                    return SenderStep.RETRY_SCHEDULED
                # Retention full: the entry remains queued and must be deferred,
                # or this refused entry is reselected forever and blocks every
                # newer alert behind it until admission itself starts failing.
                self._deferred.add(entry_id)
                _LOGGER.error(
                    "backend refused evidence entry %s with HTTP 422 and the retention "
                    "area at %s is FULL; the entry is still queued and undelivered. Run "
                    "scripts/ops/review-refused-evidence.py to drain retention",
                    entry_id,
                    queue.dead_letter_directory,
                )
                return SenderStep.RETRY_SCHEDULED
            self._deferred.add(entry_id)
            if result.disposition is DeliveryDisposition.RETRY:
                # Transient: the relay is unreachable, restarting, or answering
                # 5xx. This is precisely the condition the durable queue exists
                # to survive, so it must NOT consume the attempt budget. Counting
                # it turned an outage into mass dead-lettering of perfectly good
                # evidence -- the opposite of the guarantee.
                return SenderStep.RETRY_SCHEDULED
            self._attempts[entry_id] = attempts + 1
            return SenderStep.RETRY_SCHEDULED
        if (
            entry["kind"] == "EVENT"
            and isinstance(result, EventReceipt)
            and result.edge_event_id != entry["edge_event_id"]
        ):
            self._attempts[entry_id] = attempts + 1
            return SenderStep.RETRY_SCHEDULED
        self._attempts.pop(entry_id, None)
        try:
            queue.acknowledge(entry_id)
        except Exception:  # noqa: BLE001 - a delivered entry never stalls the queue
            # The backend has it; only our removal failed. Unguarded, a
            # filesystem fault here re-selected the same entry on every
            # iteration: the backend was flooded with duplicates of one event
            # while every newer resident event behind it never left the queue.
            # Defer so the rest drains; the entry is safely redelivered later
            # and the backend deduplicates it.
            self._deferred.add(entry_id)
            _LOGGER.exception(
                "evidence entry %s was delivered but could not be removed from "
                "the queue; deferring it so newer evidence still drains",
                entry_id,
            )
            return SenderStep.RETRY_SCHEDULED
        self._deferred.discard(entry_id)
        return _acknowledged_step(entry)

    def _send(self, entry: dict[str, object]) -> None | EventReceipt | DeliveryFailure:
        match entry["kind"]:
            case "EVENT":
                self._warn_shed_detail(entry)
                return self._transport.send_event(
                    _payload(entry), str(entry["edge_event_id"])
                )
            case "SNAPSHOT_ATTACHMENT":
                return self._transport.send_snapshot_attachment(_media_payload(entry))
            case "SNAPSHOT_DISPOSITION":
                return self._transport.send_snapshot_disposition(_media_payload(entry))
            case _:
                raise ValueError(f"unknown delivery entry kind: {entry['kind']!r}")

    def _warn_shed_detail(self, entry: dict[str, object]) -> None:
        shed_detail_keys = entry.get("shed_detail_keys", [])
        if not isinstance(shed_detail_keys, list) or not shed_detail_keys:
            return
        now = time.monotonic()
        if now - self._last_shed_detail_warning_at < _SHED_DETAIL_WARNING_INTERVAL_SECONDS:
            return
        self._last_shed_detail_warning_at = now
        _LOGGER.warning(
            "relay event detail shed before delivery: edge_event_id=%s camera_id=%s keys=%s",
            entry["edge_event_id"],
            entry["camera_id"],
            ",".join(str(key) for key in shed_detail_keys),
        )


def _payload(entry: dict[str, object]) -> str:
    """Rebuild the event body with its decision envelope; media entries are tagged.

    ``DurableEvidenceStager`` splits the staged event: it pops ``audit`` out of
    the body into ``decision_trace`` so the decision basis is admitted atomically
    with the event as its own bounded field. The wire contract the backend reads
    is still a single alert carrying ``audit``, so the two halves must be rejoined
    here. Sending ``values`` alone silently drops the decision basis -- the one
    thing the never-drop obligation exists to protect -- and the relay projection
    then records ``audit=None``.
    """
    if entry["kind"] != "EVENT":
        return json.dumps(entry, separators=(",", ":"), sort_keys=True)

    body = json.loads(base64.b64decode(str(entry["values_b64"])).decode("ascii"))
    trace = json.loads(base64.b64decode(str(entry["decision_trace_b64"])).decode("ascii"))
    if trace:
        body["audit"] = trace
    return json.dumps(body, separators=(",", ":"), sort_keys=True)


def _media_payload(entry: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in entry.items() if key not in {"entry_id", "kind"}}


def _acknowledged_step(entry: dict[str, object]) -> SenderStep:
    return SenderStep.EVENT_ACKED if entry["kind"] == "EVENT" else SenderStep.CLIP_ACKED


__all__ = ["EvidenceSender", "SenderConfig", "SenderStep"]
