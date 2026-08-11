"""Rate-limited, classified logging for relay HTTP failures (issue #184).

Every worker -> relay call site (heartbeat, runtime-status, evidence
alerts/clips/capabilities) used to log an identical bare line on every failed
attempt -- no status code, no endpoint, no way to tell "DNS/connection
refused" apart from "our config is wrong" (4xx) or "upstream is down" (5xx),
and no way to tell when it started working again. On an unhealthy relay this
produced several identical lines per second that drowned out every other
worker log line (#184).

``RelayFailureLog`` is the shared reporter each call site owns one instance
of (one per logical channel: heartbeat is per camera, runtime-status is
process-wide, evidence delivery is one per operation kind). It logs full
detail on the first failure of a kind and on every failure-class change,
folds repeats into a periodic summary line, and logs recovery exactly once.

Never logs response bodies or request headers -- only the status code (or
transport exception class already reduced to ``DeliveryFailure`` by
``shared.events.evidence_http_transport``), a static per-status hint, and the
caller-supplied endpoint path. Secrets (relay tokens, auth headers) never
reach this module.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, final

from shared.events.evidence_export_contract import DeliveryFailure

DEFAULT_SUMMARY_INTERVAL_SEC: Final = 60.0


class RelayFailureClass(StrEnum):
    """How an operator should read a relay failure."""

    TRANSPORT = "transport"  # DNS / connect / timeout -- no HTTP response at all
    CLIENT_ERROR = "client_error"  # 4xx -- our config is wrong; retrying will not fix it
    SERVER_ERROR = "server_error"  # 5xx (or an unreadable 2xx body) -- upstream problem


_CLIENT_ERROR_HINTS: Final[dict[int, str]] = {
    401: "check relay token",
    403: "check runtime enrollment / auth",
}
_DEFAULT_CLIENT_HINT: Final = "check worker relay config"
_DEFAULT_SERVER_HINT: Final = "upstream relay is down; will keep retrying"
_DEFAULT_TRANSPORT_HINT: Final = "cannot reach relay host; will keep retrying"


@dataclass(frozen=True, slots=True)
class RelayFailureOutcome:
    """One classified relay failure, safe to log as-is."""

    failure_class: RelayFailureClass
    reason: str  # "403" / "URLError: <urlopen error ...>" -- never response-body content
    hint: str


def classify_relay_failure(failure: DeliveryFailure) -> RelayFailureOutcome:
    """Classify a ``DeliveryFailure`` for diagnostic logging.

    Only reads ``status_code``/``transport_error``/``code`` -- fields already
    sanitized by ``classify_http_failure``/``bounded_request`` -- never the
    raw response body.
    """
    if failure.transport_error is not None:
        return RelayFailureOutcome(
            RelayFailureClass.TRANSPORT, failure.transport_error, _DEFAULT_TRANSPORT_HINT
        )
    status = failure.status_code
    if status is None:
        return RelayFailureOutcome(
            RelayFailureClass.SERVER_ERROR, failure.code, _DEFAULT_SERVER_HINT
        )
    if 400 <= status <= 499:
        return RelayFailureOutcome(
            RelayFailureClass.CLIENT_ERROR,
            str(status),
            _CLIENT_ERROR_HINTS.get(status, _DEFAULT_CLIENT_HINT),
        )
    return RelayFailureOutcome(RelayFailureClass.SERVER_ERROR, str(status), _DEFAULT_SERVER_HINT)


@dataclass(slots=True)
class _ActiveFailure:
    outcome: RelayFailureOutcome
    attempt: int
    first_seen: float
    last_logged: float
    since_last_summary: int


@final
class RelayFailureLog:
    """Dedupe/rate-limit relay failure logging for one logical channel.

    ``channel`` is a short human label ("heartbeat", "runtime-status",
    "relay event delivery", ...); ``method`` is the fixed HTTP verb for the
    channel. ``path`` is passed per call (it can vary, e.g. per clip id) so
    every logged line still names its endpoint (issue #184 deliverable 1).
    """

    def __init__(
        self,
        logger: logging.Logger,
        *,
        channel: str,
        method: str,
        summary_interval_sec: float = DEFAULT_SUMMARY_INTERVAL_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logger = logger
        self._channel = channel
        self._method = method
        self._summary_interval_sec = summary_interval_sec
        self._clock = clock
        self._active: _ActiveFailure | None = None

    def record_failure(self, failure: DeliveryFailure, *, path: str) -> None:
        outcome = classify_relay_failure(failure)
        now = self._clock()
        active = self._active
        if active is None or active.outcome.failure_class != outcome.failure_class:
            self._active = _ActiveFailure(outcome, 1, now, now, 0)
            self._logger.log(
                _level(outcome.failure_class),
                "relay %s %s -> %s (%s: %s) [%s, attempt 1]",
                self._method,
                path,
                outcome.reason,
                outcome.failure_class.value,
                outcome.hint,
                self._channel,
                extra={
                    "relay_channel": self._channel,
                    "relay_status_code": failure.status_code,
                    "relay_failure_class": outcome.failure_class.value,
                },
            )
            return
        active.attempt += 1
        active.since_last_summary += 1
        elapsed = now - active.last_logged
        if elapsed < self._summary_interval_sec:
            return
        self._logger.log(
            _level(outcome.failure_class),
            "relay %s %s still failing: %s (%s: %s) "
            "[%s, attempt %d, %d occurrence(s) in the last %.0fs]",
            self._method,
            path,
            outcome.reason,
            outcome.failure_class.value,
            outcome.hint,
            self._channel,
            active.attempt,
            active.since_last_summary,
            elapsed,
            extra={
                "relay_channel": self._channel,
                "relay_status_code": failure.status_code,
                "relay_failure_class": outcome.failure_class.value,
            },
        )
        active.last_logged = now
        active.since_last_summary = 0

    def record_success(self, *, path: str) -> None:
        active = self._active
        if active is None:
            return
        now = self._clock()
        self._logger.info(
            "relay %s %s recovered after %d failed attempt(s) over %.1fs [%s]",
            self._method,
            path,
            active.attempt,
            now - active.first_seen,
            self._channel,
            extra={"relay_channel": self._channel},
        )
        self._active = None


def _level(failure_class: RelayFailureClass) -> int:
    return logging.ERROR if failure_class is RelayFailureClass.CLIENT_ERROR else logging.WARNING


__all__ = [
    "DEFAULT_SUMMARY_INTERVAL_SEC",
    "RelayFailureClass",
    "RelayFailureLog",
    "RelayFailureOutcome",
    "classify_relay_failure",
]
