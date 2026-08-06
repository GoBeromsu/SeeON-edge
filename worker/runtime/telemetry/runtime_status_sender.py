"""Background delivery for the frozen runtime-status relay payload."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, final

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.evidence_http_transport import (
    bounded_request,
    classify_http_failure,
    encode_json,
    join_http_url,
    normalize_http_base,
    parse_json_object,
)
from shared.events.relay_failure_log import RelayFailureLog
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.wire import RelayRuntimeStatusPayload

LOGGER: Final = logging.getLogger(__name__)

RUNTIME_STATUS_PATH = "/api/v1/relay/runtime-status"


@dataclass(frozen=True, slots=True)
class RuntimeStatusSenderConfig:
    """Bounded cadence, retry, and shutdown settings."""

    publish_interval_sec: float = 5.0
    initial_backoff_sec: float = 1.0
    max_backoff_sec: float = 30.0


class RuntimeStatusTransport(Protocol):
    """Typed runtime-status relay boundary."""

    def send(self, payload: RelayRuntimeStatusPayload) -> int | None: ...


class RuntimeHttpRequest(Protocol):
    """Bounded HTTP request callable shared with evidence delivery."""

    def __call__(
        self,
        url: str,
        method: str,
        headers: Mapping[str, str],
        data: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes] | DeliveryFailure: ...


@final
class RelayRuntimeStatusTransport:
    """Send one closed status payload through the shared bounded HTTP transport."""

    def __init__(
        self,
        relay_url: str,
        relay_token: str,
        timeout_sec: float = 2.0,
        request: RuntimeHttpRequest = bounded_request,
    ) -> None:
        self._url = join_http_url(normalize_http_base(relay_url), RUNTIME_STATUS_PATH)
        self._relay_token = relay_token
        self._timeout_sec = timeout_sec
        self._request = request
        self._failure_log = RelayFailureLog(LOGGER, channel="runtime-status", method="POST")

    def send(self, payload: RelayRuntimeStatusPayload) -> int | None:
        result = self._request(
            self._url,
            "POST",
            {
                "Authorization": f"Bearer {self._relay_token}",
                "Content-Type": "application/json",
            },
            encode_json(payload),
            self._timeout_sec,
        )
        if isinstance(result, DeliveryFailure):
            self._failure_log.record_failure(result, path=RUNTIME_STATUS_PATH)
            return None
        status, headers, body = result
        if not 200 <= status < 300:
            self._failure_log.record_failure(
                classify_http_failure(status, headers), path=RUNTIME_STATUS_PATH
            )
            return None
        response = parse_json_object(body)
        generation = response.get("generation")
        if response.get("accepted") is not True or not isinstance(generation, int):
            failure = DeliveryFailure(
                DeliveryDisposition.RETRY, "MALFORMED_RESPONSE", status_code=status
            )
            self._failure_log.record_failure(failure, path=RUNTIME_STATUS_PATH)
            return None
        self._failure_log.record_success(path=RUNTIME_STATUS_PATH)
        return generation if generation >= 0 else None


@final
class RuntimeStatusSender:
    """Publish telemetry away from ingest and frame-processing threads."""

    def __init__(
        self,
        diagnostics: WorkerDiagnostics,
        facility_id: str | Mapping[str, str],
        transport: RuntimeStatusTransport,
        config: RuntimeStatusSenderConfig | None = None,
        *,
        before_publish: Callable[[], None] | None = None,
    ) -> None:
        resolved_config = RuntimeStatusSenderConfig() if config is None else config
        self._diagnostics = diagnostics
        self._facility_id = facility_id
        self._transport = transport
        # Invoked on every publish (the initial synchronous one in `start()`
        # and every subsequent tick in `_run()`) just before building the
        # payload, so a live-stats provider (e.g. the clip recorder's
        # `ClipRecorderStats`, #165) can push its latest snapshot into
        # `diagnostics` right before it is read -- `WorkerDiagnostics` itself
        # must not depend on `ClipRecorder` (layering), so the composition
        # root (`WorkerRuntime`) supplies this instead.
        self._before_publish = before_publish
        self._publish_interval_sec = max(0.0, resolved_config.publish_interval_sec)
        self._initial_backoff_sec = max(0.0, resolved_config.initial_backoff_sec)
        self._max_backoff_sec = max(
            self._initial_backoff_sec,
            resolved_config.max_backoff_sec,
        )
        self._snapshots: queue.Queue[list[RelayRuntimeStatusPayload]] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation_by_facility: dict[str, int] = {}
        self._seq_by_facility: dict[str, int] = {}
        self._generation: int | None = None
        self._state_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        _ = self.publish()
        self._thread = threading.Thread(
            target=self._run,
            name="runtime-status-sender",
            daemon=True,
        )
        self._thread.start()

    def publish(self) -> bool:
        """Replace the pending slot with the newest diagnostics without blocking."""
        snapshots = self._snapshots_for_publish()
        try:
            self._snapshots.put_nowait(snapshots)
        except queue.Full:
            try:
                _ = self._snapshots.get_nowait()
                self._snapshots.task_done()
            except queue.Empty:
                return False
            try:
                self._snapshots.put_nowait(snapshots)
            except queue.Full:
                return False
        return True

    def submit(self) -> bool:
        return self.publish()

    def publish_once(self) -> bool:
        """Synchronously make one bounded delivery attempt without retrying."""
        return self._post(self._snapshots_for_publish())

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def generation(self) -> int | None:
        with self._state_lock:
            return self._generation

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        pending: list[RelayRuntimeStatusPayload] | None = None
        delay = 0.0
        failures = 0
        while not self._stop_event.wait(delay):
            self._log_local_snapshot()
            _ = self.publish()
            replacement = self._take_latest()
            if replacement is not None:
                pending = replacement
            if pending is None:
                delay = self._publish_interval_sec
                continue
            if self._post(pending):
                failures = 0
                delay = self._publish_interval_sec
            else:
                failures += 1
                backoff = self._initial_backoff_sec * (2.0 ** (failures - 1))
                delay = min(
                    self._max_backoff_sec,
                    backoff,
                )

    def _log_local_snapshot(self) -> None:
        """Emit `WorkerDiagnostics.log_snapshot()` on this sender's own tick.

        Issue #207: `WorkerDiagnostics.log_snapshot()` existed but had no
        production caller anywhere, so the bed-region cache state it now
        carries (and the pre-existing stage-timing/bus/encode fields, which
        had the same problem) never actually reached a worker log line. This
        thread already runs an independent, non-per-frame background tick
        for the exact stated reason ("publish telemetry away from ingest and
        frame-processing threads"), so it is reused here rather than adding a
        second timer thread.

        Deliberately decoupled from `_post`'s relay-delivery outcome: a log
        line is not a relay call and must not inherit its retry/backoff
        rhythm or be skipped because the relay is unreachable. The one
        accepted trade-off is that a relay outage's growing backoff also
        slows this local logging tick (bounded by `_max_backoff_sec`,
        default 30s) -- still periodic, never per-frame, so it does not
        reproduce the failure mode #207 warns against; call it explicitly
        if that coupling ever needs to be broken.

        Wrapped so a `log_snapshot()` defect can never take down relay
        delivery, which is this sender's primary job.
        """
        try:
            self._diagnostics.log_snapshot()
        except Exception:  # noqa: BLE001 - local logging must never break relay delivery
            LOGGER.warning("worker diagnostics log_snapshot failed", exc_info=True)

    def _take_latest(self) -> list[RelayRuntimeStatusPayload] | None:
        latest: list[RelayRuntimeStatusPayload] | None = None
        while True:
            try:
                latest = self._snapshots.get_nowait()
            except queue.Empty:
                return latest
            self._snapshots.task_done()

    def _snapshots_for_publish(self) -> list[RelayRuntimeStatusPayload]:
        if self._before_publish is not None:
            self._before_publish()
        if isinstance(self._facility_id, str):
            return [self._diagnostics.to_payload(self._facility_id, None, 0)]
        return self._diagnostics.to_payloads(self._facility_id, None, 0)

    def _post(self, snapshots: list[RelayRuntimeStatusPayload]) -> bool:
        for snapshot in snapshots:
            facility_id = snapshot["facility_id"]
            with self._state_lock:
                seq = self._seq_by_facility.get(facility_id, 0) + 1
                self._seq_by_facility[facility_id] = seq
                generation = self._generation_by_facility.get(facility_id)
            payload = snapshot.copy()
            payload["generation"] = generation
            payload["seq"] = seq
            accepted_generation = self._transport.send(payload)
            if accepted_generation is None:
                return False
            with self._state_lock:
                self._generation_by_facility[facility_id] = accepted_generation
                self._generation = accepted_generation
        return True


__all__ = [
    "RelayRuntimeStatusTransport",
    "RuntimeHttpRequest",
    "RuntimeStatusSender",
    "RuntimeStatusSenderConfig",
    "RuntimeStatusTransport",
]
