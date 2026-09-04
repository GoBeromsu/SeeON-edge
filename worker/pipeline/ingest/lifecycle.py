from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Generic, Protocol, TypeVar, final

from worker.adapters.model.errors import FatalAcceleratorError
from worker.interfaces.bus import FrameBus
from worker.interfaces.decode import DecodeAdapter, DecodeSession
from worker.pipeline.ingest.registry import ResolvedSource, SourceRegistry
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.pipeline.ingest.rtsp_url import mask_rtsp_url
from worker.types import CURRENT_TEMPORAL_PROFILE, FramePacket

LOGGER: Final = logging.getLogger(__name__)

# Decode supervision constants, deliberately NOT config knobs (plan todo 6):
# an operator tuning these would be tuning the failure detector rather than
# fixing the camera, and a per-site value makes the DEGRADED signal
# incomparable across sites.
DECODE_SILENCE_DEADLINE_SEC: Final = 10.0
DECODE_RESPAWN_INITIAL_BACKOFF_SEC: Final = 0.5
DECODE_RESPAWN_BACKOFF_CAP_SEC: Final = 30.0
DECODE_MAX_RESPAWNS: Final = 8

_DecodeConfigT = TypeVar("_DecodeConfigT")
_DecodeConfigFactory = Callable[[str, ResolvedSource], _DecodeConfigT]


@dataclass(frozen=True, slots=True)
class CapturePolicy:
    max_failures: int = 30
    reconnect_initial_backoff_sec: float = 0.25
    reconnect_max_backoff_sec: float = 5.0
    max_total_reconnects: int | None = None
    target_fps: float = CURRENT_TEMPORAL_PROFILE.target_fps


@dataclass(frozen=True, slots=True)
class DecodeSupervisionPolicy:
    """Bounds on per-camera decode respawn.

    Only the retry budget is a construction parameter: the silence deadline
    and the backoff shape are process constants so every camera's DEGRADED
    signal means the same thing.  ``max_respawns`` exists because the
    composition root (and tests) must be able to bound how long a camera
    keeps trying before it is left permanently DEGRADED.
    """

    max_respawns: int = DECODE_MAX_RESPAWNS


class DecodeStalledError(RuntimeError):
    """A decode path produced no frame for ``DECODE_SILENCE_DEADLINE_SEC``.

    Distinct from the decoder's own dead-child errors: the child may still be
    alive and simply delivering nothing.  Both drive the same respawn, but
    the category reaches the status store separately so an operator can tell
    "decoder wedged" from "decoder exited".
    """


@final
class _SilenceWatchingSession:
    """Turn *duration* without a frame into a loud, respawnable failure.

    ``DecodeSession.read()`` returning ``None`` is the transient-silence
    signal todo 4 defined.  ``RTSPSource`` counts those to drive its RTSP
    reopen, but a count is not a duration: the same 30 ``None``s can span
    milliseconds or minutes depending on how the child behaves.  This wrapper
    adds the missing time dimension and nothing else -- it never reconnects,
    never re-opens, and never swallows anything; it raises and lets the
    supervision loop below own recovery so a respawned source still gets a
    fresh stream epoch (and therefore a rolled packet-ring epoch).
    """

    def __init__(
        self,
        session: DecodeSession,
        camera_id: str,
        *,
        clock: Callable[[], float],
        deadline_sec: float = DECODE_SILENCE_DEADLINE_SEC,
    ) -> None:
        self._session = session
        self._camera_id = camera_id
        self._clock = clock
        self._deadline_sec = deadline_sec
        self._last_frame_at = clock()

    def read(self) -> FramePacket | None:
        packet = self._session.read()
        now = self._clock()
        if packet is not None:
            self._last_frame_at = now
            return packet
        silent_for = now - self._last_frame_at
        if silent_for >= self._deadline_sec:
            raise DecodeStalledError(
                f"decode produced no frame for {silent_for:.1f}s "
                f"(deadline {self._deadline_sec:.1f}s)"
            )
        return None

    def close(self) -> None:
        self._session.close()

    def set_stream_identity(self, worker_boot_id: str, stream_epoch: int) -> None:
        """Forward stream identity so packet-ring epochs still roll per session."""
        assign = getattr(self._session, "set_stream_identity", None)
        if assign is not None:
            assign(worker_boot_id, stream_epoch)


@final
class _SilenceWatchingAdapter(Generic[_DecodeConfigT]):
    """Wrap every session this adapter opens in the silence watcher."""

    def __init__(
        self,
        decoder: DecodeAdapter[_DecodeConfigT],
        camera_id: str,
        *,
        clock: Callable[[], float],
    ) -> None:
        self._decoder = decoder
        self._camera_id = camera_id
        self._clock = clock

    def open(self, config: _DecodeConfigT) -> DecodeSession:
        return _SilenceWatchingSession(
            self._decoder.open(config),
            self._camera_id,
            clock=self._clock,
        )


@dataclass(frozen=True, slots=True)
class IngestEvent:
    camera_id: str
    event_type: str
    category: str
    detail: str


class IngestReporter(Protocol):
    def mark_starting(self, camera_id: str) -> None: ...

    def mark_ready(self, camera_id: str) -> None: ...

    def mark_degraded(self, camera_id: str, *, category: str) -> None: ...

    def emit(self, event: IngestEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class CameraIngestSpec(Generic[_DecodeConfigT]):
    camera_id: str
    source_id: str
    make_decode_config: _DecodeConfigFactory[_DecodeConfigT]
    policy: CapturePolicy = field(default_factory=CapturePolicy)
    decode_supervision: DecodeSupervisionPolicy = field(default_factory=DecodeSupervisionPolicy)


@dataclass(frozen=True, slots=True)
class CameraIngestPorts(Generic[_DecodeConfigT]):
    registry: SourceRegistry
    decoder: DecodeAdapter[_DecodeConfigT]
    bus: FrameBus
    reporter: IngestReporter


@final
class CameraIngestLoop(Generic[_DecodeConfigT]):
    def __init__(
        self,
        spec: CameraIngestSpec[_DecodeConfigT],
        ports: CameraIngestPorts[_DecodeConfigT],
        *,
        clock: Callable[[], float] = time.monotonic,
        respawn_wait: Callable[[float], bool] | None = None,
    ) -> None:
        self._spec = spec
        self._ports = ports
        self._stop_event = threading.Event()
        self._ready = False
        self._offline = False
        self._clock = clock
        self._respawn_wait = self._stop_event.wait if respawn_wait is None else respawn_wait

    @property
    def camera_id(self) -> str:
        return self._spec.camera_id

    @property
    def decode_adapter(self) -> DecodeAdapter[_DecodeConfigT]:
        """The adapter this loop actually decodes with.

        Exposed so the composition root can report the *concrete* class in
        its requested-versus-actual decode observability instead of
        re-deriving it from the profile token -- a re-derivation would agree
        with the selection logic by construction and so could never catch the
        two disagreeing.  This is the built object itself, not the internal
        silence-watching wrapper each source build puts around it.
        """
        return self._ports.decoder

    def run(self) -> None:
        """Ingest this camera, respawning its decode path on a bounded budget.

        Decode now lives in a per-camera ffmpeg child, so two new failure
        modes end a source iteration that used to run for the process's whole
        life: the child dies loudly (a ``RuntimeError`` from the decoder
        wrapper) or it goes silent (caught by ``_SilenceWatchingSession``).
        Both are camera-local and recoverable, so they respawn the whole
        source here -- a *fresh* ``RTSPSource``, which is what makes the
        replacement decode child get a new stream epoch and roll the packet
        ring, rather than a hidden in-place restart under a stale epoch.

        The budget is what keeps a permanently dead camera honest: after
        ``max_respawns`` attempts the loop returns with the camera DEGRADED
        instead of spinning, so a dead camera is visible rather than masked
        by an infinite retry.  ``FatalAcceleratorError`` still escapes
        untouched -- that is the accelerator fault boundary, not a camera gap.
        """
        self._ports.reporter.mark_starting(self.camera_id)
        respawns = 0
        while not self._stop_event.is_set():
            error = self._run_once()
            if error is None or self._stop_event.is_set():
                return
            if respawns >= self._spec.decode_supervision.max_respawns:
                self._record_respawn_exhausted(error, respawns)
                return
            delay_sec = _respawn_backoff(respawns)
            respawns += 1
            self._record_respawn(error, respawns, delay_sec)
            if self._respawn_wait(delay_sec):
                return

    def _run_once(self) -> Exception | None:
        """Run one source lifetime; return the failure that ended it, if any."""
        try:
            source = self._build_source()
        except FatalAcceleratorError:
            raise
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - boundary
            self._record_source_failure(error)
            return error

        try:
            for packet in source:
                # Per-packet call: HeartbeatReporter rate-limits to
                # camera.heartbeat_interval_sec, so a healthy stream keeps the
                # relay liveness ping periodic instead of firing once per
                # READY transition and going stale.
                self._ready = True
                if self._offline:
                    # A respawned source starts fresh, so `RTSPSource`'s own
                    # `on_recovered` callback never fires for it -- without
                    # this the camera would sit permanently "offline" in the
                    # ops event stream while happily publishing frames.
                    self._record_recovered("decode_respawned")
                else:
                    self._ports.reporter.mark_ready(self.camera_id)
                try:
                    self._ports.bus.publish(packet)
                except FatalAcceleratorError:
                    raise
                except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - boundary
                    self._record_processing_failure(error)
        except FatalAcceleratorError:
            raise
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - boundary
            self._record_source_failure(error)
            return error
        return None

    def stop(self) -> None:
        self._stop_event.set()

    def _build_source(self) -> RTSPSource[_DecodeConfigT]:
        resolved = self._ports.registry.resolve(source_id=self._spec.source_id)
        decode_config = self._spec.make_decode_config(self.camera_id, resolved)
        policy = self._spec.policy
        source = RTSPSource(
            decode_config,
            _SilenceWatchingAdapter(self._ports.decoder, self.camera_id, clock=self._clock),
            max_failures=policy.max_failures,
            reconnect_initial_backoff_sec=policy.reconnect_initial_backoff_sec,
            reconnect_max_backoff_sec=policy.reconnect_max_backoff_sec,
            max_total_reconnects=policy.max_total_reconnects,
            backoff_wait=self._stop_event.wait,
            stop_requested=self._stop_event.is_set,
            target_fps=policy.target_fps,
            pace_wait=self._stop_event.wait,
        )
        source.set_liveness_callbacks(
            on_reconnecting=self._record_reconnecting,
            on_recovered=self._record_recovered,
        )
        return source

    def _record_reconnecting(self, reason: str) -> None:
        self._ready = False
        category = "rtsp_reconnecting"
        # Issue #113: this is the only console-visible trace of a camera's
        # reconnect loop -- mark_degraded/emit below only reach the in-memory
        # status store, so without a camera_id here, an offline camera's
        # retry cadence is invisible in the worker log.
        # Issue #115: camera_id/reason must also be in the message text, not
        # just extra= -- worker/__main__.py's console formatter doesn't
        # render extra fields, so extra-only values are silently dropped.
        LOGGER.warning(
            "camera ingest reconnecting: camera_id=%s reason=%s",
            self.camera_id,
            reason,
            extra={"camera_id": self.camera_id, "reason": reason},
        )
        self._ports.reporter.mark_degraded(self.camera_id, category=category)
        self._emit_offline_once(category, reason)

    def _record_recovered(self, reason: str) -> None:
        self._ready = True
        self._ports.reporter.mark_ready(self.camera_id)
        if self._offline:
            self._ports.reporter.emit(
                IngestEvent(
                    camera_id=self.camera_id,
                    event_type="camera.recovered",
                    category="rtsp_recovered",
                    detail=reason,
                )
            )
            self._offline = False

    def _record_source_failure(self, error: Exception) -> None:
        self._ready = False
        category = type(error).__name__
        self._ports.reporter.mark_degraded(self.camera_id, category=category)
        safe_source = mask_rtsp_url(self._spec.source_id)
        self._emit_offline_once(category, f"{category} while ingesting {safe_source}")

    def _record_respawn(self, error: Exception, attempt: int, delay_sec: float) -> None:
        """Announce one decode respawn on the console, naming the camera.

        Same rule as ``_record_reconnecting`` above (issue #115): the console
        formatter renders only ``%(message)s``, so an ``extra=``-only
        camera_id is invisible exactly when an operator needs it.
        """
        category = type(error).__name__
        LOGGER.warning(
            "camera decode respawn: camera_id=%s attempt=%d/%d reason=%s backoff=%.1fs detail=%s",
            self.camera_id,
            attempt,
            self._spec.decode_supervision.max_respawns,
            category,
            delay_sec,
            _visible_decode_detail(error),
            extra={
                "camera_id": self.camera_id,
                "attempt": attempt,
                "reason": category,
                "backoff_sec": delay_sec,
            },
        )

    def _record_respawn_exhausted(self, error: Exception, respawns: int) -> None:
        """Leave a permanently dead camera loudly DEGRADED, not silently gone."""
        LOGGER.error(
            "camera decode respawn budget exhausted: camera_id=%s attempts=%d reason=%s detail=%s; "
            "camera stays DEGRADED",
            self.camera_id,
            respawns,
            type(error).__name__,
            _visible_decode_detail(error),
            extra={
                "camera_id": self.camera_id,
                "attempts": respawns,
                "reason": type(error).__name__,
            },
        )

    def _record_processing_failure(self, error: Exception) -> None:
        category = type(error).__name__
        self._ports.reporter.emit(
            IngestEvent(
                camera_id=self.camera_id,
                event_type="frame.processing_error",
                category=category,
                detail=f"{category} while publishing a decoded frame",
            )
        )

    def _emit_offline_once(self, category: str, detail: str) -> None:
        if self._offline:
            return
        self._offline = True
        self._ports.reporter.emit(
            IngestEvent(
                camera_id=self.camera_id,
                event_type="camera.offline",
                category=category,
                detail=detail,
            )
        )


class _RunnableIngest(Protocol):
    @property
    def camera_id(self) -> str: ...

    def run(self) -> None: ...

    def stop(self) -> None: ...


@final
class IngestSupervisor:
    def __init__(
        self,
        loops: Sequence[_RunnableIngest],
        *,
        restart_check: Callable[[], bool] | None = None,
        restart_poll_interval_sec: float = 1.0,
        completion_check: Callable[[], bool] | None = None,
        completion_poll_interval_sec: float = 0.1,
    ) -> None:
        self._loops = tuple(loops)
        self._threads: tuple[threading.Thread, ...] = ()
        self._restart_check = restart_check
        self._restart_poll_interval_sec = restart_poll_interval_sec
        self._restart_watcher: threading.Thread | None = None
        self._completion_check = completion_check
        self._completion_poll_interval_sec = completion_poll_interval_sec
        self._completion_watcher: threading.Thread | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        self.start()
        self.join()

    def start(self) -> None:
        if self._threads:
            return
        # Build every Thread first, but only publish it to the attribute
        # `stop()`/`join()` read (`self._threads`, `self._restart_watcher`,
        # `self._completion_watcher`) *after* `.start()` has returned. A
        # concurrent `stop()` -- e.g. a test driving `run()` on a background
        # thread and calling `stop()` once some readiness condition flips --
        # must never observe a `Thread` object that has been constructed but
        # not yet started, or `Thread.join()` raises "cannot join thread
        # before it is started".
        threads = tuple(
            threading.Thread(
                target=loop.run,
                name=f"worker-ingest-{loop.camera_id}",
                daemon=True,
            )
            for loop in self._loops
        )
        for thread in threads:
            thread.start()
        self._threads = threads
        if self._restart_check is not None:
            restart_watcher = threading.Thread(
                target=self._watch_restart,
                name="worker-ingest-restart-watch",
                daemon=True,
            )
            restart_watcher.start()
            self._restart_watcher = restart_watcher
        if self._completion_check is not None:
            completion_watcher = threading.Thread(
                target=self._watch_completion,
                name="worker-ingest-completion-watch",
                daemon=True,
            )
            completion_watcher.start()
            self._completion_watcher = completion_watcher

    def join(self, *, timeout_sec: float | None = None) -> None:
        for thread in self._threads:
            thread.join(timeout=timeout_sec)

    def wait_until_stopped(self) -> None:
        """Block until `stop()` runs, even with zero ingest loops (issue #150).

        `join()` only waits on `self._threads`, so with an empty camera
        roster there is nothing to join and it returns immediately -- a
        caller relying on it to keep the process alive would exit right
        after boot. Every path that ends this supervisor (an external
        `stop()` call, or the restart/completion watchers deciding to stop
        it themselves) sets `self._stop_event` first, so waiting on it
        directly blocks correctly regardless of loop count.
        """
        self._stop_event.wait()

    def stop(self, *, join_timeout_sec: float = 1.0) -> None:
        self._stop_event.set()
        for loop in self._loops:
            loop.stop()
        self.join(timeout_sec=join_timeout_sec)
        for watcher in (self._restart_watcher, self._completion_watcher):
            if watcher is not None and watcher is not threading.current_thread():
                watcher.join(timeout=join_timeout_sec)

    def _watch_restart(self) -> None:
        restart_check = self._restart_check
        if restart_check is None:
            return
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._restart_poll_interval_sec):
                return
            if restart_check():
                self.stop()
                return

    def _watch_completion(self) -> None:
        completion_check = self._completion_check
        if completion_check is None:
            return
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._completion_poll_interval_sec):
                return
            if completion_check():
                self.stop()
                return


def _respawn_backoff(respawns: int) -> float:
    return min(
        DECODE_RESPAWN_INITIAL_BACKOFF_SEC * (2.0 ** min(respawns, 32)),
        DECODE_RESPAWN_BACKOFF_CAP_SEC,
    )


_SAFE_DETAIL_CAUSE_LINKS: Final = 4


@contextmanager
def _logging_boundary() -> Generator[None]:
    """Contain ordinary property failures on the decode-detail log path."""
    try:
        yield
    except BaseException as error:
        if not isinstance(error, Exception):
            raise


def _safe_log_detail_of(error: BaseException) -> object:
    with _logging_boundary():
        return getattr(error, "safe_log_detail", None)
    return None  # reached only when _logging_boundary swallowed the property failure


def _visible_decode_detail(error: BaseException) -> str:
    seen: set[int] = set()
    current: BaseException | None = error
    for _ in range(_SAFE_DETAIL_CAUSE_LINKS + 1):
        if current is None:
            break
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        detail = _safe_log_detail_of(current)
        if isinstance(detail, str) and detail.strip():
            return f"{type(current).__name__}: {detail}"
        current = current.__cause__
    return "unavailable"


__all__ = [
    "DECODE_MAX_RESPAWNS",
    "DECODE_RESPAWN_BACKOFF_CAP_SEC",
    "DECODE_RESPAWN_INITIAL_BACKOFF_SEC",
    "DECODE_SILENCE_DEADLINE_SEC",
    "CameraIngestLoop",
    "CameraIngestPorts",
    "CameraIngestSpec",
    "CapturePolicy",
    "DecodeStalledError",
    "DecodeSupervisionPolicy",
    "IngestEvent",
    "IngestReporter",
    "IngestSupervisor",
]
