from __future__ import annotations

import hashlib
import logging
import queue
import socket
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, final

from worker.adapters.decode.native_au_codec import (
    AuEnvelope,
    AuFrameError,
    AuKind,
    receive_envelope,
    stream_configuration,
)
from worker.adapters.decode.native_au_mux_template import (
    distinct_parameter_sets,
    native_configuration_signature,
)
from worker.adapters.decode.native_au_progress import NativeAuProgress
from worker.types.source_packet import SourcePacket, SourceStreamConfiguration, StreamEpoch


def _components_of(envelope: AuEnvelope) -> tuple[object, ...]:
    """Signature inputs in the same order native_configuration_signature hashes."""
    return (
        envelope.codec,
        envelope.framing,
        envelope.parser_caps,
        len(distinct_parameter_sets(envelope.codec_data)),
        hashlib.sha256(distinct_parameter_sets(envelope.codec_data)).hexdigest()[:8],
        envelope.width,
        envelope.height,
        str(envelope.time_base),
    )


_COMPONENT_NAMES = (
    "codec",
    "framing",
    "parser_caps",
    "codec_data_len",
    "codec_data_hash",
    "width",
    "height",
    "time_base",
)

LOGGER = logging.getLogger(__name__)


class NativeAuSink(Protocol):
    def register_camera(self, camera_id: str) -> None: ...
    def append(self, packet: SourcePacket) -> bool: ...
    def roll_epoch(self, epoch: StreamEpoch) -> None: ...


class NativeAuGapHandler(Protocol):
    def __call__(self, camera_id: str, category: str) -> None: ...


class NativeAuStreamDeathHandler(Protocol):
    def __call__(self, category: str) -> None: ...


@dataclass(frozen=True, slots=True)
class _Gap:
    camera_id: str


_Work = AuEnvelope | _Gap


@final
class NativeAuReceiver:
    """Drain AU bytes independently from slower configuration and ring work."""

    def __init__(
        self,
        endpoint: socket.socket,
        worker_boot_id: str,
        sink: NativeAuSink,
        gap_handler: NativeAuGapHandler,
        stream_death_handler: NativeAuStreamDeathHandler | None = None,
        accept_handler: Callable[[str, int, int, int], None] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._worker_boot_id = worker_boot_id
        self._sink = sink
        self._gap_handler = gap_handler
        self._stream_death_handler = stream_death_handler or (lambda _category: None)
        self._accept_handler = accept_handler
        self._stop = threading.Event()
        self._drain_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        # Sized to ride out GIL scheduling jitter: the worker process runs a
        # policy pump per camera alongside this thread, and a half-second
        # stall at fleet rate is ~130 units. 256 shed on every stall.
        self._work: queue.Queue[_Work] = queue.Queue(maxsize=2048)
        self._epochs: dict[str, StreamEpoch] = {}
        self._sequences: dict[tuple[str, int, int], int] = {}
        self._configurations: dict[tuple[str, int, int], SourceStreamConfiguration] = {}
        self._configuration_signatures: dict[tuple[str, int, int], str] = {}
        self._timeline: dict[tuple[str, int, int], tuple[int, int]] = {}
        # Cameras with a rebuild request outstanding, keyed by the
        # (generation, epoch) the gap was reported against. A later report on
        # the same identity is a duplicate; one on a newer identity is proof
        # the rebuild landed and something else went wrong, so it is not.
        self._gaps_active: dict[str, tuple[int, int]] = {}
        # Components behind each configuration signature, kept so a signature
        # change can name the field that actually moved instead of reporting an
        # opaque digest mismatch.
        self._configuration_components: dict[
            tuple[str, int, int], tuple[object, ...]
        ] = {}
        self._retired_generations: dict[str, int] = {}
        # Gap markers the child's sender emitted per camera; the holes they
        # announce are marked by the sequence check, so this is a tally only.
        self._sender_gap_markers: dict[str, int] = {}
        # Raw signature inputs last seen per key, so the digest is recomputed
        # only when a camera actually changes them. Cameras retransmit
        # identical parameter sets on every keyframe.
        self._signature_inputs: dict[tuple[str, int, int], tuple[object, ...]] = {}
        self._tally = {"holes": 0, "shed": 0, "sender_markers": 0, "adopted": 0}
        self._tally_logged_at = time.monotonic()
        self._progress = NativeAuProgress()

    def start(self) -> None:
        self._drain_thread = threading.Thread(
            target=self._drain, name="deepstream-au-drain", daemon=True
        )
        self._process_thread = threading.Thread(
            target=self._process, name="deepstream-au-process", daemon=True
        )
        self._process_thread.start()
        self._drain_thread.start()

    def close(self) -> None:
        self._stop.set()
        with suppress(OSError):
            self._endpoint.shutdown(socket.SHUT_RDWR)
        self._endpoint.close()
        for thread in (self._drain_thread, self._process_thread):
            if thread is not None:
                thread.join(timeout=2.0)

    def retire_camera(self, camera_id: str) -> None:
        active = self._epochs.pop(camera_id, None)
        if active is not None:
            self._retired_generations[camera_id] = active.source_generation
        self._gaps_active.pop(camera_id, None)
        self._retire_keys(camera_id)
        remove = getattr(self._sink, "remove_camera", None)
        if callable(remove):
            remove(camera_id)

    def accepted_count(self, camera_id: str) -> int:
        return self._progress.count(camera_id)

    def wait_for_packets(self, camera_id: str, target: int, timeout: float) -> bool:
        return self._progress.wait(camera_id, target, timeout)

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                work: _Work = receive_envelope(self._endpoint)
            except AuFrameError as error:
                if error.camera_id:
                    work = _Gap(error.camera_id)
                else:
                    self._stream_death_handler("au_framing_unknown_camera")
                    return
            except (ConnectionError, OSError):
                if not self._stop.is_set():
                    self._stream_death_handler("au_stream_closed")
                return
            try:
                self._work.put_nowait(work)
            except queue.Full:
                # Shed the oldest unit. Its camera's sequence check marks the
                # hole on the next unit it accepts; it does not need telling.
                # This used to inject a gap marker for the INCOMING camera --
                # not the one whose unit was shed -- and every marker asked
                # for a source rebuild, which ran synchronously on the process
                # thread this queue feeds. Each rebuild stalled the drain, the
                # stall overflowed the queue, and the overflow requested more
                # rebuilds: 265 rebuilds a minute across 13 cameras, epochs in
                # the sixties four minutes after boot, no clip with video.
                with suppress(queue.Empty):
                    _ = self._work.get_nowait()
                    self._tally["shed"] += 1
                with suppress(queue.Full):
                    self._work.put_nowait(work)

    def _process(self) -> None:
        while not self._stop.is_set():
            try:
                work = self._work.get(timeout=0.2)
            except queue.Empty:
                self._log_tally()
                continue
            self._log_tally()
            if isinstance(work, _Gap):
                self._report_gap(work.camera_id, "transport_gap_marker")
                continue
            try:
                self._accept(work)
            except Exception as error:  # noqa: BLE001 - malformed camera data is isolated
                # The bare catch used to discard the exception entirely, so the
                # dominant cause of source rebuilds was invisible: 293 of 308
                # gaps in a five-minute window landed here with nothing to say
                # what had been raised.
                self._report_gap(
                    work.camera_id,
                    f"malformed_envelope:{type(error).__name__}:{error}"[:200],
                )

    def _signature_delta(self, key: tuple[str, int, int], envelope: AuEnvelope) -> str:
        """Name which signature inputs changed, so the cause is not a guess.

        The signature is a digest over seven inputs, so a mismatch alone says
        nothing about which one moved. Cameras retransmit parameter sets
        periodically, which is expected and should not look like a geometry or
        codec change.
        """
        previous = self._configuration_components.get(key)
        current = _components_of(envelope)
        if previous is None:
            return "no_previous_components"
        moved = [
            f"{name}:{old}->{new}"
            for name, old, new in zip(_COMPONENT_NAMES, previous, current, strict=True)
            if old != new
        ]
        return ",".join(moved) if moved else "digest_only"

    def _report_gap(
        self,
        camera_id: str,
        reason: str,
        identity: tuple[int, int] | None = None,
    ) -> None:
        """Ask for a source rebuild, naming the condition that prompted it.

        The wire category stays ``"parser"`` because the child validates it
        against a fixed whitelist, but that label describes none of these
        conditions and sent one investigation into the C++ AU parser, which
        turned out to be emitting nothing at all. The real reason goes to the
        log so the next reader is not misdirected the same way.

        ``identity`` is the (generation, epoch) the condition was observed on;
        transport gaps carry none and fall back to the adopted identity. A
        report is suppressed only while one is outstanding for the same or a
        newer identity. Suppressing by camera alone left rings stranded on the
        live fleet: the rebuild landed, its units were refused for a reason of
        their own, and the refusal could never be reported because the camera
        was still marked pending from the gap that caused the rebuild.
        """
        if not camera_id:
            return
        if identity is None:
            active = self._epochs.get(camera_id)
            identity = (
                (active.source_generation, active.stream_epoch)
                if active is not None
                else (0, 0)
            )
        pending = self._gaps_active.get(camera_id)
        if pending is not None and pending >= identity:
            return
        self._gaps_active[camera_id] = identity
        LOGGER.warning(
            "native au gap: camera_id=%s reason=%s", camera_id, reason
        )
        self._gap_handler(camera_id, "parser")

    def _accept(self, envelope: AuEnvelope) -> None:
        retired = self._retired_generations.get(envelope.camera_id, -1)
        if envelope.generation <= retired:
            return
        self._retired_generations.pop(envelope.camera_id, None)
        active = self._epochs.get(envelope.camera_id)
        if active is not None and (
            envelope.generation < active.source_generation
            or (
                envelope.generation == active.source_generation
                and envelope.epoch < active.stream_epoch
            )
        ):
            # Access units already in flight when the epoch rolled arrive
            # carrying the superseded identity. That is expected debris, not a
            # gap: reporting it asks for another rebuild, which advances the
            # epoch again and strands the next batch of in-flight units, so the
            # loop feeds itself. Measured on the live fleet this produced 301
            # rebuilds in five minutes with zero child-reported failures.
            # Retired generations are already dropped silently a few lines up;
            # a superseded epoch gets the same treatment.
            return
        identity = StreamEpoch(
            self._worker_boot_id, envelope.camera_id, envelope.epoch, envelope.generation
        )
        key = (envelope.camera_id, envelope.generation, envelope.epoch)
        observed = (envelope.generation, envelope.epoch)
        if envelope.kind is AuKind.GAP:
            # The child's sender shed units under backpressure. That is a hole,
            # and the sequence check marks it on the next unit exactly as it
            # does for units shed on this side. Rebuilding for it restarted the
            # source, whose opening burst congested the sender again, which
            # emitted the next marker: five rebuilds a minute, each stalling
            # this thread long enough to shed a second of every camera.
            self._sender_gap_markers[envelope.camera_id] = (
                self._sender_gap_markers.get(envelope.camera_id, 0) + 1
            )
            self._tally["sender_markers"] += 1
            return
        expected = self._sequences.get(key, 0) + 1
        discontinuity: str | None = None
        if envelope.sequence != expected:
            if key in self._sequences:
                if envelope.sequence < expected:
                    return  # duplicate or reordered; already accounted for
                # Units lost inside an epoch leave a hole in the ring. The
                # packet that follows the hole carries the mark, and clip
                # selection refuses a window that crosses it -- one clip
                # window, not a source rebuild. Rebuilding for a hole was the
                # engine of the storm described in _drain.
                discontinuity = f"sequence_gap:{expected}->{envelope.sequence}"
                self._tally["holes"] += 1
                LOGGER.warning(
                    "native au sequence gap: camera_id=%s generation=%d epoch=%d "
                    "expected=%d got=%d",
                    envelope.camera_id, envelope.generation, envelope.epoch,
                    expected, envelope.sequence,
                )
            else:
                self._tally["adopted"] += 1
                # A unit for an identity newer than anything adopted, arriving
                # after sequence 1. The identity itself proves the rebuild
                # landed; only the opening unit was lost, and the sender-side
                # reservation cannot protect it once it is inside this
                # process. Refusing it and asking for another rebuild fed the
                # storm and, with the old pending-gap suppression, stranded
                # ten of thirteen rings for half an hour. Adopt it: the ring
                # starts at the next keyframe regardless.
                LOGGER.warning(
                    "native au epoch adopted mid-stream: camera_id=%s generation=%d "
                    "epoch=%d first_sequence=%d",
                    envelope.camera_id, envelope.generation, envelope.epoch,
                    envelope.sequence,
                )
        # A DTS jump on a CONTIGUOUS sequence is the camera's clock moving and
        # needs a fresh stream. After a sequence gap the jump is just the hole,
        # already marked on this packet; rebuilding for it turned every stall
        # into thirteen rebuilds.
        if discontinuity is None and self._timestamp_discontinuous(key, envelope):
            self._report_gap(envelope.camera_id, "timestamp_discontinuity", observed)
            return
        if active != identity:
            if active is not None and envelope.generation > active.source_generation:
                remove = getattr(self._sink, "remove_camera", None)
                if callable(remove):
                    remove(envelope.camera_id)
            self._sink.register_camera(envelope.camera_id)
            self._sink.roll_epoch(identity)
            self._retire_keys(envelope.camera_id, keep=key)
            self._epochs[envelope.camera_id] = identity
        self._sequences[key] = envelope.sequence
        inputs = (
            envelope.codec, envelope.framing, envelope.parser_caps, envelope.codec_data,
            envelope.width, envelope.height, envelope.time_base,
        )
        configuration = self._configurations.get(key)
        if configuration is not None and self._signature_inputs.get(key) == inputs:
            signature = self._configuration_signatures[key]
        else:
            signature = native_configuration_signature(*inputs)
            self._signature_inputs[key] = inputs
        if configuration is None:
            configuration = stream_configuration(envelope)
            self._configurations[key] = configuration
            self._configuration_signatures[key] = signature
            self._configuration_components[key] = _components_of(envelope)
        elif self._configuration_signatures[key] != signature:
            self._report_gap(
                envelope.camera_id,
                f"configuration_signature_changed:{self._signature_delta(key, envelope)}",
                observed,
            )
            return
        packet = SourcePacket(
            identity, configuration, 0, envelope.pts, envelope.dts, envelope.duration,
            envelope.keyframe, envelope.payload, envelope.sequence - 1,
            discontinuity=discontinuity,
        )
        if not self._sink.append(packet):
            self._report_gap(envelope.camera_id, "ring_append_rejected", observed)
            return
        self._timeline[key] = (envelope.dts, envelope.duration)
        self._gaps_active.pop(envelope.camera_id, None)
        self._progress.accept(envelope.camera_id)
        if self._accept_handler is not None:
            self._accept_handler(
                envelope.camera_id,
                envelope.pts,
                envelope.sequence,
                envelope.generation,
            )

    def _log_tally(self) -> None:
        now = time.monotonic()
        if now - self._tally_logged_at < 60.0:
            return
        self._tally_logged_at = now
        LOGGER.info(
            "native au receiver tally (60s): holes=%d shed=%d sender_markers=%d adopted=%d",
            self._tally["holes"], self._tally["shed"],
            self._tally["sender_markers"], self._tally["adopted"],
        )
        for name in self._tally:
            self._tally[name] = 0

    def _timestamp_discontinuous(self, key: tuple[str, int, int], envelope: AuEnvelope) -> bool:
        previous = self._timeline.get(key)
        if previous is None:
            return False
        previous_dts, previous_duration = previous
        delta = envelope.dts - previous_dts
        maximum = max(previous_duration * 120, envelope.time_base.denominator * 5)
        return delta <= 0 or delta > maximum

    def _retire_keys(self, camera_id: str, keep: tuple[str, int, int] | None = None) -> None:
        for mapping in (
            self._sequences, self._configurations, self._configuration_signatures,
            self._timeline, self._signature_inputs,
        ):
            for key in tuple(mapping):
                if key[0] == camera_id and key != keep:
                    del mapping[key]


__all__ = [
    "NativeAuGapHandler", "NativeAuReceiver", "NativeAuSink", "NativeAuStreamDeathHandler",
]
