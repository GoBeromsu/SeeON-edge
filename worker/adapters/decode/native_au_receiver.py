from __future__ import annotations

import queue
import socket
import threading
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
from worker.adapters.decode.native_au_mux_template import native_configuration_signature
from worker.adapters.decode.native_au_progress import NativeAuProgress
from worker.types.source_packet import SourcePacket, SourceStreamConfiguration, StreamEpoch


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
        self._work: queue.Queue[_Work] = queue.Queue(maxsize=256)
        self._epochs: dict[str, StreamEpoch] = {}
        self._sequences: dict[tuple[str, int, int], int] = {}
        self._configurations: dict[tuple[str, int, int], SourceStreamConfiguration] = {}
        self._configuration_signatures: dict[tuple[str, int, int], str] = {}
        self._timeline: dict[tuple[str, int, int], tuple[int, int]] = {}
        self._gaps_active: set[str] = set()
        self._retired_generations: dict[str, int] = {}
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
        self._gaps_active.discard(camera_id)
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
                camera = work.camera_id
                with suppress(queue.Empty):
                    _ = self._work.get_nowait()
                with suppress(queue.Full):
                    self._work.put_nowait(_Gap(camera))

    def _process(self) -> None:
        while not self._stop.is_set():
            try:
                work = self._work.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(work, _Gap):
                self._report_gap(work.camera_id)
                continue
            try:
                self._accept(work)
            except Exception:  # noqa: BLE001 - malformed camera data is isolated
                self._report_gap(work.camera_id)

    def _report_gap(self, camera_id: str) -> None:
        if not camera_id or camera_id in self._gaps_active:
            return
        self._gaps_active.add(camera_id)
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
            self._report_gap(envelope.camera_id)
            return
        identity = StreamEpoch(
            self._worker_boot_id, envelope.camera_id, envelope.epoch, envelope.generation
        )
        key = (envelope.camera_id, envelope.generation, envelope.epoch)
        expected = self._sequences.get(key, 0) + 1
        if envelope.kind is AuKind.GAP or envelope.sequence != expected:
            self._report_gap(envelope.camera_id)
            return
        if self._timestamp_discontinuous(key, envelope):
            self._report_gap(envelope.camera_id)
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
        signature = native_configuration_signature(
            envelope.codec, envelope.framing, envelope.parser_caps, envelope.codec_data,
            envelope.width, envelope.height, envelope.time_base,
        )
        configuration = self._configurations.get(key)
        if configuration is None:
            configuration = stream_configuration(envelope)
            self._configurations[key] = configuration
            self._configuration_signatures[key] = signature
        elif self._configuration_signatures[key] != signature:
            self._report_gap(envelope.camera_id)
            return
        packet = SourcePacket(
            identity, configuration, 0, envelope.pts, envelope.dts, envelope.duration,
            envelope.keyframe, envelope.payload, envelope.sequence - 1,
        )
        if not self._sink.append(packet):
            self._report_gap(envelope.camera_id)
            return
        self._timeline[key] = (envelope.dts, envelope.duration)
        self._gaps_active.discard(envelope.camera_id)
        self._progress.accept(envelope.camera_id)
        if self._accept_handler is not None:
            self._accept_handler(
                envelope.camera_id,
                envelope.pts,
                envelope.sequence,
                envelope.generation,
            )

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
            self._sequences, self._configurations, self._configuration_signatures, self._timeline,
        ):
            for key in tuple(mapping):
                if key[0] == camera_id and key != keep:
                    del mapping[key]


__all__ = [
    "NativeAuGapHandler", "NativeAuReceiver", "NativeAuSink", "NativeAuStreamDeathHandler",
]
