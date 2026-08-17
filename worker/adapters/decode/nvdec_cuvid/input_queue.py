from __future__ import annotations

import heapq
import queue
import threading
from fractions import Fraction
from typing import Final, final

from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError
from worker.adapters.decode.nvdec_cuvid.process import DecoderProcess
from worker.types.source_packet import SourcePacket

_INPUT_QUEUE_CAPACITY: Final = 32
_QUEUE_WAIT_SEC: Final = 0.05


@final
class DecoderInputQueue:
    """Keep compressed decoder input off the demux thread.

    Evidence is appended before ``offer`` is called. If this bounded queue
    fills, queued decoder-only packets are discarded and input resumes at a
    keyframe so an arbitrary H.264 reference-chain gap is never forwarded.
    """

    def __init__(self, process: DecoderProcess) -> None:
        self._process = process
        self._packets: queue.Queue[SourcePacket] = queue.Queue(
            maxsize=_INPUT_QUEUE_CAPACITY
        )
        self._timings: list[tuple[Fraction, int, SourcePacket]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._finish_requested = False
        self._awaiting_keyframe = False
        self._overflow_count = 0
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._write_packets,
            name="nvdec-cuvid-stdin-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def overflow_count(self) -> int:
        with self._lock:
            return self._overflow_count

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def offer(self, packet: SourcePacket) -> bool:
        with self._lock:
            if self._finish_requested or self._stop.is_set():
                return False
            if self._awaiting_keyframe:
                if not packet.is_keyframe:
                    return False
                self._awaiting_keyframe = False
            try:
                self._packets.put_nowait(packet)
            except queue.Full:
                self._overflow_count += 1
                self._discard_backlog()
                self._awaiting_keyframe = not packet.is_keyframe
                if self._awaiting_keyframe:
                    return False
                self._packets.put_nowait(packet)
            return True

    def raise_if_failed(self) -> None:
        error = self.error
        if error is not None:
            raise RuntimeError(
                f"packet-preserving NVDEC decode failed ({type(error).__name__})"
            ) from error
        returncode = getattr(self._process, "failure_returncode", None)
        if returncode is not None:
            raise NvdecUnavailableError(
                "ffmpeg decoder exited before a complete frame was available",
                returncode=returncode,
            )

    def pop_timing(self) -> SourcePacket:
        with self._lock:
            if not self._timings:
                raise NvdecUnavailableError(
                    "ffmpeg decoder produced a frame without source packet timing"
                )
            return heapq.heappop(self._timings)[2]

    def finish(self) -> None:
        with self._lock:
            self._finish_requested = True

    def abort(self) -> None:
        self._stop.set()
        self._discard_backlog()

    def join(self, timeout: float) -> None:
        self._thread.join(timeout=timeout)

    def _publish_timing(self, packet: SourcePacket) -> None:
        """Make a packet's timing readable before its frame can be emitted."""
        with self._lock:
            heapq.heappush(
                self._timings,
                (packet.presentation_time, packet.arrival_index, packet),
            )

    def _withdraw_timing(self, packet: SourcePacket) -> None:
        """Drop a timing whose packet never reached the decoder."""
        with self._lock:
            remaining = [
                entry for entry in self._timings if entry[1] != packet.arrival_index
            ]
            if len(remaining) != len(self._timings):
                self._timings = remaining
                heapq.heapify(self._timings)

    def _discard_backlog(self) -> None:
        while True:
            try:
                self._packets.get_nowait()
            except queue.Empty:
                return

    def _write_packets(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    packet = self._packets.get(timeout=_QUEUE_WAIT_SEC)
                except queue.Empty:
                    with self._lock:
                        if not self._finish_requested:
                            continue
                    return
                self._publish_timing(packet)
                try:
                    self._process.write_packet(packet.payload)
                except Exception:
                    self._withdraw_timing(packet)
                    raise
        except Exception as error:  # noqa: BLE001 - writer thread boundary
            if not self._stop.is_set():
                with self._lock:
                    self._error = error
        finally:
            self._process.close_input()


__all__ = ["DecoderInputQueue"]
