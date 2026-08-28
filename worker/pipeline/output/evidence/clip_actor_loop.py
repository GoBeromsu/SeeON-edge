"""Serialized clip actor message loop and lease cleanup."""

from __future__ import annotations

import queue
import threading
import time
from typing import Protocol, assert_never

from worker.pipeline.output.evidence.clip_actor import ClipActor
from worker.pipeline.output.evidence.clip_recorder_models import (
    EpochRollMessage,
    EventMessage,
    FlushMessage,
    FrameMessage,
    RecorderMessage,
)

# How often the loop performs traffic-independent upkeep. `rotate` has its own
# 30s interval guard, so this only bounds how often that guard is consulted.
_UPKEEP_INTERVAL_SEC = 0.1


class _Rotate(Protocol):
    """The maintenance seam, declaring the keyword-only `force` contract.

    This is a static contract, NOT an import-time or runtime check: a Protocol
    annotation binds nothing at call time. It exists so a type checker rejects a
    mismatched callable, where the previous `Callable[..., None]` accepted any
    signature and deferred the failure to the background recorder thread, where
    it surfaces as a dead actor rather than an error at the wiring site.
    """

    def __call__(self, *, force: bool) -> None: ...


def run_actor_loop(
    actor: ClipActor,
    messages: queue.Queue[RecorderMessage],
    stop: threading.Event,
    rotate: _Rotate,
) -> None:
    # Upkeep must be driven by elapsed time, NEVER by the queue running dry.
    #
    # Two things live here that recover state nothing else recovers: `rotate`
    # clears the disk-pressure suspension that blocks every new clip, and
    # `expire` closes an active clip whose camera stopped sending. Both used to
    # run only in the `queue.Empty` branch below.
    #
    # That branch needs 100ms of total silence across the shared queue. Thirteen
    # cameras at 15fps put roughly 195 messages/s into it, so in production it
    # effectively never fires -- and a single healthy camera is enough to starve
    # it. Recovery would then depend on traffic going idle, which is no better
    # than the original defect where it depended on a clip finalizing.
    #
    # Measured live: the store hit 80.68% against the 0.80 watermark and every
    # fall for ninety minutes recorded no footage. Reclaiming 264GB did not lift
    # it; only a restart did, because the constructor resets the flag.
    next_upkeep = time.monotonic()

    def upkeep() -> None:
        nonlocal next_upkeep
        now = time.monotonic()
        if now < next_upkeep:
            return
        next_upkeep = now + _UPKEEP_INTERVAL_SEC
        actor.expire()
        rotate(force=False)

    try:
        while True:
            upkeep()
            try:
                message = messages.get(timeout=0.1)
            except queue.Empty:
                if stop.is_set() and messages.empty():
                    break
                continue
            try:
                match message:
                    case FrameMessage():
                        actor.handle_frame(message)
                    case EventMessage():
                        actor.handle_event(message)
                    case EpochRollMessage(previous=previous, done=done):
                        try:
                            actor.handle_epoch_roll(previous)
                        finally:
                            done.set()
                    case FlushMessage(done=done):
                        try:
                            actor.flush()
                            rotate(force=True)
                        finally:
                            done.set()
                    case unreachable:
                        assert_never(unreachable)
            finally:
                _release(message)
                messages.task_done()
    finally:
        actor.shutdown()


def release_pending(messages: queue.Queue[RecorderMessage]) -> None:
    while True:
        try:
            message = messages.get_nowait()
        except queue.Empty:
            return
        try:
            _release(message)
        finally:
            messages.task_done()


def _release(message: RecorderMessage) -> None:
    if isinstance(message, FrameMessage):
        message.packet.release()
    elif isinstance(message, EventMessage):
        message.trigger_packet.release()


__all__ = ["release_pending", "run_actor_loop"]
