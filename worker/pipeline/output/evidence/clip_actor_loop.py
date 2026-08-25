"""Serialized clip actor message loop and lease cleanup."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import assert_never

from worker.pipeline.output.evidence.clip_actor import ClipActor
from worker.pipeline.output.evidence.clip_recorder_models import (
    EpochRollMessage,
    EventMessage,
    FlushMessage,
    FrameMessage,
    RecorderMessage,
)


def run_actor_loop(
    actor: ClipActor,
    messages: queue.Queue[RecorderMessage],
    stop: threading.Event,
    rotate: Callable[[], None],
) -> None:
    try:
        while True:
            try:
                message = messages.get(timeout=0.1)
            except queue.Empty:
                if stop.is_set() and messages.empty():
                    break
                actor.expire()
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
                            rotate()
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
