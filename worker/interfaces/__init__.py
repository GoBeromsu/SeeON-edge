"""Worker seam Protocols: one per replaceable boundary."""

from __future__ import annotations

from worker.interfaces.bus import FrameBus, FrameSubscription
from worker.interfaces.decision import Decider
from worker.interfaces.decode import DecodeAdapter, DecodeSession
from worker.interfaces.encode import ClipEncoder, ClipFinalizer, EncoderSession
from worker.interfaces.extract import Extractor
from worker.interfaces.frame import FrameMaterializer, HostFrameView
from worker.interfaces.output import EventSink
from worker.interfaces.perception import PerceptionFrameAdapter
from worker.interfaces.render import OverlaySceneRenderer
from worker.interfaces.serving import BatchServingClient, ServingClient
from worker.interfaces.thumbnail import ThumbnailGenerator

__all__ = [
    "BatchServingClient",
    "ClipEncoder",
    "ClipFinalizer",
    "Decider",
    "DecodeAdapter",
    "DecodeSession",
    "EncoderSession",
    "EventSink",
    "Extractor",
    "FrameBus",
    "FrameMaterializer",
    "FrameSubscription",
    "HostFrameView",
    "OverlaySceneRenderer",
    "PerceptionFrameAdapter",
    "ServingClient",
    "ThumbnailGenerator",
]
