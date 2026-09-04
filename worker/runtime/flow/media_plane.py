"""Composition-owned Flow media plane and Smart Record bridge."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Final

from worker.adapters.deepstream.service_maker import (
    DeepStreamMediaPlane,
    DeepStreamMediaPlaneConfig,
    FlowFactory,
)
from worker.interfaces.media_plane import MediaPlane, RecordingInfo, SnapshotUnavailable
from worker.native.deepstream.metadata import LatestMetadataSlot, SourceBinding
from worker.pipeline.output.evidence.smart_record_actor import ClipSealed, SmartRecordActor
from worker.pipeline.output.live_view import LatestFrameStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FlowMediaPlaneConfig:
    infer_config_path: str
    tracker_config_path: str
    tracker_library_path: str
    record_dir: Path
    record_cache_seconds: int
    frame_width: int
    frame_height: int
    source_silence_timeout_sec: float = 30.0

    def adapter_config(self) -> DeepStreamMediaPlaneConfig:
        return DeepStreamMediaPlaneConfig(
            infer_config_path=self.infer_config_path,
            tracker_config_path=self.tracker_config_path,
            tracker_library_path=self.tracker_library_path,
            record_dir=self.record_dir,
            record_cache_seconds=self.record_cache_seconds,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )


class FlowMediaPlane:
    """Runtime-owned facade exposing the control subset used by policy pumps."""

    def __init__(
        self,
        config: FlowMediaPlaneConfig,
        *,
        flow_factory: FlowFactory | None = None,
        snapshot_encoder: Callable[[str], bytes] | None = None,
        live_frames: LatestFrameStore | None = None,
        worker_boot_id: str | None = None,
    ) -> None:
        self.config = config
        self.metadata = LatestMetadataSlot()
        kwargs: dict[str, object] = {
            "metadata_slot": self.metadata,
            "snapshot_encoder": snapshot_encoder,
            "jpeg_publisher": self._publish_jpeg,
            "worker_boot_id": worker_boot_id,
        }
        if flow_factory is not None:
            kwargs["flow_factory"] = flow_factory
        self.plane = DeepStreamMediaPlane(config.adapter_config(), **kwargs)
        self._actors: dict[str, SmartRecordActor] = {}
        self._live_frames = live_frames
        if live_frames is not None:
            live_frames.set_demand_listener(self._refresh_live_frame)

    def start(self) -> None:
        self.plane.start()

    def stop(self) -> None:
        self.plane.stop()

    def snapshot(self, camera_id: str) -> bytes:
        jpeg = self.plane.snapshot(camera_id)
        self._publish_jpeg(camera_id, jpeg)
        return jpeg

    def bind_live_frames(self, live_frames: LatestFrameStore) -> None:
        if self._live_frames is not None and self._live_frames is not live_frames:
            raise RuntimeError("Flow live-frame store is already bound")
        self._live_frames = live_frames
        live_frames.set_demand_listener(self._refresh_live_frame)

    def _refresh_live_frame(
        self, camera_id: str, viewers: int, mode: str, snapshot_requested: bool
    ) -> None:
        del viewers, mode, snapshot_requested
        # This callback runs on an HTTP thread, never on Flow's probe thread.
        try:
            self.snapshot(camera_id)
        except SnapshotUnavailable:
            # Expected before the first frame; the HTTP layer answers with its
            # own typed unavailable response.
            return
        except (RuntimeError, OSError) as error:
            # Anything else is a real fault of the preview path and must be
            # visible, but a demand signal must not crash the HTTP server.
            LOGGER.warning("live-frame refresh failed for camera_id=%s: %s", camera_id, error)

    def _publish_jpeg(self, camera_id: str, jpeg: bytes) -> None:
        if self._live_frames is not None:
            self._live_frames.publish_jpeg(camera_id, jpeg, frame_index=0)

    def add_source(self, camera_id: str, uri: str) -> SourceBinding:
        return self.plane.add_source(camera_id, uri)

    def remove_source(self, camera_id: str) -> None:
        self.plane.remove_source(camera_id)

    def source_failure(self, camera_id: str, category: str) -> SourceBinding:
        return self.plane.source_failure(camera_id, category)

    def status(self):
        return self.plane.status()

    def clear_preview(self, camera_id: str) -> None:
        if self._live_frames is not None:
            self._live_frames.clear_camera(camera_id)

    #: The clip the acceptance criteria name is 60 s: this cached lookback plus
    #: the plane's forward window. It lives here, with the recorder, so a single
    #: place owns the contract instead of a literal at the call site.
    DEFAULT_LOOKBACK_SEC: Final = 15

    def smart_recorder(
        self,
        camera_id: str,
        *,
        sink: Callable[[ClipSealed], None],
        lookback_sec: int = DEFAULT_LOOKBACK_SEC,
        clock: Callable[[], float] = monotonic,
    ) -> SmartRecordActor:
        if camera_id in self._actors:
            raise ValueError(f"Flow smart recorder already exists for {camera_id}")

        def on_sealed(item: ClipSealed | BaseException) -> None:
            if isinstance(item, BaseException):
                raise item
            sink(item)

        actor = SmartRecordActor(
            camera_id=camera_id,
            media_plane=self.plane,
            clock=clock,
            sink=on_sealed,
            lookback_sec=lookback_sec,
        )
        self._actors[camera_id] = actor
        return actor

    def recorder_counters(self, camera_id: str) -> tuple[int, int, int]:
        actor = self._actors.get(camera_id)
        if actor is None:
            return (0, 0, 0)
        return (
            actor.smart_record_extended_total,
            actor.smart_record_extension_raced_total,
            actor.smart_record_start_refused_total,
        )

    @property
    def media_plane(self) -> MediaPlane:
        return self.plane

    def sealed_recording(self, info: RecordingInfo) -> None:
        """Test seam for delivering a Flow recording completion to its actor."""
        actor = self._actors.get(info.camera_id)
        if actor is None:
            raise ValueError(f"Flow smart recorder is absent for {info.camera_id}")
        actor.on_sealed(info)


__all__ = ["FlowMediaPlane", "FlowMediaPlaneConfig"]
