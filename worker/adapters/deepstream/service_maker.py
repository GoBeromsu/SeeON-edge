"""pyservicemaker implementation of the vendor-neutral media plane."""

from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from worker.adapters.deepstream.metadata import convert_frame
from worker.adapters.deepstream.sources import SourceTable
from worker.adapters.deepstream.tensor_rows import rows_from_tensor
from worker.interfaces.media_plane import (
    MediaPlane,
    MediaPlaneStatus,
    RecordingInfo,
    RecordingRefused,
    SourceStatus,
)
from worker.native.deepstream.metadata import LatestMetadataSlot, SourceBinding


class FlowFactory(Protocol):
    def __call__(self, config: DeepStreamMediaPlaneConfig) -> _FlowHandle: ...


@dataclass(frozen=True, slots=True)
class DeepStreamMediaPlaneConfig:
    infer_config_path: str
    tracker_config_path: str
    tracker_library_path: str
    record_dir: Path
    record_cache_seconds: int
    frame_width: int
    frame_height: int
    transform_id: str = "deepstream-flow.v1"
    pipeline_name: str = "deepstream-media-plane"


@dataclass(slots=True)
class _Recording:
    session_id: int
    on_sealed: Callable[[RecordingInfo], None]
    sealed: bool = False


@dataclass(frozen=True, slots=True)
class _FlowHandle:
    flow: Any
    pipeline: Any


def _default_flow_factory(config: DeepStreamMediaPlaneConfig) -> _FlowHandle:
    from pyservicemaker import Flow, Pipeline

    pipeline = Pipeline(config.pipeline_name)
    return _FlowHandle(flow=Flow(pipeline), pipeline=pipeline)


class _Probe:
    def __init__(self, plane: DeepStreamMediaPlane) -> None:
        self._plane = plane

    def handle_metadata(self, batch_meta: Any) -> None:
        for frame_meta in batch_meta.frame_items:
            self._plane.publish_frame(frame_meta)


class DeepStreamMediaPlane(MediaPlane):
    """A single Flow worker; vendor calls are contained in this adapter."""

    def __init__(
        self,
        config: DeepStreamMediaPlaneConfig,
        *,
        metadata_slot: LatestMetadataSlot,
        flow_factory: FlowFactory = _default_flow_factory,
        snapshot_encoder: Callable[[str], bytes] | None = None,
        worker_boot_id: str | None = None,
        child_instance_id: str | None = None,
    ) -> None:
        self._config = config
        self._slot = metadata_slot
        handle = flow_factory(config)
        self._flow = handle.flow
        self._pipeline = handle.pipeline
        self._sources = SourceTable(
            worker_boot_id=worker_boot_id or str(uuid.uuid4()),
            child_instance_id=child_instance_id or str(uuid.uuid4()),
            transform_id=config.transform_id,
        )
        self._snapshot_encoder = snapshot_encoder
        self._commands: queue.Queue[
            tuple[Callable[[], Any], threading.Event | None, list[Any] | None]
        ] = queue.Queue()
        self._command_thread = threading.Thread(
            target=self._run_commands, name="deepstream-flow-commands", daemon=True
        )
        self._command_thread.start()
        self._live: set[str] = set()
        self._publish_sequence: dict[str, int] = {}
        self._recordings: dict[str, _Recording] = {}
        self._active_encode_sessions: set[int] = set()
        self._started = False
        self._probe = _Probe(self)

    def start(self) -> None:
        if self._started:
            return
        self._build_flow()
        self._started = True
        self._flow.start()

    def stop(self) -> None:
        if self._started:
            self._flow.stop()
            self._started = False

    def status(self) -> MediaPlaneStatus:
        return MediaPlaneStatus(
            sources=tuple(
                SourceStatus(
                    camera_id, self._sources.binding(camera_id), camera_id in self._live, 0
                )
                for camera_id in self._sources.camera_ids()
            ),
            engine_identity="pyservicemaker-flow",
            nvenc_sessions_active=len(self._active_encode_sessions),
        )

    def add_source(self, camera_id: str, uri: str) -> SourceBinding:
        binding = self._sources.add(camera_id, uri)
        self._slot.register_source(binding)
        if self._started:
            self._enqueue(lambda: self._flow.add_source(uri))
        return binding

    def remove_source(self, camera_id: str) -> None:
        source_name = self._sources.source_name(camera_id)
        self._slot.remove_source(camera_id)
        self._live.discard(camera_id)
        self._recordings.pop(camera_id, None)
        self._sources.remove(camera_id)
        if self._started:
            self._enqueue(lambda: self._flow.remove_source(source_name))

    def source_failure(self, camera_id: str, category: str) -> SourceBinding:
        del category
        binding = self._sources.rebuild(camera_id)
        self._slot.register_source(binding)
        self._live.discard(camera_id)
        return binding

    def snapshot(self, camera_id: str) -> bytes:
        if self._snapshot_encoder is None:
            raise RuntimeError("OSD snapshot encoder is not configured")
        return self._snapshot_encoder(camera_id)

    def start_recording(
        self,
        camera_id: str,
        *,
        lookback_sec: int,
        duration_sec: int,
        on_sealed: Callable[[RecordingInfo], None],
    ) -> int:
        if camera_id not in self._live:
            raise RecordingRefused(f"source has not published a frame: {camera_id}")
        existing = self._recordings.get(camera_id)
        if existing is not None:
            return existing.session_id
        session_id = self._start_signal(camera_id, lookback_sec, duration_sec)
        self._recordings[camera_id] = _Recording(session_id, on_sealed)
        return session_id

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        recording = self._recordings.get(camera_id)
        if recording is None or recording.session_id != session_id:
            raise RecordingRefused(f"unknown recording session {session_id} for {camera_id}")
        self._call_on_pipeline(lambda: self._source_element(camera_id).emit("stop-sr", session_id))

    def _enqueue(self, command: Callable[[], Any]) -> None:
        self._commands.put((command, None, None))

    def _call_on_pipeline(self, command: Callable[[], Any]) -> Any:
        done = threading.Event()
        result: list[Any] = []
        self._commands.put((command, done, result))
        done.wait()
        value = result[0]
        if isinstance(value, BaseException):
            raise value
        return value

    def _run_commands(self) -> None:
        while True:
            command, done, result = self._commands.get()
            try:
                value: Any = command()
            except Exception as error:  # noqa: BLE001 - the caller re-raises on its own thread
                value = error
            if result is not None:
                result.append(value)
            if done is not None:
                done.set()

    def _build_flow(self) -> None:
        camera_ids = self._sources.camera_ids()
        uris = [self._sources.uri(camera_id) for camera_id in camera_ids]
        if not uris:
            return
        from pyservicemaker import RecordConfig, RenderMode

        record = RecordConfig(
            recording_type="local",
            rec_cache=self._config.record_cache_seconds,
            rec_dir_path=str(self._config.record_dir),
        )
        self._flow.batch_capture(
            uris,
            record_config=record,
            width=self._config.frame_width,
            height=self._config.frame_height,
        ).infer(self._config.infer_config_path).track(
            ll_config_file=self._config.tracker_config_path,
            ll_lib_file=self._config.tracker_library_path,
        ).attach(self._probe).render(mode=RenderMode.DISCARD)
        for camera_id in camera_ids:
            source = self._source_element(camera_id)
            source.set(
                {
                    "select-rtp-protocol": 4,
                    "latency": 200,
                    "init-rtsp-reconnect-interval": 5,
                    "rtsp-reconnect-interval": 5,
                }
            )
            source.connect(
                "sr-done", lambda info, camera_id=camera_id: self._recording_done(camera_id, info)
            )

    def _source_element(self, camera_id: str) -> Any:
        return self._pipeline[self._sources.source_name(camera_id)]

    def _start_signal(self, camera_id: str, lookback_sec: int, duration_sec: int) -> int:
        # Flow commands originate off the probe callback.  Start needs its
        # returned session immediately, so the source action is the command.
        return int(
            self._call_on_pipeline(
                lambda: self._source_element(camera_id).emit(
                    "start-sr", lookback_sec, duration_sec, None
                )
            )
        )

    def publish_frame(self, frame_meta: Any) -> None:
        """Probe entry: convert one accepted frame and publish it to the slot."""
        camera_id = self._sources.camera_id(int(frame_meta.pad_index))
        rows = None
        for tensor_meta in frame_meta.tensor_items:
            layer = tensor_meta.as_tensor_output().get_layers()["output0"]
            rows = rows_from_tensor(layer)
            break
        if rows is None:
            return
        sequence = self._publish_sequence.get(camera_id, 0) + 1
        self._publish_sequence[camera_id] = sequence
        binding = self._sources.binding(camera_id)
        metadata = convert_frame(
            frame_meta,
            rows=rows,
            binding=binding,
            frame_w=self._config.frame_width,
            frame_h=self._config.frame_height,
            publish_sequence=sequence,
            boot_id=binding.worker_boot_id,
        )
        self._live.add(camera_id)
        self._slot.publish(metadata)

    def _recording_done(self, camera_id: str, info: Any) -> None:
        recording = self._recordings.get(camera_id)
        if recording is None or recording.sealed:
            return
        recording.sealed = True
        path = str(Path(str(info.dirpath)) / str(info.filename))
        recording.on_sealed(
            RecordingInfo(
                session_id=int(info.session_id),
                camera_id=camera_id,
                path=path,
                duration_ms=int(info.duration),
                width=int(info.width),
                height=int(info.height),
            )
        )
        self._recordings.pop(camera_id, None)


__all__ = ["DeepStreamMediaPlane", "DeepStreamMediaPlaneConfig", "FlowFactory", "_FlowHandle"]
