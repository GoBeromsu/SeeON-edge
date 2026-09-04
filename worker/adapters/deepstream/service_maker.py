"""pyservicemaker implementation of the vendor-neutral media plane."""

from __future__ import annotations

import io
import logging
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from worker.adapters.deepstream.metadata import convert_frame
from worker.adapters.deepstream.sources import SourceTable
from worker.adapters.deepstream.tensor_rows import host_array_from_tensor, rows_from_tensor
from worker.interfaces.media_plane import (
    EarlyStopUnsupported,
    MediaPlane,
    MediaPlaneStatus,
    RecordingInfo,
    RecordingRefused,
    SnapshotUnavailable,
    SourceRosterFixed,
    SourceStatus,
)
from worker.native.deepstream.metadata import LatestMetadataSlot, SourceBinding

_MAX_PREVIEW_JPEG_BYTES = 2 * 1024 * 1024
#: How often the plane reports what perception is actually producing.
_PERCEPTION_HEARTBEAT_FRAMES = 900

#: A frame whose inference produced no tensor metadata still has real tracks.
_EMPTY_POSE_ROWS: Final = np.zeros((0, 57), dtype=np.float32)
LOGGER = logging.getLogger(__name__)


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
    preview_jpeg_stride: int = 10
    #: The preview retriever costs the pipeline its throughput (see _build_flow);
    #: it stays off until a non-throttling frame path exists.
    preview_retriever_enabled: bool = False
    transform_id: str = "deepstream-flow.v1"
    pipeline_name: str = "deepstream-media-plane"


@dataclass(slots=True)
class _Recording:
    session_id: int
    on_sealed: Callable[[RecordingInfo], None]
    sealed: bool = False


@dataclass(frozen=True, slots=True)
class _FlowHandle:
    """Everything the plane needs from the SDK, so tests can supply fakes."""

    flow: Any
    pipeline: Any
    record_config: Callable[..., Any]
    render_mode_discard: Any
    make_probe: Callable[[str, _Probe], Any]
    make_retriever: Callable[[_JpegRetriever], Any] | None = None


def _default_flow_factory(config: DeepStreamMediaPlaneConfig) -> _FlowHandle:
    from pyservicemaker import Flow, Pipeline, Probe, RecordConfig, RenderMode

    pipeline = Pipeline(config.pipeline_name)
    return _FlowHandle(
        flow=Flow(pipeline),
        pipeline=pipeline,
        record_config=RecordConfig,
        render_mode_discard=RenderMode.DISCARD,
        make_probe=lambda name, probe: Probe(name, _batch_operator(probe)),
        make_retriever=_buffer_retriever,
    )


class _Probe:
    """Vendor-neutral half of the probe; wrapped in a BatchMetadataOperator at build time."""

    def __init__(self, plane: DeepStreamMediaPlane) -> None:
        self._plane = plane

    def handle_metadata(self, batch_meta: Any) -> None:
        for frame_meta in batch_meta.frame_items:
            self._plane.publish_frame(frame_meta)


def _batch_operator(probe: _Probe) -> Any:
    """Subclass the SDK operator lazily so this module imports without pyservicemaker."""
    from pyservicemaker import BatchMetadataOperator

    class _Operator(BatchMetadataOperator):
        def __init__(self) -> None:
            super().__init__()

        def handle_metadata(self, batch_meta: Any) -> None:
            probe.handle_metadata(batch_meta)

    return _Operator()


def _buffer_retriever(retriever: _JpegRetriever) -> Any:
    """Create the SDK subclass lazily so imports work without DeepStream."""
    from pyservicemaker import BufferRetriever

    class _Retriever(BufferRetriever):
        def __init__(self) -> None:
            super().__init__()

        def consume(self, buffer: Any) -> int:
            return retriever.consume(buffer)

    return _Retriever()


class _JpegRetriever:
    """Best-effort preview encoder invoked from Flow's batched-buffer thread."""

    def __init__(self, plane: DeepStreamMediaPlane, stride: int) -> None:
        """Encode only what a viewer asked for.

        ``retrieve`` is the Flow's terminal sink, so this callback runs on the
        pipeline thread and its cost is the pipeline's cost: copying and
        encoding every stride-th frame collapsed throughput from 30 fps to
        roughly 2 fps and starved the decision layer of frames. The live view
        is demand-driven, so nothing is copied until a snapshot is requested,
        and then only for the camera that was asked for.
        """
        if stride < 1:
            raise ValueError("preview_jpeg_stride must be at least one")
        self._plane = plane
        self._stride = stride
        self._frames: dict[str, int] = {}

    def consume(self, buffer: Any) -> int:
        try:
            for batch_id in range(int(buffer.batch_size)):
                camera_id = self._plane.camera_id_for_pad(batch_id)
                if camera_id is None:
                    continue
                if not self._plane.preview_wanted(camera_id):
                    continue
                count = self._frames.get(camera_id, 0) + 1
                self._frames[camera_id] = count
                if count % self._stride:
                    continue
                # The extracted surface is already an HWC uint8 frame; wrapping
                # it in a colour format is what the SDK rejects ("Color format
                # not compatible with tensor layout"), so copy it as-is.
                jpeg = _encode_preview_jpeg(host_array_from_tensor(buffer.extract(batch_id)))
                self._plane.publish_jpeg(camera_id, jpeg)
        except Exception:  # noqa: BLE001 - preview work must not stop inference
            LOGGER.warning("dropping DeepStream preview frame", exc_info=True)
        return 0


def _encode_preview_jpeg(pixels: Any) -> bytes:
    """Encode an RGB/RGBA uint8 frame, reducing quality/size to the slot cap."""
    from PIL import Image

    if pixels.ndim == 4 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.dtype.name != "uint8":
        raise ValueError(f"preview tensor must be HWC uint8 RGB/RGBA, received {pixels.shape}")
    image = Image.fromarray(pixels[:, :, :3], "RGB")
    for quality in (80, 60, 40):
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        jpeg = output.getvalue()
        if len(jpeg) <= _MAX_PREVIEW_JPEG_BYTES:
            return jpeg
    while image.width > 1 and image.height > 1:
        image = image.resize((image.width // 2, image.height // 2))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=40, optimize=True)
        jpeg = output.getvalue()
        if len(jpeg) <= _MAX_PREVIEW_JPEG_BYTES:
            return jpeg
    raise ValueError("preview JPEG exceeds maximum size")


class DeepStreamMediaPlane(MediaPlane):
    """A single Flow worker; vendor calls are contained in this adapter."""

    def __init__(
        self,
        config: DeepStreamMediaPlaneConfig,
        *,
        metadata_slot: LatestMetadataSlot,
        flow_factory: FlowFactory = _default_flow_factory,
        snapshot_encoder: Callable[[str], bytes] | None = None,
        jpeg_publisher: Callable[[str, bytes], None] | None = None,
        worker_boot_id: str | None = None,
        child_instance_id: str | None = None,
    ) -> None:
        if config.preview_jpeg_stride < 1:
            raise ValueError("preview_jpeg_stride must be at least one")
        self._config = config
        self._slot = metadata_slot
        handle = flow_factory(config)
        self._handle = handle
        self._flow = handle.flow
        self._pipeline = handle.pipeline
        self._sources = SourceTable(
            worker_boot_id=worker_boot_id or str(uuid.uuid4()),
            child_instance_id=child_instance_id or str(uuid.uuid4()),
            transform_id=config.transform_id,
        )
        self._snapshot_encoder = snapshot_encoder
        self._jpeg_publisher = jpeg_publisher
        self._latest_jpegs: dict[str, bytes] = {}
        self._jpeg_lock = threading.Lock()
        # A pose path that produces nothing must not be silent: without these a
        # dead parser looks exactly like an empty room for hours.
        self._frames_without_pose_tensor = 0
        self._objects_observed = 0
        self._matched_tracks = 0
        self._accepting = True
        self._probe_failures = 0
        self._cameras_warned_without_pose_tensor: set[str] = set()
        self._commands: queue.Queue[
            tuple[Callable[[], Any], threading.Event | None, list[Any] | None]
        ] = queue.Queue()
        self._command_thread = threading.Thread(
            target=self._run_commands, name="deepstream-flow-commands", daemon=True
        )
        self._command_thread.start()
        self._live: set[str] = set()
        self._publish_sequence: dict[str, int] = {}
        self._unmapped_pads: set[int] = set()
        self._preview_requested: set[str] = set()
        self._recordings: dict[str, _Recording] = {}
        self._active_encode_sessions: set[int] = set()
        self._started = False
        self._flow_thread: threading.Thread | None = None
        self._flow_error: Exception | None = None
        self._flow_finished = threading.Event()
        self._probe = _Probe(self)
        self._jpeg_retriever = _JpegRetriever(self, config.preview_jpeg_stride)

    def start(self) -> None:
        """Build the Flow from the registered roster and run it on its own thread.

        A pyservicemaker Flow fixes its sources when it is built and blocks
        when called, so the media plane runs it on a dedicated thread and ends
        it through ``Pipeline.stop()``. Sources registered after this point are
        refused (``SourceRosterFixed``): the worker restarts to change its
        roster, exactly as the nvidia child does.
        """
        if self._started:
            return
        self._build_flow()
        self._started = True
        self._flow_thread = threading.Thread(
            target=self._run_flow, name="deepstream-flow", daemon=True
        )
        self._flow_thread.start()

    def _run_flow(self) -> None:
        try:
            self._flow()
        except Exception as error:  # noqa: BLE001 - surfaced through status, never swallowed
            self._flow_error = error
        finally:
            self._flow_finished.set()

    def stop(self) -> None:
        if self._started:
            # Disarm the probe first. Teardown removes sources from the table
            # while the SDK is still delivering buffers, and a probe that keeps
            # converting against an emptied table both floods the log with
            # unmapped-pad warnings and races the SDK's own stream removal.
            self._accepting = False
            self._pipeline.stop()
            self._started = False
            if self._flow_thread is not None:
                self._flow_thread.join(timeout=10.0)
                if self._flow_thread.is_alive():
                    LOGGER.error(
                        "the DeepStream Flow did not stop within 10s; "
                        "the pipeline may still be delivering buffers"
                    )

    def camera_ids_for_preview(self) -> tuple[str, ...]:
        """Cameras this plane can encode a preview for."""
        return self._sources.camera_ids()

    def request_preview(self, camera_id: str) -> None:
        """Arm one preview encode for this camera on the next batched buffer."""
        with self._jpeg_lock:
            self._preview_requested.add(camera_id)

    def preview_wanted(self, camera_id: str) -> bool:
        """Whether a viewer has asked for this camera since the last encode."""
        with self._jpeg_lock:
            return camera_id in self._preview_requested

    def camera_id_for_pad(self, pad_index: int) -> str | None:
        """Which camera a batch/pad index belongs to, or None when unmapped."""
        return self._sources.camera_id_for_pad(pad_index)

    def published_frames(self, camera_id: str) -> int:
        """Frames this plane has published for a camera since it started.

        Monotonic and never consumed. The metadata slot is a capacity-one
        mailbox that the policy pump drains, so peeking at it reports silence
        the moment the pump keeps up; this counter is the liveness signal that
        survives consumption.
        """
        return self._publish_sequence.get(camera_id, 0)

    def perception_counters(self) -> tuple[int, int]:
        """Objects the tracker delivered, and frames that carried no pose tensor.

        A pose path that silently produces nothing is indistinguishable from an
        empty room, which is exactly how a mis-bound output layer hid for a
        whole bring-up. These make the difference observable.
        """
        return self._objects_observed, self._frames_without_pose_tensor

    def probe_failures(self) -> int:
        """Frames dropped because their conversion raised inside the SDK probe."""
        return self._probe_failures

    def status(self) -> MediaPlaneStatus:
        error = self._flow_error
        return MediaPlaneStatus(
            fatal_error=None if error is None else f"{type(error).__name__}: {error}",
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
        if self._started:
            raise SourceRosterFixed(
                f"cannot add {camera_id!r}: the Flow's sources are fixed once it runs; "
                "restart the worker to change the roster"
            )
        binding = self._sources.add(camera_id, uri)
        self._slot.register_source(binding)
        return binding

    def remove_source(self, camera_id: str) -> None:
        if self._started:
            raise SourceRosterFixed(
                f"cannot remove {camera_id!r}: the Flow's sources are fixed once it runs; "
                "restart the worker to change the roster"
            )
        self._slot.remove_source(camera_id)
        self._live.discard(camera_id)
        with self._jpeg_lock:
            self._latest_jpegs.pop(camera_id, None)
        self._recordings.pop(camera_id, None)
        self._sources.remove(camera_id)

    def source_failure(self, camera_id: str, category: str) -> SourceBinding:
        """Rotate this camera's stream identity; the plugin owns media recovery.

        A Flow fixes its sources when it is built, so this does not and cannot
        rebuild the pipeline element. What it does is the part the decision
        layer needs: a new ``stream_epoch``/``source_generation`` so frames
        arriving after the outage are never mistaken for the old stream, and a
        cleared live/preview state so nothing stale is served. Media recovery
        is ``nvurisrcbin``'s own RTSP reconnect, configured on every source
        (measured in the spike: a stalled source recovers within one interval).
        The canonical camera id never changes across this.
        """
        del category
        binding = self._sources.rebuild(camera_id)
        self._slot.register_source(binding)
        self._live.discard(camera_id)
        with self._jpeg_lock:
            self._latest_jpegs.pop(camera_id, None)
        return binding

    def snapshot(self, camera_id: str) -> bytes:
        if camera_id not in self._sources.camera_ids():
            raise SnapshotUnavailable(f"unknown source has no OSD snapshot: {camera_id}")
        # Arm the next batched buffer to produce a frame for this camera; the
        # pipeline thread does no preview work until something asks.
        self.request_preview(camera_id)
        if self._snapshot_encoder is not None:
            # The encoder is a bounded, non-probe seam.  Invoke it once for this
            # request and retain the resulting OSD JPEG for other consumers.
            self.publish_jpeg(camera_id, self._snapshot_encoder(camera_id))
        with self._jpeg_lock:
            jpeg = self._latest_jpegs.get(camera_id)
        if jpeg is None:
            raise SnapshotUnavailable(f"source has not produced an OSD snapshot: {camera_id}")
        return jpeg

    def publish_jpeg(self, camera_id: str, jpeg: bytes) -> None:
        """Accept one encoded OSD frame and satisfy the pending preview request."""
        if camera_id not in self._sources.camera_ids():
            raise SnapshotUnavailable(f"unknown source has no OSD snapshot: {camera_id}")
        with self._jpeg_lock:
            self._preview_requested.discard(camera_id)
        if not jpeg:
            raise SnapshotUnavailable(f"source produced an empty OSD snapshot: {camera_id}")
        if len(jpeg) > _MAX_PREVIEW_JPEG_BYTES:
            raise SnapshotUnavailable(
                f"source produced an oversized OSD snapshot: {camera_id} ({len(jpeg)} bytes)"
            )
        with self._jpeg_lock:
            self._latest_jpegs[camera_id] = jpeg
        if self._jpeg_publisher is not None:
            self._jpeg_publisher(camera_id, jpeg)

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
        # Measured (docs/research/pyservicemaker-p1b-spike.md): the SDK's
        # Pipeline.stop_recording returns True but never seals, and the binding
        # exposes no way to emit the element's stop-sr action signal. The clip
        # therefore runs to the duration given at start and seals itself.
        raise EarlyStopUnsupported(
            f"pyservicemaker cannot stop session {session_id} on {camera_id} early; "
            "the recording seals at its start duration"
        )

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
        record = self._handle.record_config(
            recording_type="local",
            rec_cache=self._config.record_cache_seconds,
            rec_dir_path=str(self._config.record_dir),
        )
        flow = (
            self._flow.batch_capture(
                uris,
                record_config=record,
                width=self._config.frame_width,
                height=self._config.frame_height,
            )
            .infer(self._config.infer_config_path)
            .track(
                ll_config_file=self._config.tracker_config_path,
                ll_lib_file=self._config.tracker_library_path,
            )
        )
        flow.attach(what=self._handle.make_probe("media-plane-probe", self._probe))
        # `render(DISCARD)` is the terminal sink because it is the only one that
        # keeps up: measured in this image, a `retrieve()` sink delivers roughly
        # 2 batched buffers/s where `render` delivers 30, which starves the
        # decision layer of frames. A preview branch must never be paid for with
        # the safety pipeline's throughput, so the JPEG retriever is opt-in and
        # off by default (ML_WORKER_FLOW_PREVIEW_RETRIEVER=1) until a sink that
        # does not throttle the graph is wired.
        if self._handle.make_retriever is None or not self._config.preview_retriever_enabled:
            # A clock-synced sink paces the whole graph, which on a live RTSP
            # source backs the reader up until the server drops it as too slow.
            flow.render(mode=self._handle.render_mode_discard, enable_osd=False, sync=False)
        else:
            # `retrieve` IS the terminal sink: it pulls each batched buffer into
            # Python, which is where the preview JPEG comes from. Measured in
            # the shipped image: attach(probe) + retrieve() delivers metadata
            # and buffers one-for-one through the full infer/track chain, while
            # forking a second branch beside a render sink delivers nothing.
            flow.retrieve(self._handle.make_retriever(self._jpeg_retriever))
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

    def _source_element(self, camera_id: str) -> Any:
        return self._pipeline[self._sources.source_name(camera_id)]

    def _start_signal(self, camera_id: str, lookback_sec: int, duration_sec: int) -> int:
        # Pipeline.start_recording is the SDK's working primitive: it returns
        # the session id and delivers RecordingInfo to the callback when the
        # clip seals (measured live: start(5,6) -> sr-done at +6.06 s).
        source_name = self._sources.source_name(camera_id)
        return int(
            self._call_on_pipeline(
                lambda: self._pipeline.start_recording(
                    source_name,
                    lookback_sec,
                    duration_sec,
                    lambda info, camera_id=camera_id: self._recording_done(camera_id, info),
                )
            )
        )

    def publish_frame(self, frame_meta: Any) -> None:
        """Probe entry: convert one accepted frame and publish it to the slot.

        Any exception escaping this callback aborts the process inside the SDK,
        so a frame that cannot be converted is dropped and reported rather than
        allowed to take the worker down with it.
        """
        try:
            self._publish_frame(frame_meta)
        except Exception:  # noqa: BLE001 - an SDK probe must never raise
            self._probe_failures += 1
            if self._probe_failures == 1:
                LOGGER.exception("dropping a frame whose conversion failed inside the SDK probe")

    def _publish_frame(self, frame_meta: Any) -> None:
        if not self._accepting:
            # Stopping: the source table is being emptied out from under this
            # callback, so there is nothing meaningful left to convert.
            return
        pad_index = int(frame_meta.pad_index)
        camera_id = self._sources.camera_id_for_pad(pad_index)
        if camera_id is None:
            # Never raise from a probe callback: the SDK aborts the process.
            if pad_index not in self._unmapped_pads:
                self._unmapped_pads.add(pad_index)
                LOGGER.warning(
                    "dropping frames from unmapped mux pad %d; known pads are %s",
                    pad_index,
                    sorted(self._sources.pad_index(name) for name in self._sources.camera_ids()),
                )
            return
        rows = None
        for tensor_meta in frame_meta.tensor_items:
            layer = tensor_meta.as_tensor_output().get_layers()["output0"]
            rows = rows_from_tensor(layer)
            break
        objects = int(getattr(frame_meta, "num_obj_meta", 0) or 0)
        if objects == 0:
            objects = sum(1 for _ in frame_meta.object_items)
        self._objects_observed += objects
        if rows is None:
            self._frames_without_pose_tensor += 1
            if camera_id not in self._cameras_warned_without_pose_tensor:
                self._cameras_warned_without_pose_tensor.add(camera_id)
                LOGGER.warning(
                    "no pose tensor metadata on the first frame from camera %s "
                    "(%d tracked objects on it); every track on such a frame is unmatched",
                    camera_id,
                    objects,
                )
            # nvinfer attaches tensor metadata per inferred frame; a frame can
            # arrive without it (skipped inference interval, exhausted tensor
            # pool). The tracked objects are still real, so publish the frame
            # with no pose rows - every track is then explicitly unmatched -
            # rather than dropping it, which would hide the frame from the
            # decision layer and make the camera look silent.
            rows = _EMPTY_POSE_ROWS
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
        self._matched_tracks += len(metadata.frame.association.track_ids)
        if sequence % _PERCEPTION_HEARTBEAT_FRAMES == 0:
            # Operator-visible in the message itself: this is the one line that
            # distinguishes an empty room from a perception path that is dead.
            LOGGER.info(
                "perception heartbeat camera_id=%s frames=%d objects=%d matched_tracks=%d "
                "frames_without_pose_tensor=%d",
                camera_id,
                sequence,
                self._objects_observed,
                self._matched_tracks,
                self._frames_without_pose_tensor,
            )
        self._slot.publish(metadata)

    def _recording_done(self, camera_id: str, info: Any) -> None:
        recording = self._recordings.get(camera_id)
        if recording is None or recording.sealed:
            return
        # The SDK's RecordingInfo names these file_directory/file_name; the
        # older dirpath/filename spelling aborts the process from the callback.
        path = str(Path(str(info.file_directory)) / str(info.file_name))
        try:
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
        except Exception as error:  # noqa: BLE001 - an SDK callback must not abort the process
            # This runs on the pipeline thread: an exception here terminates the
            # worker and takes every camera down with it. The sealed media and
            # its contributor sidecar are already on disk, so the publication is
            # replayable; the recording is left unsealed for that retry.
            LOGGER.exception(
                "clip publication failed for camera_id=%s session=%s; media is retained at %s "
                "and will be republished (%s)",
                camera_id,
                info.session_id,
                path,
                type(error).__name__,
            )
            self._recordings.pop(camera_id, None)
            return
        recording.sealed = True
        self._recordings.pop(camera_id, None)


__all__ = ["DeepStreamMediaPlane", "DeepStreamMediaPlaneConfig", "FlowFactory", "_FlowHandle"]
