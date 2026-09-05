"""pyservicemaker implementation of the vendor-neutral media plane."""

from __future__ import annotations

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
    MetadataSlot,
    RecordingInfo,
    RecordingRefused,
    SnapshotUnavailable,
    SourceRosterFixed,
    SourceStatus,
)
from worker.types.metadata import SourceBinding

#: How often the plane reports what perception is actually producing.
_PERCEPTION_HEARTBEAT_FRAMES = 900
#: Isolated malformed SDK metadata is tolerated; a broken conversion contract is fatal.
_PROBE_CONSECUTIVE_FAILURE_THRESHOLD = 3

#: A frame whose inference produced no tensor metadata still has real tracks.
_EMPTY_POSE_ROWS: Final = np.zeros((0, 57), dtype=np.float32)
LOGGER = logging.getLogger(__name__)


class DeepStreamFlowStopTimeout(RuntimeError):
    """The SDK Flow still owns media resources after requested shutdown."""


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
    #: The forked OSD/JPEG snapshot branch. Off by default until its measured
    #: throughput cost is accepted for a deployment.
    snapshot_branch_enabled: bool = False


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
    make_jpeg_retriever: Callable[[str, DeepStreamMediaPlane], Any] | None = None


def _default_flow_factory(config: DeepStreamMediaPlaneConfig) -> _FlowHandle:
    from pyservicemaker import (
        BufferRetriever,
        Flow,
        Pipeline,
        Probe,
        Receiver,
        RecordConfig,
        RenderMode,
    )

    def make_jpeg_retriever(camera_id: str, plane: DeepStreamMediaPlane) -> Any:
        class _Retriever(BufferRetriever):
            def consume(self, buffer: Any) -> int:
                try:
                    jpeg = host_array_from_tensor(buffer.extract(0)).tobytes()
                    plane.publish_jpeg(camera_id, jpeg)
                except Exception as error:  # noqa: BLE001 - SDK callback containment
                    plane.snapshot_failed(camera_id, error)
                return 0

        return Receiver(f"snapshot-receiver-{camera_id}", _Retriever())

    pipeline = Pipeline(config.pipeline_name)
    return _FlowHandle(
        flow=Flow(pipeline),
        pipeline=pipeline,
        record_config=RecordConfig,
        render_mode_discard=RenderMode.DISCARD,
        make_probe=lambda name, probe: Probe(name, _batch_operator(probe)),
        make_jpeg_retriever=make_jpeg_retriever,
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


class DeepStreamMediaPlane(MediaPlane):
    """A single Flow worker; vendor calls are contained in this adapter."""

    def __init__(
        self,
        config: DeepStreamMediaPlaneConfig,
        *,
        metadata_slot: MetadataSlot,
        flow_factory: FlowFactory = _default_flow_factory,
        snapshot_encoder: Callable[[str], bytes] | None = None,
        worker_boot_id: str | None = None,
        child_instance_id: str | None = None,
    ) -> None:
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
        # A pose path that produces nothing must not be silent: without these a
        # dead parser looks exactly like an empty room for hours.
        self._frames_without_pose_tensor: dict[str, int] = {}
        self._objects_observed: dict[str, int] = {}
        self._matched_tracks: dict[str, int] = {}
        self._accepting = True
        self._probe_failures: dict[str, int] = {}
        self._probe_fatal_error: str | None = None
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
        self._recordings: dict[str, _Recording] = {}
        self._active_encode_sessions: set[int] = set()
        self._started = False
        self._flow_thread: threading.Thread | None = None
        self._flow_error: Exception | None = None
        self._flow_finished = threading.Event()
        self._probe = _Probe(self)
        self._snapshot_encoder = snapshot_encoder
        self._snapshot_requests: dict[str, tuple[threading.Event, list[bytes | BaseException]]] = {}
        self._snapshot_lock = threading.Lock()

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
            if self._flow_thread is not None:
                self._flow_thread.join(timeout=10.0)
                if self._flow_thread.is_alive():
                    raise DeepStreamFlowStopTimeout(
                        "DeepStream Flow did not stop within 10 seconds; "
                        "the pipeline may still be delivering buffers"
                    )
            self._started = False

    def published_frames(self, camera_id: str) -> int:
        """Frames this plane has published for a camera since it started.

        Monotonic and never consumed. The metadata slot is a capacity-one
        mailbox that the policy pump drains, so peeking at it reports silence
        the moment the pump keeps up; this counter is the liveness signal that
        survives consumption.
        """
        return self._publish_sequence.get(camera_id, 0)

    def perception_counters(self, camera_id: str) -> tuple[int, int]:
        """Objects the tracker delivered, and frames that carried no pose tensor.

        A pose path that silently produces nothing is indistinguishable from an
        empty room, which is exactly how a mis-bound output layer hid for a
        whole bring-up. These make the difference observable.
        """
        return (
            self._objects_observed.get(camera_id, 0),
            self._frames_without_pose_tensor.get(camera_id, 0),
        )

    def status(self) -> MediaPlaneStatus:
        error = self._flow_error
        return MediaPlaneStatus(
            fatal_error=(
                f"{type(error).__name__}: {error}" if error is not None else self._probe_fatal_error
            ),
            sources=tuple(
                SourceStatus(
                    camera_id, self._sources.binding(camera_id), camera_id in self._live, 0
                )
                for camera_id in self._sources.camera_ids()
            ),
            engine_identity="pyservicemaker-flow",
            nvenc_sessions_active=len(self._active_encode_sessions),
        )

    def camera_id_for_pad(self, pad_index: int) -> str | None:
        """Which camera a batch/pad index belongs to, or None when unmapped."""
        return self._sources.camera_id_for_pad(pad_index)

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
        return binding

    def snapshot(self, camera_id: str) -> bytes:
        if camera_id not in self._sources.camera_ids():
            raise SnapshotUnavailable(f"unknown source has no OSD snapshot: {camera_id}")
        if self._snapshot_encoder is not None:
            return self._snapshot_encoder(camera_id)
        if not self._started:
            raise SnapshotUnavailable(
                f"source has not started its OSD snapshot branch: {camera_id}"
            )
        done = threading.Event()
        result: list[bytes | BaseException] = []
        with self._snapshot_lock:
            if camera_id in self._snapshot_requests:
                raise SnapshotUnavailable(
                    f"OSD snapshot is already pending for source: {camera_id}"
                )
            self._snapshot_requests[camera_id] = (done, result)
        self._call_on_pipeline(
            lambda: self._pipeline[f"snapshot-valve-{self._sources.pad_index(camera_id)}"].set(
                {"drop": False}
            )
        )
        if not done.wait(timeout=2.0):
            self._call_on_pipeline(
                lambda: self._pipeline[
                    f"snapshot-valve-{self._sources.pad_index(camera_id)}"
                ].set({"drop": True})
            )
            with self._snapshot_lock:
                self._snapshot_requests.pop(camera_id, None)
            raise SnapshotUnavailable(f"OSD snapshot branch timed out for source: {camera_id}")
        value = result[0]
        if isinstance(value, BaseException):
            raise SnapshotUnavailable(
                f"OSD snapshot branch failed for source {camera_id}: {value}"
            ) from value
        return value

    def publish_jpeg(self, camera_id: str, jpeg: bytes) -> None:
        with self._snapshot_lock:
            request = self._snapshot_requests.pop(camera_id, None)
        if request is None:
            return
        done, result = request
        result.append(jpeg)
        self._enqueue(
            lambda: self._pipeline[f"snapshot-valve-{self._sources.pad_index(camera_id)}"].set(
                {"drop": True}
            )
        )
        done.set()

    def snapshot_failed(self, camera_id: str, error: BaseException) -> None:
        LOGGER.exception("OSD snapshot encoding failed for camera_id=%s", camera_id, exc_info=error)
        with self._snapshot_lock:
            request = self._snapshot_requests.pop(camera_id, None)
        if request is None:
            return
        done, result = request
        result.append(error)
        self._enqueue(
            lambda: self._pipeline[f"snapshot-valve-{self._sources.pad_index(camera_id)}"].set(
                {"drop": True}
            )
        )
        done.set()

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
        terminal_flow = self._build_snapshot_branch(flow, camera_ids)
        # `render(DISCARD)` is the terminal sink because it is the only one that
        # keeps up: measured in this image, a `retrieve()` sink delivers roughly
        # 2 batched buffers/s where `render` delivers 30, which starves the
        # decision layer of frames. The binding exposes pixels only through that
        # terminal retriever, so it cannot provide a non-throttling OSD snapshot.
        terminal_flow.render(mode=self._handle.render_mode_discard, enable_osd=False, sync=False)
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

    def _build_snapshot_branch(self, flow: Any, camera_ids: tuple[str, ...]) -> Any:
        """Build one normally-closed OSD/JPEG branch per source.

        The terminal Flow remains the discard sink.  A valve upstream of each
        encoder is opened only by ``snapshot`` and is immediately closed by
        the appsink callback, so idle branches do not encode frames.
        """
        if not self._config.snapshot_branch_enabled:
            return flow
        if self._handle.make_jpeg_retriever is None:
            return flow
        fork = flow.fork()
        tee = fork._streams[0].originator  # noqa: SLF001 - Flow has no public stream endpoint
        demux = "snapshot-demux"
        self._pipeline.add("nvstreamdemux", demux)
        self._pipeline.link(tee, demux)
        for index, camera_id in enumerate(camera_ids):
            queue_name = f"snapshot-queue-{index}"
            valve_name = f"snapshot-valve-{index}"
            convert_name = f"snapshot-convert-{index}"
            osd_name = f"snapshot-osd-{index}"
            post_osd_convert = f"snapshot-post-osd-convert-{index}"
            caps_name = f"snapshot-caps-{index}"
            encoder_name = f"snapshot-encoder-{index}"
            sink_name = f"snapshot-sink-{index}"
            self._pipeline.add("queue", queue_name, {"leaky": 2, "max-size-buffers": 1})
            self._pipeline.add("valve", valve_name, {"drop": True})
            self._pipeline.add("nvvideoconvert", convert_name, {"gpu-id": 0, "compute-hw": 1})
            self._pipeline.add("nvdsosd", osd_name, {"gpu-id": 0, "display-mask": True})
            self._pipeline.add(
                "nvvideoconvert", post_osd_convert, {"gpu-id": 0, "compute-hw": 1}
            )
            self._pipeline.add(
                "capsfilter", caps_name, {"caps": "video/x-raw(memory:NVMM), format=I420"}
            )
            self._pipeline.add("nvjpegenc", encoder_name)
            self._pipeline.add("appsink", sink_name, {"emit-signals": True, "sync": False}).attach(
                sink_name,
                self._handle.make_jpeg_retriever(camera_id, self),
                tips="new-sample",
            )
            self._pipeline.link((demux, queue_name), (f"src_{index}", "sink"))
            self._pipeline.link(
                queue_name,
                valve_name,
                convert_name,
                osd_name,
                post_osd_convert,
                caps_name,
                encoder_name,
                sink_name,
            )
        return fork

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
        try:
            self._publish_frame(frame_meta, camera_id)
        except Exception:  # noqa: BLE001 - an SDK probe must never raise
            self._record_probe_failure(camera_id)

    def _record_probe_failure(self, camera_id: str) -> None:
        failures = self._probe_failures.get(camera_id, 0) + 1
        self._probe_failures[camera_id] = failures
        if failures == 1:
            LOGGER.exception(
                "dropping frame whose conversion failed inside the SDK probe "
                "camera_id=%s consecutive_failures=%d",
                camera_id,
                failures,
            )
        if failures == _PROBE_CONSECUTIVE_FAILURE_THRESHOLD:
            self._probe_fatal_error = (
                "DeepStream probe conversion failed consecutively "
                f"camera_id={camera_id} failures={failures}"
            )
            LOGGER.error("%s", self._probe_fatal_error)

    def _publish_frame(self, frame_meta: Any, camera_id: str) -> None:
        rows = None
        for tensor_meta in frame_meta.tensor_items:
            layer = tensor_meta.as_tensor_output().get_layers()["output0"]
            rows = rows_from_tensor(layer)
            break
        objects = sum(1 for _ in frame_meta.object_items)
        self._objects_observed[camera_id] = self._objects_observed.get(camera_id, 0) + objects
        if rows is None:
            self._frames_without_pose_tensor[camera_id] = (
                self._frames_without_pose_tensor.get(camera_id, 0) + 1
            )
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
        self._matched_tracks[camera_id] = self._matched_tracks.get(camera_id, 0) + len(
            metadata.frame.association.track_ids
        )
        self._probe_failures.pop(camera_id, None)
        if sequence % _PERCEPTION_HEARTBEAT_FRAMES == 0:
            # Operator-visible in the message itself: this is the one line that
            # distinguishes an empty room from a perception path that is dead.
            LOGGER.info(
                "perception heartbeat camera_id=%s frames=%d objects=%d matched_tracks=%d "
                "frames_without_pose_tensor=%d",
                camera_id,
                sequence,
                self._objects_observed[camera_id],
                self._matched_tracks[camera_id],
                self._frames_without_pose_tensor.get(camera_id, 0),
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


__all__ = [
    "DeepStreamFlowStopTimeout",
    "DeepStreamMediaPlane",
    "DeepStreamMediaPlaneConfig",
    "FlowFactory",
    "_FlowHandle",
]
