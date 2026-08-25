"""Optional loopback MJPEG server for camera-focused development views."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import HTTPServer
from typing import Final

from worker.domains.fall import FallModelProtocol
from worker.pipeline.output._mjpeg_http import (
    BED_ZONE_FRAME_TIMEOUT_SECONDS,
    BedZoneNotFoundError,
    BedZonePayload,
    BedZoneRecognizer,
    ClipDeletionControl,
    MjpegProbe,
    MjpegProbeError,
    MjpegProbePayload,
    build_http_server,
)
from worker.pipeline.output.live_view import LatestFrameStore

DEV_MJPEG_ENV: Final = "ML_WORKER_DEV_MJPEG"
DEV_MJPEG_HOST_ENV: Final = "ML_WORKER_DEV_MJPEG_HOST"
DEV_MJPEG_PORT_ENV: Final = "ML_WORKER_DEV_MJPEG_PORT"


@dataclass(frozen=True, slots=True)
class MjpegServerConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8090
    probe_token: str | None = None


class MjpegServer:
    """Own the bounded HTTP server and its single daemon serving thread."""

    def __init__(
        self,
        store: LatestFrameStore,
        config: MjpegServerConfig,
        probe: MjpegProbe | None = None,
        bed_zone_recognizer: BedZoneRecognizer | None = None,
        clip_deletion_control: ClipDeletionControl | None = None,
        replay_fall_model: FallModelProtocol | None = None,
        *,
        bed_zone_frame_timeout_s: float = BED_ZONE_FRAME_TIMEOUT_SECONDS,
    ) -> None:
        self.store = store
        self.host = config.host
        self.probe_token = config.probe_token
        self.probe = probe if probe is not None else _unavailable_probe
        self.bed_zone_recognizer = bed_zone_recognizer
        self._server: HTTPServer = build_http_server(
            store,
            host=self.host,
            port=config.port,
            probe_token=self.probe_token,
            probe=self.probe,
            bed_zone_recognizer=self.bed_zone_recognizer,
            clip_deletion_control=clip_deletion_control,
            replay_fall_model=replay_fall_model,
            bed_zone_frame_timeout_s=bed_zone_frame_timeout_s,
        )
        self.port = int(self._server.server_port)
        self._thread: threading.Thread | None = None
        self._is_open = True
        self._lifecycle_lock = threading.Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None or not self._is_open:
                return
            thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
            )
            thread.start()
            self._thread = thread

    def stop(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            is_open = self._is_open
            if is_open:
                self._is_open = False
        if thread is not None:
            self._server.shutdown()
            thread.join(timeout=2)
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None
        if is_open:
            self._server.server_close()


MjpegServerFactory = Callable[
    [
        LatestFrameStore,
        MjpegServerConfig,
        MjpegProbe | None,
        BedZoneRecognizer | None,
        ClipDeletionControl | None,
        FallModelProtocol | None,
    ],
    MjpegServer,
]


def dev_mjpeg_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(DEV_MJPEG_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def dev_mjpeg_host(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(DEV_MJPEG_HOST_ENV, "127.0.0.1").strip() or "127.0.0.1"


def dev_mjpeg_port(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw_port = env.get(DEV_MJPEG_PORT_ENV, "8090").strip() or "8090"
    return int(raw_port)


def dev_mjpeg_config(
    environ: Mapping[str, str] | None = None,
) -> MjpegServerConfig:
    return MjpegServerConfig(
        enabled=dev_mjpeg_enabled(environ),
        host=dev_mjpeg_host(environ),
        port=dev_mjpeg_port(environ),
    )


def start_optional_mjpeg_server(
    store: LatestFrameStore,
    config: MjpegServerConfig | None = None,
    *,
    probe: MjpegProbe | None = None,
    bed_zone_recognizer: BedZoneRecognizer | None = None,
    clip_deletion_control: ClipDeletionControl | None = None,
    replay_fall_model: FallModelProtocol | None = None,
    factory: MjpegServerFactory = MjpegServer,
) -> MjpegServer | None:
    resolved = dev_mjpeg_config() if config is None else config
    if not resolved.enabled:
        return None
    try:
        server = factory(
            store,
            resolved,
            probe,
            bed_zone_recognizer,
            clip_deletion_control,
            replay_fall_model,
        )
    except OSError:
        return None
    try:
        server.start()
    except OSError:
        server.stop()
        return None
    return server


def _unavailable_probe(_rtsp_url: str) -> MjpegProbePayload:
    raise MjpegProbeError("decode")


__all__ = [
    "BedZoneNotFoundError",
    "BedZonePayload",
    "BedZoneRecognizer",
    "ClipDeletionControl",
    "MjpegProbe",
    "MjpegProbeError",
    "MjpegProbePayload",
    "MjpegServer",
    "MjpegServerConfig",
    "dev_mjpeg_config",
    "dev_mjpeg_enabled",
    "dev_mjpeg_host",
    "start_optional_mjpeg_server",
]
