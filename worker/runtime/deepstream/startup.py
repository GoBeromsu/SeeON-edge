"""Bounded control and metadata handshake for a spawned dark child."""

from __future__ import annotations

from dataclasses import dataclass

from worker.native.deepstream.control import (
    ChildControlError,
    ControlIdentity,
    DeepStreamControlClient,
)
from worker.native.deepstream.metadata import LatestMetadataSlot, MetadataReceiver
from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.errors import ChildStartupError
from worker.runtime.deepstream.readiness import wait_for_ready
from worker.runtime.deepstream.source_control import DarkSourceController
from worker.runtime.deepstream.transport import ChildTransport

_TRANSFORM_ID = "seeon-perception-v1"


@dataclass(frozen=True, slots=True)
class ConnectedSession:
    control: DeepStreamControlClient
    receiver: MetadataReceiver
    sources: DarkSourceController


def connect_session(
    config: ChildConfig,
    transport: ChildTransport,
    metadata: LatestMetadataSlot,
) -> ConnectedSession:
    if not wait_for_ready(
        transport.ready_fd,
        transport.process,
        config.startup_timeout_sec,
    ):
        transport.control.close()
        transport.wake.close()
        transport.failures.close()
        raise ChildStartupError("ready_failed", "gpu-0")
    control = DeepStreamControlClient(
        transport.control,
        ControlIdentity(
            config.worker_boot_id,
            config.child_instance_id,
            _TRANSFORM_ID,
        ),
    )
    try:
        control.connect()
        _ = control.status()
        receiver = MetadataReceiver(transport.wake, metadata, control)
        _ = receiver.__enter__()
    except (ChildControlError, OSError) as error:
        control.close()
        transport.wake.close()
        transport.failures.close()
        raise ChildStartupError("handshake_failed", "control") from error
    return ConnectedSession(
        control,
        receiver,
        DarkSourceController(control, metadata, receiver),
    )


__all__ = ["ConnectedSession", "connect_session"]
