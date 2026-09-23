"""WireRecord construction shared by producer payload builders."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Final

from shared.events.execution_records import ExecutionRecordContractError, WireRecord
from worker.interfaces.execution_records import ExecutionRecordSink
from worker.types.metadata import MetadataFrame

LOGGER = logging.getLogger(__name__)


FRAME_UNIT_WINDOW: Final = 30
PRODUCER_SDK = "sdk"
PRODUCER_MODEL = "model"
PRODUCER_POLICY = "policy"
PRODUCER_EVENT = "event"
PRODUCER_BACKEND = "backend"


def try_emit(sink: ExecutionRecordSink | None, record: WireRecord | None) -> bool:
    if sink is None or record is None:
        return False
    try:
        return bool(sink.try_emit(record))
    except Exception:  # noqa: BLE001 - producers must never raise
        LOGGER.warning(
            "execution-record sink.try_emit failed camera_id=%s record_kind=%s",
            record.camera_id,
            record.record_kind,
        )
        return False


def frame_causal_unit_id(camera_id: str, worker_boot_id: str, stream_epoch: int, seq: int) -> str:
    """Pre-Gate-R placeholder: bucket frames by the deployed 30-frame window."""
    return f"{camera_id}:{worker_boot_id}:{stream_epoch}:frame:{seq // FRAME_UNIT_WINDOW}"


NO_TRACK: Final = "no-track"
NO_GENERATION: Final = "no-generation"


def fall_causal_unit_id(
    camera_id: str,
    worker_boot_id: str,
    stream_epoch: int,
    track_id: int | None,
    generation: int | None,
) -> str:
    """Logical unit of one fall decision: camera, boot, epoch, track, generation.

    A snapshot with no track (window-gated, outside-detection-window) or a
    track the classifier has not yet given a generation carries explicit
    NO_TRACK / NO_GENERATION tokens. 0 is a real NVDCF track id and the real
    first generation, so coercing absence to 0 would alias those records onto
    a live unit and hide the absence in the join key. Pre-Gate-R placeholder
    membership rule; see worker/pipeline/diagnostics/AGENTS.md.
    """
    track = NO_TRACK if track_id is None else str(track_id)
    gen = NO_GENERATION if generation is None else str(generation)
    return f"{camera_id}:{worker_boot_id}:{stream_epoch}:{track}:{gen}"


def observed_time(metadata: MetadataFrame, observed_at_ns: int | None) -> tuple[str, int]:
    if observed_at_ns is not None:
        return "monotonic", observed_at_ns
    pts = metadata.identity.source_pts
    if pts is not None:
        return "pts", pts
    return "monotonic", time.monotonic_ns()


def monotonic_or(observed_at_ns: int | None) -> int:
    return time.monotonic_ns() if observed_at_ns is None else observed_at_ns


def make_record(
    *,
    record_kind: str,
    camera_id: str,
    worker_boot_id: str,
    source_generation: int,
    stream_epoch: int,
    producer: str,
    observed_at_ns: int,
    time_quality: str,
    causal_unit_id: str,
    outcome: str,
    payload: Mapping[str, object],
    frame_seq: int | None = None,
    source_pts_ns: int | None = None,
) -> WireRecord | None:
    try:
        return WireRecord(
            record_kind=record_kind,
            camera_id=camera_id,
            worker_boot_id=worker_boot_id,
            source_generation=source_generation,
            stream_epoch=stream_epoch,
            producer=producer,
            producer_sequence=0,
            observed_at_ns=observed_at_ns,
            time_quality=time_quality,
            causal_unit_id=causal_unit_id,
            outcome=outcome,
            payload=payload,
            frame_seq=frame_seq,
            source_pts_ns=source_pts_ns,
        )
    except ExecutionRecordContractError as error:
        LOGGER.warning(
            "execution-record contract rejected camera_id=%s record_kind=%s %s",
            camera_id,
            record_kind,
            error,
        )
        return None


__all__ = [
    "FRAME_UNIT_WINDOW",
    "NO_GENERATION",
    "NO_TRACK",
    "PRODUCER_BACKEND",
    "PRODUCER_EVENT",
    "PRODUCER_MODEL",
    "PRODUCER_POLICY",
    "PRODUCER_SDK",
    "fall_causal_unit_id",
    "frame_causal_unit_id",
    "make_record",
    "monotonic_or",
    "observed_time",
    "try_emit",
]
