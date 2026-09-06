"""Bounded policy-input coverage observations for one native camera pump."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from worker.types.metadata import MetadataFrame, SourceBinding

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    worker_boot_id: str
    camera_id: str
    source_generation: int
    stream_epoch: int


@dataclass(frozen=True, slots=True)
class ActualMetadataObservation:
    identity: ObservationIdentity
    seq: int
    source_pts_ns: int | None
    native_publish_sequence: int
    host_time: float


@dataclass(frozen=True, slots=True)
class ObservationGap:
    identity: ObservationIdentity
    last_actual: ActualMetadataObservation | None
    loss_detected_host_time: float


@dataclass(frozen=True, slots=True)
class ObservationRecovery:
    gap: ObservationGap
    next_actual: ActualMetadataObservation
    host_observation_duration: float | None
    source_duration_ns: int | None
    native_publish_sequence_gap: int | None


class ObservationCoverage:
    """Track only the latest actual input and at most one open coverage gap."""

    def __init__(
        self,
        binding: SourceBinding,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._expected_identity = _binding_identity(binding)
        self._monotonic = monotonic
        self._last_actual: ActualMetadataObservation | None = None
        self._open_gap: ObservationGap | None = None

    @property
    def last_actual(self) -> ActualMetadataObservation | None:
        return self._last_actual

    @property
    def open_gap(self) -> ObservationGap | None:
        return self._open_gap

    def rebind(self, binding: SourceBinding) -> None:
        self._expected_identity = _binding_identity(binding)

    def observe(
        self,
        metadata: MetadataFrame,
        *,
        host_time: float | None = None,
    ) -> ObservationRecovery | None:
        observed_at = self._monotonic() if host_time is None else host_time
        frame = metadata.frame
        actual = ActualMetadataObservation(
            identity=ObservationIdentity(
                worker_boot_id=frame.identity.worker_boot_id,
                camera_id=frame.identity.camera_id,
                source_generation=metadata.source_generation,
                stream_epoch=frame.identity.stream_epoch,
            ),
            seq=frame.identity.seq,
            source_pts_ns=frame.identity.source_pts,
            native_publish_sequence=metadata.native_publish_sequence,
            host_time=observed_at,
        )
        if actual.identity != self._expected_identity:
            raise ValueError(
                "policy input observation identity does not match the expected source binding"
            )
        previous = self._last_actual
        publication_gap = _native_publish_sequence_gap(previous, actual)
        if previous is not None and publication_gap is not None and publication_gap > 0:
            LOGGER.debug(
                "policy input native publication sequence gap observed: "
                "coverage_scope=policy_input camera_id=%s worker_boot_id=%s "
                "source_generation=%d stream_epoch=%d previous_frame_seq=%d "
                "next_frame_seq=%d native_publish_sequence_gap=%d "
                "gap_cause=unknown capture_loss=unknown person_count_claim=unknown",
                actual.identity.camera_id,
                actual.identity.worker_boot_id,
                actual.identity.source_generation,
                actual.identity.stream_epoch,
                previous.seq,
                actual.seq,
                publication_gap,
            )
        gap = self._open_gap
        recovery = None if gap is None else _close_gap(gap, actual)
        self._last_actual = actual
        self._open_gap = None
        if recovery is not None:
            LOGGER.info(
                "policy input coverage gap closed: "
                "coverage_scope=policy_input gap_state=closed camera_id=%s "
                "gap_expected_worker_boot_id=%s gap_expected_camera_id=%s "
                "gap_expected_source_generation=%d gap_expected_stream_epoch=%d "
                "last_actual_worker_boot_id=%s last_actual_camera_id=%s "
                "last_actual_source_generation=%s last_actual_stream_epoch=%s "
                "last_actual_frame_seq=%s last_actual_source_pts_ns=%s "
                "next_actual_worker_boot_id=%s next_actual_camera_id=%s "
                "next_actual_source_generation=%d next_actual_stream_epoch=%d "
                "next_actual_frame_seq=%d next_actual_source_pts_ns=%s "
                "last_actual_frame_host_time=%s loss_detected_host_time=%.9f "
                "next_actual_frame_host_time=%.9f host_observation_duration=%s "
                "source_duration_ns=%s native_publish_sequence_gap=%s "
                "gap_cause=unknown capture_loss=unknown "
                "person_count_claim=unknown fall_state_claim=unknown",
                actual.identity.camera_id,
                gap.identity.worker_boot_id,
                gap.identity.camera_id,
                gap.identity.source_generation,
                gap.identity.stream_epoch,
                _optional_text(
                    None if gap.last_actual is None else gap.last_actual.identity.worker_boot_id
                ),
                _optional_text(
                    None if gap.last_actual is None else gap.last_actual.identity.camera_id
                ),
                _optional_number(
                    None if gap.last_actual is None else gap.last_actual.identity.source_generation
                ),
                _optional_number(
                    None if gap.last_actual is None else gap.last_actual.identity.stream_epoch
                ),
                _optional_number(None if gap.last_actual is None else gap.last_actual.seq),
                _optional_number(
                    None if gap.last_actual is None else gap.last_actual.source_pts_ns
                ),
                actual.identity.worker_boot_id,
                actual.identity.camera_id,
                actual.identity.source_generation,
                actual.identity.stream_epoch,
                actual.seq,
                _optional_number(actual.source_pts_ns),
                _optional_float(gap.last_actual),
                gap.loss_detected_host_time,
                actual.host_time,
                _optional_number(recovery.host_observation_duration),
                _optional_number(recovery.source_duration_ns),
                _optional_number(recovery.native_publish_sequence_gap),
            )
        return recovery

    def detect_gap(self, *, host_time: float | None = None) -> ObservationGap | None:
        if self._open_gap is not None:
            return None
        detected_at = self._monotonic() if host_time is None else host_time
        gap = ObservationGap(
            identity=self._expected_identity,
            last_actual=self._last_actual,
            loss_detected_host_time=detected_at,
        )
        self._open_gap = gap
        LOGGER.warning(
            "policy input coverage gap opened: "
            "coverage_scope=policy_input gap_state=open camera_id=%s "
            "expected_worker_boot_id=%s expected_camera_id=%s "
            "expected_source_generation=%d expected_stream_epoch=%d "
            "last_actual_worker_boot_id=%s last_actual_camera_id=%s "
            "last_actual_source_generation=%s last_actual_stream_epoch=%s "
            "last_actual_frame_seq=%s last_actual_source_pts_ns=%s "
            "last_actual_frame_host_time=%s loss_detected_host_time=%.9f "
            "next_actual_frame_host_time=unknown source_duration_ns=unknown "
            "gap_cause=unknown capture_loss=unknown "
            "person_count_claim=unknown fall_state_claim=unknown",
            gap.identity.camera_id,
            gap.identity.worker_boot_id,
            gap.identity.camera_id,
            gap.identity.source_generation,
            gap.identity.stream_epoch,
            _optional_text(
                None if gap.last_actual is None else gap.last_actual.identity.worker_boot_id
            ),
            _optional_text(None if gap.last_actual is None else gap.last_actual.identity.camera_id),
            _optional_number(
                None if gap.last_actual is None else gap.last_actual.identity.source_generation
            ),
            _optional_number(
                None if gap.last_actual is None else gap.last_actual.identity.stream_epoch
            ),
            _optional_number(None if gap.last_actual is None else gap.last_actual.seq),
            _optional_number(None if gap.last_actual is None else gap.last_actual.source_pts_ns),
            _optional_float(gap.last_actual),
            gap.loss_detected_host_time,
        )
        return gap


def _binding_identity(binding: SourceBinding) -> ObservationIdentity:
    return ObservationIdentity(
        worker_boot_id=binding.worker_boot_id,
        camera_id=binding.camera_id,
        source_generation=binding.source_generation,
        stream_epoch=binding.stream_epoch,
    )


def _close_gap(
    gap: ObservationGap,
    actual: ActualMetadataObservation,
) -> ObservationRecovery:
    previous = gap.last_actual
    same_boot = (
        previous is not None
        and previous.identity.worker_boot_id == actual.identity.worker_boot_id
        and previous.identity.camera_id == actual.identity.camera_id
    )
    same_source_epoch = (
        same_boot
        and previous.identity.source_generation == actual.identity.source_generation
        and previous.identity.stream_epoch == actual.identity.stream_epoch
    )
    source_duration = None
    if (
        same_source_epoch
        and previous.source_pts_ns is not None
        and actual.source_pts_ns is not None
        and actual.source_pts_ns >= previous.source_pts_ns
    ):
        source_duration = actual.source_pts_ns - previous.source_pts_ns
    publication_gap = _native_publish_sequence_gap(previous, actual)
    return ObservationRecovery(
        gap=gap,
        next_actual=actual,
        host_observation_duration=(
            None
            if not same_boot or actual.host_time < previous.host_time
            else actual.host_time - previous.host_time
        ),
        source_duration_ns=source_duration,
        native_publish_sequence_gap=publication_gap,
    )


def _native_publish_sequence_gap(
    previous: ActualMetadataObservation | None,
    actual: ActualMetadataObservation,
) -> int | None:
    if (
        previous is None
        or previous.identity != actual.identity
        or actual.native_publish_sequence <= previous.native_publish_sequence
    ):
        return None
    return max(
        actual.native_publish_sequence - previous.native_publish_sequence - 1,
        0,
    )


def _optional_float(observation: ActualMetadataObservation | None) -> str:
    return "unknown" if observation is None else f"{observation.host_time:.9f}"


def _optional_text(value: str | None) -> str:
    return "unknown" if value is None else value


def _optional_number(value: float | int | None) -> str:
    return "unknown" if value is None else str(value)


__all__ = [
    "ActualMetadataObservation",
    "ObservationCoverage",
    "ObservationGap",
    "ObservationIdentity",
    "ObservationRecovery",
]
