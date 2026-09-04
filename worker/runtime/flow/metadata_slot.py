"""Capacity-one mailbox for Flow perception metadata."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias, final

from worker.types.metadata import MetadataCounters, MetadataFrame, SourceBinding

CounterName: TypeAlias = Literal[
    "accepted",
    "overwritten",
    "late",
    "unknown_source",
    "generation_mismatch",
    "epoch_mismatch",
    "boot_mismatch",
    "child_mismatch",
    "transform_mismatch",
    "malformed",
    "pull_failures",
]


@dataclass(frozen=True, slots=True)
class AcceptanceToken:
    binding: SourceBinding
    native_publish_sequence: int


def _increment(counters: MetadataCounters, name: CounterName) -> MetadataCounters:
    return replace(counters, **{name: getattr(counters, name) + 1})


def _matches(metadata: MetadataFrame, binding: SourceBinding) -> CounterName | None:
    identity = metadata.identity
    if identity.worker_boot_id != binding.worker_boot_id:
        return "boot_mismatch"
    if str(metadata.child_instance_id) != binding.child_instance_id:
        return "child_mismatch"
    if metadata.source_generation != binding.source_generation:
        return "generation_mismatch"
    if identity.stream_epoch != binding.stream_epoch:
        return "epoch_mismatch"
    if metadata.transform_id != binding.transform_id:
        return "transform_mismatch"
    return None


@final
class LatestMetadataSlot:
    """Capacity-one metadata and exact accepted-frame conditions per camera."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._expected: dict[str, SourceBinding] = {}
        self._latest: dict[str, MetadataFrame] = {}
        self._high_water: dict[tuple[str, int, int], tuple[int, int, int]] = {}
        self._counters = MetadataCounters()

    def register_source(self, binding: SourceBinding) -> AcceptanceToken:
        with self._condition:
            self._expected[binding.camera_id] = binding
            _ = self._latest.pop(binding.camera_id, None)
            self._high_water = {
                key: high_water
                for key, high_water in self._high_water.items()
                if key[0] != binding.camera_id
            }
            self._condition.notify_all()
            return AcceptanceToken(binding, 0)

    def remove_source(self, camera_id: str) -> None:
        with self._condition:
            _ = self._expected.pop(camera_id, None)
            _ = self._latest.pop(camera_id, None)
            self._high_water = {
                binding: high_water
                for binding, high_water in self._high_water.items()
                if binding[0] != camera_id
            }
            self._condition.notify_all()

    def expected_binding(self, camera_id: str) -> SourceBinding | None:
        with self._lock:
            return self._expected.get(camera_id)

    def subscribe(self, binding: SourceBinding) -> AcceptanceToken:
        with self._lock:
            current = self._latest.get(binding.camera_id)
            publish_sequence = 0 if current is None else current.native_publish_sequence
            return AcceptanceToken(binding, publish_sequence)

    def wait_accepted(self, token: AcceptanceToken, *, timeout_sec: float) -> MetadataFrame:
        def accepted() -> bool:
            current = self._latest.get(token.binding.camera_id)
            return (
                current is not None
                and _matches(current, token.binding) is None
                and current.native_publish_sequence > token.native_publish_sequence
            )

        with self._condition:
            if not self._condition.wait_for(accepted, timeout=timeout_sec):
                raise TimeoutError("metadata binding deadline elapsed")
            return self._latest[token.binding.camera_id]

    def publish(self, metadata: MetadataFrame) -> bool:
        camera_id = metadata.identity.camera_id
        if camera_id == "":
            self.mark_malformed()
            return False
        with self._condition:
            expected = self._expected.get(camera_id)
            counters = self._counters
            if expected is None:
                self._counters = _increment(counters, "unknown_source")
                return False
            mismatch = _matches(metadata, expected)
            if mismatch is not None:
                self._counters = _increment(counters, mismatch)
                return False
            identity = (
                metadata.identity.source_pts or 0,
                metadata.identity.seq,
                metadata.native_publish_sequence,
            )
            binding_key = (camera_id, metadata.source_generation, metadata.identity.stream_epoch)
            high_water = self._high_water.get(binding_key)
            if high_water is not None and any(
                current <= previous for current, previous in zip(identity, high_water, strict=True)
            ):
                self._counters = _increment(counters, "late")
                return False
            if camera_id in self._latest:
                counters = _increment(counters, "overwritten")
            self._latest[camera_id] = metadata
            self._high_water[binding_key] = identity
            self._counters = _increment(counters, "accepted")
            self._condition.notify_all()
            return True

    def peek(self, camera_id: str) -> MetadataFrame | None:
        with self._lock:
            return self._latest.get(camera_id)

    def take(self, camera_id: str) -> MetadataFrame | None:
        with self._lock:
            return self._latest.pop(camera_id, None)

    def mark_malformed(self) -> None:
        with self._lock:
            self._counters = _increment(self._counters, "malformed")

    def mark_pull_failure(self) -> None:
        with self._lock:
            self._counters = _increment(self._counters, "pull_failures")

    def counters(self) -> MetadataCounters:
        with self._lock:
            return self._counters


__all__ = ["AcceptanceToken", "LatestMetadataSlot"]
