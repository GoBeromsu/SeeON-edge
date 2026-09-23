"""Worker -> Backend execution-record (diagnostics) wire contract.

One JSON body per POST to ``/api/v1/relay/execution-records``. The worker
builds it from its bounded in-memory lanes; the Backend validates it into the
``diagnostics`` feature's store types. This module is the single source of the
field names and value rules on both sides, so neither side hand-copies them.

Every record is immutable and self-identified: ``record_id`` is the SHA-256 of
its canonical JSON without the id itself, and ``batch_id`` is the SHA-256 of
the sorted record ids plus the gap ranges. Both ids are what make retries
idempotent. Ids are computed here so a worker cannot accidentally send two
different bodies under one id.

Nothing here talks HTTP. ``shared.events`` may import only ``contracts`` and
itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

RELAY_EXECUTION_RECORDS_PATH: Final = "/relay/execution-records"
MAX_EXECUTION_RECORD_BODY_BYTES: Final = 1024 * 1024

RECORD_KINDS: Final = frozenset(
    {
        "sdk.frame",
        "policy.consume",
        "model.score",
        "policy.decision",
        "event.delivery",
        "backend.acceptance",
    }
)
TIME_QUALITIES: Final = frozenset({"monotonic", "wall", "pts", "unknown"})

# Record kinds that are produced by a process-scoped observer (the durable
# delivery-queue drainer) rather than by a camera stream. Their rows carry the
# observing boot id and PROCESS_SCOPE for source_generation and stream_epoch.
# This is declared vocabulary: the Backend keys those rows into a per-boot
# process-scope segment, and the join to the originating stream happens through
# causal_unit_id (the edge_event_id), which the event.delivery record shares.
# It is not a fallback for a missing stream identity on any other kind.
PROCESS_SCOPED_KINDS: Final = frozenset({"backend.acceptance"})
PROCESS_SCOPE: Final = 0

# event.delivery has two producers with different scopes and is deliberately
# NOT in PROCESS_SCOPED_KINDS: the Flow evidence binding records queue admission
# with the real frame identity (stream-scoped), while the durable-queue drainer
# records each attempt outcome with the observing boot and PROCESS_SCOPE
# (process-scoped, same convention as backend.acceptance). Both share
# causal_unit_id == edge_event_id. A consumer must not assume one scope for
# this kind.
STORAGE_STATES: Final = frozenset({"committed", "STORAGE_UNAVAILABLE"})

_JSON_KW: Final = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False}
_IDENTITY_MAX: Final = 128


class ExecutionRecordContractError(ValueError):
    """The body violates the wire contract; the Backend answers 422."""


def canonical_json(value: object) -> str:
    return json.dumps(value, **_JSON_KW)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _identity(value: object, what: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _IDENTITY_MAX or "\x00" in value:
        raise ExecutionRecordContractError(f"invalid {what}")
    return value


def _non_negative(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutionRecordContractError(f"invalid {what}")
    return value


def _optional_non_negative(value: object, what: str) -> int | None:
    return None if value is None else _non_negative(value, what)


def _sha256_hex(value: object, what: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ExecutionRecordContractError(f"invalid {what}")
    return value


@dataclass(frozen=True, slots=True)
class WireProvenance:
    """Execution identity every record in the batch points at."""

    worker_build_revision: str
    worker_image_digest: str
    model_digest: str
    calibration_digest: str
    preprocessing_identity: str
    config_digest: str
    policy_identity: str

    def __post_init__(self) -> None:
        for name in self.__slots__:
            _identity(getattr(self, name), name)

    def to_json(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_json(cls, value: object) -> WireProvenance:
        if not isinstance(value, Mapping):
            raise ExecutionRecordContractError("provenance must be an object")
        try:
            return cls(**{name: value[name] for name in cls.__slots__})
        except KeyError as error:
            raise ExecutionRecordContractError(f"provenance missing {error.args[0]}") from None


@dataclass(frozen=True, slots=True)
class WireRecord:
    """One immutable producer observation.

    ``record_id`` is derived, never supplied: it is the SHA-256 of the canonical
    JSON of every other field. ``payload`` must already be JSON-serializable and
    bounded by the producer; the Backend rejects oversize payloads per its
    retention budget, it does not truncate them.
    """

    record_kind: str
    camera_id: str
    worker_boot_id: str
    source_generation: int
    stream_epoch: int
    producer: str
    producer_sequence: int
    observed_at_ns: int
    time_quality: str
    causal_unit_id: str
    outcome: str
    payload: Mapping[str, object]
    frame_seq: int | None = None
    source_pts_ns: int | None = None
    parent_record_id: str | None = None
    reason: str | None = None
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.record_kind not in RECORD_KINDS:
            raise ExecutionRecordContractError("invalid record_kind")
        if self.time_quality not in TIME_QUALITIES:
            raise ExecutionRecordContractError("invalid time_quality")
        if self.record_kind in PROCESS_SCOPED_KINDS and (
            self.source_generation != PROCESS_SCOPE or self.stream_epoch != PROCESS_SCOPE
        ):
            raise ExecutionRecordContractError(
                f"{self.record_kind} is process-scoped and must use PROCESS_SCOPE"
            )
        for name in ("camera_id", "worker_boot_id", "producer", "causal_unit_id", "outcome"):
            _identity(getattr(self, name), name)
        for name in ("source_generation", "stream_epoch", "producer_sequence", "observed_at_ns"):
            _non_negative(getattr(self, name), name)
        _optional_non_negative(self.frame_seq, "frame_seq")
        if self.source_pts_ns is not None and (
            isinstance(self.source_pts_ns, bool) or not isinstance(self.source_pts_ns, int)
        ):
            raise ExecutionRecordContractError("invalid source_pts_ns")
        if self.parent_record_id is not None:
            _sha256_hex(self.parent_record_id, "parent_record_id")
        if self.reason is not None:
            _identity(self.reason, "reason")
        if not isinstance(self.payload, Mapping):
            raise ExecutionRecordContractError("payload must be an object")
        body = self._body()
        try:
            text = canonical_json(body)
        except (TypeError, ValueError) as error:
            raise ExecutionRecordContractError("payload is not JSON-serializable") from error
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "record_id", _sha256(text))

    def _body(self) -> dict[str, object]:
        return {
            "record_kind": self.record_kind,
            "camera_id": self.camera_id,
            "worker_boot_id": self.worker_boot_id,
            "source_generation": self.source_generation,
            "stream_epoch": self.stream_epoch,
            "producer": self.producer,
            "producer_sequence": self.producer_sequence,
            "observed_at_ns": self.observed_at_ns,
            "time_quality": self.time_quality,
            "causal_unit_id": self.causal_unit_id,
            "outcome": self.outcome,
            "payload": dict(self.payload),
            "frame_seq": self.frame_seq,
            "source_pts_ns": self.source_pts_ns,
            "parent_record_id": self.parent_record_id,
            "reason": self.reason,
        }

    def to_json(self) -> dict[str, object]:
        body = self._body()
        body["record_id"] = self.record_id
        return body

    @classmethod
    def from_json(cls, value: object) -> WireRecord:
        if not isinstance(value, Mapping):
            raise ExecutionRecordContractError("record must be an object")
        supplied = value.get("record_id")
        fields = {k: v for k, v in value.items() if k != "record_id"}
        try:
            record = cls(**fields)
        except TypeError as error:
            raise ExecutionRecordContractError(f"record fields invalid: {error}") from None
        if supplied is not None and supplied != record.record_id:
            raise ExecutionRecordContractError("record_id does not match record content")
        return record


@dataclass(frozen=True, slots=True)
class WireGap:
    """A known drop the worker itself observed (lane overflow, export failure)."""

    producer: str
    from_sequence: int
    to_sequence: int
    from_ns: int
    to_ns: int
    record_count: int
    cause: str

    def __post_init__(self) -> None:
        _identity(self.producer, "producer")
        _identity(self.cause, "cause")
        for name in ("from_sequence", "to_sequence", "from_ns", "to_ns", "record_count"):
            _non_negative(getattr(self, name), name)
        if self.to_sequence < self.from_sequence or self.to_ns < self.from_ns:
            raise ExecutionRecordContractError("invalid gap range")

    def to_json(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_json(cls, value: object) -> WireGap:
        if not isinstance(value, Mapping):
            raise ExecutionRecordContractError("gap must be an object")
        try:
            return cls(**{name: value[name] for name in cls.__slots__})
        except KeyError as error:
            raise ExecutionRecordContractError(f"gap missing {error.args[0]}") from None


@dataclass(frozen=True, slots=True)
class WireBatch:
    """One POST body. ``batch_id`` is derived from the record ids and gaps."""

    camera_id: str
    worker_boot_id: str
    provenance: WireProvenance
    records: tuple[WireRecord, ...]
    gaps: tuple[WireGap, ...] = ()
    batch_id: str = field(init=False)

    def __post_init__(self) -> None:
        _identity(self.camera_id, "camera_id")
        _identity(self.worker_boot_id, "worker_boot_id")
        records = tuple(self.records)
        gaps = tuple(self.gaps)
        if not records and not gaps:
            raise ExecutionRecordContractError("batch has no records and no gaps")
        for record in records:
            if record.camera_id != self.camera_id or record.worker_boot_id != self.worker_boot_id:
                raise ExecutionRecordContractError("record camera/boot does not match batch")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "gaps", gaps)
        identity = {
            "camera_id": self.camera_id,
            "worker_boot_id": self.worker_boot_id,
            "provenance": self.provenance.to_json(),
            "record_ids": sorted(record.record_id for record in records),
            "gaps": [gap.to_json() for gap in gaps],
        }
        object.__setattr__(self, "batch_id", _sha256(canonical_json(identity)))

    def to_json(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "camera_id": self.camera_id,
            "worker_boot_id": self.worker_boot_id,
            "provenance": self.provenance.to_json(),
            "records": [record.to_json() for record in self.records],
            "gaps": [gap.to_json() for gap in self.gaps],
        }

    def encode(self) -> bytes:
        return canonical_json(self.to_json()).encode()

    @classmethod
    def from_json(cls, value: object) -> WireBatch:
        if not isinstance(value, Mapping):
            raise ExecutionRecordContractError("batch must be an object")
        records = value.get("records", ())
        gaps = value.get("gaps", ())
        if not isinstance(records, Sequence) or isinstance(records, str | bytes):
            raise ExecutionRecordContractError("records must be an array")
        if not isinstance(gaps, Sequence) or isinstance(gaps, str | bytes):
            raise ExecutionRecordContractError("gaps must be an array")
        batch = cls(
            camera_id=value.get("camera_id"),  # type: ignore[arg-type]
            worker_boot_id=value.get("worker_boot_id"),  # type: ignore[arg-type]
            provenance=WireProvenance.from_json(value.get("provenance")),
            records=tuple(WireRecord.from_json(item) for item in records),
            gaps=tuple(WireGap.from_json(item) for item in gaps),
        )
        supplied = value.get("batch_id")
        if supplied is not None and supplied != batch.batch_id:
            raise ExecutionRecordContractError("batch_id does not match batch content")
        return batch


@dataclass(frozen=True, slots=True)
class WireBatchReceipt:
    """Backend answer. ``committed`` means the diagnostics commit succeeded.

    It is never a Hub acceptance and never an event receipt; a worker must not
    treat it as either.
    """

    batch_id: str
    accepted: int
    duplicates: int
    rejected: tuple[tuple[str, str], ...]
    storage_state: str
    committed_at_ns: int

    def __post_init__(self) -> None:
        _sha256_hex(self.batch_id, "batch_id")
        if self.storage_state not in STORAGE_STATES:
            raise ExecutionRecordContractError("invalid storage_state")
        for name in ("accepted", "duplicates", "committed_at_ns"):
            _non_negative(getattr(self, name), name)
        rejected = tuple((str(r), str(why)) for r, why in self.rejected)
        object.__setattr__(self, "rejected", rejected)

    def to_json(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "rejected": [list(item) for item in self.rejected],
            "storage_state": self.storage_state,
            "committed_at_ns": self.committed_at_ns,
        }

    @classmethod
    def from_json(cls, value: object) -> WireBatchReceipt:
        if not isinstance(value, Mapping):
            raise ExecutionRecordContractError("receipt must be an object")
        rejected = value.get("rejected", ())
        if not isinstance(rejected, Sequence) or isinstance(rejected, str | bytes):
            raise ExecutionRecordContractError("rejected must be an array")
        try:
            return cls(
                batch_id=value["batch_id"],
                accepted=value["accepted"],
                duplicates=value["duplicates"],
                rejected=tuple(tuple(item) for item in rejected),  # type: ignore[arg-type]
                storage_state=value["storage_state"],
                committed_at_ns=value["committed_at_ns"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ExecutionRecordContractError(f"receipt invalid: {error}") from None


__all__ = [
    "MAX_EXECUTION_RECORD_BODY_BYTES",
    "PROCESS_SCOPE",
    "PROCESS_SCOPED_KINDS",
    "RECORD_KINDS",
    "RELAY_EXECUTION_RECORDS_PATH",
    "STORAGE_STATES",
    "TIME_QUALITIES",
    "ExecutionRecordContractError",
    "WireBatch",
    "WireBatchReceipt",
    "WireGap",
    "WireProvenance",
    "WireRecord",
    "canonical_json",
]
