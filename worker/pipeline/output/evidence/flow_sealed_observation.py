"""Typed sealed Flow observations and their shared sidecar JSON schema."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode
from worker.pipeline.output.evidence.flow_sealed_json import (
    event_payload,
    finite_number,
    reject_constant,
    text_field,
    unique_object,
)
from worker.pipeline.output.evidence.smart_record_actor import (
    ClipBoundary,
    ClipContributor,
    ClipSealed,
)
from worker.types import BusinessEvent

_I32_RANGE = range(-(1 << 31), 1 << 31)
_I64_RANGE = range(-(1 << 63), 1 << 63)
_U64_RANGE = range(1 << 64)
_UNAVAILABLE_KEYS = frozenset(
    (
        "clip_id",
        "camera_id",
        "duration_ms",
        "boundary",
        "contributors",
        "events",
        "native_result",
        "contains_video",
        "path",
    )
)


class FlowSealedMalformedError(ValueError):
    """Sidecar bytes are not a valid sealed Flow observation."""


@dataclass(frozen=True, slots=True)
class ClipSealedUnavailable:
    """A native recording that completed without a publishable ready clip.

    ``path`` is the locator the media plane actually reported, or ``None``
    when none was reported. Duration is the actual unsigned value, zero
    included; nothing here is a fabricated media fact.
    """

    clip_id: str
    duration_ms: int
    contributors: tuple[ClipContributor, ...]
    boundary: ClipBoundary
    native_result: int
    contains_video: bool
    path: str | None = None

    def __post_init__(self) -> None:
        if not valid_clip_id(self.clip_id):
            raise ValueError("unavailable observation clip id is invalid")
        if self.path is not None and (type(self.path) is not str or not self.path):
            raise ValueError("unavailable observation path must be a nonempty reported locator")
        if type(self.duration_ms) is not int or self.duration_ms not in _U64_RANGE:
            raise ValueError("unavailable observation duration must be an unsigned 64-bit integer")
        if type(self.native_result) is not int or self.native_result not in _I32_RANGE:
            raise ValueError("unavailable observation native result must be a 32-bit integer")
        if type(self.contains_video) is not bool:
            raise ValueError("unavailable observation contains_video must be a boolean")
        if self.native_result == 0 and self.contains_video:
            raise ValueError("a successful native result with video is not unavailable")
        if type(self.boundary) is not str:
            raise ValueError("unavailable observation boundary must be a string")
        if not self.contributors or not all(
            isinstance(item, ClipContributor) for item in self.contributors
        ):
            raise ValueError("unavailable observation requires contributors")


FlowSealedObservation = ClipSealed | ClipSealedUnavailable


def unavailable_reason(observation: ClipSealedUnavailable) -> EvidenceReasonCode:
    """Map the native result onto the existing outbound reason vocabulary."""
    if observation.native_result != 0:
        return EvidenceReasonCode.ENCODER_FAILED
    return EvidenceReasonCode.NO_FRAMES


def valid_clip_id(clip_id: object) -> bool:
    return (
        type(clip_id) is str
        and bool(clip_id)
        and not clip_id.startswith(".")
        and "/" not in clip_id
        and "\0" not in clip_id
    )


def encode_sidecar(
    observation: FlowSealedObservation, events: Mapping[str, BusinessEvent]
) -> bytes:
    """Render canonical sidecar bytes; READY bytes are the established format."""
    if not valid_clip_id(observation.clip_id):
        raise ValueError("sealed Flow clip id is invalid")
    contributors = tuple(sorted(observation.contributors, key=lambda item: item.detected_at))
    missing = [item.event_ref for item in contributors if item.event_ref not in events]
    if missing:
        raise ValueError(f"sealed Flow clip has unknown contributor {missing[0]}")
    ordered = [events[item.event_ref] for item in contributors]
    camera_id = _attributed_camera(contributors, ordered)
    if isinstance(observation, ClipSealed):
        payload: dict[str, object] = {
            "clip_id": observation.clip_id,
            "path": observation.path,
            "duration_ms": observation.duration_ms,
            "camera_id": camera_id,
            "boundary": observation.boundary,
            "contributors": [asdict(item) for item in contributors],
            "events": [event_payload(event, exact=False) for event in ordered],
        }
    else:
        if not camera_id.strip():
            raise ValueError("unavailable observation has no attributed camera")
        fields: dict[str, object] = {
            "clip_id": observation.clip_id,
            "camera_id": camera_id,
            "duration_ms": observation.duration_ms,
            "boundary": observation.boundary,
            "contributors": [asdict(item) for item in contributors],
            "events": [event_payload(event, exact=True) for event in ordered],
            "native_result": observation.native_result,
            "contains_video": observation.contains_video,
        }
        if observation.path is not None:
            fields["path"] = observation.path
        payload = {"unavailable": fields}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode_sidecar(
    data: bytes, file_name: str
) -> tuple[FlowSealedObservation, dict[str, BusinessEvent], str]:
    """Strictly read one sidecar into its observation, events and camera."""
    try:
        document = json.loads(
            data.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant
        )
        observation, camera_id, rows = _observation(document)
        events = _events(rows, observation, camera_id)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise FlowSealedMalformedError(f"sealed Flow sidecar is malformed: {file_name}") from exc
    if file_name != f"{observation.clip_id}.json":
        raise FlowSealedMalformedError(f"sidecar name does not match its clip id: {file_name}")
    return observation, events, camera_id


def _observation(document: object) -> tuple[FlowSealedObservation, str, object]:
    if not isinstance(document, dict):
        raise TypeError("sidecar is not an object")
    if "unavailable" in document:
        fields = document["unavailable"]
        if len(document) != 1 or not isinstance(fields, dict) or set(fields) - _UNAVAILABLE_KEYS:
            raise ValueError("unavailable envelope has unknown fields")
        camera_id = text_field(fields, "camera_id")
        if not camera_id.strip():
            raise ValueError("unavailable observation has no attributed camera")
        path = fields.get("path")
        if "path" in fields and type(path) is not str:
            raise ValueError("unavailable path is not a string")
        unavailable = ClipSealedUnavailable(
            clip_id=text_field(fields, "clip_id"),
            duration_ms=fields["duration_ms"],
            contributors=_contributors(fields["contributors"]),
            boundary=text_field(fields, "boundary"),  # type: ignore[arg-type]
            native_result=fields["native_result"],
            contains_video=fields["contains_video"],
            path=path,
        )
        return unavailable, camera_id, fields["events"]
    if "native_result" in document or "contains_video" in document:
        raise ValueError("ready sidecar carries unavailable fields")
    duration_ms = document["duration_ms"]
    if type(duration_ms) is not int or duration_ms not in _I64_RANGE:
        raise ValueError("ready duration is not a signed 64-bit integer")
    sealed = ClipSealed(
        clip_id=text_field(document, "clip_id"),
        path=text_field(document, "path"),
        duration_ms=duration_ms,
        contributors=_contributors(document["contributors"]),
        boundary=text_field(document, "boundary"),  # type: ignore[arg-type]
    )
    if not valid_clip_id(sealed.clip_id):
        raise ValueError("ready clip id is invalid")
    return sealed, text_field(document, "camera_id"), document["events"]


def _contributors(value: object) -> tuple[ClipContributor, ...]:
    if not isinstance(value, list):
        raise TypeError("contributors are not an array")
    return tuple(
        ClipContributor(
            event_ref=text_field(item, "event_ref"), detected_at=text_field(item, "detected_at")
        )
        for item in value
    )


def _events(
    rows: object, observation: FlowSealedObservation, camera_id: str
) -> dict[str, BusinessEvent]:
    references = [item.event_ref for item in observation.contributors]
    if any(not ref.strip() for ref in references) or len(set(references)) != len(references):
        raise ValueError("contributor references are blank or repeated")
    if not isinstance(rows, list):
        raise TypeError("events are not an array")
    if isinstance(observation, ClipSealedUnavailable) and not rows:
        raise ValueError("unavailable observation has no events")
    events: dict[str, BusinessEvent] = {}
    for item in rows:
        event = BusinessEvent(
            domain=text_field(item, "domain"),
            event_type=text_field(item, "event_type"),
            identity=text_field(item, "identity"),
            camera_id=text_field(item, "camera_id"),
            facility_id=text_field(item, "facility_id"),
            time_sec=finite_number(item["time_sec"]),
            probability=None if item["probability"] is None else finite_number(item["probability"]),
        )
        identity = str(event.identity)
        if (
            not identity.strip()
            or not event.camera_id.strip()
            or event.camera_id != camera_id
            or identity not in references
            or identity in events
        ):
            raise ValueError("event attribution contradicts the contributors")
        events[identity] = event
    if len(events) != len(references):
        raise ValueError("a contributor has no event")
    return events


def _attributed_camera(
    contributors: tuple[ClipContributor, ...], events: list[BusinessEvent]
) -> str:
    identities: set[str] = set()
    for contributor, event in zip(contributors, events, strict=True):
        identity = str(event.identity)
        if (
            not contributor.event_ref.strip()
            or not identity.strip()
            or identity != contributor.event_ref
            or identity in identities
            or not event.camera_id.strip()
            or event.camera_id != events[0].camera_id
        ):
            raise ValueError("sealed Flow clip attribution is contradictory")
        identities.add(identity)
    return events[0].camera_id if events else ""


__all__ = [
    "ClipSealedUnavailable",
    "FlowSealedMalformedError",
    "FlowSealedObservation",
    "decode_sidecar",
    "encode_sidecar",
    "unavailable_reason",
    "valid_clip_id",
]
