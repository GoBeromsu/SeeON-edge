"""Read-only admission for content-addressed model bundles.

This module deliberately has no dependency on ``worker.tools``: provisioning and
runtime admission have opposite authority and lifecycle boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from contracts.model_selection import (
    ContractError,
    ModelSelection,
    canonical_json_bytes,
    parse_model_selection,
    validate_evaluation_receipt_identity,
    validate_field_receipt_identity,
)

_BUNDLE_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")
_IDENTITY_FIELDS: Final = (
    "dataset",
    "evaluation",
    "field",
    "calibration",
    "conformance",
    "class",
    "input",
    "policy",
    "members",
)
_RECEIPT_IDENTITY_FIELDS: Final = ("evaluation", "field")
_BUNDLE_IDENTITY_FIELDS: Final = tuple(
    field
    for field in _IDENTITY_FIELDS
    if field not in _RECEIPT_IDENTITY_FIELDS
)
_REQUIRED_MEMBERS: Final = frozenset(
    {
        "model.pt",
        "arch.json",
        "metadata.yaml",
        "input-contract.json",
        "fall-policy-v2.json",
        "calibration.json",
        "conformance.json",
    }
)
_REQUIRED_RECEIPTS: Final = frozenset({"evaluation-receipt.json", "field-evaluation-receipt.json"})


class ModelBundleAdmissionError(RuntimeError):
    """A model bundle is unsafe or does not exactly satisfy desired state."""


@dataclass(frozen=True, slots=True)
class DesiredModelBundle:
    """The configured desired state; no on-disk value can expand this authority."""

    bundle_sha256: str
    identities: Mapping[str, object]
    selection: ModelSelection | None = None

    def __post_init__(self) -> None:
        if _BUNDLE_RE.fullmatch(self.bundle_sha256) is None:
            raise ModelBundleAdmissionError("desired bundle identity is invalid")
        identities = dict(self.identities)
        if set(identities) != set(_IDENTITY_FIELDS):
            raise ModelBundleAdmissionError(
                "desired identities must contain exactly the required fields"
            )
        _canonical_json(identities)
        object.__setattr__(self, "identities", _freeze(identities))


@dataclass(frozen=True, slots=True)
class ModelBundleProof:
    """Immutable observed and applied facts, suitable for later provenance output."""

    observed: Mapping[str, object]
    applied: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RuntimeModelObservations:
    worker_image_digest: str
    config_revision: str
    restart_revision: str

    def __post_init__(self) -> None:
        for value in (self.worker_image_digest, self.config_revision, self.restart_revision):
            if _BUNDLE_RE.fullmatch(value) is None:
                raise ModelBundleAdmissionError("runtime observation identity is invalid")

    @property
    def identities(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "worker": self.worker_image_digest,
                "config": self.config_revision,
                "restart": self.restart_revision,
            }
        )


def desired_model_bundle_from_selection_document(raw: object) -> DesiredModelBundle:
    """Map the canonical selection parser into admission identities."""
    try:
        selection = parse_model_selection(raw)
    except ContractError as exc:
        raise ModelBundleAdmissionError(str(exc)) from exc
    return DesiredModelBundle(
        selection.model_publication.bundle_sha256,
        {
            "dataset": selection.dataset_publication.payload_digest,
            "evaluation": selection.evaluation_receipt_digest,
            "field": selection.field_evaluation_receipt_digest,
            "calibration": selection.calibration_digest,
            "conformance": selection.conformance_digest,
            "class": selection.output_class_semantics_digest,
            "input": selection.input_observation_schema,
            "policy": selection.policy_digest,
            "members": selection.bundle_members_digest,
        },
        selection,
    )


def runtime_revision_digest(kind: str, integer: int) -> str:
    if (
        not isinstance(kind, str)
        or not kind
        or not isinstance(integer, int)
        or isinstance(integer, bool)
    ):
        raise ModelBundleAdmissionError("runtime revision input is invalid")
    return hashlib.sha256(_canonical_json({"kind": kind, "revision": integer}).encode()).hexdigest()


def admit_model_bundle(
    models_root: Path, desired: DesiredModelBundle, observations: RuntimeModelObservations
) -> ModelBundleProof:
    """Verify one published bundle without mutating it or selecting an alternative.

    Call this before model construction/warmup.  Any failed comparison is fatal.
    """
    _require_directory(models_root, "models root")
    bundles_root = models_root / "bundles"
    _require_directory(bundles_root, "bundles root")
    root = bundles_root / desired.bundle_sha256
    _require_directory(root, "bundle root")
    _require_below(bundles_root, root)
    manifest_path = root / "manifest.json"
    raw = _read_regular(manifest_path, "manifest")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelBundleAdmissionError("bundle manifest is invalid JSON") from exc
    if not isinstance(document, dict) or _canonical_json(document).encode() + b"\n" != raw:
        raise ModelBundleAdmissionError("bundle manifest is not canonical")
    if document.get("schema_version") != 1:
        raise ModelBundleAdmissionError("bundle manifest schema mismatch")
    if document.get("bundle_sha256") != desired.bundle_sha256:
        raise ModelBundleAdmissionError("bundle identity mismatch")
    if (
        desired.selection is not None
        and document.get("runtime_format") != desired.selection.runtime_format
    ):
        raise ModelBundleAdmissionError("bundle runtime format mismatch")
    members = document.get("members")
    receipts = document.get("receipts")
    payload = document.get("payload")
    if (
        not isinstance(members, list)
        or not isinstance(receipts, list)
        or not isinstance(payload, dict)
    ):
        raise ModelBundleAdmissionError("bundle manifest shape mismatch")
    identities = payload.get("identities")
    if not isinstance(identities, dict):
        raise ModelBundleAdmissionError("bundle identities missing")
    for field in _BUNDLE_IDENTITY_FIELDS:
        if identities.get(field) != desired.identities[field]:
            raise ModelBundleAdmissionError(f"{field} identity mismatch")
    if set(identities) != set(_BUNDLE_IDENTITY_FIELDS):
        raise ModelBundleAdmissionError("bundle identities contain unknown fields")
    _validate_bundle_identity(document, desired.bundle_sha256)
    observed_members = _verify_members(root, members)
    observed_receipts = _verify_members(root, receipts)
    receipt_identities = _verify_required_members(root, members, receipts, desired)
    _verify_exact_tree(root, {"manifest.json", *observed_members, *observed_receipts})
    frozen_static = _freeze({**dict(identities), **receipt_identities})
    observed = _freeze(
        {
            "bundle_sha256": desired.bundle_sha256,
            "members": tuple(observed_members),
            "receipts": tuple(observed_receipts),
            "identities": frozen_static,
        }
    )
    applied_identities = {**dict(identities), **receipt_identities}
    for field in _IDENTITY_FIELDS:
        if applied_identities[field] != desired.identities[field]:
            raise ModelBundleAdmissionError(f"{field} identity mismatch")
    applied = _freeze({"bundle_sha256": desired.bundle_sha256, "identities": applied_identities})
    return ModelBundleProof(observed=observed, applied=applied)


def _validate_bundle_identity(document: Mapping[str, object], expected: str) -> None:
    members = document["members"]
    payload = document["payload"]
    if not isinstance(members, list) or not isinstance(payload, dict):
        raise ModelBundleAdmissionError("bundle manifest shape mismatch")
    canonical_members: list[dict[str, object]] = []
    for member in members:
        if not isinstance(member, dict):
            raise ModelBundleAdmissionError("bundle member is invalid")
        try:
            canonical_members.append(
                {"path": member["path"], "sha256": member["sha256"], "size": member["size"]}
            )
        except KeyError as exc:
            raise ModelBundleAdmissionError("bundle member is invalid") from exc
    actual = hashlib.sha256(
        _canonical_json({"members": canonical_members, "payload": payload}).encode()
    ).hexdigest()
    if actual != expected:
        raise ModelBundleAdmissionError("bundle content identity mismatch")


def _verify_members(root: Path, members: list[object]) -> tuple[str, ...]:
    paths: list[str] = []
    for member in members:
        if not isinstance(member, dict):
            raise ModelBundleAdmissionError("bundle member is invalid")
        path = member.get("path")
        size = member.get("size")
        digest = member.get("sha256")
        if (
            not isinstance(path, str)
            or _RELATIVE_RE.fullmatch(path) is None
            or path == "manifest.json"
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or _BUNDLE_RE.fullmatch(digest) is None
            or path in paths
        ):
            raise ModelBundleAdmissionError("bundle member is invalid")
        member_path = root / path
        _require_below(root, member_path)
        content = _read_regular(member_path, f"member {path}")
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ModelBundleAdmissionError(f"member mismatch: {path}")
        paths.append(path)
    if not paths:
        raise ModelBundleAdmissionError("bundle members missing")
    return tuple(paths)


def _verify_required_members(
    root: Path, members: list[object], receipts: list[object], desired: DesiredModelBundle
) -> Mapping[str, str]:
    if desired.selection is None:
        return MappingProxyType({})
    by_path = {member["path"]: member for member in members if isinstance(member, dict)}
    by_receipt = {member["path"]: member for member in receipts if isinstance(member, dict)}
    if set(by_path) != _REQUIRED_MEMBERS or set(by_receipt) != _REQUIRED_RECEIPTS:
        raise ModelBundleAdmissionError("bundle required members mismatch")
    observed: dict[str, str] = {}
    for path, validator in (
        ("evaluation-receipt.json", validate_evaluation_receipt_identity),
        ("field-evaluation-receipt.json", validate_field_receipt_identity),
    ):
        try:
            raw = _read_regular(root / path, path)
            receipt = json.loads(raw)
            if canonical_json_bytes(receipt) + b"\n" != raw:
                raise ModelBundleAdmissionError(f"{path} is not canonical JSON")
            validator(desired.selection, receipt)
            observed["evaluation" if path == "evaluation-receipt.json" else "field"] = (
                hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
            )
        except (ContractError, json.JSONDecodeError) as exc:
            raise ModelBundleAdmissionError(f"{path} is not a valid desired-bound receipt") from exc
    return MappingProxyType(observed)


def _verify_exact_tree(root: Path, expected: set[str]) -> None:
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    found_files: set[str] = set()
    found_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            raise ModelBundleAdmissionError(f"bundle contains unsafe path: {relative}")
        if stat.S_ISREG(info.st_mode):
            found_files.add(relative)
        else:
            found_directories.add(relative)
    if found_files != expected or found_directories != expected_directories:
        raise ModelBundleAdmissionError("bundle tree contains missing or extra filesystem nodes")


def _require_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ModelBundleAdmissionError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ModelBundleAdmissionError(f"{label} is not a regular directory")


def _read_regular(path: Path, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ModelBundleAdmissionError(f"{label} is not a regular file")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1 << 20):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except ModelBundleAdmissionError:
        raise
    except OSError as exc:
        raise ModelBundleAdmissionError(f"{label} is unavailable") from exc


def _require_below(root: Path, path: Path) -> None:
    try:
        resolved_root = root.resolve(strict=True)
        if os.path.commonpath((str(resolved_root), str(path.resolve(strict=False)))) != str(
            resolved_root
        ):
            raise ModelBundleAdmissionError("bundle path escapes its root")
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
                raise ModelBundleAdmissionError("bundle contains symlink path")
    except OSError as exc:
        raise ModelBundleAdmissionError("bundle path is unavailable") from exc


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ModelBundleAdmissionError("bundle manifest contains non-JSON values") from exc


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(member) for key, member in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(member) for member in value)
    return value


__all__ = [
    "DesiredModelBundle",
    "ModelBundleAdmissionError",
    "ModelBundleProof",
    "RuntimeModelObservations",
    "admit_model_bundle",
    "desired_model_bundle_from_selection_document",
    "runtime_revision_digest",
]
