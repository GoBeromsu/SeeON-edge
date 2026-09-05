"""Committed model manifest: pinned sources, artifact list, bundled sidecars.

The manifest is data, not policy: every artifact names the source it comes
from, its path under the models root, its byte size, and its SHA-256. Changing
a weight means changing this file in the same commit, which is what makes a
fetch reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MANIFEST_PATH: Final = Path(__file__).resolve().parent / "manifest.json"
SIDECAR_ROOT: Final = Path(__file__).resolve().parent / "sidecars"
SUPPORTED_SCHEMA_VERSION: Final = 1

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_HEX40_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_RELATIVE_PATH_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")


class ManifestError(ValueError):
    """The committed manifest is malformed; nothing is fetched."""


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    source_locator: str
    ref: str

    def url_for(self, remote_path: str) -> str:
        if self.kind == "huggingface":
            return f"https://huggingface.co/{self.source_locator}/resolve/{self.ref}/{remote_path}"
        if self.kind == "github-release":
            return f"https://github.com/{self.source_locator}/releases/download/{self.ref}/{remote_path}"
        raise ManifestError(f"source {self.name!r} has unsupported kind {self.kind!r}")

    @property
    def wants_hf_token(self) -> bool:
        return self.kind == "huggingface"


@dataclass(frozen=True)
class Artifact:
    path: str
    source: Source
    remote_path: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return self.source.url_for(self.remote_path)


@dataclass(frozen=True)
class Manifest:
    sources: Mapping[str, Source]
    artifacts: tuple[Artifact, ...]
    sidecars: tuple[str, ...]
    bundles: tuple[Bundle, ...] = ()
    published_bundles: tuple[PublishedBundle, ...] = ()


@dataclass(frozen=True)
class Bundle:
    """A content-addressed payload collection plus external receipt descriptors.

    Loader-specific required member names are a runtime admission concern; this
    generic provisioning schema preserves every declared payload artifact.
    """

    sha256: str
    members: tuple[Artifact, ...]
    payload: Mapping[str, object]
    receipts: tuple[Artifact, ...] = ()
    runtime_format: str | None = None

    @property
    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "members": [
                    {"path": member.path, "sha256": member.sha256, "size": member.size}
                    for member in self.members
                ],
                "payload": self.payload,
            }
        )

    @property
    def manifest_bytes(self) -> bytes:
        return (
            canonical_json(
                {
                    "bundle_sha256": self.sha256,
                    "members": [
                        {"path": member.path, "sha256": member.sha256, "size": member.size}
                        for member in self.members
                    ],
                    "payload": self.payload,
                    "receipts": [
                        {"path": receipt.path, "sha256": receipt.sha256, "size": receipt.size}
                        for receipt in self.receipts
                    ],
                    **(
                        {}
                        if self.runtime_format is None
                        else {"runtime_format": self.runtime_format}
                    ),
                    "schema_version": 1,
                }
            )
            + "\n"
        ).encode()


@dataclass(frozen=True)
class PublishedBundle:
    """Published model lineage, independent of the fetch transport."""

    source_locator: str
    revision: str
    bundle_sha256: str


def bundle_from_published_manifest(raw: bytes, source: Source) -> Bundle:
    """Parse the canonical bundle descriptor published beside its payload."""
    try:
        document = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"published bundle manifest is invalid JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ManifestError("published bundle manifest must be an object")
    parsed_document = {
        **document,
        "sha256": document.get("bundle_sha256"),
    }
    parsed_document.pop("bundle_sha256", None)
    bundle = _parse_bundle(
        0,
        {
            **parsed_document,
            "members": [
                {**member, "source": source.name, "remote_path": member["path"]}
                for member in document.get("members", [])
                if isinstance(member, Mapping)
            ],
            "receipts": [
                {**receipt, "source": source.name, "remote_path": receipt["path"]}
                for receipt in document.get("receipts", [])
                if isinstance(receipt, Mapping)
            ],
        },
        {source.name: source},
    )
    if raw != bundle.manifest_bytes:
        raise ManifestError("published bundle manifest is not canonical")
    return bundle


def _require(mapping: Mapping[str, object], key: str, where: str) -> object:
    if key not in mapping:
        raise ManifestError(f"{where}: missing {key!r}")
    return mapping[key]


def _relative_path(value: object, where: str) -> str:
    if not isinstance(value, str) or _RELATIVE_PATH_RE.fullmatch(value) is None:
        raise ManifestError(f"{where}: path must be a plain relative path, got {value!r}")
    return value


def _parse_source(name: str, raw: object) -> Source:
    where = f"sources[{name!r}]"
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: must be an object")
    kind = _require(raw, "kind", where)
    source_locator = _require(raw, "source_locator", where)
    if kind == "huggingface":
        ref = _require(raw, "revision", where)
        if not isinstance(ref, str) or _HEX40_RE.fullmatch(ref) is None:
            raise ManifestError(f"{where}: revision must be a 40-hex commit, got {ref!r}")
    elif kind == "github-release":
        ref = _require(raw, "tag", where)
        if not isinstance(ref, str) or not ref:
            raise ManifestError(f"{where}: tag must be a non-empty string")
    else:
        raise ManifestError(f"{where}: unsupported kind {kind!r}")
    if (
        not isinstance(source_locator, str)
        or source_locator.count("/") != 1
        or source_locator.startswith("/")
    ):
        raise ManifestError(f"{where}: source_locator must be 'owner/name', got {source_locator!r}")
    return Source(name=name, kind=kind, source_locator=source_locator, ref=ref)


def _parse_artifact(index: int, raw: object, sources: Mapping[str, Source]) -> Artifact:
    where = f"artifacts[{index}]"
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: must be an object")
    path = _relative_path(_require(raw, "path", where), where)
    source_name = _require(raw, "source", where)
    if not isinstance(source_name, str) or source_name not in sources:
        raise ManifestError(f"{where}: unknown source {source_name!r}")
    remote_path = _relative_path(_require(raw, "remote_path", where), where)
    size = _require(raw, "size", where)
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ManifestError(f"{where}: size must be a positive integer")
    sha256 = _require(raw, "sha256", where)
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise ManifestError(f"{where}: sha256 must be 64 lowercase hex characters")
    return Artifact(
        path=path,
        source=sources[source_name],
        remote_path=remote_path,
        size=size,
        sha256=sha256,
    )


def canonical_json(value: object) -> str:
    """Return the sole serialization accepted for content-addressed payloads."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"bundle payload must be JSON data: {exc}") from exc


def _parse_bundle(index: int, raw: object, sources: Mapping[str, Source]) -> Bundle:
    where = f"bundles[{index}]"
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: must be an object")
    sha256 = _require(raw, "sha256", where)
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise ManifestError(f"{where}: sha256 must be 64 lowercase hex characters")
    raw_members = _require(raw, "members", where)
    if not isinstance(raw_members, list) or not raw_members:
        raise ManifestError(f"{where}: members must be a non-empty list")
    members = tuple(
        _parse_artifact(member_index, member, sources)
        for member_index, member in enumerate(raw_members)
    )
    paths = [member.path for member in members]
    raw_receipts = raw.get("receipts", [])
    if not isinstance(raw_receipts, list):
        raise ManifestError(f"{where}: receipts must be a list")
    if raw_receipts == [] and "receipts" in raw:
        raise ManifestError(f"{where}: receipts must be non-empty when present")
    receipts = tuple(
        _parse_artifact(receipt_index, receipt, sources)
        for receipt_index, receipt in enumerate(raw_receipts)
    )
    receipt_paths = [receipt.path for receipt in receipts]
    if len(paths) != len(set(paths)) or len(receipt_paths) != len(set(receipt_paths)):
        raise ManifestError(f"{where}: duplicate member paths")
    if set(paths) & set(receipt_paths):
        raise ManifestError(f"{where}: receipt paths overlap payload members")
    payload = _require(raw, "payload", where)
    if not isinstance(payload, Mapping):
        raise ManifestError(f"{where}: payload must be an object")
    runtime_format = raw.get("runtime_format")
    if runtime_format is not None and (not isinstance(runtime_format, str) or not runtime_format):
        raise ManifestError(f"{where}: runtime_format must be a non-empty string")
    bundle = Bundle(
        sha256=sha256,
        members=members,
        payload=dict(payload),
        receipts=receipts,
        runtime_format=runtime_format,
    )
    actual = hashlib.sha256(bundle.canonical_payload.encode()).hexdigest()
    if actual != sha256:
        raise ManifestError(f"{where}: sha256 does not match canonical members and payload")
    return bundle


def parse_manifest(raw: object) -> Manifest:
    if not isinstance(raw, Mapping):
        raise ManifestError("manifest must be a JSON object")
    version = _require(raw, "schema_version", "manifest")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema_version {version!r} is not {SUPPORTED_SCHEMA_VERSION}"
        )
    raw_sources = _require(raw, "sources", "manifest")
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ManifestError("manifest: sources must be a non-empty object")
    sources = {name: _parse_source(name, value) for name, value in raw_sources.items()}
    raw_artifacts = _require(raw, "artifacts", "manifest")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ManifestError("manifest: artifacts must be a non-empty list")
    artifacts = tuple(
        _parse_artifact(index, value, sources) for index, value in enumerate(raw_artifacts)
    )
    raw_sidecars = raw.get("sidecars", [])
    if not isinstance(raw_sidecars, list):
        raise ManifestError("manifest: sidecars must be a list")
    sidecars = tuple(_relative_path(value, "sidecars") for value in raw_sidecars)
    paths = [artifact.path for artifact in artifacts] + list(sidecars)
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        raise ManifestError(f"manifest: duplicate destination paths {duplicates}")
    raw_bundles = raw.get("bundles", [])
    if not isinstance(raw_bundles, list):
        raise ManifestError("manifest: bundles must be a list")
    bundles = tuple(_parse_bundle(index, value, sources) for index, value in enumerate(raw_bundles))
    bundle_hashes = [bundle.sha256 for bundle in bundles]
    if len(bundle_hashes) != len(set(bundle_hashes)):
        raise ManifestError("manifest: duplicate bundle identities")
    raw_published = raw.get("published_bundles", [])
    if not isinstance(raw_published, list):
        raise ManifestError("manifest: published_bundles must be a list")
    published: list[PublishedBundle] = []
    for index, publication in enumerate(raw_published):
        where = f"published_bundles[{index}]"
        if not isinstance(publication, Mapping):
            raise ManifestError(f"{where}: must be an object")
        if set(publication) != {"source_locator", "revision", "bundle_sha256"}:
            raise ManifestError(f"{where}: fields must be source_locator, revision, bundle_sha256")
        source_locator = publication["source_locator"]
        revision = publication["revision"]
        bundle_sha256 = publication["bundle_sha256"]
        if not isinstance(source_locator, str) or source_locator.count("/") != 1:
            raise ManifestError(f"{where}: source_locator must be owner/name")
        if not isinstance(revision, str) or _HEX40_RE.fullmatch(revision) is None:
            raise ManifestError(f"{where}: revision must be a 40-hex commit")
        if not isinstance(bundle_sha256, str) or _SHA256_RE.fullmatch(bundle_sha256) is None:
            raise ManifestError(f"{where}: bundle_sha256 must be 64 lowercase hex characters")
        if not any(
            source.source_locator == source_locator and source.ref == revision
            for source in sources.values()
        ):
            raise ManifestError(f"{where}: publication must name a pinned source")
        published.append(PublishedBundle(source_locator, revision, bundle_sha256))
    if len({publication.bundle_sha256 for publication in published}) != len(published):
        raise ManifestError("manifest: duplicate published bundle identities")
    return Manifest(
        sources=sources,
        artifacts=artifacts,
        sidecars=sidecars,
        bundles=bundles,
        published_bundles=tuple(published),
    )


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    return parse_manifest(raw)
