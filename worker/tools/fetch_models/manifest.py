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
    repo: str
    ref: str

    def url_for(self, remote_path: str) -> str:
        if self.kind == "huggingface":
            return f"https://huggingface.co/{self.repo}/resolve/{self.ref}/{remote_path}"
        if self.kind == "github-release":
            return f"https://github.com/{self.repo}/releases/download/{self.ref}/{remote_path}"
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


@dataclass(frozen=True)
class Bundle:
    """A content-addressed collection which is published as one unit."""

    sha256: str
    members: tuple[Artifact, ...]
    payload: Mapping[str, object]

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
                    "schema_version": 1,
                }
            )
            + "\n"
        ).encode()


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
    repo = _require(raw, "repo", where)
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
    if not isinstance(repo, str) or repo.count("/") != 1 or repo.startswith("/"):
        raise ManifestError(f"{where}: repo must be 'owner/name', got {repo!r}")
    return Source(name=name, kind=kind, repo=repo, ref=ref)


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
    if len(paths) != len(set(paths)):
        raise ManifestError(f"{where}: duplicate member paths")
    payload = _require(raw, "payload", where)
    if not isinstance(payload, Mapping):
        raise ManifestError(f"{where}: payload must be an object")
    bundle = Bundle(sha256=sha256, members=members, payload=dict(payload))
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
    return Manifest(sources=sources, artifacts=artifacts, sidecars=sidecars, bundles=bundles)


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    return parse_manifest(raw)
