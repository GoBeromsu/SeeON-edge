"""Immutable worker image admission for real canary execution."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import ClassVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

_REVISION = re.compile(r"^[0-9a-f]{40}$")


class WorkerImageAdmissionError(ValueError):
    """Requested worker identity is absent, mutable, or does not match Docker."""


@dataclass(frozen=True, slots=True)
class WorkerImageAdmission:
    image: str | None
    expected_revision: str | None
    render_only: bool


@dataclass(frozen=True, slots=True)
class WorkerImageBinding:
    image: str
    digest: str
    revision: str | None


class _InspectConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    labels: dict[str, str] | None = Field(
        default=None,
        validation_alias=AliasChoices("Labels", "labels"),
    )


class _ImageInspect(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    repo_digests: tuple[str, ...] = Field(validation_alias="RepoDigests")
    config: _InspectConfig = Field(validation_alias="Config")


def _digest(image: str) -> str:
    match = re.fullmatch(r"([^@\s]+)@sha256:([0-9a-f]{64})", image)
    if match is None:
        raise WorkerImageAdmissionError("worker image must be repository@sha256:<digest>")
    return f"sha256:{match.group(2)}"


def admit_worker_image(request: WorkerImageAdmission) -> WorkerImageBinding:
    """Resolve and verify a requested image before any real canary side effect."""
    if request.image is None or not request.image.strip():
        raise WorkerImageAdmissionError("worker image is required")
    image = request.image.strip()
    digest = _digest(image)
    if request.expected_revision is not None and _REVISION.fullmatch(
        request.expected_revision
    ) is None:
        raise WorkerImageAdmissionError("expected revision must be 40 lowercase hex characters")
    if request.render_only:
        return WorkerImageBinding(image, digest, request.expected_revision)
    if request.expected_revision is None:
        raise WorkerImageAdmissionError("expected revision is required for real canary execution")
    completed = subprocess.run(
        ("docker", "image", "inspect", image),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise WorkerImageAdmissionError("worker image inspect failed")
    try:
        inspections = tuple(
            _ImageInspect.model_validate(item) for item in json.loads(completed.stdout)
        )
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise WorkerImageAdmissionError("worker image inspect output is invalid") from error
    if len(inspections) != 1 or image not in inspections[0].repo_digests:
        raise WorkerImageAdmissionError("worker image digest mismatch")
    labels = inspections[0].config.labels or {}
    observed_revision = labels.get("org.opencontainers.image.revision")
    if observed_revision != request.expected_revision:
        raise WorkerImageAdmissionError("worker image revision mismatch")
    return WorkerImageBinding(image, digest, observed_revision)
