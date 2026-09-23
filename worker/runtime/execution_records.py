"""Compose the optional Worker -> Backend execution-record export path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import PurePath

from pydantic import SecretBytes, SecretStr

from shared.events.execution_records_client import ExecutionRecordsClient
from worker.pipeline.diagnostics.exporter import ExecutionRecordExporter
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.pipeline.diagnostics.provenance import (
    ExecutionRecordProvenanceError,
    build_wire_provenance,
)
from worker.runtime.config.execution_records import (
    ExecutionRecordsSettings,
    execution_records_settings_from_environment,
)
from worker.runtime.config.worker_models import WorkerConfig


def compose_execution_records(
    config: WorkerConfig,
    *,
    env: Mapping[str, str],
    build_revision: str | None,
    image_digest: str | None,
    model_digest: str | None,
    calibration_digest: str | None,
    preprocessing_identity: str | None,
    policy_identity: str | None,
) -> tuple[
    ExecutionRecordsSettings | None,
    ExecutionRecordLanes | None,
    ExecutionRecordExporter | None,
]:
    settings = execution_records_settings_from_environment(env)
    if settings is None:
        return None, None, None
    token = config.relay.token.get_secret_value().strip()
    if not config.relay.url.strip() or not token:
        raise ExecutionRecordProvenanceError(
            "execution records enabled but relay URL/token missing"
        )
    provenance = build_wire_provenance(
        worker_build_revision=build_revision,
        worker_image_digest=image_digest,
        model_digest=model_digest,
        calibration_digest=calibration_digest,
        preprocessing_identity=preprocessing_identity,
        config_digest=_config_digest(config),
        policy_identity=policy_identity,
    )
    lanes = ExecutionRecordLanes(lane_capacity=settings.lane_capacity)
    exporter = ExecutionRecordExporter(
        lanes=lanes,
        client=ExecutionRecordsClient(config.relay.url, token),
        provenance=provenance,
        batch_max=settings.batch_max,
        flush_ms=settings.flush_ms,
    )
    return settings, lanes, exporter


def _config_digest(config: WorkerConfig) -> str:
    """Stable content digest of the effective worker config.

    ``model_dump(mode="python")`` keeps nested frozen dataclasses and their
    ``MappingProxyType`` fields (``PulledWorkerConfig.detection_windows``) as
    Python objects; pydantic's JSON mode cannot serialize those, and that only
    shows up with a real pulled config. ``_jsonable`` maps every such value to
    a deterministic JSON shape instead.
    """
    payload = config.model_dump(mode="python")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_jsonable).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: object) -> object:
    """JSON shape for the non-JSON values a worker config can carry."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in fields(value)}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (PurePath, datetime, date)):
        return str(value)
    if isinstance(value, (SecretStr, SecretBytes)):
        # Secrets never enter a digest: neither their value nor anything
        # derived from it. A rotated token must not look like a config change.
        return "<secret>"
    raise TypeError(f"config value of type {type(value).__name__} has no digest shape")


__all__ = ["compose_execution_records"]
