"""Compose the optional Worker -> Backend execution-record export path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

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
    worker_boot_id: str,
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
    del worker_boot_id
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
    payload = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["compose_execution_records"]
