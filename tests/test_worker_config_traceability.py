"""Traceability coverage for Issue #35 PR C: an evidence event's
`audit.config_version` must resolve against `config_history` as the worker
config in effect at emission time -- both tables live in the same
worker-state.sqlite3, so the join is a local SQL query, not a cross-store
correlation (acceptance criterion 7).

Open-question verification (plan §5 PR C, "verify handoff open questions"):

1. Does `eldercare-fall-ai` bump `configVersion` on detection-window change?
   Yes. In the sibling repo, `MlConfigService.updateNightWindow`
   (backend/src/ml-config/ml-config.service.ts:99, in the upsert's `update`
   branch) sets `configVersion: { increment: 1 }` on every night-window
   update. Camera add/edit/remove independently bumps it too, via
   `bumpMlConfigVersion` (backend/src/ml-config/ml-config.version.ts:3,
   called from backend/src/cameras/cameras.service.ts:52,99,137,158). So an
   observed `config_version` change on the worker reliably signals a real
   backend config change.

2. Do both `fall`/`bed_exit` domains always emit `payload["audit"]` as a
   Mapping? Yes, by construction, not convention. Both domains share the
   identical `_audit_snapshot` `audit_metadata_provider`
   (worker/domains/registry.py) -- there is no per-domain divergence.
   `WorkerRuntime._build_domain_audit` (worker/runtime/worker.py:1128)
   unconditionally returns `build_audit_envelope(...)` (always a `dict`) for
   every enabled domain, or `{}` when no provider is registered -- never
   anything else. `AlertEvidenceAttacher.attach` (worker/pipeline/output/
   evidence_attacher.py:78) only ever sets `BusinessEvent.audit` (typed
   `Mapping[str, object] | None`) from that same dict, or leaves it
   untouched (`None`) when empty. So `evidence_stager.py`'s
   `isinstance(event_audit, Mapping)` guard in `_canonical_payload` is
   either always-true or the `audit` key is cleanly absent -- a non-Mapping
   value can never reach it. The
   `test_audit_config_version_is_locally_joinable_against_config_history`
   test below exercises this end-to-end via the `fall` domain's event shape.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimLease,
    EdgeEventId,
    EvidenceOutbox,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.runtime.config.lkg_store import CONFIG_HISTORY_RETENTION_COUNT, WorkerConfigLkgStore
from worker.runtime.config.restart import RestartDirective

EVENT_ID = EdgeEventId("00000000-0000-4000-8000-000000000002")


def _config_payload(config_version: int) -> dict[str, object]:
    return {
        "registry_version": config_version,
        "config_version": config_version,
        "restart_epoch": 1,
        "cameras": [],
    }


def _event() -> dict[str, object]:
    return {
        "edge_event_id": EVENT_ID,
        "event_type": "fall",
        "probability": 0.9,
        "detected_at": "2026-08-02T00:00:00Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "audit": {"model_version": "model-1"},
    }


def test_audit_config_version_is_locally_joinable_against_config_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-state.sqlite3"

    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=7), RestartDirective(generation=1, version=7)
    )

    stager = DurableEvidenceStager(
        database,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        config_version=7,
        clock=lambda: 100.0,
    )
    stager.stage(_event())

    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            """
            SELECT config_history.config_version, config_history.payload_json
            FROM evidence_events
            JOIN config_history
              ON config_history.config_version = CAST(
                     json_extract(evidence_events.payload_json, '$.audit.config_version')
                     AS INTEGER
                 )
            WHERE evidence_events.edge_event_id = ?
            """,
            (str(EVENT_ID),),
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    joined_config_version, payload_json = row
    assert joined_config_version == 7
    assert json.loads(payload_json)["config_version"] == 7


def test_config_history_retains_unacked_reference_beyond_retention_window(
    tmp_path: Path,
) -> None:
    """Retention policy: "unacked event + last N". A config_history row still
    referenced by an evidence_events row that hasn't reached
    delivery_state='ACKED' must survive pruning even after
    CONFIG_HISTORY_RETENTION_COUNT newer configs have been saved."""
    database = tmp_path / "worker-state.sqlite3"
    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=1), RestartDirective(generation=1, version=1)
    )

    stager = DurableEvidenceStager(
        database,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        config_version=1,
        clock=lambda: 100.0,
    )
    stager.stage(_event())  # stays delivery_state='PENDING' -- never claimed/acked

    for version in range(2, CONFIG_HISTORY_RETENTION_COUNT + 5):
        assert config_store.save(
            _config_payload(config_version=version),
            RestartDirective(generation=1, version=version),
        )

    remaining = _history_versions(database)
    assert 1 in remaining, "unacked reference must survive pruning"


def test_config_history_prunes_acked_reference_beyond_retention_window(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-state.sqlite3"
    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=1), RestartDirective(generation=1, version=1)
    )

    stager = DurableEvidenceStager(
        database,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        config_version=1,
        clock=lambda: 100.0,
    )
    stager.stage(_event())
    stager.complete(EVENT_ID, None)
    with EvidenceOutbox.open(database) as outbox:
        claim = outbox.claim(ClaimLease("sender", 100.0, 10.0))
        assert claim is not None
        assert outbox.acknowledge(claim)

    for version in range(2, CONFIG_HISTORY_RETENTION_COUNT + 5):
        assert config_store.save(
            _config_payload(config_version=version),
            RestartDirective(generation=1, version=version),
        )

    remaining = _history_versions(database)
    assert 1 not in remaining, "acked + outside the retention window must be pruned"
    assert len(remaining) == CONFIG_HISTORY_RETENTION_COUNT


def _history_versions(database: Path) -> set[int]:
    connection = sqlite3.connect(database)
    try:
        return {
            int(row[0])
            for row in connection.execute("SELECT config_version FROM config_history").fetchall()
        }
    finally:
        connection.close()
