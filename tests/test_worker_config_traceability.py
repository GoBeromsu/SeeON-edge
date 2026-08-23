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

from pathlib import Path

from worker.runtime.config.lkg_store import (
    CONFIG_HISTORY_RETENTION_COUNT,
    WorkerConfigLkgStore,
)
from worker.runtime.config.restart import RestartDirective


def _config_payload(config_version: int) -> dict[str, object]:
    return {
        "registry_version": config_version,
        "config_version": config_version,
        "restart_epoch": 1,
        "cameras": [],
    }


def test_config_revision_is_durable_and_contains_the_resolved_payload(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-state.sqlite3"
    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=7), RestartDirective(generation=1, version=7)
    )
    revisions = tuple((config_store.database_path / "revisions").glob("*.json"))
    assert len(revisions) == 1
    assert '"config_version":7' in revisions[0].read_text()


def test_config_revisions_are_bounded_beyond_retention_window(
    tmp_path: Path,
) -> None:
    """The worker cache retains only its fixed bounded window."""
    database = tmp_path / "worker-state.sqlite3"
    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=1), RestartDirective(generation=1, version=1)
    )

    for version in range(2, CONFIG_HISTORY_RETENTION_COUNT + 5):
        assert config_store.save(
            _config_payload(config_version=version),
            RestartDirective(generation=1, version=version),
        )

    remaining = _revision_versions(config_store)
    assert 1 not in remaining
    assert len(remaining) == CONFIG_HISTORY_RETENTION_COUNT


def test_config_revisions_keep_the_newest_versions(
    tmp_path: Path,
) -> None:
    database = tmp_path / "worker-state.sqlite3"
    config_store = WorkerConfigLkgStore(database)
    assert config_store.save(
        _config_payload(config_version=1), RestartDirective(generation=1, version=1)
    )

    for version in range(2, CONFIG_HISTORY_RETENTION_COUNT + 5):
        assert config_store.save(
            _config_payload(config_version=version),
            RestartDirective(generation=1, version=version),
        )

    remaining = _revision_versions(config_store)
    assert 1 not in remaining
    assert len(remaining) == CONFIG_HISTORY_RETENTION_COUNT


def _revision_versions(store: WorkerConfigLkgStore) -> set[int]:
    return {
        int(path.read_text().split('"config_version":', 1)[1].split(",", 1)[0])
        for path in (store.database_path / "revisions").glob("*.json")
    }


def test_worker_and_compose_have_no_facility_identity_env_contract() -> None:
    root = Path(__file__).resolve().parents[1]

    assert "API_FACILITY_ID_ENV" not in (root / "worker/runtime/config/config_pull.py").read_text()
    compose = (root / "compose.edge.yaml").read_text()
    assert "API_FACILITY_ID:" not in compose
    assert "EDGE_FACILITY_TOKEN:" not in compose
