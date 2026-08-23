from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.qa.store import (
    QaConflictError,
    QaLabelDisposition,
    QaStore,
)


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return database


def _run_payload(threshold: float) -> dict[str, object]:
    return {
        "camera_id": "camera-a",
        "module_qualified_id": "fall.v1",
        "policy_qualified_id": "fall.policy.v1",
        "frames": [
            {
                "frame_key": ["boot-a", "camera-a", 1, 1],
                "events": [],
                "operating_threshold": threshold,
            }
        ],
    }


def test_record_run_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = QaStore(_database(tmp_path))
    payload = _run_payload(0.7)

    first = store.record_run(
        camera_id="camera-a",
        module_qualified_id="fall.v1",
        policy_qualified_id="fall.policy.v1",
        effective_policy_id="e" * 64,
        frame_count=1,
        event_count=0,
        source_kind="captured",
        source_run_id=None,
        requested_by="operator-1",
        requested_at="2026-08-14T00:00:00Z",
        result=payload,
    )
    second = store.record_run(
        camera_id="camera-a",
        module_qualified_id="fall.v1",
        policy_qualified_id="fall.policy.v1",
        effective_policy_id="e" * 64,
        frame_count=1,
        event_count=0,
        source_kind="captured",
        source_run_id=None,
        requested_by="operator-2",
        requested_at="2026-08-14T00:01:00Z",
        result=payload,
    )

    # Identical result content resolves to the identical run_id (content-addressed).
    assert first.run_id == second.run_id
    fetched = store.get_run(first.run_id)
    assert fetched is not None
    assert fetched.result == payload


def test_record_run_rejects_source_kind_contract_violation(tmp_path: Path) -> None:
    store = QaStore(_database(tmp_path))
    with pytest.raises(ValueError, match="source_run_id"):
        store.record_run(
            camera_id="camera-a",
            module_qualified_id="fall.v1",
            policy_qualified_id="fall.policy.v1",
            effective_policy_id="e" * 64,
            frame_count=1,
            event_count=0,
            source_kind="replay",
            source_run_id=None,
            requested_by="operator-1",
            requested_at="2026-08-14T00:00:00Z",
            result=_run_payload(0.7),
        )


def test_record_comparison_requires_existing_runs_and_is_immutable(tmp_path: Path) -> None:
    store = QaStore(_database(tmp_path))
    baseline = store.record_run(
        camera_id="camera-a",
        module_qualified_id="fall.v1",
        policy_qualified_id="fall.policy.v1",
        effective_policy_id="e" * 64,
        frame_count=1,
        event_count=1,
        source_kind="captured",
        source_run_id=None,
        requested_by="operator-1",
        requested_at="2026-08-14T00:00:00Z",
        result=_run_payload(0.7),
    )
    candidate = store.record_run(
        camera_id="camera-a",
        module_qualified_id="fall.v1",
        policy_qualified_id="fall.policy.v1",
        effective_policy_id="f" * 64,
        frame_count=1,
        event_count=0,
        source_kind="replay",
        source_run_id=baseline.run_id,
        requested_by="operator-1",
        requested_at="2026-08-14T00:02:00Z",
        result=_run_payload(0.9),
    )

    comparison_payload: dict[str, object] = {
        "identical": False,
        "mismatches": [
            {"frame_key": ["boot-a", "camera-a", 1, 1], "reason": "event-count-differs"}
        ],
    }
    comparison = store.record_comparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        created_at="2026-08-14T00:03:00Z",
        comparison=comparison_payload,
    )
    assert comparison.identical is False
    assert comparison.mismatch_count == 1

    with pytest.raises(QaConflictError, match="unknown replay run"):
        store.record_comparison(
            baseline_run_id="0" * 64,
            candidate_run_id=candidate.run_id,
            created_at="2026-08-14T00:04:00Z",
            comparison=comparison_payload,
        )

    with pytest.raises(ValueError, match="identical flag"):
        store.record_comparison(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            created_at="2026-08-14T00:05:00Z",
            comparison={"identical": True, "mismatches": [{"frame_key": [], "reason": "x"}]},
        )

    refetched = store.get_comparison(comparison.comparison_id)
    assert refetched is not None
    assert refetched.comparison == comparison_payload


def test_label_versions_are_sequential_cas_and_conflict_on_stale_version(tmp_path: Path) -> None:
    store = QaStore(_database(tmp_path))
    baseline = store.record_run(
        camera_id="camera-a",
        module_qualified_id="fall.v1",
        policy_qualified_id="fall.policy.v1",
        effective_policy_id="e" * 64,
        frame_count=1,
        event_count=1,
        source_kind="captured",
        source_run_id=None,
        requested_by="operator-1",
        requested_at="2026-08-14T00:00:00Z",
        result=_run_payload(0.7),
    )
    candidate = store.record_run(
        camera_id="camera-a",
        module_qualified_id="fall.v1",
        policy_qualified_id="fall.policy.v1",
        effective_policy_id="f" * 64,
        frame_count=1,
        event_count=0,
        source_kind="replay",
        source_run_id=baseline.run_id,
        requested_by="operator-1",
        requested_at="2026-08-14T00:02:00Z",
        result=_run_payload(0.9),
    )
    comparison = store.record_comparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        created_at="2026-08-14T00:03:00Z",
        comparison={
            "identical": False,
            "mismatches": [
                {"frame_key": ["boot-a", "camera-a", 1, 1], "reason": "event-count-differs"}
            ],
        },
    )

    assert store.current_label(comparison.comparison_id) is None

    first_label = store.label(
        comparison_id=comparison.comparison_id,
        expected_version=0,
        actor_id="reviewer-1",
        labeled_at="2026-08-14T01:00:00Z",
        disposition=QaLabelDisposition.FALSE_NEGATIVE,
        notes="threshold change silenced a real onset",
    )
    assert first_label.version == 1

    with pytest.raises(QaConflictError, match="expected version"):
        store.label(
            comparison_id=comparison.comparison_id,
            expected_version=0,
            actor_id="reviewer-2",
            labeled_at="2026-08-14T01:05:00Z",
            disposition=QaLabelDisposition.TRUE_POSITIVE,
            notes=None,
        )

    second_label = store.label(
        comparison_id=comparison.comparison_id,
        expected_version=1,
        actor_id="reviewer-2",
        labeled_at="2026-08-14T01:10:00Z",
        disposition=QaLabelDisposition.TRUE_POSITIVE,
        notes="reviewed footage, agrees with baseline",
    )
    assert second_label.version == 2

    current = store.current_label(comparison.comparison_id)
    assert current is not None
    assert current.disposition is QaLabelDisposition.TRUE_POSITIVE
    assert current.version == 2

    history = store.label_history(comparison.comparison_id)
    assert [label.disposition for label in history] == [
        QaLabelDisposition.FALSE_NEGATIVE,
        QaLabelDisposition.TRUE_POSITIVE,
    ]
