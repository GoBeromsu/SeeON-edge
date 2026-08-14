from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.evidence import evidence_outbox_stage
from worker.pipeline.output.evidence.evidence_stager import (
    DurableEvidenceStager,
    RuntimeManifestReferenceError,
    RuntimeManifestReferenceFailure,
)

_RUNTIME_MANIFEST_SHA256 = "a" * 64
_EVENT_ID = "event:provenance"


def _event() -> dict[str, object]:
    return {
        "edge_event_id": _EVENT_ID,
        "event_type": "fall",
        "probability": 0.93,
        "detected_at": "2026-08-13T00:00:00Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "audit": {"runtime_manifest_sha256": _RUNTIME_MANIFEST_SHA256},
    }


def _stager(database: Path) -> DurableEvidenceStager:
    return DurableEvidenceStager(
        database_path=database,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        config_version=7,
        clock=lambda: 100.0,
        runtime_manifest_sha256=_RUNTIME_MANIFEST_SHA256,
    )


def _insert_manifest_contents(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO runtime_manifest_contents (
                manifest_sha256, manifest_schema_version, canonical_json, created_at
            ) VALUES (?, 1, ?, '2026-08-13T00:00:00Z')
            """,
            (
                _RUNTIME_MANIFEST_SHA256,
                json.dumps({"manifest_schema_version": 1}, separators=(",", ":")),
            ),
        )


def _event_count(database: Path) -> int:
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT count(*) FROM evidence_events").fetchone()
    assert row is not None
    return int(row[0])


def test_stager_accepts_reference_to_existing_immutable_manifest_contents(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _insert_manifest_contents(database)

    _stager(database).stage(_event())

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT payload_json FROM evidence_events WHERE edge_event_id = ?",
            (_EVENT_ID,),
        ).fetchone()
    assert row is not None
    assert json.loads(str(row[0]))["audit"]["runtime_manifest_sha256"] == (_RUNTIME_MANIFEST_SHA256)


def test_manifest_check_and_outbox_insert_share_one_write_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _insert_manifest_contents(database)
    transaction_states: list[bool] = []
    statements: list[str] = []
    original_check = evidence_outbox_stage.require_runtime_manifest_contents

    def observe_check(connection: sqlite3.Connection, manifest_sha256: str) -> None:
        transaction_states.append(connection.in_transaction)
        connection.set_trace_callback(statements.append)
        original_check(connection, manifest_sha256)

    monkeypatch.setattr(
        evidence_outbox_stage,
        "require_runtime_manifest_contents",
        observe_check,
    )

    _stager(database).stage(_event())

    assert transaction_states == [True]
    select_index = next(
        index
        for index, statement in enumerate(statements)
        if "SELECT 1 FROM runtime_manifest_contents" in statement
    )
    insert_index = next(
        index
        for index, statement in enumerate(statements)
        if "INSERT INTO evidence_events" in statement
    )
    assert select_index < insert_index
    assert statements[-1] == "COMMIT"


def test_stager_rejects_orphan_manifest_reference_without_partial_outbox_state(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)

    with pytest.raises(RuntimeManifestReferenceError) as raised:
        _stager(database).stage(_event())

    assert raised.value.failure is RuntimeManifestReferenceFailure.MISSING
    assert raised.value.manifest_sha256 == _RUNTIME_MANIFEST_SHA256
    assert _event_count(database) == 0


def test_stager_rejects_unavailable_provenance_without_creating_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"

    with pytest.raises(RuntimeManifestReferenceError) as raised:
        _stager(database).stage(_event())

    assert raised.value.failure is RuntimeManifestReferenceFailure.UNAVAILABLE
    assert raised.value.manifest_sha256 == _RUNTIME_MANIFEST_SHA256
    assert not database.exists()


def test_event_audit_reference_is_checked_when_stager_has_no_injected_reference(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    stager = DurableEvidenceStager(
        database_path=database,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        config_version=7,
        clock=lambda: 100.0,
    )

    with pytest.raises(RuntimeManifestReferenceError) as raised:
        stager.stage(_event())

    assert raised.value.failure is RuntimeManifestReferenceFailure.MISSING
    assert _event_count(database) == 0
