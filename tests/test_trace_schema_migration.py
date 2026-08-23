from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app.edge_db.compatibility import CURRENT_SCHEMA_RANGE
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import MIGRATIONS, SCHEMA_VERSION


def test_forward_migration_preserves_old_component_rows_and_admits_truthful_states(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:7])
    analysis_id = "1" * 64
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO runtime_analysis_traces (
                trace_id, trace_schema_version, worker_boot_id, camera_id,
                stream_epoch, frame_seq, pts, pts_missing_reason,
                source_time_sec, source_time_missing_reason, frame_width,
                frame_height, bed_region_provenance, storage_bytes
            ) VALUES (?, 1, 'boot-a', 'camera-a', 1, 1, 1.0, NULL,
                      1.0, NULL, 4, 4, 'fresh', 1)
            """,
            (analysis_id,),
        )
        connection.execute(
            "INSERT INTO runtime_analysis_components VALUES (?, 0, 'pose', 'observed')",
            (analysis_id,),
        )

    result = migrate_database(database)

    assert result.previous_version == 7
    assert result.current_version == CURRENT_SCHEMA_RANGE.maximum == SCHEMA_VERSION
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO runtime_analysis_components VALUES (?, 1, 'classifier', 'executed')",
            (analysis_id,),
        )
        connection.execute(
            "INSERT INTO runtime_analysis_components VALUES (?, 2, 'window', 'not-applicable')",
            (analysis_id,),
        )
        assert connection.execute(
            "SELECT ordinal, observation_state FROM runtime_analysis_components ORDER BY ordinal"
        ).fetchall() == [(0, "observed"), (1, "executed"), (2, "not-applicable")]


def test_migrator_authorizer_allows_only_alter_internal_quick_check(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"

    assert migrate_database(database).current_version == SCHEMA_VERSION
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute(
            "SELECT count(*) FROM sqlite_schema WHERE name = 'runtime_analysis_components_v7'"
        ).fetchone() == (0,)
