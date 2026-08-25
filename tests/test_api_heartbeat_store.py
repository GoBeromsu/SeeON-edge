from __future__ import annotations

import inspect
import sqlite3
import threading
from pathlib import Path

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.status.heartbeat_store import NEVER_SEEN, ONLINE, STALE, HeartbeatStore


def test_never_seen_when_in_inventory_without_heartbeat() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    inventory = {"cam-a": {"camera_id": "cam-a", "facility_id": "fac-1"}}

    snap = store.snapshot(inventory, now=1000.0)

    assert snap["cameras"]["cam-a"]["status"] == NEVER_SEEN
    assert snap["cameras"]["cam-a"]["facility_id"] == "fac-1"
    assert snap["cameras"]["cam-a"]["last_heartbeat_at"] is None


def test_online_within_stale_window() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-a", "fac-1", received_at=1000.0)

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}}, now=1060.0)

    assert snap["cameras"]["cam-a"]["status"] == ONLINE
    assert snap["cameras"]["cam-a"]["age_sec"] == 60.0


def test_stale_after_window() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-a", "fac-1", received_at=1000.0)

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}}, now=1200.0)

    assert snap["cameras"]["cam-a"]["status"] == STALE


def test_seen_camera_outside_inventory_is_reported() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-z", "fac-9", received_at=1000.0)

    snap = store.snapshot({}, now=1010.0)

    assert snap["cameras"]["cam-z"]["status"] == ONLINE
    assert snap["cameras"]["cam-z"]["facility_id"] == "fac-9"


def test_latest_heartbeat_wins() -> None:
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-a", "fac-1", received_at=1000.0)
    store.record("cam-a", "fac-1", received_at=1100.0)

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}}, now=1110.0)

    assert snap["cameras"]["cam-a"]["last_heartbeat_at"] == 1100.0
    assert snap["cameras"]["cam-a"]["age_sec"] == 10.0


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_missing_observation_is_never_a_fabricated_zero() -> None:
    store = HeartbeatStore(stale_after_sec=90.0, clock=_Clock(1_000.0))

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}})

    camera = snap["cameras"]["cam-a"]
    assert camera["status"] == NEVER_SEEN
    assert camera["last_heartbeat_at"] is None
    assert camera["age_sec"] is None
    assert camera["config_version"] is None
    assert 0 not in (camera["last_heartbeat_at"], camera["age_sec"], camera["config_version"])


def test_stale_is_distinct_from_valid_empty_and_missing() -> None:
    clock = _Clock(1_000.0)
    store = HeartbeatStore(stale_after_sec=90.0, clock=clock)
    store.record("cam-stale", "fac-1", received_at=900.0)
    clock.now = 1_000.0

    snap = store.snapshot(
        {
            "cam-stale": {"facility_id": "fac-1"},
            "cam-missing": {"facility_id": "fac-1"},
        }
    )

    assert snap["cameras"]["cam-stale"]["status"] == STALE
    assert snap["cameras"]["cam-stale"]["last_heartbeat_at"] == 900.0
    assert snap["cameras"]["cam-missing"]["status"] == NEVER_SEEN
    assert snap["cameras"]["cam-missing"]["last_heartbeat_at"] is None
    assert "cam-absent" not in snap["cameras"]


def test_future_timestamp_is_rejected_and_does_not_become_online() -> None:
    clock = _Clock(1_000.0)
    store = HeartbeatStore(stale_after_sec=90.0, clock=clock)
    store.record("cam-a", "fac-1", received_at=2_000.0)

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}})

    assert snap["cameras"]["cam-a"]["status"] == NEVER_SEEN
    assert snap["cameras"]["cam-a"]["last_heartbeat_at"] is None


def test_clock_rollback_marks_existing_observation_stale() -> None:
    clock = _Clock(2_000.0)
    store = HeartbeatStore(stale_after_sec=90.0, clock=clock)
    store.record("cam-a", "fac-1", received_at=2_000.0)
    clock.now = 1_000.0

    snap = store.snapshot({"cam-a": {"facility_id": "fac-1"}})

    assert snap["cameras"]["cam-a"]["status"] == STALE
    assert snap["cameras"]["cam-a"]["last_heartbeat_at"] == 2_000.0


def test_stale_entries_are_evicted_after_retention_window() -> None:
    clock = _Clock(1_000.0)
    store = HeartbeatStore(stale_after_sec=90.0, retain_after_sec=180.0, clock=clock)
    store.record("cam-old", "fac-1", received_at=800.0)
    store.record("cam-kept", "fac-1", received_at=950.0)
    clock.now = 1_000.0

    snap = store.snapshot(
        {
            "cam-old": {"facility_id": "fac-1"},
            "cam-kept": {"facility_id": "fac-1"},
        }
    )

    assert snap["cameras"]["cam-old"]["status"] == NEVER_SEEN
    assert snap["cameras"]["cam-old"]["last_heartbeat_at"] is None
    assert snap["cameras"]["cam-kept"]["status"] == ONLINE


def test_cardinality_cap_evicts_oldest_first() -> None:
    clock = _Clock(1_000.0)
    store = HeartbeatStore(stale_after_sec=90.0, max_cameras=2, clock=clock)
    store.record("cam-a", "fac-1", received_at=980.0)
    store.record("cam-b", "fac-1", received_at=990.0)
    store.record("cam-c", "fac-1", received_at=1_000.0)

    snap = store.snapshot(now=1_000.0)

    assert set(snap["cameras"]) == {"cam-b", "cam-c"}
    assert "cam-a" not in snap["cameras"]


def test_one_hundred_heartbeat_ids_are_memory_only_and_lost_on_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    store = HeartbeatStore(stale_after_sec=90.0, clock=_Clock(1_000.0))
    for index in range(100):
        store.record(f"cam-{index}", f"fac-{index}", received_at=1_000.0)

    first = store.snapshot(now=1_000.0)
    restarted = HeartbeatStore(stale_after_sec=90.0, clock=_Clock(1_000.0))
    lost = restarted.snapshot(
        {f"cam-{index}": {"facility_id": f"fac-{index}"} for index in range(100)}
    )

    assert len(first["cameras"]) == 100
    assert all(row["status"] == ONLINE for row in first["cameras"].values())
    assert all(row["status"] == NEVER_SEEN for row in lost["cameras"].values())
    assert all(row["last_heartbeat_at"] is None for row in lost["cameras"].values())
    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert "control_heartbeats" not in tables
        assert "runtime_latency" not in tables


def test_concurrent_records_keep_latest_timestamp() -> None:
    store = HeartbeatStore(stale_after_sec=1_000.0)
    barrier = threading.Barrier(17)
    threads = [
        threading.Thread(
            target=lambda stamp=stamp: (
                barrier.wait(),
                store.record("cam-a", "fac-1", received_at=float(stamp)),
            )
        )
        for stamp in range(1, 17)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    snap = store.snapshot(now=20.0)
    assert snap["cameras"]["cam-a"]["last_heartbeat_at"] == 16.0


def test_heartbeat_store_has_no_sqlite_surface() -> None:
    source = inspect.getsource(HeartbeatStore)
    module = Path(__file__).resolve().parents[1] / "backend/app/features/status/heartbeat_store.py"
    text = module.read_text(encoding="utf-8")
    assert "sqlite3" not in text
    assert "CREATE TABLE" not in text
    assert "control_heartbeats" not in text
    assert "database_path" not in source
