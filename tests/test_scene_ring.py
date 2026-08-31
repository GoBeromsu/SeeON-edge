from __future__ import annotations

from fractions import Fraction
from threading import Event, Thread

from worker.pipeline.output.evidence.scene_repository import SceneRingRepository
from worker.pipeline.output.evidence.scene_ring import CameraSceneRing, SceneRingLimits
from worker.types import SceneRecord


def _record(
    camera_id: str,
    seq: int,
    *,
    epoch: int = 1,
    pts: int | Fraction | None = None,
    size: int = 4,
) -> SceneRecord:
    return SceneRecord(
        worker_boot_id="boot",
        camera_id=camera_id,
        stream_epoch=epoch,
        generation=0,
        source_pts_sec=Fraction(seq if pts is None else pts),
        seq=seq,
        payload=b"x" * size,
        size_bytes=size,
        detail_shed=False,
    )


def test_ring_evicts_by_frame_byte_and_duration_limits() -> None:
    frame_limits = SceneRingLimits(max_frames=2, max_bytes=100, max_duration_seconds=70)
    by_frame = CameraSceneRing("camera", frame_limits)
    for seq in range(3):
        assert by_frame.append(_record("camera", seq))
    assert tuple(record.seq for record in by_frame.select(1, Fraction(0), Fraction(3))) == (1, 2)

    byte_limits = SceneRingLimits(max_frames=10, max_bytes=8, max_duration_seconds=70)
    by_bytes = CameraSceneRing("camera", byte_limits)
    assert by_bytes.append(_record("camera", 0))
    assert by_bytes.append(_record("camera", 1))
    assert by_bytes.append(_record("camera", 2))
    assert tuple(record.seq for record in by_bytes.select(1, Fraction(0), Fraction(3))) == (1, 2)

    duration_limits = SceneRingLimits(max_frames=10, max_bytes=100, max_duration_seconds=2)
    by_duration = CameraSceneRing("camera", duration_limits)
    for seq in range(4):
        assert by_duration.append(_record("camera", seq))
    selected = by_duration.select(1, Fraction(0), Fraction(4))
    assert tuple(record.seq for record in selected) == (1, 2, 3)


def test_roll_epoch_discards_history_and_select_includes_boundaries() -> None:
    ring = CameraSceneRing("camera")
    assert ring.append(_record("camera", 1, pts=Fraction(1, 2)))
    assert ring.append(_record("camera", 2, pts=Fraction(3, 2)))
    assert tuple(record.seq for record in ring.select(1, Fraction(1, 2), Fraction(3, 2))) == (1, 2)
    ring.roll_epoch(2)
    assert ring.select(1, Fraction(0), Fraction(10)) == ()
    assert ring.append(_record("camera", 3, epoch=2))
    assert tuple(record.seq for record in ring.select(2, Fraction(0), Fraction(10))) == (3,)


def test_repository_evicts_fattest_skips_active_then_falls_back() -> None:
    limits = SceneRingLimits(max_frames=10, max_bytes=100, max_duration_seconds=70)
    repository = SceneRingRepository(
        ("active", "idle"), per_camera_limits=limits, global_max_bytes=20
    )
    assert repository.append(_record("active", 1, size=10))
    assert repository.append(_record("idle", 1, size=10))
    repository.mark_active("active")
    assert repository.append(_record("idle", 2, size=5))
    active_rows = repository.select("active", 1, Fraction(0), Fraction(2))
    assert tuple(record.seq for record in active_rows) == (1,)
    idle_rows = repository.select("idle", 1, Fraction(0), Fraction(2))
    assert tuple(record.seq for record in idle_rows) == (2,)

    repository.mark_active("idle")
    assert repository.append(_record("idle", 3, size=10))
    assert repository.metrics.active_ring_evictions == 1
    assert repository.total_bytes <= 20


def test_selection_copy_remains_valid_during_concurrent_mutation() -> None:
    copy_limits = SceneRingLimits(max_frames=20, max_bytes=80, max_duration_seconds=70)
    ring = CameraSceneRing("camera", copy_limits)
    for seq in range(20):
        assert ring.append(_record("camera", seq))
    selected = ring.select(1, Fraction(0), Fraction(19))
    done = Event()

    def mutate() -> None:
        for seq in range(20, 100):
            ring.append(_record("camera", seq))
            ring.evict_oldest()
        done.set()

    worker = Thread(target=mutate)
    worker.start()
    assert done.wait(timeout=2)
    worker.join(timeout=2)
    assert tuple(record.seq for record in selected) == tuple(range(20))
    assert all(record.payload == b"xxxx" for record in selected)
