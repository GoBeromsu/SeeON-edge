"""RED contract for the concrete frame-index task scheduler."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from worker.pipeline.bus import Scheduler


def test_default_task_intervals_preserve_legacy_insertion_order() -> None:
    # Given a scheduler with no explicit task configuration
    scheduler = Scheduler()

    # Then the legacy defaults and their insertion order are preserved.
    assert list(scheduler.task_intervals.items()) == [("pose", 1), ("bed", 90)]


def test_default_scheduler_returns_due_tasks_in_mapping_order() -> None:
    # Given the default frame-index scheduler
    scheduler = Scheduler()

    # When frame zero is evaluated, both default tasks are due.
    # Then tasks are returned in mapping insertion order.
    assert scheduler.tasks_for_frame(0) == ("pose", "bed")

    # When a non-bed frame is evaluated, only the per-frame task is due.
    assert scheduler.tasks_for_frame(1) == ("pose",)

    # When the bed interval elapses, the same order is retained.
    assert scheduler.tasks_for_frame(90) == ("pose", "bed")


def test_tasks_run_exactly_at_positive_interval_boundaries() -> None:
    # Given explicitly configured positive task intervals
    scheduler = Scheduler({"fast": 2, "slow": 3})

    # Then each task runs exactly when frame_index modulo interval is zero.
    assert scheduler.tasks_for_frame(1) == ()
    assert scheduler.tasks_for_frame(2) == ("fast",)
    assert scheduler.tasks_for_frame(3) == ("slow",)
    assert scheduler.tasks_for_frame(6) == ("fast", "slow")


def test_nonpositive_intervals_are_always_skipped() -> None:
    # Given tasks with zero and negative intervals alongside a valid task
    scheduler = Scheduler({"zero": 0, "negative": -2, "valid": 2})

    # Then invalid intervals never run, while valid intervals retain order.
    assert scheduler.tasks_for_frame(0) == ("valid",)
    assert scheduler.tasks_for_frame(2) == ("valid",)
    assert scheduler.should_run("zero", 0) is False
    assert scheduler.should_run("negative", 0) is False


def test_should_run_matches_tasks_for_frame_membership() -> None:
    # Given a scheduler with distinct task intervals
    scheduler = Scheduler({"pose": 1, "bed": 30})

    # Then should_run agrees with the frame's due-task tuple.
    assert scheduler.should_run("pose", 7) is True
    assert scheduler.should_run("bed", 7) is False
    assert scheduler.should_run("bed", 30) is True
    assert scheduler.should_run("unknown", 30) is False


def test_task_mapping_order_is_preserved_for_custom_configuration() -> None:
    # Given a custom mapping whose insertion order differs from task names' order
    scheduler = Scheduler({"bed": 2, "pose": 2, "alert": 2})

    # Then due tasks follow the supplied mapping order, not a sorted order.
    assert scheduler.tasks_for_frame(2) == ("bed", "pose", "alert")


def test_scheduler_instances_are_independent_per_camera() -> None:
    # Given separate schedulers for two camera pipelines
    camera_a = Scheduler({"pose": 1})
    camera_b = Scheduler({"bed": 30})

    # When the same frame index is evaluated independently
    # Then each camera sees only its own configured schedule.
    assert camera_a.tasks_for_frame(30) == ("pose",)
    assert camera_b.tasks_for_frame(30) == ("bed",)
    assert camera_a.task_intervals is not camera_b.task_intervals


def test_scheduler_is_frozen_and_slotted_dataclass_like() -> None:
    # Given a concrete scheduler instance
    scheduler = Scheduler()

    # Then it has the legacy frozen/slotted dataclass shape.
    assert is_dataclass(scheduler)
    assert not hasattr(scheduler, "__dict__")
    field_name = "task_intervals"
    with pytest.raises(FrozenInstanceError):
        setattr(scheduler, field_name, {"changed": 1})
