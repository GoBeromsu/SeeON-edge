

def test_upkeep_runs_under_continuous_traffic_not_only_when_the_queue_drains() -> None:
    """The load-bearing case: 13 cameras never let the queue go quiet.

    Disk-pressure suspension is cleared only by `rotate`, and an active clip on
    a stalled camera is closed only by `expire`. Both used to run exclusively in
    the `queue.Empty` branch, which needs 100ms of total silence across a queue
    shared by every camera. Thirteen cameras at 15fps put roughly 195 messages/s
    into it, so that branch effectively never fires in production and one
    healthy camera is enough to starve it.

    An idle-queue test passes happily against that bug, which is exactly what
    happened here: the first version of this fix was measured "working" on the
    live stack when in truth the restart had reset the flag.

    So this test never lets the queue drain.
    """
    import queue as queue_mod
    import threading

    from worker.pipeline.output.evidence.clip_actor_loop import run_actor_loop
    from worker.pipeline.output.evidence.clip_recorder_models import FrameMessage

    calls: list[bool] = []
    stop = threading.Event()

    class _ReleasablePacket:
        def release(self) -> None: ...

    class _NeverEmpty(queue_mod.Queue):
        """Always yields a frame; `queue.Empty` is never raised."""

        def __init__(self) -> None:
            super().__init__()
            self.served = 0

        def get(self, block: bool = True, timeout: float | None = None) -> object:
            del block, timeout
            self.served += 1
            if self.served > 400:
                stop.set()
                raise queue_mod.Empty
            return FrameMessage(packet=_ReleasablePacket())

        def task_done(self) -> None: ...

        def empty(self) -> bool:
            return self.served > 400

    class _Actor:
        def __init__(self) -> None:
            self.expiries = 0

        def handle_frame(self, message: object) -> None:
            del message

        def expire(self) -> None:
            self.expiries += 1

        def shutdown(self) -> None: ...

    def rotate(*, force: bool) -> None:
        calls.append(force)

    actor = _Actor()
    messages = _NeverEmpty()
    run_actor_loop(actor, messages, stop, rotate)

    assert messages.served > 400, "the queue must never have drained"
    assert calls, (
        "retention was never re-evaluated under continuous traffic, so disk "
        "pressure stays latched and no clip can ever be recorded again -- the "
        "queue.Empty branch cannot carry this, 13 cameras never let it fire"
    )
    assert not any(calls), (
        "upkeep must not force a rotate; rotate()'s own interval guard is what "
        "keeps this cheap"
    )
    assert actor.expiries, (
        "a stalled camera's active clip could never wall-clock expire while "
        "another camera kept the shared queue busy"
    )


def test_pressure_clears_under_continuous_traffic_without_a_restart(tmp_path, monkeypatch) -> None:
    """End to end: the real maintenance object, driven by the real loop.

    The two halves were each pinned separately -- the loop calls rotate on an
    elapsed deadline, and rotate assigns the current pressure state rather than
    OR-ing it. This composes them, because the claim that matters to a nursing
    home is neither half on its own: after the disk recovers, a suspended
    recorder must start admitting clips again WITHOUT a restart, while thirteen
    cameras are still feeding the shared queue.

    That last clause is what the first version of this fix got wrong and what a
    live measurement could not distinguish, since restarting resets the flag.
    """
    import queue as queue_mod
    import threading
    from pathlib import Path
    from typing import NamedTuple

    from worker.pipeline.output.evidence import clip_actor_loop as loop_module
    from worker.pipeline.output.evidence.clip_actor_loop import run_actor_loop
    from worker.pipeline.output.evidence.clip_maintenance import ClipMaintenance
    from worker.pipeline.output.evidence.clip_recorder_models import (
        ClipRecorderConfig,
        ClipRecorderStats,
        FrameMessage,
    )

    class _Usage(NamedTuple):
        total: int
        used: int
        free: int

    (tmp_path / "clips").mkdir()
    config = ClipRecorderConfig(store_dir=tmp_path, rotate_min_interval_seconds=0.0)
    stats = ClipRecorderStats()
    over = {"pressure": True}

    def disk_usage(_path: Path) -> _Usage:
        # 90% while pressured, then 10% once the operator reclaims space.
        return _Usage(total=100, used=90 if over["pressure"] else 10, free=10)

    maintenance = ClipMaintenance(
        config=config,
        stats=stats,
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=disk_usage,
    )

    # Drive the deadline from a fake clock. Real wall time would let 400
    # messages fly past in well under one upkeep interval, so the loop would
    # tick once and the test would prove nothing about recovery -- while in
    # production those same 400 messages span about two seconds.
    ticks = {"now": 0.0}

    def fake_monotonic() -> float:
        ticks["now"] += 0.02
        return ticks["now"]

    monkeypatch.setattr(loop_module.time, "monotonic", fake_monotonic)

    stop = threading.Event()

    class _ReleasablePacket:
        def release(self) -> None: ...

    class _NeverEmpty(queue_mod.Queue):
        """Thirteen cameras: the shared queue never goes quiet."""

        def __init__(self) -> None:
            super().__init__()
            self.served = 0

        def get(self, block: bool = True, timeout: float | None = None) -> object:
            del block, timeout
            self.served += 1
            if self.served == 200:
                # The suspension must be real before the disk recovers,
                # otherwise the clear below proves nothing.
                assert stats.recording_suspended, (
                    "disk pressure never suspended recording, so this test "
                    "cannot show that recovery lifts it"
                )
                over["pressure"] = False
            if self.served > 400:
                stop.set()
                raise queue_mod.Empty
            return FrameMessage(packet=_ReleasablePacket())

        def task_done(self) -> None: ...

        def empty(self) -> bool:
            return self.served > 400

    class _Actor:
        def handle_frame(self, message: object) -> None:
            del message

        def expire(self) -> None: ...

        def shutdown(self) -> None: ...

    run_actor_loop(_Actor(), _NeverEmpty(), stop, maintenance.rotate)

    assert not stats.recording_suspended, (
        "recording stayed suspended after the disk recovered, so every fall "
        "records no footage until someone restarts the worker"
    )
