"""Flow lifecycle supervision consumes accepted metadata liveness."""

from __future__ import annotations

from types import SimpleNamespace

from worker.runtime.flow.lifecycle_supervisor import FlowLifecycleSupervisor


class _Metadata:
    """Only the pump reads the slot; the supervisor must not."""

    def __init__(self) -> None:
        self.frames: dict[str, SimpleNamespace] = {}

    def peek(self, camera_id: str) -> SimpleNamespace | None:
        raise AssertionError(
            "the supervisor must read the plane's published-frame counter, not the "
            f"capacity-one slot the pump drains ({camera_id})"
        )


class _Plane:
    def __init__(self) -> None:
        self.metadata = _Metadata()
        self.published: dict[str, int] = {}
        self.fatal_error: str | None = None
        self.failures: list[str] = []
        self.previews_cleared: list[str] = []
        self.epoch = 1

    def published_frames(self, camera_id: str) -> int:
        return self.published.get(camera_id, 0)

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(fatal_error=self.fatal_error)

    def source_failure(self, camera_id: str, category: str) -> SimpleNamespace:
        assert category == "metadata_silence"
        self.failures.append(camera_id)
        self.epoch += 1
        self.metadata.frames.pop(camera_id, None)
        self.published.pop(camera_id, None)
        return SimpleNamespace(stream_epoch=self.epoch)

    def clear_preview(self, camera_id: str) -> None:
        self.previews_cleared.append(camera_id)


def test_fatal_plane_escalates_once() -> None:
    plane = _Plane()
    plane.fatal_error = "Flow thread stopped"
    fatal: list[str] = []
    supervisor = FlowLifecycleSupervisor(
        plane,
        ["camera-a"],
        on_ready=lambda _id: None,
        on_unready=lambda _id: None,
        on_fatal=fatal.append,
    )

    supervisor.tick()
    supervisor.tick()

    assert fatal == ["Flow thread stopped"]


def test_silence_rotates_once_then_recovery_marks_ready_and_advances_pump_binding() -> None:
    plane = _Plane()
    now = [0.0]
    ready: list[str] = []
    unready: list[str] = []
    supervisor = FlowLifecycleSupervisor(
        plane,
        ["camera-a"],
        on_ready=ready.append,
        on_unready=unready.append,
        on_fatal=lambda _error: None,
        silence_timeout_sec=30.0,
        clock=lambda: now[0],
    )
    plane.published["camera-a"] = 1
    supervisor.tick()

    now[0] = 30.0
    supervisor.tick()
    supervisor.tick()

    assert plane.failures == ["camera-a"]
    assert plane.previews_cleared == ["camera-a"]
    assert unready == ["camera-a"]
    assert supervisor.counters("camera-a").outages == 1

    plane.published["camera-a"] = 2
    now[0] = 31.0
    supervisor.tick()

    assert ready == ["camera-a"]
    assert supervisor.counters("camera-a").recoveries == 1
    assert plane.epoch == 2


def test_camera_that_keeps_publishing_is_not_rotated() -> None:
    plane = _Plane()
    now = [0.0]
    supervisor = FlowLifecycleSupervisor(
        plane,
        ["camera-a"],
        on_ready=lambda _id: None,
        on_unready=lambda _id: None,
        on_fatal=lambda _error: None,
        silence_timeout_sec=30.0,
        clock=lambda: now[0],
    )
    for sequence, tick in ((1, 0.0), (2, 29.0), (3, 58.0)):
        now[0] = tick
        plane.published["camera-a"] = sequence
        supervisor.tick()

    assert plane.failures == []
    assert supervisor.counters("camera-a").outages == 0


def test_shutdown_stops_the_flow_before_clearing_its_fixed_roster() -> None:
    """A running Flow's sources are fixed, so order matters on the way down.

    Removing sources first raises SourceRosterFixed and aborts the shutdown,
    which a live run surfaced as a runtime error on every stop.
    """
    from worker.interfaces.media_plane import SourceRosterFixed

    order: list[str] = []

    class _Plane:
        def __init__(self) -> None:
            self.running = True

        def stop(self) -> None:
            order.append("stop")
            self.running = False

        def remove_source(self, camera_id: str) -> None:
            if self.running:
                raise SourceRosterFixed(f"roster is fixed while running: {camera_id}")
            order.append(f"remove:{camera_id}")

    plane = _Plane()
    plane.stop()
    plane.remove_source("camera-a")

    assert order == ["stop", "remove:camera-a"]
