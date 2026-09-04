"""Flow-profile activation order (P1b-AC4).

A pyservicemaker Flow fixes its sources when it is built, so the worker
registers the roster, starts the plane, and then requires one accepted
metadata frame before any pump or readiness exists. Engines are verified,
never built (ADR-0002); the first accepted frame is the real-batch warmup.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from worker.runtime.flow.cold_start import FlowWarmupTimeout
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.worker import WorkerRuntime


class _Metadata:
    def __init__(self, calls: list[str], *, bindings: dict[str, object], accept_after: int) -> None:
        self._calls = calls
        self._bindings = bindings
        self._remaining = accept_after

    def expected_binding(self, camera_id: str) -> object | None:
        return self._bindings.get(camera_id)

    def subscribe(self, binding: object) -> object:
        self._calls.append(f"subscribe:{binding}")
        return binding

    def wait_accepted(self, token: object, *, timeout_sec: float) -> object:
        assert timeout_sec == 1.0
        self._calls.append(f"wait:{token}")
        if self._remaining <= 0:
            raise TimeoutError("metadata binding deadline elapsed")
        self._remaining -= 1
        if self._remaining == 0:
            return object()
        raise TimeoutError("metadata binding deadline elapsed")


class _Plane:
    def __init__(self, *, bindings: dict[str, object], accept_after: int) -> None:
        self.calls: list[str] = []
        self.metadata = _Metadata(self.calls, bindings=bindings, accept_after=accept_after)


def _runtime(plane: _Plane) -> WorkerRuntime:
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime._shared_graph = object()  # noqa: SLF001 - isolated warmup branch
    runtime._boot = SimpleNamespace(profile=SimpleNamespace(name="flow"))  # noqa: SLF001
    runtime.fall_model = object()
    runtime._flow_media_plane = plane  # noqa: SLF001
    runtime._warm_one = lambda model, device: plane.calls.append(f"warm:{device}")
    runtime._warmed_component_ids = frozenset({"fall-classifier"})  # noqa: SLF001
    return runtime


def _pump(camera_id: str) -> SimpleNamespace:
    return SimpleNamespace(camera_id=camera_id)


def test_flow_warm_models_warms_only_the_cpu_fall_model() -> None:
    plane = _Plane(bindings={}, accept_after=0)
    runtime = _runtime(plane)
    assert runtime._warm_models() == ("fall-classifier",)  # noqa: SLF001
    # No bootstrap source, no subscription: the accepted-frame warmup belongs
    # to activation, after the roster is registered and the Flow is running.
    assert plane.calls == ["warm:cpu"]


def test_first_frame_from_any_roster_source_completes_the_warmup() -> None:
    plane = _Plane(bindings={"a": "bind-a", "b": "bind-b"}, accept_after=2)
    runtime = _runtime(plane)
    runtime._await_flow_first_frame([_pump("a"), _pump("b")])  # noqa: SLF001
    assert plane.calls == ["subscribe:bind-a", "subscribe:bind-b", "wait:bind-a", "wait:bind-b"]


def test_warmup_without_registered_sources_is_a_typed_boot_failure() -> None:
    plane = _Plane(bindings={}, accept_after=5)
    runtime = _runtime(plane)
    with pytest.raises(FlowWarmupTimeout, match="no registered source"):
        runtime._await_flow_first_frame([_pump("a")])  # noqa: SLF001


def test_warmup_times_out_typed_when_no_source_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 0.0, 31.0])
    monkeypatch.setattr("worker.runtime.worker.time.monotonic", lambda: next(clock))
    plane = _Plane(bindings={"a": "bind-a"}, accept_after=0)
    runtime = _runtime(plane)
    with pytest.raises(FlowWarmupTimeout, match="accepted metadata frame from any source"):
        runtime._await_flow_first_frame([_pump("a")])  # noqa: SLF001
    assert plane.calls == ["subscribe:bind-a", "wait:bind-a"]


def test_flow_status_tick_reads_actor_and_plane_counters() -> None:
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime.diagnostics = WorkerDiagnostics()
    runtime._flow_media_plane = SimpleNamespace(
        media_plane=SimpleNamespace(
            status=lambda: SimpleNamespace(
                sources=(SimpleNamespace(camera_id="camera-a"),),
                nvenc_sessions_active=2,
            )
        ),
        recorder_counters=lambda camera_id: (3, 4, 5),
    )

    runtime._refresh_flow_recording_telemetry()  # noqa: SLF001

    snapshot = runtime.diagnostics.snapshot().cameras[0]
    assert snapshot.smart_record_extended_total == 3
    assert snapshot.smart_record_extension_raced_total == 4
    assert snapshot.smart_record_start_refused_total == 5
    assert snapshot.nvenc_sessions_active == 2
