from __future__ import annotations

from types import SimpleNamespace

import pytest

from worker.runtime.flow.cold_start import FlowWarmupTimeout
from worker.runtime.worker import WorkerRuntime


class _Metadata:
    def __init__(self, calls: list[str], *, timeout: bool = False) -> None:
        self._calls = calls
        self._timeout = timeout

    def subscribe(self, binding: object) -> object:
        self._calls.append("subscribe")
        return binding

    def wait_accepted(self, token: object, *, timeout_sec: float) -> object:
        del token
        assert timeout_sec == 10.0
        self._calls.append("accepted")
        if self._timeout:
            raise TimeoutError("metadata binding deadline elapsed")
        return object()


class _Plane:
    def __init__(self, *, timeout: bool = False) -> None:
        self.calls: list[str] = []
        self.metadata = _Metadata(self.calls, timeout=timeout)

    def add_source(self, camera_id: str, uri: str) -> object:
        assert (camera_id, uri) == ("_bootstrap_warmup", "loopback://bootstrap")
        self.calls.append("add_source")
        return object()

    def remove_source(self, camera_id: str) -> None:
        assert camera_id == "_bootstrap_warmup"
        self.calls.append("remove_source")


def _runtime(plane: _Plane) -> WorkerRuntime:
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime._shared_graph = object()  # noqa: SLF001 - isolated warmup branch
    runtime._boot = SimpleNamespace(profile=SimpleNamespace(name="flow"))  # noqa: SLF001
    runtime.fall_model = object()
    runtime._flow_media_plane = plane  # noqa: SLF001
    runtime._warm_one = lambda model, device: plane.calls.append(f"warm:{device}")
    runtime._warmed_component_ids = frozenset({"fall-classifier"})  # noqa: SLF001
    return runtime


def test_flow_warmup_orders_cpu_model_then_source_acceptance_and_removal() -> None:
    plane = _Plane()
    runtime = _runtime(plane)
    assert runtime._warm_models() == ("fall-classifier",)  # noqa: SLF001
    assert plane.calls == ["warm:cpu", "add_source", "subscribe", "accepted", "remove_source"]


def test_flow_warmup_timeout_is_a_typed_boot_failure_and_removes_source() -> None:
    plane = _Plane(timeout=True)
    runtime = _runtime(plane)
    with pytest.raises(FlowWarmupTimeout, match="accepted metadata frame"):
        runtime._warm_models()  # noqa: SLF001
    assert plane.calls == ["warm:cpu", "add_source", "subscribe", "accepted", "remove_source"]
