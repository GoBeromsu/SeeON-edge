"""A thread must not be visible to its own stop path before it has started.

Three separate runs of the default suite failed with::

    RuntimeError: cannot join thread before it is started

from ``tests/test_worker_zero_camera_boot.py``, and each time it was dismissed
as a load-dependent flake. It is not a flake. Every one of these components
assigned the thread to ``self._thread`` and only then called ``start()``. A
concurrent ``stop()`` landing inside that window sees a non-``None`` thread,
calls ``join()`` on it, and raises.

On a loaded edge device this window is a real shutdown hazard, so the ordering is
pinned here rather than left to timing. The rule is: build locally, start, then
publish.
"""

from __future__ import annotations

import ast
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from worker.pipeline.output.evidence.clip_recorder import ClipRecorder
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.pipeline.output.evidence.evidence_runtime import EvidenceExportRuntime
from worker.pipeline.output.evidence.evidence_sender import SenderStep
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig

_ROOT = Path(__file__).resolve().parents[1]

#: Every module that owns a ``self._thread`` lifecycle.
_THREAD_OWNERS = (
    "worker/pipeline/output/evidence/evidence_runtime.py",
    "worker/pipeline/output/evidence/clip_recorder.py",
    "worker/pipeline/output/mjpeg_server.py",
    "worker/pipeline/trace/writer.py",
)


def _is_self_thread(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_thread"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _thread_lifecycle_events(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[dict[str, int], dict[str, list[int]], dict[str, list[int]]]:
    """Return construction, publication, and start lines by local thread name."""
    constructed: dict[str, int] = {}
    published: dict[str, list[int]] = {}
    started: dict[str, list[int]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "Thread"
            ):
                constructed[node.targets[0].id] = node.lineno
            if isinstance(node.value, ast.Name) and any(
                _is_self_thread(target) for target in node.targets
            ):
                published.setdefault(node.value.id, []).append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start"
            and isinstance(node.func.value, ast.Name)
        ):
            started.setdefault(node.func.value.id, []).append(node.lineno)
    return constructed, published, started


def _publishes_before_start(tree: ast.AST) -> list[int]:
    """Return publication lines that precede their local thread's ``start``."""
    offenders: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(_is_self_thread(target) for target in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "Thread"
        ):
            offenders.append(node.lineno)
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        constructed, published, started = _thread_lifecycle_events(function)
        for name in constructed:
            for publication in published.get(name, []):
                if not any(start < publication for start in started.get(name, [])):
                    offenders.append(publication)
    return offenders


@pytest.mark.parametrize("relative", _THREAD_OWNERS)
def test_a_thread_is_started_before_it_becomes_visible(relative: str) -> None:
    """Every published local thread must have been started first."""
    source = (_ROOT / relative).read_text(encoding="utf-8")
    offenders = _publishes_before_start(ast.parse(source))

    assert not offenders, (
        f"{relative} assigns a freshly constructed Thread to self._thread at "
        f"line(s) {offenders}. A concurrent stop() would join an unstarted "
        f"thread and raise RuntimeError. Build into a local, start it, then "
        f"assign self._thread."
    )


@pytest.mark.parametrize("relative", _THREAD_OWNERS)
def test_the_started_thread_is_still_published(relative: str) -> None:
    """Every locally started lifecycle thread must be published afterwards."""
    source = (_ROOT / relative).read_text(encoding="utf-8")
    tree = ast.parse(source)
    unpublished: list[int] = []
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        constructed, published, started = _thread_lifecycle_events(function)
        for name in constructed:
            if started.get(name) and not any(
                publication > min(started[name])
                for publication in published.get(name, [])
            ):
                unpublished.extend(started[name])

    assert not unpublished, (
        f"{relative} starts local lifecycle thread(s) at line(s) {unpublished} "
        "without subsequently publishing them to self._thread, so stop() "
        "cannot join them."
    )


class _RetryingSender:
    def run_once(self) -> SenderStep:
        return SenderStep.RETRY_SCHEDULED


@pytest.mark.parametrize(
    ("factory", "start_name", "stop_name"),
    [
        (
            lambda path: EvidenceExportRuntime(
                store_dir=path,
                queue_directory=path,
                sender=_RetryingSender(),
            ),
            "start_sender",
            "stop_sender",
        ),
        (
            lambda path: ClipRecorder(ClipRecorderConfig(store_dir=path)),
            "start",
            "stop",
        ),
        (
            lambda _path: MjpegServer(
                LatestFrameStore(),
                MjpegServerConfig(port=0),
            ),
            "start",
            "stop",
        ),
    ],
)
def test_stop_waits_for_start_to_publish_its_started_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[Path], EvidenceExportRuntime | ClipRecorder | MjpegServer],
    start_name: str,
    stop_name: str,
) -> None:
    """A stop racing publication must join the thread which start already ran."""
    component = factory(tmp_path)
    if isinstance(component, EvidenceExportRuntime):
        component.initialize_under_lock()

    real_thread = threading.Thread
    started = threading.Event()
    release_start = threading.Event()
    joined: list[object] = []

    class ThreadAtPublicationBoundary:
        def __init__(
            self,
            *,
            target: Callable[..., object] | None,
            daemon: bool,
            name: str | None = None,
        ) -> None:
            self._inner = real_thread(target=target, daemon=daemon, name=name)

        def start(self) -> None:
            self._inner.start()
            started.set()
            assert release_start.wait(2)

        def join(self, timeout: float | None = None) -> None:
            joined.append(self)
            self._inner.join(timeout)

        def is_alive(self) -> bool:
            return self._inner.is_alive()

    module = __import__(type(component).__module__, fromlist=["threading"])
    monkeypatch.setattr(module.threading, "Thread", ThreadAtPublicationBoundary)

    start_thread = real_thread(target=getattr(component, start_name))
    start_thread.start()
    assert started.wait(2)
    stop_returned = threading.Event()

    def stop() -> None:
        getattr(component, stop_name)()
        stop_returned.set()

    stop_thread = real_thread(target=stop)
    stop_thread.start()
    assert not stop_returned.wait(0.05)
    release_start.set()
    start_thread.join(2)
    stop_thread.join(2)

    assert stop_returned.is_set()
    assert joined, "stop returned without joining the thread start had already started"
