from __future__ import annotations

import hashlib
from pathlib import Path

from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    DerivativeArtifact,
    DerivativeKind,
)
from worker.pipeline.trace.models import DetailUnavailableReason
from worker.runtime.derivative_runtime import (
    DerivativeCommand,
    DerivativeCommandExecutor,
    DerivativeOutcome,
)
from worker.types.overlay_scene import (
    CoordinateTransform,
    ObservationSemantics,
    OverlayScene,
    SceneFrameIdentity,
    SceneValue,
)


class _Renderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(
        self, job: AnnotatedDerivativeJob, destination: Path, *, cancelled: object = None
    ) -> DerivativeArtifact:
        del cancelled
        self.calls += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"derivative")
        return DerivativeArtifact.from_path(
            destination,
            mime_type=job.derivative_kind.mime_type,
            width=1,
            height=1,
            start_time_ms=0,
            end_time_ms=0,
            render_backend="test",
            render_version="test",
            scene_id=job.scenes[0].scene_id,
        )


def _job(tmp_path: Path) -> AnnotatedDerivativeJob:
    source = tmp_path / "primary.mp4"
    source.write_bytes(b"primary")
    scene = OverlayScene(
        "scene",
        SceneFrameIdentity(
            "boot",
            "camera",
            1,
            1,
            SceneValue(0.0, ObservationSemantics.PRESENT),
            SceneValue(0.0, ObservationSemantics.PRESENT),
            "config",
        ),
        (1, 1),
        "source-pixels",
        CoordinateTransform(1, 1, 1, 1, 1.0, 1.0, 0.0, 0.0),
        (),
        (),
        (),
        (),
        (),
    )
    return AnnotatedDerivativeJob(
        "incident",
        "clip",
        source,
        hashlib.sha256(b"primary").hexdigest(),
        "a" * 64,
        "b" * 64,
        (scene,),
        len(b"primary"),
        derivative_kind=DerivativeKind.STILL,
    )


def test_same_command_returns_identical_receipt_without_double_production(tmp_path: Path) -> None:
    renderer = _Renderer()
    executor = DerivativeCommandExecutor(tmp_path / "store", still_renderer=renderer)
    command = DerivativeCommand(_job(tmp_path))

    first = executor.execute(command)
    second = executor.execute(command)

    assert first == second
    assert first.outcome is DerivativeOutcome.AVAILABLE
    assert renderer.calls == 1
    assert len(tuple((tmp_path / "store" / "derivatives" / "objects").glob("*.jpg"))) == 1


def test_trace_detail_loss_is_an_explicit_typed_unavailable_receipt(tmp_path: Path) -> None:
    renderer = _Renderer()
    command = DerivativeCommand(_job(tmp_path), DetailUnavailableReason.RETENTION_BOUND)

    receipt = DerivativeCommandExecutor(tmp_path / "store", still_renderer=renderer).execute(
        command
    )

    assert receipt.outcome is DerivativeOutcome.UNAVAILABLE
    assert receipt.reason == DetailUnavailableReason.RETENTION_BOUND.value
    assert renderer.calls == 0
