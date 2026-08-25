from __future__ import annotations

from test_derivative_job_store import _job, _Renderer

from worker.runtime.derivative_runtime import (
    DerivativeCommand,
    DerivativeCommandExecutor,
    DerivativeControlService,
    DerivativeOutcome,
)


def test_production_executor_returns_a_receipt(tmp_path):
    renderer = _Renderer()
    receipt = DerivativeCommandExecutor(tmp_path / "store", still_renderer=renderer).execute(
        DerivativeCommand(_job(tmp_path))
    )

    assert receipt.outcome is DerivativeOutcome.AVAILABLE
    assert receipt.artifact is not None
    assert renderer.calls == 1


def test_production_control_submits_event_derivative_asynchronously(tmp_path):
    renderer = _Renderer()
    control = DerivativeControlService(
        DerivativeCommandExecutor(tmp_path / "store", still_renderer=renderer)
    )

    receipt = control.submit(DerivativeCommand(_job(tmp_path))).result(timeout=2.0)

    assert receipt.outcome is DerivativeOutcome.AVAILABLE
    assert renderer.calls == 1
