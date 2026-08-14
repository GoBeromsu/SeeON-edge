"""Composition-root projection: adapter telemetry -> local diagnostics model.

Lives in ``worker.runtime`` (the sole package import-linter allows to depend
on both ``worker.adapters`` and its own ``worker.runtime.telemetry``
submodules) rather than in either endpoint: the adapter-level
``DeviceResidencyTelemetrySnapshot`` (``worker.adapters.decode.nvdec_device``)
must never depend on ``worker.runtime`` (forbidden by "worker adapters do not
depend on pipeline, domains, or runtime"), and the telemetry model
(``worker.runtime.telemetry.models``) stays a plain dataclass with no adapter
import of its own, matching every other ``CameraDiagnosticsSnapshot`` field's
convention (e.g. ``BedRegionDiagnostics`` is built the same way, in
``worker.pipeline.analytics.composite``, not in ``worker.runtime.telemetry``
itself -- this module is that same seam for a runtime-owned counter object).
"""

from __future__ import annotations

from collections.abc import Callable
from time import time

from worker.adapters.decode.nvdec_device.telemetry import DeviceResidencyTelemetrySnapshot
from worker.runtime.telemetry.models import DeviceResidencyDiagnostics


def device_residency_diagnostics(
    snapshot: DeviceResidencyTelemetrySnapshot,
    *,
    residency_path: str,
    unavailable_reason: str | None = None,
    wall_clock: Callable[[], float] = time,
) -> DeviceResidencyDiagnostics:
    """Project one pool's telemetry snapshot into the local diagnostics shape.

    ``residency_path`` names where in the pipeline this pool's frames stay
    device-resident (e.g. ``"decode->preprocess->inference"``) -- a plain
    string, not a memory-path object, matching every other local-only
    diagnostics field's convention of carrying only bounded, log-safe values.
    """
    if not residency_path:
        raise ValueError("residency_path must be non-empty")
    return DeviceResidencyDiagnostics(
        residency_path=residency_path,
        h2d_transfers=snapshot.h2d_transfers,
        h2d_bytes=snapshot.h2d_bytes,
        d2h_transfers=snapshot.d2h_transfers,
        d2h_bytes=snapshot.d2h_bytes,
        pool_capacity=snapshot.pool_capacity,
        pool_outstanding=snapshot.pool_outstanding,
        pool_high_watermark=snapshot.pool_high_watermark,
        pool_exhaustion_events=snapshot.pool_exhaustion_events,
        decode_time_ms_total=snapshot.decode_time_ms_total,
        decode_samples=snapshot.decode_samples,
        inference_time_ms_total=snapshot.inference_time_ms_total,
        inference_samples=snapshot.inference_samples,
        unavailable_reason=unavailable_reason,
        updated_at_sec=wall_clock(),
    )


__all__ = ["device_residency_diagnostics"]
