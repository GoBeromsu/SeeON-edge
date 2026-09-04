"""Independent binary capacity-gate math over immutable raw receipts."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from worker.tools.deepstream_canary.models import (
    CanaryMode,
    GateCheck,
    GatePolicy,
    GateReport,
    RungReceipt,
)


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _check(name: str, passed: bool, actual: float | int | bool, required: str) -> GateCheck:
    return GateCheck(name=name, passed=passed, actual=str(actual), required=required)


def _expected_camera_count(rung: str) -> int:
    match rung:
        case "zero":
            return 0
        case "loopback":
            return 1
        case "1" | "4" | "8" | "13":
            return int(rung)
        case _:
            return -1


def _required_duration(receipt: RungReceipt, policy: GatePolicy) -> int:
    match receipt.rung:
        case "zero":
            return policy.zero_clean_seconds
        case "loopback":
            return policy.loopback_clean_seconds
        case "1" | "4":
            return policy.standard_rung_clean_seconds
        case "8" | "13":
            return policy.candidate_rung_clean_seconds
        case _:
            return policy.candidate_rung_clean_seconds


def _camera_checks(receipt: RungReceipt, policy: GatePolicy) -> Iterable[GateCheck]:
    for camera in receipt.cameras:
        prefix = f"camera.{camera.camera_id}"
        yield _check(
            f"{prefix}.fps_p05",
            _percentile(camera.fps_windows, 0.05) >= policy.fps_p05_min,
            _percentile(camera.fps_windows, 0.05),
            f">={policy.fps_p05_min}",
        )
        yield _check(
            f"{prefix}.fps_p50",
            _percentile(camera.fps_windows, 0.50) >= policy.fps_p50_min,
            _percentile(camera.fps_windows, 0.50),
            f">={policy.fps_p50_min}",
        )
        yield _check(
            f"{prefix}.fps_p95",
            _percentile(camera.fps_windows, 0.95) >= policy.fps_p95_min,
            _percentile(camera.fps_windows, 0.95),
            f">={policy.fps_p95_min}",
        )
        for name, actual, maximum in (
            ("p50", camera.latency_ms.p50, policy.latency_p50_max_ms),
            ("p95", camera.latency_ms.p95, policy.latency_p95_max_ms),
            ("p99", camera.latency_ms.p99, policy.latency_p99_max_ms),
            ("max", camera.latency_ms.max, policy.latency_absolute_max_ms),
        ):
            yield _check(f"{prefix}.latency_{name}", actual <= maximum, actual, f"<={maximum}")
        yield _check(f"{prefix}.au_gaps", camera.au_gaps == 0, camera.au_gaps, "0")
        yield _check(
            f"{prefix}.config_discontinuities",
            camera.config_discontinuities == 0,
            camera.config_discontinuities,
            "0",
        )
        yield _check(
            f"{prefix}.timestamp_discontinuities",
            camera.timestamp_discontinuities == 0,
            camera.timestamp_discontinuities,
            "0",
        )
        overwrite_fraction = camera.metadata_overwritten / camera.metadata_published
        yield _check(
            f"{prefix}.metadata_overwrite_fraction",
            overwrite_fraction <= policy.metadata_overwrite_fraction_max,
            overwrite_fraction,
            f"<={policy.metadata_overwrite_fraction_max}",
        )
        yield _check(
            f"{prefix}.event_evidence_parity",
            camera.event_evidence_parity,
            camera.event_evidence_parity,
            "true",
        )
        yield _check(f"{prefix}.preview", camera.preview_ok, camera.preview_ok, "true")
        yield _check(f"{prefix}.derivative", camera.derivative_ok, camera.derivative_ok, "true")


def evaluate_receipt(receipt: RungReceipt, policy: GatePolicy) -> GateReport:
    """Recompute a binary verdict; missing required fields fail during parsing."""
    expected_count = _expected_camera_count(receipt.rung)
    checks = [
        _check(
            "camera_count",
            receipt.camera_count == expected_count == len(receipt.cameras),
            receipt.camera_count,
            str(expected_count),
        ),
        _check(
            "clean_steady_seconds",
            receipt.clean_steady_seconds >= _required_duration(receipt, policy),
            receipt.clean_steady_seconds,
            f">={_required_duration(receipt, policy)}",
        ),
        *_camera_checks(receipt, policy),
        _check(
            "workload.phase_offsets",
            len(receipt.workload.camera_phase_offsets_ms) == receipt.camera_count,
            len(receipt.workload.camera_phase_offsets_ms),
            str(receipt.camera_count),
        ),
        _check(
            "gpu.warmup_peak_mib",
            receipt.gpu.warmup_peak_mib <= policy.gpu_warmup_peak_max_mib,
            receipt.gpu.warmup_peak_mib,
            f"<={policy.gpu_warmup_peak_max_mib}",
        ),
        _check(
            "gpu.steady_p95_mib",
            receipt.gpu.steady_p95_mib <= policy.gpu_steady_p95_max_mib,
            receipt.gpu.steady_p95_mib,
            f"<={policy.gpu_steady_p95_max_mib}",
        ),
        _check(
            "gpu.recovery_mib",
            receipt.gpu.recovery_mib <= policy.gpu_recovery_max_mib,
            receipt.gpu.recovery_mib,
            f"<={policy.gpu_recovery_max_mib}",
        ),
        _check(
            "gpu.minimum_slack_mib",
            receipt.gpu.minimum_slack_mib >= policy.minimum_gpu_slack_mib,
            receipt.gpu.minimum_slack_mib,
            f">={policy.minimum_gpu_slack_mib}",
        ),
        _check(
            "gpu.utilization_p95",
            receipt.gpu.utilization_p95 <= policy.gpu_utilization_p95_max,
            receipt.gpu.utilization_p95,
            f"<={policy.gpu_utilization_p95_max}",
        ),
        _check(
            "gpu.utilization_max",
            receipt.gpu.utilization_max <= policy.gpu_utilization_absolute_max,
            receipt.gpu.utilization_max,
            f"<={policy.gpu_utilization_absolute_max}",
        ),
        _check("gpu.new_xids", not receipt.gpu.new_xids, len(receipt.gpu.new_xids), "0"),
        _check(
            "nvdec.hardware_branches",
            receipt.nvdec.hardware_branches == receipt.camera_count,
            receipt.nvdec.hardware_branches,
            str(receipt.camera_count),
        ),
        _check(
            "nvdec.software_fallbacks",
            receipt.nvdec.software_fallbacks == 0,
            receipt.nvdec.software_fallbacks,
            "0",
        ),
    ]
    protection_values = (
        ("container_restarts", receipt.live_protection.container_restarts),
        ("camera_stale_transitions", receipt.live_protection.camera_stale_transitions),
        ("evidence_drop_increase", receipt.live_protection.evidence_drop_increase),
        ("relay_sentinel_leaks", receipt.live_protection.relay_sentinel_leaks),
        ("mount_intersections", receipt.live_protection.mount_intersections),
        ("kernel_faults", receipt.live_protection.kernel_faults),
    )
    for name, actual in protection_values:
        checks.append(_check(f"live_protection.{name}", actual == 0, actual, "0"))
    timeline_kinds = {entry.kind for entry in receipt.timeline if entry.playable}
    required_timeline = {"event", "evidence", "preview", "derivative"}
    timeline_passed = receipt.rung == "zero" or required_timeline <= timeline_kinds
    checks.append(
        _check(
            "timeline.playable_and_hashed",
            timeline_passed,
            timeline_passed,
            "true",
        )
    )
    verdict = "PASS" if all(check.passed for check in checks) else "FAIL"
    policy_digest = hashlib.sha256(policy.model_dump_json().encode()).hexdigest()
    return GateReport(
        schema_version=1,
        verdict=verdict,
        rung=receipt.rung,
        claim_eligible=verdict == "PASS" and receipt.mode is CanaryMode.COMMISSIONING,
        gate_policy_sha256=policy_digest,
        checks=tuple(checks),
    )
