from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "worker.tools.deepstream_canary", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _policy(path: Path) -> Path:
    source = Path("scripts/qa/deepstream-canary/gate-policy.v1.json")
    path.write_bytes(source.read_bytes())
    return path


def _passing_receipt(camera_count: int = 1) -> dict[str, object]:
    cameras = [
        {
            "camera_id": f"loop-{index:02d}",
            "fps_windows": [14.8, 15.0, 15.2],
            "latency_ms": {"p50": 90.0, "p95": 150.0, "p99": 190.0, "max": 220.0},
            "au_gaps": 0,
            "config_discontinuities": 0,
            "timestamp_discontinuities": 0,
            "metadata_published": 1000,
            "metadata_overwritten": 10,
            "event_evidence_parity": True,
            "preview_ok": True,
            "derivative_ok": True,
        }
        for index in range(camera_count)
    ]
    return {
        "schema_version": 1,
        "rung": "loopback" if camera_count == 1 else str(camera_count),
        "mode": "commissioning",
        "camera_count": camera_count,
        "clean_steady_seconds": 900 if camera_count == 1 else 7200,
        "cameras": cameras,
        "gpu": {
            "child_pid": 222,
            "warmup_peak_mib": 5000.0,
            "steady_p50_mib": 4500.0,
            "steady_p95_mib": 4700.0,
            "recovery_mib": 4300.0,
            "global_used_mib": 5000.0,
            "total_mib": 16384.0,
            "minimum_slack_mib": 11000.0,
            "utilization_p50": 35.0,
            "utilization_p95": 55.0,
            "utilization_max": 70.0,
            "new_xids": [],
        },
        "nvdec": {"hardware_branches": camera_count, "software_fallbacks": 0},
        "timeline": [
            {"kind": kind, "sha256": character * 64, "playable": True}
            for kind, character in (
                ("event", "a"),
                ("evidence", "b"),
                ("preview", "c"),
                ("derivative", "d"),
            )
        ],
        "live_protection": {
            "container_restarts": 0,
            "camera_stale_transitions": 0,
            "evidence_drop_increase": 0,
            "relay_sentinel_leaks": 0,
            "mount_intersections": 0,
            "kernel_faults": 0,
        },
        "fault_windows": [],
        "workload": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 15.0,
            "gop": 30,
            "camera_phase_offsets_ms": [index * 67 for index in range(camera_count)],
        },
        "artifacts": {
            "worker_image": "sha256:" + "b" * 64,
            "support_images": ("sha256:" + "e" * 64,),
            "models_manifest": "f" * 64,
            "engine_manifest": "1" * 64,
            "corpus": "2" * 64,
            "gate_policy": "3" * 64,
            "compose": "4" * 64,
        },
    }


def test_cli_renders_isolated_zero_and_loopback_compose(tmp_path: Path) -> None:
    # Given: no facility authorization and a fresh run directory.
    evidence = tmp_path / "run"

    # When: the default non-facility contract is rendered without starting Docker.
    result = _run_cli(
        "run",
        "--rungs",
        "zero,loopback",
        "--evidence-dir",
        str(evidence),
        "--render-only",
        "--worker-image",
        "seeon-edge@sha256:" + "b" * 64,
    )

    # Then: compose is isolated, publisher-local, loopback-bound, and immutable.
    assert result.returncode == 0, result.stderr
    rendered = (evidence / "compose.rendered.yaml").read_text()
    assert "name: seeon-ds-canary" in rendered
    assert "internal: true" in rendered
    assert '127.0.0.1:18090:8090' in rendered
    assert "publisher-01:" in rendered
    assert "ml-api" in rendered
    assert (evidence / "receipt-manifest.json").is_file()
    compose_check = subprocess.run(
        ["docker", "compose", "-f", str(evidence / "compose.rendered.yaml"), "config", "-q"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CANARY_RELAY_TOKEN": "test-only-render-token"},
        timeout=30,
    )
    assert compose_check.returncode == 0, compose_check.stderr


def test_live_rung_refuses_without_authorization_before_docker(tmp_path: Path) -> None:
    # Given: a requested facility rung without an owner artifact.
    evidence = tmp_path / "unauthorized"

    # When: the operator requests rung 1.
    result = _run_cli(
        "run",
        "--rungs",
        "1",
        "--evidence-dir",
        str(evidence),
        "--render-only",
        "--worker-image",
        "seeon-edge@sha256:" + "b" * 64,
    )

    # Then: the request fails closed and records the refusal.
    assert result.returncode == 2
    refusal = json.loads((evidence / "authorization-refusal.json").read_text())
    assert refusal["code"] == "authorization_required"


def test_rung_13_requires_current_authorization_eight_pass_and_slack(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.authorization import AuthorizationRequest, authorize_rungs
    from worker.tools.deepstream_canary.models import AuthorizationArtifact, CanaryMode

    # Given: a current artifact lacking an 8-pass binding and projected slack.
    artifact = AuthorizationArtifact(
        schema_version=1,
        appliance_id="edge-lab-01",
        worker_image="sha256:" + "b" * 64,
        camera_ids=tuple(f"camera-{index:02d}" for index in range(13)),
        owner="owner@example.invalid",
        issue="https://github.com/SeniorAILab/SeeON-edge/issues/381",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        authorized_rungs=(1, 4, 8, 13),
        eight_pass_report_sha256=None,
        projected_slack_mib=3071.0,
    )

    request = AuthorizationRequest(
        rungs=(13,),
        mode=CanaryMode.COMMISSIONING,
        appliance_id="edge-lab-01",
        worker_image_digest="sha256:" + "b" * 64,
        camera_ids=tuple(f"camera-{index:02d}" for index in range(13)),
        now=datetime.now(UTC),
        minimum_projected_slack_mib=3072.0,
    )

    # When / Then: rung 13 is refused independently of Compose execution.
    with pytest.raises(ValueError, match="eight_pass_report_required"):
        authorize_rungs(request, artifact)


def test_mount_guard_rejects_live_volume_ancestor(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.safety import (
        LiveContainer,
        LiveSnapshot,
        refuse_mount_overlap,
    )

    # Given: a canary state directory nested under a live bind mount.
    live = tmp_path / "live"
    snapshot = LiveSnapshot((LiveContainer("live-1", 0, (live,)),), 0)

    # When / Then: mount admission fails before Compose starts.
    with pytest.raises(RuntimeError, match="live_mount_intersection"):
        refuse_mount_overlap((live / "canary",), snapshot)


def test_live_gpu_protection_attributes_container_and_allows_pid_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worker.tools.deepstream_canary.safety as safety_module
    from worker.tools.deepstream_canary.safety import (
        LiveContainer,
        LiveSnapshot,
        SafetyLimits,
        compare_live_snapshot,
    )

    # Given: one protected SeeON GPU container plus an unrelated host process.
    before = LiveSnapshot(
        containers=(LiveContainer("live-1", 0, (), True, ("123",)),),
        xid_count=0,
        gpu_processes=("123, python, 100", "999, desktop, 10"),
    )
    after = LiveSnapshot(
        containers=(LiveContainer("live-1", 0, (), True, ("456",)),),
        xid_count=0,
        gpu_processes=("456, ffmpeg, 100",),
    )
    monkeypatch.setattr(safety_module, "capture_live_snapshot", lambda: after)

    # When / Then: unrelated exit and attributed in-container PID churn are allowed.
    observed: dict[str, float] = {}
    compare_live_snapshot(before, SafetyLimits(0, 100, False), observed)
    assert observed == {}


def test_live_gpu_protection_applies_bounded_container_loss_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worker.tools.deepstream_canary.safety as safety_module
    from worker.tools.deepstream_canary.safety import (
        LiveContainer,
        LiveSnapshot,
        SafetyLimits,
        compare_live_snapshot,
    )

    # Given: a protected SeeON container temporarily has no attributed GPU process.
    before = LiveSnapshot((LiveContainer("live-1", 0, (), True, ("123",)),), 0)
    after = LiveSnapshot((LiveContainer("live-1", 0, (), True, ()),), 0)
    monkeypatch.setattr(safety_module, "capture_live_snapshot", lambda: after)
    clock = iter((0.0, 31.0))
    monkeypatch.setattr(safety_module.time, "monotonic", lambda: next(clock))
    observed: dict[str, float] = {}
    limits = SafetyLimits(0, 100, False, gpu_process_loss_grace_seconds=30)

    # When / Then: the first miss is tolerated, but sustained loss fails closed.
    compare_live_snapshot(before, limits, observed)
    with pytest.raises(RuntimeError, match="live_container_gpu_process_missing"):
        compare_live_snapshot(before, limits, observed)


def test_verifier_recomputes_pass_and_rejects_intentional_red(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.gates import evaluate_receipt
    from worker.tools.deepstream_canary.models import GatePolicy, RungReceipt

    # Given: raw per-camera signals and the versioned policy.
    policy = GatePolicy.model_validate_json(_policy(tmp_path / "policy.json").read_bytes())
    passing = RungReceipt.model_validate(_passing_receipt())

    # When: gate math is recomputed.
    report = evaluate_receipt(passing, policy)

    # Then: clean signals pass, while a missing PTS mapping is an intentional RED.
    assert report.verdict == "PASS"
    malformed = passing.model_dump_json().replace(
        '\"latency_ms\":{\"p50\":90.0,\"p95\":150.0,\"p99\":190.0,\"max\":220.0},',
        "",
    )
    with pytest.raises(ValueError):
        RungReceipt.model_validate_json(malformed)


def test_product_image_explicitly_excludes_canary_module() -> None:
    # Given / When: Docker build inputs are inspected.
    dockerignore = Path(".dockerignore").read_text()
    dockerfile = Path("Dockerfile.edge").read_text()

    # Then: COPY worker cannot ship the operator-only harness.
    assert "worker/tools/deepstream_canary" in dockerignore
    assert "test ! -e /app/worker/tools/deepstream_canary" in dockerfile


def _verify(kind: str, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/qa/verify_deepstream_delivery.py",
            kind,
            "--evidence-root",
            str(root),
            "--output",
            str(root / f"{kind}.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _render_evidence(root: Path) -> None:
    result = _run_cli(
        "run",
        "--rungs",
        "zero,loopback",
        "--evidence-dir",
        str(root),
        "--render-only",
        "--worker-image",
        "seeon-edge@sha256:" + "b" * 64,
    )
    assert result.returncode == 0


def test_canary_verifier_rejects_missing_requested_rung(tmp_path: Path) -> None:
    # Given: immutable zero+loopback intent but only a valid zero receipt.
    root = tmp_path / "missing-loopback"
    (root / "raw").mkdir(parents=True)
    zero = _passing_receipt(camera_count=0)
    zero.update(
        {
            "rung": "zero",
            "clean_steady_seconds": 120,
            "nvdec": {"hardware_branches": 0, "software_fallbacks": 0},
            "timeline": [],
            "workload": {
                "codec": "h264",
                "width": 1280,
                "height": 720,
                "fps": 15.0,
                "gop": 30,
                "camera_phase_offsets_ms": [],
            },
        }
    )
    (root / "raw" / "rung-zero.json").write_text(json.dumps(zero))
    policy_digest = hashlib.sha256(
        Path("scripts/qa/deepstream-canary/gate-policy.v1.json").read_bytes()
    ).hexdigest()
    (root / "run-request.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "requested_rungs": ["zero", "loopback"],
                "policy_sha256": policy_digest,
            }
        )
        + "\n"
    )

    # When: the independent verifier evaluates the immutable requested set.
    result = _verify("canary", root)

    # Then: an available zero PASS cannot hide the absent loopback receipt.
    assert result.returncode == 1
    verdict = json.loads((root / "canary.json").read_text())
    assert verdict["findings"] == ["requested_rung_missing:loopback"]


def test_canary_verifier_rejects_low_per_camera_fps(tmp_path: Path) -> None:
    # Given: a raw receipt whose aggregate shape looks complete but camera FPS is below policy.
    root = tmp_path / "canary-red"
    (root / "raw").mkdir(parents=True)
    receipt = json.dumps(_passing_receipt()).replace(
        '"fps_windows": [14.8, 15.0, 15.2]', '"fps_windows": [1.0, 1.0, 1.0]'
    )
    (root / "raw" / "rung-loopback.json").write_text(receipt)

    # When: the independent canary verifier recomputes gate math.
    result = _verify("canary", root)

    # Then: intentional RED exits non-zero and cannot trust a self-reported PASS.
    assert result.returncode == 1
    assert json.loads((root / "canary.json").read_text())["verdict"] == "FAIL"


def test_compliance_verifier_rejects_receipt_digest_tamper(tmp_path: Path) -> None:
    # Given: rendered evidence whose tracked compose bytes are changed after manifesting.
    root = tmp_path / "compliance-red"
    _render_evidence(root)
    with (root / "compose.rendered.yaml").open("a") as target:
        _ = target.write("# tampered\n")

    # When: the independent compliance verifier hashes raw bytes.
    result = _verify("compliance", root)

    # Then: intentional RED is detected.
    assert result.returncode == 1


def test_scope_verifier_rejects_non_loopback_port(tmp_path: Path) -> None:
    # Given: a rendered compose with a public host bind.
    root = tmp_path / "scope-red"
    _render_evidence(root)
    compose = root / "compose.rendered.yaml"
    compose.write_text(compose.read_text().replace("127.0.0.1:18090", "0.0.0.0:18090"))

    # When: scope invariants are independently checked.
    result = _verify("scope", root)

    # Then: intentional RED is detected.
    assert result.returncode == 1


def test_quality_verifier_rejects_failed_gate_report(tmp_path: Path) -> None:
    # Given: an independently generated FAIL report from a low-FPS raw receipt.
    root = tmp_path / "quality-red"
    (root / "raw").mkdir(parents=True)
    receipt = json.dumps(_passing_receipt()).replace(
        '"fps_windows": [14.8, 15.0, 15.2]', '"fps_windows": [1.0]'
    )
    (root / "raw" / "rung-loopback.json").write_text(receipt)
    assert _verify("canary", root).returncode == 1

    # When: the quality verifier evaluates the canonical report.
    result = _verify("quality", root)

    # Then: intentional RED is detected.
    assert result.returncode == 1


def test_report_filename_is_content_addressed(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.report import write_canonical_report

    # Given: a machine-consumed report.
    report = {"schema_version": 1, "verdict": "PASS", "checks": []}

    # When: it is persisted.
    path = write_canonical_report(tmp_path, report)

    # Then: the filename digest equals the canonical bytes.
    assert path.name == f"gate-report.{hashlib.sha256(path.read_bytes()).hexdigest()}.json"
    assert os.stat(path).st_mode & 0o222 == 0
