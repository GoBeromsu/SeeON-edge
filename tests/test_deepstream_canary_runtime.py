from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_publishers_wait_for_mediamtx_health_when_compose_is_rendered(tmp_path: Path) -> None:
    # Given: a fresh isolated run root.
    evidence = tmp_path / "canary"

    # When: one loopback publisher is generated.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worker.tools.deepstream_canary",
            "run",
            "--rungs",
            "zero,loopback",
            "--evidence-dir",
            str(evidence),
            "--render-only",
            "--worker-image",
            "seeon-edge@sha256:" + "b" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: publisher startup cannot race the digest-pinned RTSP server.
    assert completed.returncode == 0, completed.stderr
    rendered = (evidence / "compose.rendered.yaml").read_text()
    publisher = rendered.split("  publisher-01:", maxsplit=1)[1]
    assert "mediamtx:\n        condition: service_healthy" in publisher


def test_render_copies_prepared_content_addressed_engine_cache(tmp_path: Path) -> None:
    # Given: an immutable previously prepared plan cache.
    source = tmp_path / "prepared"
    plan = source / "c7-plan-key"
    plan.mkdir(parents=True)
    (plan / ".identity.json").write_text("{}\n")
    evidence = tmp_path / "reuse"

    # When: the isolated run is rendered with that explicit cache source.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worker.tools.deepstream_canary",
            "run",
            "--rungs",
            "zero,loopback",
            "--evidence-dir",
            str(evidence),
            "--render-only",
            "--engine-cache-source",
            str(source),
            "--worker-image",
            "seeon-edge@sha256:" + "b" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: a read-only source is mounted only into the explicit prepare service.
    assert completed.returncode == 0, completed.stderr
    rendered = (evidence / "compose.rendered.yaml").read_text()
    assert f"{source.resolve()}:/prepared-cache:ro" in rendered
    assert not (evidence / "run/engine-cache/c7-plan-key").exists()
    assert (plan / ".identity.json").is_file()


def test_engine_prepare_is_profiled_before_steady_capacity_gate(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.models import GatePolicy

    # Given: the tracked policy and a rendered canary project.
    evidence = tmp_path / "prepare"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worker.tools.deepstream_canary",
            "run",
            "--rungs",
            "zero,loopback",
            "--evidence-dir",
            str(evidence),
            "--render-only",
            "--worker-image",
            "seeon-edge@sha256:" + "b" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # When: preparation and steady-state declarations are inspected.
    assert completed.returncode == 0, completed.stderr
    rendered = (evidence / "compose.rendered.yaml").read_text()
    policy = GatePolicy.model_validate_json(
        Path("scripts/qa/deepstream-canary/gate-policy.v1.json").read_bytes()
    )

    # Then: engine build is explicit and does not weaken the unchanged runtime threshold.
    engine = rendered.split("  engine-builder:", maxsplit=1)[1].split(
        "  ml-worker:", maxsplit=1
    )[0]
    worker = rendered.split("  ml-worker:", maxsplit=1)[1].split(
        "  publisher-01:", maxsplit=1
    )[0]
    assert "profiles: [prepare]" in engine
    assert "engine-builder:" not in worker
    assert policy.engine_preparation.utilization_gate_active is False
    assert policy.gpu_utilization_absolute_max == 95.0


def test_runner_emits_verifiable_rung_receipt_from_recorded_telemetry(tmp_path: Path) -> None:
    from worker.tools.deepstream_canary.models import ArtifactBindings
    from worker.tools.deepstream_canary.telemetry import emit_rung_receipt

    # Given: recorded per-window telemetry and immutable artifact identities.
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    telemetry = Path("tests/fixtures_deepstream_canary_telemetry.json")
    artifacts = ArtifactBindings(
        worker_image="sha256:" + "1" * 64,
        support_images=("sha256:" + "2" * 64,),
        models_manifest="3" * 64,
        engine_manifest="4" * 64,
        corpus="5" * 64,
        gate_policy="6" * 64,
        compose="7" * 64,
    )

    # When: the runner's telemetry transformation emits a raw rung receipt.
    receipt = emit_rung_receipt(telemetry, evidence, artifacts)

    # Then: the independent verifier recomputes PASS without trusting execution status.
    assert receipt.name == "rung-loopback.json"
    policy_digest = hashlib.sha256(
        Path("scripts/qa/deepstream-canary/gate-policy.v1.json").read_bytes()
    ).hexdigest()
    (evidence / "run-request.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "requested_rungs": ["loopback"],
                "policy_sha256": policy_digest,
            }
        )
        + "\n"
    )
    verified = subprocess.run(
        [
            sys.executable,
            "scripts/qa/verify_deepstream_delivery.py",
            "canary",
            "--evidence-root",
            str(evidence),
            "--output",
            str(evidence / "verified.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert verified.returncode == 0, verified.stderr
    assert '"verdict":"PASS"' in verified.stdout


def test_authorization_rejects_camera_count_that_does_not_match_rung() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from worker.tools.deepstream_canary.authorization import AuthorizationRequest, authorize_rungs
    from worker.tools.deepstream_canary.models import AuthorizationArtifact, CanaryMode

    # Given: rung 4 authorization carrying only one camera identity.
    artifact = AuthorizationArtifact(
        schema_version=1,
        appliance_id="edge-lab-01",
        worker_image="sha256:" + "b" * 64,
        camera_ids=("camera-01",),
        owner="owner@example.invalid",
        issue="https://github.com/SeniorAILab/SeeON-edge/issues/381",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        authorized_rungs=(4,),
        projected_slack_mib=4096.0,
    )
    request = AuthorizationRequest(
        rungs=(4,),
        mode=CanaryMode.COMMISSIONING,
        appliance_id="edge-lab-01",
        worker_image_digest="sha256:" + "b" * 64,
        camera_ids=("camera-01",),
        now=datetime.now(UTC),
        minimum_projected_slack_mib=3072.0,
    )

    # When / Then: authorization fails before Docker.
    with pytest.raises(ValueError, match="camera_count_mismatch"):
        authorize_rungs(request, artifact)
