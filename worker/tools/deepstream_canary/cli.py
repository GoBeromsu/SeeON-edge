"""CLI boundary for isolated canary rendering, authorization, and execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, final

from pydantic import ValidationError

from worker.tools.deepstream_canary.authorization import (
    AuthorizationRequest,
    CanaryAuthorizationError,
    authorize_rungs,
)
from worker.tools.deepstream_canary.compose import RenderRequest, render_compose
from worker.tools.deepstream_canary.models import AuthorizationArtifact, CanaryMode, GatePolicy
from worker.tools.deepstream_canary.report import JsonValue, canonical_json, write_once
from worker.tools.deepstream_canary.runner import (
    ExecutionArtifactSources,
    ExecutionRequest,
    execute_canary,
)
from worker.tools.deepstream_canary.safety import (
    CanarySafetyError,
    SafetyLimits,
    capture_live_snapshot,
    encode_live_snapshot,
    refuse_existing_project,
    refuse_mount_overlap,
)

DEFAULT_WORKER_IMAGE: Final = (
    "seeon-edge@sha256:23b9cc63520aef5106d828b6db8f481db66aeaaead66b66c61b509135ace33ad"
)
POLICY_PATH: Final = Path("scripts/qa/deepstream-canary/gate-policy.v1.json")
SUPPORT_IMAGE_DIGESTS: Final = (
    "sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a",
    "sha256:2001b73159ec146478df64d5fe5a973ae0d978a1648138cbbc6bc6f9c4cc9c82",
)


class CanaryArgumentError(ValueError):
    """Invalid operator input that must fail before Docker."""


@final
class RunArguments(argparse.Namespace):
    command: str = "run"
    rungs: str = ""
    evidence_dir: Path = Path()
    mode: str = CanaryMode.COMMISSIONING
    authorization: Path | None = None
    worker_image: str = DEFAULT_WORKER_IMAGE
    model_dir: Path = Path("models")
    engine_cache_source: Path | None = None
    policy: Path = POLICY_PATH
    appliance_id: str = "unbound-canary-appliance"
    camera_ids: str = ""
    render_only: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m worker.tools.deepstream_canary")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    _ = run.add_argument("--rungs", required=True)
    _ = run.add_argument("--evidence-dir", type=Path, required=True)
    _ = run.add_argument("--mode", choices=tuple(CanaryMode), default=CanaryMode.COMMISSIONING)
    _ = run.add_argument("--authorization", type=Path)
    _ = run.add_argument("--worker-image", default=DEFAULT_WORKER_IMAGE)
    _ = run.add_argument("--model-dir", type=Path, default=Path("models"))
    _ = run.add_argument("--engine-cache-source", type=Path)
    _ = run.add_argument("--policy", type=Path, default=POLICY_PATH)
    _ = run.add_argument("--appliance-id", default="unbound-canary-appliance")
    _ = run.add_argument("--camera-ids", default="")
    _ = run.add_argument("--render-only", action="store_true")
    return parser


def _parse_rungs(raw: str) -> tuple[str, ...]:
    rungs = tuple(item.strip() for item in raw.split(",") if item.strip())
    allowed = {"zero", "loopback", "1", "4", "8", "13"}
    if not rungs or any(item not in allowed for item in rungs):
        raise CanaryArgumentError("rungs must be a comma list from zero,loopback,1,4,8,13")
    if "13" in rungs and rungs[-1] != "13":
        raise CanaryArgumentError("rung 13 must be last")
    return rungs


def _load_authorization(path: Path | None) -> AuthorizationArtifact | None:
    if path is None:
        return None
    return AuthorizationArtifact.model_validate_json(path.read_bytes())


def _image_digest(image: str) -> str:
    marker = "@sha256:"
    if marker not in image:
        raise CanaryArgumentError("worker image must be digest-pinned")
    digest = image.split(marker, maxsplit=1)[1]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CanaryArgumentError("worker image digest is invalid")
    return f"sha256:{digest}"


def _rung_durations(rungs: tuple[str, ...], policy: GatePolicy) -> tuple[tuple[str, int], ...]:
    durations: list[tuple[str, int]] = []
    for rung in rungs:
        match rung:
            case "zero":
                seconds = policy.zero_clean_seconds
            case "loopback":
                seconds = policy.loopback_clean_seconds
            case "1" | "4":
                seconds = policy.warmup_seconds + policy.standard_rung_clean_seconds
            case "8" | "13":
                seconds = policy.warmup_seconds + policy.candidate_rung_clean_seconds
            case unreachable:
                raise CanaryArgumentError(f"unsupported rung {unreachable}")
        durations.append((rung, seconds))
    return tuple(durations)


def _run(arguments: RunArguments) -> int:
    evidence_dir: Path = arguments.evidence_dir
    evidence_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    rungs = _parse_rungs(arguments.rungs)
    mode = CanaryMode(arguments.mode)
    policy_path = arguments.policy.resolve()
    policy_content = policy_path.read_bytes()
    policy = GatePolicy.model_validate_json(policy_content)
    policy_digest = hashlib.sha256(policy_content).hexdigest()
    worker_image: str = arguments.worker_image
    camera_ids = tuple(item.strip() for item in arguments.camera_ids.split(",") if item.strip())
    live_rungs = tuple(int(item) for item in rungs if item.isdigit())
    artifact = _load_authorization(arguments.authorization)
    request = AuthorizationRequest(
        rungs=live_rungs,
        mode=mode,
        appliance_id=arguments.appliance_id,
        worker_image_digest=_image_digest(worker_image),
        camera_ids=camera_ids,
        now=datetime.now(UTC),
        minimum_projected_slack_mib=policy.minimum_gpu_slack_mib,
    )
    try:
        authorize_rungs(request, artifact)
    except CanaryAuthorizationError as error:
        rung_values: list[JsonValue] = [item for item in rungs]
        refusal: dict[str, JsonValue] = {
            "schema_version": 1,
            "code": error.code,
            "rungs": rung_values,
        }
        write_once(evidence_dir / "authorization-refusal.json", canonical_json(refusal))
        print(str(error), file=sys.stderr)
        return 2
    requested_rungs: list[JsonValue] = [item for item in rungs]
    write_once(
        evidence_dir / "run-request.json",
        canonical_json(
            {
                "schema_version": 1,
                "requested_rungs": requested_rungs,
                "policy_sha256": policy_digest,
            }
        ),
    )
    refuse_existing_project()
    maximum = max((int(item) for item in rungs if item.isdigit()), default=0)
    camera_count = max(maximum, 1 if "loopback" in rungs else 0)
    relay_token = secrets.token_urlsafe(48)
    compose_path, compose_digest = render_compose(
        RenderRequest(
            evidence_dir=evidence_dir,
            worker_image=worker_image,
            relay_token=relay_token,
            camera_count=camera_count,
            model_dir=arguments.model_dir,
            engine_cache_source=arguments.engine_cache_source,
        )
    )
    print(json.dumps({"compose_sha256": compose_digest, "project": "seeon-ds-canary"}))
    if arguments.render_only:
        return 0
    baseline = capture_live_snapshot()
    write_once(evidence_dir / "raw" / "live-baseline.json", encode_live_snapshot(baseline))
    if mode is CanaryMode.COMMISSIONING and baseline.healthy_runtime_camera_ids:
        raise CanarySafetyError(
            "commissioning_live_cameras_present",
            str(len(baseline.healthy_runtime_camera_ids)),
        )
    run_root = evidence_dir.resolve() / "run"
    refuse_mount_overlap(
        tuple(
            run_root / name
            for name in ("state", "sockets", "engine-cache", "models", "scratch", "clips")
        ),
        baseline,
    )
    return execute_canary(
        ExecutionRequest(
            compose_path=compose_path,
            evidence_dir=evidence_dir,
            baseline=baseline,
            rung_durations=_rung_durations(rungs, policy),
            publisher_count=camera_count,
            relay_token=relay_token,
            safety_limits=SafetyLimits(
                minimum_gpu_slack_mib=policy.minimum_gpu_slack_mib,
                maximum_gpu_utilization=policy.gpu_utilization_absolute_max,
                require_live_status=mode is CanaryMode.SHARED_HOST_SMOKE,
                gpu_process_loss_grace_seconds=(
                    policy.protected_gpu_process_loss_grace_seconds
                ),
            ),
            mode=mode,
            rungs=rungs,
            artifacts=ExecutionArtifactSources(
                worker_image=_image_digest(worker_image),
                support_images=SUPPORT_IMAGE_DIGESTS,
                gate_policy=policy_digest,
                compose=compose_digest,
            ),
        )
    )


def main() -> int:
    arguments = RunArguments()
    _ = _parser().parse_args(namespace=arguments)
    try:
        return _run(arguments)
    except (
        CanaryArgumentError,
        CanarySafetyError,
        FileExistsError,
        OSError,
        ValidationError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2
