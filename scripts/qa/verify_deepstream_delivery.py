# /// script
# requires-python = ">=3.11"
# ///
# --- How to run ---
# uv run python scripts/qa/verify_deepstream_delivery.py canary \
#   --evidence-root <dir> --output <file>

"""Independently recompute C8 canary and final F1/F2/F4 verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import ClassVar, Literal, assert_never, final

from pydantic import BaseModel, ConfigDict, ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qa.deepstream_final_verification import compliance, quality, scope  # noqa: E402
from worker.tools.deepstream_canary.gates import evaluate_receipt  # noqa: E402
from worker.tools.deepstream_canary.models import GatePolicy, GateReport, RungReceipt  # noqa: E402
from worker.tools.deepstream_canary.report import write_canonical_report  # noqa: E402

POLICY = Path("scripts/qa/deepstream-canary/gate-policy.v1.json")


class RunRequestReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    requested_rungs: tuple[str, ...]
    policy_sha256: str
    worker_image: str | None = None
    worker_image_digest: str | None = None
    expected_revision: str | None = None


class DeliveryVerdict(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    verdict: Literal["PASS", "FAIL", "APPROVE", "REJECT"]
    verifier: Literal["canary", "compliance", "quality", "scope"]
    reports: tuple[GateReport, ...] = ()
    findings: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()


def _canary(root: Path, policy_path: Path) -> DeliveryVerdict:
    policy_content = policy_path.read_bytes()
    policy = GatePolicy.model_validate_json(policy_content)
    request = RunRequestReceipt.model_validate_json((root / "run-request.json").read_bytes())
    policy_digest = hashlib.sha256(policy_content).hexdigest()
    receipts = tuple(
        RungReceipt.model_validate_json(path.read_bytes())
        for path in sorted((root / "raw").glob("rung-*.json"))
    )
    by_rung = {receipt.rung: receipt for receipt in receipts}
    findings = [
        f"requested_rung_missing:{rung}"
        for rung in request.requested_rungs
        if rung not in by_rung
    ]
    if request.policy_sha256 != policy_digest:
        findings.append("run_request_policy_digest_mismatch")
    identity_artifacts: list[str] = []
    if request.worker_image_digest is not None:
        expected_suffix = f"@{request.worker_image_digest}"
        if request.worker_image is None or not request.worker_image.endswith(expected_suffix):
            findings.append("run_request_worker_image_identity_mismatch")
        identity_artifacts.append(f"worker-image:{request.worker_image_digest}")
        findings.extend(
            f"worker_image_digest_mismatch:{receipt.rung}"
            for receipt in receipts
            if receipt.artifacts.worker_image != request.worker_image_digest
        )
    if request.expected_revision is not None:
        if re.fullmatch(r"[0-9a-f]{40}", request.expected_revision) is None:
            findings.append("run_request_expected_revision_invalid")
        identity_artifacts.append(f"expected-revision:{request.expected_revision}")
    reports = tuple(
        evaluate_receipt(by_rung[rung], policy)
        for rung in request.requested_rungs
        if rung in by_rung
    )
    for report in reports:
        _ = write_canonical_report(root, report.model_dump(mode="json"))
    passed = (
        not findings
        and bool(reports)
        and len(reports) == len(request.requested_rungs)
        and all(report.verdict == "PASS" for report in reports)
    )
    return DeliveryVerdict(
        verdict="PASS" if passed else "FAIL",
        verifier="canary",
        reports=reports,
        findings=tuple(findings),
        artifacts=tuple(identity_artifacts),
    )


def _legacy_scope(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    compose = (root / "compose.rendered.yaml").read_text(encoding="utf-8")
    findings: list[str] = []
    required = ("name: seeon-ds-canary", "internal: true", "127.0.0.1:18090:8090")
    findings.extend(f"missing:{token}" for token in required if token not in compose)
    if ".env.edge.prod" in compose:
        findings.append("production_env_reference")
    published: list[str] = re.findall(
        r"- [\"']?([^\n\"']+:[0-9]+:[0-9]+)[\"']?", compose
    )
    findings.extend(
        f"non_loopback_port:{port}" for port in published if not port.startswith("127.0.0.1:")
    )
    compose_digest = hashlib.sha256(compose.encode()).hexdigest()
    return tuple(findings), (f"compose.rendered.yaml:{compose_digest}",)


@final
class VerifyArguments(argparse.Namespace):
    verifier: Literal["canary", "compliance", "quality", "scope"] = "canary"
    evidence_root: Path | None = None
    output: Path = Path()
    policy: Path = POLICY
    plan: Path | None = None
    ledger: Path | None = None
    require_tasks: str = ""
    require_prs: int = 0
    inject: str | None = None
    base_sha: str | None = None
    head_sha: str | None = None
    allowed_plan: Path | None = None
    inject_forbidden_path: str | None = None


def _arguments() -> VerifyArguments:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("verifier", choices=("canary", "compliance", "quality", "scope"))
    _ = parser.add_argument("--evidence-root", type=Path)
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--policy", type=Path, default=POLICY)
    _ = parser.add_argument("--plan", type=Path)
    _ = parser.add_argument("--ledger", type=Path)
    _ = parser.add_argument("--require-tasks", default="")
    _ = parser.add_argument("--require-prs", type=int, default=0)
    _ = parser.add_argument("--inject", choices=("missing-native-ctest",))
    _ = parser.add_argument("--base-sha")
    _ = parser.add_argument("--head-sha")
    _ = parser.add_argument("--allowed-plan", type=Path)
    _ = parser.add_argument("--inject-forbidden-path")
    arguments = VerifyArguments()
    _ = parser.parse_args(namespace=arguments)
    return arguments


def _final_verdict(
    verifier: Literal["compliance", "quality", "scope"],
    result: tuple[tuple[str, ...], tuple[str, ...]],
) -> DeliveryVerdict:
    findings, artifacts = result
    return DeliveryVerdict(
        verdict="APPROVE" if not findings else "REJECT",
        verifier=verifier,
        findings=findings,
        artifacts=artifacts,
    )


def main() -> int:
    arguments = _arguments()
    try:
        match arguments.verifier:
            case "canary":
                if arguments.evidence_root is None:
                    raise FileNotFoundError("--evidence-root is required for canary")
                verdict = _canary(arguments.evidence_root, arguments.policy)
            case "compliance":
                if arguments.evidence_root is None:
                    raise FileNotFoundError("--evidence-root is required for compliance")
                tasks = tuple(int(item) for item in arguments.require_tasks.split(",") if item)
                verdict = _final_verdict(
                    "compliance",
                    compliance(
                        arguments.evidence_root,
                        arguments.plan,
                        arguments.ledger,
                        tasks,
                        arguments.require_prs,
                    ),
                )
            case "quality":
                if arguments.evidence_root is None:
                    raise FileNotFoundError("--evidence-root is required for quality")
                verdict = _final_verdict(
                    "quality", quality(arguments.evidence_root, arguments.inject)
                )
            case "scope":
                if (
                    arguments.base_sha is not None
                    and arguments.head_sha is not None
                    and arguments.allowed_plan is not None
                ):
                    result = scope(
                        arguments.base_sha,
                        arguments.head_sha,
                        arguments.allowed_plan,
                        arguments.inject_forbidden_path,
                    )
                elif arguments.evidence_root is not None:
                    result = _legacy_scope(arguments.evidence_root)
                else:
                    raise FileNotFoundError("scope requires Git range inputs or --evidence-root")
                verdict = _final_verdict("scope", result)
            case unreachable:
                assert_never(unreachable)
    except (OSError, ValidationError, ValueError, KeyError, json.JSONDecodeError) as error:
        verdict = DeliveryVerdict(
            verdict="FAIL" if arguments.verifier == "canary" else "REJECT",
            verifier=arguments.verifier,
            findings=(str(error),),
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(verdict.model_dump_json() + "\n", encoding="utf-8")
    print(verdict.model_dump_json())
    return 0 if verdict.verdict in {"PASS", "APPROVE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
