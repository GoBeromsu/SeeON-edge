# /// script
# requires-python = ">=3.11"
# ///
# --- How to run ---
# uv run python scripts/qa/verify_deepstream_delivery.py canary \\
#   --evidence-root <dir> --output <file>

"""Independently recompute C8 canary, compliance, quality, and scope verdicts."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import ClassVar, Literal, assert_never, final

from pydantic import BaseModel, ConfigDict, ValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from worker.tools.deepstream_canary.gates import evaluate_receipt  # noqa: E402
from worker.tools.deepstream_canary.models import GatePolicy, GateReport, RungReceipt  # noqa: E402
from worker.tools.deepstream_canary.report import write_canonical_report  # noqa: E402

POLICY = Path("scripts/qa/deepstream-canary/gate-policy.v1.json")


class ManifestEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    sha256: str
    size: int


class ReceiptManifest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    files: dict[str, ManifestEntry]


class RunRequestReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    requested_rungs: tuple[str, ...]
    policy_sha256: str


class DeliveryVerdict(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    verdict: Literal["PASS", "FAIL"]
    verifier: Literal["canary", "compliance", "quality", "scope"]
    reports: tuple[GateReport, ...] = ()
    findings: tuple[str, ...] = ()


def _canary(root: Path, policy_path: Path) -> DeliveryVerdict:
    policy = GatePolicy.model_validate_json(policy_path.read_bytes())
    request = RunRequestReceipt.model_validate_json((root / "run-request.json").read_bytes())
    policy_digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    receipts = tuple(
        RungReceipt.model_validate_json(path.read_bytes())
        for path in sorted((root / "raw").glob("rung-*.json"))
    )
    by_rung = {receipt.rung: receipt for receipt in receipts}
    findings = tuple(
        [
            f"requested_rung_missing:{rung}"
            for rung in request.requested_rungs
            if rung not in by_rung
        ]
        + (
            []
            if request.policy_sha256 == policy_digest
            else ["run_request_policy_digest_mismatch"]
        )
    )
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
        findings=findings,
    )


def _compliance(root: Path) -> DeliveryVerdict:
    manifest = ReceiptManifest.model_validate_json((root / "receipt-manifest.json").read_bytes())
    findings: list[str] = []
    for relative, expected in manifest.files.items():
        path = root / relative
        if not path.is_file():
            findings.append(f"missing:{relative}")
            continue
        content = path.read_bytes()
        if len(content) != expected.size or hashlib.sha256(content).hexdigest() != expected.sha256:
            findings.append(f"digest_mismatch:{relative}")
    return DeliveryVerdict(
        verdict="PASS" if not findings else "FAIL",
        verifier="compliance",
        findings=tuple(findings),
    )


def _quality(root: Path) -> DeliveryVerdict:
    reports = tuple(
        GateReport.model_validate_json(path.read_bytes())
        for path in sorted(root.glob("gate-report.*.json"))
    )
    findings: list[str] = []
    if not reports:
        findings.append("gate_reports_missing")
    for report in reports:
        if report.verdict != "PASS" or not report.checks or not all(
            check.passed for check in report.checks
        ):
            findings.append(f"non_passing_report:{report.rung}")
    return DeliveryVerdict(
        verdict="PASS" if not findings else "FAIL",
        verifier="quality",
        reports=reports,
        findings=tuple(findings),
    )


def _scope(root: Path) -> DeliveryVerdict:
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
        f"non_loopback_port:{port}"
        for port in published
        if not port.startswith("127.0.0.1:")
    )
    return DeliveryVerdict(
        verdict="PASS" if not findings else "FAIL",
        verifier="scope",
        findings=tuple(findings),
    )


@final
class VerifyArguments(argparse.Namespace):
    verifier: Literal["canary", "compliance", "quality", "scope"] = "canary"
    evidence_root: Path = Path()
    output: Path = Path()
    policy: Path = POLICY


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "verifier", choices=("canary", "compliance", "quality", "scope")
    )
    _ = parser.add_argument("--evidence-root", type=Path, required=True)
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--policy", type=Path, default=POLICY)
    arguments = VerifyArguments()
    _ = parser.parse_args(namespace=arguments)
    try:
        match arguments.verifier:
            case "canary":
                verdict = _canary(arguments.evidence_root, arguments.policy)
            case "compliance":
                verdict = _compliance(arguments.evidence_root)
            case "quality":
                verdict = _quality(arguments.evidence_root)
            case "scope":
                verdict = _scope(arguments.evidence_root)
            case unreachable:
                assert_never(unreachable)
    except (OSError, ValidationError) as error:
        verdict = DeliveryVerdict(
            verdict="FAIL",
            verifier=arguments.verifier,
            findings=(str(error),),
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(verdict.model_dump_json() + "\n", encoding="utf-8")
    print(verdict.model_dump_json())
    return 0 if verdict.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
