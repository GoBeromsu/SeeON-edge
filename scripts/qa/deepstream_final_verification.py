"""F1/F2/F4 final-verification contracts with byte-bound evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PR_URL = re.compile(r"https://github\.com/SeniorAILab/SeeON-edge/pull/[0-9]+")
_REQUIRED_QUALITY = frozenset(
    {"ruff", "lint-imports", "pytest", "docker-build", "native-ctest", "native-sanitizer"}
)
_FORBIDDEN_PREFIXES = ("backend/", "front/", "contracts/")


class QualityCheck(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    artifact: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exit_code: int


class QualityReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    checks: tuple[QualityCheck, ...]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_from_text(text: str) -> int | None:
    match = re.search(r"(?:todo|task)[ -]?(\d+)", text, re.IGNORECASE)
    return None if match is None else int(match.group(1))


def _task_receipts(root: Path, task: int, phase: str) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(root.rglob(f"task-{task}-*"))
        if path.is_file() and phase in path.name.lower() and path.stat().st_size > 0
    )


def compliance(
    root: Path,
    plan: Path | None,
    ledger: Path | None,
    required_tasks: tuple[int, ...],
    required_prs: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Verify either the legacy C8 manifest or complete F1 delivery evidence."""
    if plan is None or ledger is None:
        manifest = json.loads((root / "receipt-manifest.json").read_text(encoding="utf-8"))
        findings: list[str] = []
        artifacts: list[str] = []
        for relative, expected in manifest["files"].items():
            path = root / relative
            if not path.is_file():
                findings.append(f"missing:{relative}")
                continue
            digest = _digest(path)
            artifacts.append(f"{relative}:{digest}")
            if path.stat().st_size != expected["size"] or digest != expected["sha256"]:
                findings.append(f"digest_mismatch:{relative}")
        return tuple(findings), tuple(artifacts)

    plan_content = plan.read_text(encoding="utf-8")
    ledger_lines = tuple(line for line in ledger.read_text(encoding="utf-8").splitlines() if line)
    parsed = tuple(json.loads(line) for line in ledger_lines)
    findings = []
    artifacts = [f"plan:{_digest(plan)}", f"ledger:{_digest(ledger)}"]
    task_entries: dict[int, list[str]] = {}
    pr_urls: set[str] = set()
    for line, entry in zip(ledger_lines, parsed, strict=True):
        task = _task_from_text(str(entry.get("task", "")))
        if task is not None:
            task_entries.setdefault(task, []).append(line)
        pr_urls.update(_PR_URL.findall(line))
    for task in required_tasks:
        if re.search(rf"(?m)^- \[[ xX]\] {task}\.", plan_content) is None:
            findings.append(f"task_{task}_plan_missing")
        if task not in task_entries:
            findings.append(f"task_{task}_ledger_missing")
        for phase in ("red", "green", "cleanup"):
            receipts = _task_receipts(root, task, phase)
            if not receipts:
                findings.append(f"task_{task}_{phase}_missing")
            artifacts.extend(
                f"{path.relative_to(root)}:{_digest(path)}" for path in receipts
            )
        entries = task_entries.get(task, ())
        if entries and not any(re.search(r"\b[0-9a-f]{40}\b", line) for line in entries):
            findings.append(f"task_{task}_sha_missing")
    if len(pr_urls) < required_prs:
        findings.append(f"pr_urls_missing:{len(pr_urls)}/{required_prs}")
    artifacts.extend(f"pr:{url}" for url in sorted(pr_urls))
    return tuple(findings), tuple(artifacts)


def quality(root: Path, injection: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Verify successful command receipts and their immutable output logs."""
    receipt_path = root / "quality-receipt.json"
    if not receipt_path.is_file():
        return ("quality_receipt_missing",), ()
    receipt = QualityReceipt.model_validate_json(receipt_path.read_bytes())
    findings: list[str] = []
    artifacts = [f"quality-receipt.json:{_digest(receipt_path)}"]
    by_name = {check.name: check for check in receipt.checks}
    for name in sorted(_REQUIRED_QUALITY):
        check = by_name.get(name)
        if check is None:
            findings.append(f"quality_check_missing:{name}")
            continue
        path = root / check.artifact
        if not path.is_file():
            findings.append(f"quality_artifact_missing:{name}")
            continue
        digest = _digest(path)
        artifacts.append(f"{check.artifact}:{digest}")
        if digest != check.sha256:
            findings.append(f"quality_digest_mismatch:{name}")
        if check.exit_code != 0:
            findings.append(f"quality_exit_nonzero:{name}:{check.exit_code}")
    if injection == "missing-native-ctest":
        findings.append("quality_check_missing:native-ctest")
    return tuple(findings), tuple(artifacts)


def scope(
    base_sha: str,
    head_sha: str,
    allowed_plan: Path,
    injected_path: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Verify the Git range stays within the worker migration's approved scope."""
    plan_digest = _digest(allowed_plan)
    completed = subprocess.run(
        ("git", "diff", "--name-only", base_sha, head_sha),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    findings: list[str] = []
    if completed.returncode != 0:
        findings.append("git_diff_failed")
        paths: tuple[str, ...] = ()
    else:
        paths = tuple(line for line in completed.stdout.splitlines() if line)
    if injected_path is not None:
        paths = (*paths, injected_path)
    for path in paths:
        if path.startswith(_FORBIDDEN_PREFIXES) or path in {
            ".env.edge.prod",
            "compose.edge.yaml",
        }:
            findings.append(f"forbidden_path:{path}")
    artifacts = (f"allowed-plan:{plan_digest}", *(f"path:{path}" for path in paths))
    return tuple(findings), artifacts
