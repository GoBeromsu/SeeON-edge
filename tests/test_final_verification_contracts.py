from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
VERIFIER = REPOSITORY_ROOT / "scripts/qa/verify_deepstream_delivery.py"
REVISION = "1" * 40


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFIER), *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _verdict(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compliance_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    plan = tmp_path / "plan.md"
    plan.write_text("\n".join(f"- [x] {task}. C{task}" for task in range(1, 10)))
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(
                {
                    "plan": str(plan),
                    "task": f"todo {task}",
                    "commit": str(task) * 40,
                    "pr": f"https://github.com/SeniorAILab/SeeON-edge/pull/{390 + task}",
                    "cleanup": ["complete"],
                }
            )
            for task in range(1, 10)
        )
        + "\n"
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for task in range(1, 10):
        for phase in ("red", "green", "cleanup"):
            (evidence / f"task-{task}-{phase}.log").write_text(
                f"task={task} phase={phase} exit=0\n"
            )
    return plan, ledger, evidence


def test_f1_exact_cli_approves_hashed_complete_receipts_and_rejects_missing_task(
    tmp_path: Path,
) -> None:
    # Given: nine ledger entries and independently hashed RED/GREEN/cleanup receipts.
    plan, ledger, evidence = _compliance_inputs(tmp_path)
    output = tmp_path / "f1.json"
    command = (
        "compliance",
        "--plan",
        str(plan),
        "--ledger",
        str(ledger),
        "--evidence-root",
        str(evidence),
        "--require-tasks",
        "1,2,3,4,5,6,7,8,9",
        "--require-prs",
        "9",
        "--output",
        str(output),
    )

    # When: the exact F1 command verifies complete inputs, then task 9 is omitted.
    approved = _run(*command)
    (evidence / "task-9-cleanup.log").unlink()
    rejected = _run(*command)

    # Then: only the complete, byte-bound evidence receives exact APPROVE.
    assert approved.returncode == 0, approved.stderr
    assert _verdict(output)["verdict"] == "REJECT"
    assert '"verdict":"APPROVE"' in approved.stdout
    assert rejected.returncode == 1
    assert "task_9_cleanup_missing" in rejected.stdout


def _quality_receipt(root: Path) -> None:
    checks = []
    for name in (
        "ruff",
        "lint-imports",
        "pytest",
        "docker-build",
        "native-ctest",
        "native-sanitizer",
    ):
        artifact = root / f"{name}.log"
        artifact.write_text(f"{name}: exit=0\n")
        checks.append(
            {
                "name": name,
                "artifact": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "exit_code": 0,
            }
        )
    (root / "quality-receipt.json").write_text(
        json.dumps({"schema_version": 1, "checks": checks}) + "\n"
    )


def test_f2_exact_cli_rejects_missing_native_ctest_injection_and_approves_receipt(
    tmp_path: Path,
) -> None:
    # Given: hashed successful receipts for every F2 quality gate.
    evidence = tmp_path / "quality"
    evidence.mkdir()
    _quality_receipt(evidence)
    red_output = tmp_path / "f2-red.json"
    green_output = tmp_path / "f2-green.json"

    # When: the exact RED injection and real commands are independently evaluated.
    rejected = _run(
        "quality",
        "--evidence-root",
        str(evidence),
        "--inject",
        "missing-native-ctest",
        "--output",
        str(red_output),
    )
    approved = _run(
        "quality",
        "--evidence-root",
        str(evidence),
        "--output",
        str(green_output),
    )

    # Then: injection is REJECT and complete byte-bound evidence is APPROVE.
    assert rejected.returncode == 1
    assert _verdict(red_output)["verdict"] == "REJECT"
    assert approved.returncode == 0, approved.stderr
    assert _verdict(green_output)["verdict"] == "APPROVE"


def test_f4_exact_cli_rejects_injected_forbidden_path_and_approves_empty_diff(
    tmp_path: Path,
) -> None:
    # Given: a parsed plan and an empty Git revision range.
    plan = tmp_path / "plan.md"
    plan.write_text("# approved worker-only plan\n")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    red_output = tmp_path / "f4-red.json"
    green_output = tmp_path / "f4-green.json"

    # When: a forbidden backend path is injected, then the real empty range is checked.
    rejected = _run(
        "scope",
        "--base-sha",
        head,
        "--head-sha",
        head,
        "--allowed-plan",
        str(plan),
        "--inject-forbidden-path",
        "backend/AGENTS.md",
        "--output",
        str(red_output),
    )
    approved = _run(
        "scope",
        "--base-sha",
        head,
        "--head-sha",
        head,
        "--allowed-plan",
        str(plan),
        "--output",
        str(green_output),
    )

    # Then: the exact final scope verdicts distinguish forbidden from approved paths.
    assert rejected.returncode == 1
    assert "forbidden_path:backend/AGENTS.md" in rejected.stdout
    assert approved.returncode == 0, approved.stderr
    assert _verdict(green_output)["verdict"] == "APPROVE"
