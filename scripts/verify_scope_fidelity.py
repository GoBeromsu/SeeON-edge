#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# --- How to run ---
# uv run python scripts/verify_scope_fidelity.py --fixture
# uv run python scripts/verify_scope_fidelity.py --plan <plan> --evidence <evidence>
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

IDENTITY_PATTERN: Final = (
    re.compile(r"\bAPI_FACILITY_ID\b|\bEDGE_FACILITY_TOKEN\b"),
    "environment facility identity",
)
OPS_PATTERNS: Final = (
    (
        re.compile(
            r"\b(?:DROP|TRUNCATE)\s+TABLE\b|\bDELETE\s+FROM\b", re.IGNORECASE
        ),
        "destructive SQL",
    ),
    (re.compile(r"\bssh\b[^\n]*\bjnu(?:-oss)?\b", re.IGNORECASE), "JNU target"),
    (re.compile(r"\b(?:latest|:dev)\b"), "mutable image tag"),
)
RUNTIME_PATHS: Final = ("backend/app", "worker", "shared")
DEPLOYMENT_PATHS: Final = (
    "compose.edge.yaml",
    "compose.edge.cpu.yaml",
    ".env.edge.prod.example",
)
OPS_PATHS: Final = (
    "scripts/ops",
)


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    category: str


class ScopeError(Exception):
    pass


def tracked_files(root: Path, paths: tuple[str, ...]) -> tuple[Path, ...]:
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return tuple(root / item for item in output.splitlines() if item)


def scan_file(
    root: Path,
    path: Path,
    patterns: tuple[tuple[re.Pattern[str], str], ...],
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8", errors="strict")
    relative = path.relative_to(root).as_posix()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if relative.endswith("verify_scope_fidelity.py"):
            continue
        if line.lstrip().startswith("#"):
            continue
        for pattern, category in patterns:
            if pattern.search(line):
                findings.append(Finding(relative, line_number, category))
    return tuple(findings)


def scan(root: Path) -> tuple[Finding, ...]:
    identity_findings = tuple(
        finding
        for path in tracked_files(root, RUNTIME_PATHS + DEPLOYMENT_PATHS + OPS_PATHS)
        if path.is_file()
        for finding in scan_file(root, path, (IDENTITY_PATTERN,))
    )
    deployment_findings = tuple(
        finding
        for path in tracked_files(root, DEPLOYMENT_PATHS + OPS_PATHS)
        if path.is_file()
        for finding in scan_file(root, path, OPS_PATTERNS)
    )
    return identity_findings + deployment_findings


def verify_plan_and_evidence(plan: Path, evidence: Path) -> None:
    plan_text = plan.read_text(encoding="utf-8")
    required = (
        "facility-bound",
        "ML-first rollback",
        "approved_plan_sha256",
        "one camera per room",
    )
    missing = tuple(value for value in required if value not in plan_text)
    if missing:
        raise ScopeError(f"plan scope markers missing: {', '.join(missing)}")
    for task in range(1, 18):
        path = evidence / f"task-{task}-edge-driven-facility-provisioning.txt"
        if not path.is_file():
            raise ScopeError(f"task evidence missing: {path.name}")
    task_17 = (evidence / "task-17-edge-driven-facility-provisioning.txt").read_text(
        encoding="utf-8"
    )
    for marker in ("provider=codex", "mode=fast", "exclusive_paths="):
        if marker not in task_17:
            raise ScopeError(f"task 17 ownership marker missing: {marker}")


def run_fixture() -> None:
    with tempfile.TemporaryDirectory(prefix="scope-fidelity-") as temporary:
        root = Path(temporary)
        _ = subprocess.run(["git", "init", "-q", str(root)], check=True)
        safe = root / "backend" / "app"
        safe.mkdir(parents=True)
        _ = (safe / "runtime.py").write_text(
            '# Historical API_FACILITY_ID note is not executable residue.\n'
            'endpoint = "/api/v1/connection/sync-cameras"\n',
            encoding="utf-8",
        )
        _ = subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        if scan(root):
            raise ScopeError("safe API-first fixture was rejected")
        cases = {
            "env.py": 'facility = os.environ["API_FACILITY_ID"]\n',
            "sql.py": 'statement = "DELETE FROM spaces"\n',
            "host.py": 'command = "ssh jnu-oss"\n',
            "image.py": 'image = "worker:latest"\n',
        }
        for name, content in cases.items():
            parent = safe if name == "env.py" else root / "scripts" / "ops"
            parent.mkdir(parents=True, exist_ok=True)
            path = parent / name
            _ = path.write_text(content, encoding="utf-8")
            _ = subprocess.run(["git", "-C", str(root), "add", str(path)], check=True)
            if not scan(root):
                raise ScopeError(f"negative scope fixture passed: {name}")
            path.unlink()
            _ = subprocess.run(
                ["git", "-C", str(root), "rm", "--cached", "-q", str(path)],
                check=True,
            )
    print("SCOPE_FIDELITY_FIXTURE_OK")


def parse_cli(argv: list[str]) -> tuple[bool, Path | None, Path | None]:
    fixture = False
    plan: Path | None = None
    evidence: Path | None = None
    index = 0
    while index < len(argv):
        match argv[index]:
            case "--fixture":
                fixture = True
                index += 1
            case ("--plan" | "--evidence") as flag:
                if index + 1 >= len(argv):
                    raise ScopeError(f"missing value for {flag}")
                value = Path(argv[index + 1])
                if flag == "--plan":
                    plan = value
                else:
                    evidence = value
                index += 2
            case unknown:
                raise ScopeError(f"unknown argument: {unknown}")
    return fixture, plan, evidence


def main() -> int:
    fixture, plan, evidence = parse_cli(sys.argv[1:])
    if fixture:
        run_fixture()
        return 0
    if plan is None or evidence is None:
        raise ScopeError("--plan and --evidence are required")
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    if findings:
        details = ", ".join(
            f"{finding.path}:{finding.line} ({finding.category})" for finding in findings
        )
        raise ScopeError(details)
    verify_plan_and_evidence(plan.resolve(), evidence.resolve())
    print("SCOPE_FIDELITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
