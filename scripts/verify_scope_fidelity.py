#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# --- How to run ---
# uv run python scripts/verify_scope_fidelity.py --fixture
# uv run python scripts/verify_scope_fidelity.py --repo
# uv run python scripts/verify_scope_fidelity.py --plan <plan> --evidence <evidence>
from __future__ import annotations

import ast
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
# The camera roster is owned by the Edge camera_registry DB (dashboard-entered)
# and pulled by ml-worker from ml-api. It must never be provisionable through
# the environment, because an env var reaches the runtime via compose, and
# compose is in Git. `--config` remains a CLI-only developer/e2e path.
ROSTER_PATTERN: Final = (
    re.compile(r"\bEDGE_CAMERA_CONFIG(?:_FILE)?\b|\bAPI_CAMERA_INVENTORY\b"),
    "environment camera roster",
)
NAME_ONLY_MARKER: Final = "scope-fidelity: name-only"
ENVIRONMENT_CATEGORIES: Final = frozenset(
    {"environment facility identity", "environment camera roster"}
)
OPS_PATTERNS: Final = (
    (
        re.compile(r"\b(?:DROP|TRUNCATE)\s+TABLE\b|\bDELETE\s+FROM\b", re.IGNORECASE),
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
OPS_PATHS: Final = ("scripts/ops",)


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


def _contains_raise(nodes: list[ast.stmt]) -> bool:
    return any(isinstance(node, ast.Raise) for statement in nodes for node in ast.walk(statement))


def _is_collection_declaration(value: ast.expr) -> bool:
    if isinstance(value, (ast.List, ast.Set, ast.Tuple)):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"frozenset", "set", "tuple"}
        and len(value.args) == 1
        and not value.keywords
        and isinstance(value.args[0], (ast.List, ast.Set, ast.Tuple))
    )


def _assignment_parts(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target.id, statement.value
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.value is not None
    ):
        return statement.target.id, statement.value
    return None


def _is_membership_test(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(candidate, ast.Compare)
        and isinstance(candidate.left, ast.Name)
        and candidate.left.id in names
        and any(isinstance(operator, (ast.In, ast.NotIn)) for operator in candidate.ops)
        for candidate in ast.walk(node)
    )


def _loop_is_rejection(loop: ast.For | ast.AsyncFor) -> bool:
    targets = {node.id for node in ast.walk(loop.target) if isinstance(node, ast.Name)}
    guards = tuple(
        node
        for statement in loop.body
        for node in ast.walk(statement)
        if isinstance(node, ast.If)
        and _is_membership_test(node.test, targets)
        and _contains_raise(node.body)
    )
    if not guards:
        return False

    for statement in loop.body:
        for reference in ast.walk(statement):
            if not (
                isinstance(reference, ast.Name)
                and isinstance(reference.ctx, ast.Load)
                and reference.id in targets
            ):
                continue
            if any(reference in ast.walk(guard.test) for guard in guards):
                continue
            if any(
                reference in ast.walk(raised)
                for guard in guards
                for raised in ast.walk(guard)
                if isinstance(raised, ast.Raise)
            ):
                continue
            return False
    return True


def _flows_to_raising_guard(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    tree: ast.Module,
) -> bool:
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.If) and any(current is item for item in ast.walk(parent.test)):
            return _contains_raise(parent.body)
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            parts = _assignment_parts(parent)
            if parts is None:
                return False
            assigned_name, _ = parts
            return any(
                isinstance(candidate, ast.If)
                and isinstance(candidate.test, ast.Name)
                and candidate.test.id == assigned_name
                and _contains_raise(candidate.body)
                for candidate in ast.walk(tree)
            )
        current = parent
    return False


def _is_rejection_collection_use(
    reference: ast.Name,
    parents: dict[ast.AST, ast.AST],
    tree: ast.Module,
) -> bool:
    parent = parents.get(reference)
    if isinstance(parent, ast.Compare):
        for operator, comparator in zip(parent.ops, parent.comparators, strict=True):
            if comparator is reference and isinstance(operator, (ast.In, ast.NotIn)):
                return _flows_to_raising_guard(parent, parents, tree)
    if (
        isinstance(parent, ast.Attribute)
        and parent.value is reference
        and parent.attr in {"intersection", "isdisjoint"}
    ):
        call = parents.get(parent)
        return isinstance(call, ast.Call) and _flows_to_raising_guard(call, parents, tree)
    if isinstance(parent, (ast.For, ast.AsyncFor)) and parent.iter is reference:
        return _loop_is_rejection(parent)
    return False


def _docstring_nodes(tree: ast.Module) -> set[ast.Constant]:
    docstrings: set[ast.Constant] = set()
    for owner in ast.walk(tree):
        if (
            not isinstance(
                owner,
                (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
            or not owner.body
        ):
            continue
        first = owner.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(first.value)
    return docstrings


def _python_environment_findings(
    relative: str,
    text: str,
    patterns: tuple[tuple[re.Pattern[str], str], ...],
) -> tuple[Finding, ...]:
    tree = ast.parse(text)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    docstrings = _docstring_nodes(tree)
    matches = {
        node: category
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstrings
        for pattern, category in patterns
        if pattern.search(node.value)
    }
    safe_sources: set[ast.Constant] = set()
    lines = text.splitlines()

    for statement in tree.body:
        parts = _assignment_parts(statement)
        if parts is None:
            continue
        binding, value = parts
        sources = {
            node for node in ast.walk(value) if isinstance(node, ast.Constant) and node in matches
        }
        if not sources:
            continue
        references = tuple(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == binding
        )
        if (
            _is_collection_declaration(value)
            and references
            and all(
                _is_rejection_collection_use(reference, parents, tree) for reference in references
            )
        ):
            safe_sources.update(sources)
            continue

        marker_line = lines[statement.lineno - 1]
        if (
            NAME_ONLY_MARKER in marker_line
            and isinstance(value, ast.Constant)
            and value in sources
            and not references
        ):
            safe_sources.update(sources)

    findings = {
        Finding(relative, source.lineno, category)
        for source, category in matches.items()
        if source not in safe_sources
    }
    return tuple(sorted(findings, key=lambda item: (item.line, item.category)))


def scan_file(
    root: Path,
    path: Path,
    patterns: tuple[tuple[re.Pattern[str], str], ...],
) -> tuple[Finding, ...]:
    text = path.read_text(encoding="utf-8", errors="strict")
    relative = path.relative_to(root).as_posix()
    if relative.endswith("verify_scope_fidelity.py"):
        return ()

    environment_patterns = tuple(item for item in patterns if item[1] in ENVIRONMENT_CATEGORIES)
    findings = list(
        _python_environment_findings(relative, text, environment_patterns)
        if path.suffix == ".py" and environment_patterns
        else ()
    )
    line_patterns = tuple(
        item for item in patterns if path.suffix != ".py" or item[1] not in ENVIRONMENT_CATEGORIES
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#") or NAME_ONLY_MARKER in line:
            continue
        for pattern, category in line_patterns:
            if pattern.search(line):
                findings.append(Finding(relative, line_number, category))
    return tuple(findings)


def scan(root: Path) -> tuple[Finding, ...]:
    identity_findings = tuple(
        finding
        for path in tracked_files(root, RUNTIME_PATHS + DEPLOYMENT_PATHS + OPS_PATHS)
        if path.is_file()
        for finding in scan_file(root, path, (IDENTITY_PATTERN, ROSTER_PATTERN))
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
            "".join(
                (
                    "# Historical API_FACILITY_ID note is not executable residue.\n",
                    'endpoint = "/api/v1/connection/sync-cameras"\n',
                    "RETIRED_ENVIRONMENT_KEYS = frozenset(\n",
                    '    {"API_FACILITY_ID", "EDGE_CAMERA_CONFIG_FILE"}\n',
                    ")\n",
                    "\n",
                    "def reject_retired_environment(environ):\n",
                    "    for key in RETIRED_ENVIRONMENT_KEYS:\n",
                    "        if key in environ:\n",
                    '            raise ValueError(f"retired key: {key}")\n',
                )
            ),
            encoding="utf-8",
        )
        _ = subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        if scan(root):
            raise ScopeError("safe API-first fixture was rejected")
        cases = {
            "env.py": 'facility = os.environ["API_FACILITY_ID"]\n',
            "getenv.py": 'facility = os.getenv("API_FACILITY_ID")\n',
            "pydantic_alias.py": "".join(
                (
                    "facility_id: str | None = Field(\n",
                    '    default=None, validation_alias="API_FACILITY_ID"\n',
                    ")\n",
                )
            ),
            "hidden_alias.py": "".join(
                (
                    'FACILITY_KEY = "API_FACILITY_ID"  # scope-fidelity: name-only\n',
                    "facility = os.environ[FACILITY_KEY]\n",
                )
            ),
            "denylist_config_read.py": "".join(
                (
                    'DENIED_KEYS = frozenset({"EDGE_CAMERA_CONFIG"})\n',
                    "camera_config = os.environ[next(iter(DENIED_KEYS))]\n",
                )
            ),
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


def scan_repo(root: Path) -> None:
    findings = scan(root)
    if findings:
        details = ", ".join(
            f"{finding.path}:{finding.line} ({finding.category})" for finding in findings
        )
        raise ScopeError(details)


def parse_cli(argv: list[str]) -> tuple[bool, bool, Path | None, Path | None]:
    fixture = False
    repo = False
    plan: Path | None = None
    evidence: Path | None = None
    index = 0
    while index < len(argv):
        match argv[index]:
            case "--fixture":
                fixture = True
                index += 1
            case "--repo":
                repo = True
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
    return fixture, repo, plan, evidence


def main() -> int:
    fixture, repo, plan, evidence = parse_cli(sys.argv[1:])
    if fixture:
        run_fixture()
        return 0
    root = Path(__file__).resolve().parents[1]
    # `--repo` runs the residue scan on its own. The --plan/--evidence mode
    # below also scans, but requires a plan file plus 17 evidence files, so it
    # cannot run in CI -- which is why the scan never actually guarded the
    # tree and env residue (EDGE_FACILITY_TOKEN in compose.edge.yaml) survived
    # despite already matching IDENTITY_PATTERN.
    if repo:
        scan_repo(root)
        print("SCOPE_FIDELITY_REPO_OK")
        return 0
    if plan is None or evidence is None:
        raise ScopeError("--plan and --evidence are required")
    scan_repo(root)
    verify_plan_and_evidence(plan.resolve(), evidence.resolve())
    print("SCOPE_FIDELITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
