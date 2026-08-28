"""Architecture-audit H3 structural guard: `worker/domains/**` must not import
or expose NumPy, OpenCV, CUDA, or TensorRT types.

Detection modules interpret numeric observations into business events; the
domain layer is deliberately numeric/hardware-agnostic and independent of
which infrastructure profile (CPU/MPS/CUDA) or inference library
(NumPy/OpenCV/PyTorch/TensorRT) an adapter happens to use underneath. This
test parses every `worker/domains/**/*.py` module with `ast` (no import,
since importing would require the banned packages to be installed to even
fail correctly) and asserts:

1. No module-level `import`/`from ... import ...` statement names a banned
   package (or a CUDA-flavored submodule of an otherwise-allowed package).
2. No public function/method signature annotates a parameter or return
   value with a bare `Any` (or `typing.Any`) -- an escape hatch that would
   let a hardware/library type slip through unexamined.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DOMAINS_ROOT: Final = REPO_ROOT / "worker" / "domains"

# Named explicitly by the architecture-audit H3 finding: NumPy, OpenCV,
# CUDA, TensorRT. `pycuda`/`cupy`/`onnxruntime`/`torch` are included as the
# concrete packages that would carry those types into Python; any import
# whose dotted path contains "cuda" is banned regardless of top-level
# package (covers `torch.cuda`, `numba.cuda`, etc.).
_BANNED_ROOT_MODULES: Final = frozenset(
    {
        "numpy",
        "cv2",
        "torch",
        "tensorrt",
        "pycuda",
        "cupy",
        "onnxruntime",
        "ultralytics",
    }
)


def _domain_source_files() -> tuple[Path, ...]:
    return tuple(sorted(DOMAINS_ROOT.rglob("*.py")))


def _imported_dotted_names(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return tuple(names)


def _is_banned_import(dotted_name: str) -> bool:
    root = dotted_name.split(".", maxsplit=1)[0]
    return root in _BANNED_ROOT_MODULES or "cuda" in dotted_name.lower()


@pytest.mark.parametrize(
    "path",
    _domain_source_files(),
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_domain_module_imports_no_banned_hardware_library(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = tuple(name for name in _imported_dotted_names(tree) if _is_banned_import(name))
    assert offenders == (), (
        f"{path.relative_to(REPO_ROOT)} imports banned hardware/library "
        f"module(s) {offenders!r}; worker/domains must stay numeric/"
        "hardware-agnostic (architecture-audit H3)"
    )


def _is_any_annotation(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id == "Any"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "Any"
    if isinstance(annotation, ast.Subscript):
        return _is_any_annotation(annotation.value)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _is_any_annotation(annotation.left) or _is_any_annotation(annotation.right)
    return False


def _public_function_defs(tree: ast.Module) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    )


@pytest.mark.parametrize(
    "path",
    _domain_source_files(),
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_domain_public_signatures_never_annotate_any(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for function in _public_function_defs(tree):
        annotated = list(function.args.args) + list(function.args.kwonlyargs)
        for argument in annotated:
            if argument.arg in ("self", "cls"):
                continue
            if _is_any_annotation(argument.annotation):
                offenders.append(f"{function.name}({argument.arg})")
        if _is_any_annotation(function.returns):
            offenders.append(f"{function.name}() -> Any")
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} exposes `Any` on public signature(s) "
        f"{offenders!r}; worker/domains public signatures must not use `Any` "
        "(architecture-audit H3)"
    )
