"""Vendor drift firewall (ADR-0006), fall-ai side.

this repository stays canonical for ``contracts``; eldercare-dataset-ops keeps
a vendored copy. (``features`` moved into ``edge/features/`` in the 3-instance
refactor and is no longer drift-checked here.) This test byte-compares
this repo's canonical copies against the sibling dataset-ops checkout on
every run so the two never silently diverge (the mirror-image of dataset-ops's
own ``tests/test_vendor_drift.py``, which runs the same check from its side).

Sibling path: ``DATASET_OPS_REPO``, defaulting to ``../eldercare-dataset-ops``
relative to this repo's root. When the sibling checkout is absent (e.g. this
repo is checked out in isolation, or in a worktree lane with no adjacent
dataset-ops checkout), there is nothing to diff against -- the test is
SKIPPED with a loud warning rather than failing.
"""

from __future__ import annotations

import ast
import os
import warnings
from pathlib import Path

import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
VENDORED_PACKAGES = ("contracts",)

# reacquire's default legitimately differs per repo (each names its own
# retrain command); every other ModelMetadata field must match exactly.
_SCHEMA_DIVERGENT_FIELDS = {"reacquire"}

_IGNORED_DIR_NAMES = {"__pycache__"}
_IGNORED_SUFFIXES = (".pyc",)


def _dataset_ops_ml_root() -> Path:
    override = os.environ.get("DATASET_OPS_REPO")
    base = Path(override).expanduser() if override else (ML_ROOT.parent / "eldercare-dataset-ops")
    return (base / "ml").resolve()


def _snapshot(root: Path) -> dict[str, Path]:
    """Map relpath -> file Path for every real file under *root* (caches excluded)."""
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        files[str(path.relative_to(root))] = path
    return files


def _field_signatures(path: Path, class_name: str) -> dict[str, tuple[str, str | None]]:
    """Map field name -> (annotation_src, default_src) for a dataclass in *path*.

    Parsed via ``ast`` rather than imported: dataset-ops's ``training/`` pulls
    in its own dependency set, which this repo does not (and should not) need
    installed just to run this check.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            sig: dict[str, tuple[str, str | None]] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    default = ast.unparse(stmt.value) if stmt.value is not None else None
                    sig[stmt.target.id] = (ast.unparse(stmt.annotation), default)
            return sig
    raise AssertionError(f"class {class_name!r} not found in {path}")


DATASET_OPS_ML = _dataset_ops_ml_root()

if not DATASET_OPS_ML.is_dir():
    _warning = (
        f"DATASET_OPS_REPO sibling checkout not found at {DATASET_OPS_ML} -- SKIPPING "
        "the vendor drift check against eldercare-dataset-ops's ml/contracts, "
        "ml/features, and ModelMetadata schema. This test cannot confirm those stay "
        "in sync with the dataset-ops copy. Set DATASET_OPS_REPO to the sibling repo "
        "path if it lives somewhere other than ../eldercare-dataset-ops (e.g. from a "
        "worktree lane)."
    )
    if os.environ.get("DATASET_OPS_REQUIRED") == "1":
        raise RuntimeError(f"DATASET_OPS_REQUIRED=1: {_warning}")
    warnings.warn(_warning, stacklevel=1)
    print(f"\nWARNING: {_warning}\n")  # noqa: T201 -- loud on purpose, warnings can be swallowed
    pytestmark = pytest.mark.skip(reason=_warning)


@pytest.mark.parametrize("package", VENDORED_PACKAGES)
def test_vendor_package_matches_dataset_ops(package: str) -> None:
    ours_root = ML_ROOT / package  # canonical, owned by this repo
    theirs_root = DATASET_OPS_ML / package  # dataset-ops's vendored mirror
    assert ours_root.is_dir(), f"canonical package missing in this repo: {ours_root}"
    assert theirs_root.is_dir(), f"vendored copy missing in sibling repo: {theirs_root}"

    ours = _snapshot(ours_root)
    theirs = _snapshot(theirs_root)

    missing_there = sorted(set(ours) - set(theirs))
    extra_there = sorted(set(theirs) - set(ours))
    assert not missing_there, (
        f"dataset-ops's ml/{package}/ is missing files present in this repo's "
        f"canonical copy: {missing_there}"
    )
    assert not extra_there, (
        f"dataset-ops's ml/{package}/ has extra files not present in this repo's "
        f"canonical copy: {extra_there}"
    )

    mismatched = sorted(
        relpath
        for relpath, path in ours.items()
        if path.read_bytes() != theirs[relpath].read_bytes()
    )
    assert not mismatched, (
        f"dataset-ops's ml/{package}/ has drifted from this repo's canonical copy "
        f"(re-sync dataset-ops from fall-ai, ADR-0006): {mismatched}"
    )
