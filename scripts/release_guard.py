"""Refuse to release unless every version carrier agrees with the tag.

A release of this repository is cut by pushing an annotated tag shaped
``seeon-edge-v<semver>``. That tag is the only thing an operator sees, so it
must not be able to disagree with what the tree says about itself. This module
is the guard: it reads the PRODUCT version out of every file that states one,
requires them all to be identical, and — when a tag is being released —
requires the tag to be exactly ``seeon-edge-v`` plus that version.

It fails loudly: every carrier and its value is printed on the failure path, so
the operator sees which file is out of step instead of a bare mismatch.

Run it by hand before tagging:

    python3 scripts/release_guard.py                       # lockstep only
    python3 scripts/release_guard.py --tag seeon-edge-v0.1.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

TAG_PREFIX = "seeon-edge-v"

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every file in the tree that states the PRODUCT version, and how to read it.
#: Keep this exhaustive — the guard is worth exactly as much as this list is.
TOML_CARRIERS: tuple[str, ...] = (
    "pyproject.toml",
    "backend/pyproject.toml",
    "worker/pyproject.toml",
    "shared/pyproject.toml",
)
JSON_CARRIERS: tuple[str, ...] = ("front/package.json",)

# Deliberately NOT carriers — do not add them, and do not "fix" them to match a
# release tag:
#
#   front/src/shared/releaseIdentity.ts
#       EDGE_DATABASE_FORMAT_IDENTITY = 'seeon-edge-v1' is the on-disk DATABASE
#       FORMAT identity, paired with EDGE_DATABASE_SCHEMA_VERSION = 18. It only
#       coincidentally spells like the tag. It moves when the SQLite format
#       lineage changes, never when the product ships; bumping it to track a
#       release would tell every edge device its existing database belongs to a
#       different format lineage.
#   worker/runtime/provenance/environment.py
#       Reports torch / CUDA / NVIDIA driver versions — other people's versions.
#       MANIFEST_SCHEMA_VERSION is the export manifest's own schema, and
#       exporter_version is ultralytics'.

_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _read_toml_version(path: Path) -> str:
    return str(tomllib.loads(path.read_text(encoding="utf-8"))["project"]["version"])


def _read_json_version(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["version"])


def read_carriers(root: Path = REPO_ROOT) -> dict[str, str]:
    """Map every carrier's repo-relative path to the version it states."""
    carriers: dict[str, str] = {}
    for relative in TOML_CARRIERS:
        carriers[relative] = _read_toml_version(root / relative)
    for relative in JSON_CARRIERS:
        carriers[relative] = _read_json_version(root / relative)
    return carriers


def check(carriers: dict[str, str], tag: str | None) -> list[str]:
    """Return every reason this tree must not be released. Empty means go."""
    problems: list[str] = []
    versions = set(carriers.values())
    if len(versions) != 1:
        problems.append("version carriers are not in lockstep with each other")
    version = carriers["pyproject.toml"]
    if not _SEMVER.match(version):
        problems.append(f"pyproject.toml version {version!r} is not <major>.<minor>.<patch>")
    if tag is not None and tag != f"{TAG_PREFIX}{version}":
        expected = f"{TAG_PREFIX}{version}"
        problems.append(f"tag {tag!r} does not match the tree (expected {expected!r})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default=None,
        help="the tag being released; omit to check carrier lockstep only",
    )
    parser.add_argument(
        "--print-expected-tag",
        action="store_true",
        help="print the tag this tree expects and nothing else (still guards lockstep)",
    )
    args = parser.parse_args(argv)

    carriers = read_carriers()
    problems = check(carriers, args.tag)

    if args.print_expected_tag:
        if problems:
            for problem in problems:
                print(f"::error::{problem}", file=sys.stderr)
            return 1
        print(f"{TAG_PREFIX}{carriers['pyproject.toml']}")
        return 0

    width = max(len(name) for name in carriers)
    report = "\n".join(f"  {name:<{width}} = {value}" for name, value in carriers.items())
    expected = f"{TAG_PREFIX}{carriers['pyproject.toml']}"

    if not problems:
        print("version carriers agree:")
        print(report)
        print(f"  {'expected tag':<{width}} = {expected}")
        if args.tag is not None:
            print(f"  {'tag':<{width}} = {args.tag}")
        return 0

    for problem in problems:
        print(f"::error::{problem}")
    print("every version carrier in the tree:", file=sys.stderr)
    print(report, file=sys.stderr)
    print(f"  {'expected tag':<{width}} = {expected}", file=sys.stderr)
    if args.tag is not None:
        print(f"  {'tag':<{width}} = {args.tag}", file=sys.stderr)
    print(
        "Bring every carrier to the same version and retag; do not weaken this guard.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
