"""Report every surviving reference to a deleted tree.

P2 deletes whole packages. A deletion is only finished when nothing still names
the thing that is gone - not just Python imports, but the string and path
references that live in Dockerfiles, compose files, CI workflows, shell scripts,
manifests and docs, none of which any type checker will catch.

The scan walks tracked text files (``git ls-files``), skips binaries and the
handful of generated artefacts that legitimately embed old paths, and prints one
line per surviving reference. Empty output for every listed tree is P2-AC5.

    uv run python scripts/deletion_closure_scan.py
    uv run python scripts/deletion_closure_scan.py --tree worker/native
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

#: Trees P2 removes. A reference to any of these after deletion is a leak.
DELETED_TREES: tuple[str, ...] = (
    "worker/native",
    "worker/runtime/deepstream",
    "worker/adapters/decode",
    "worker/adapters/encode",
    "worker/pipeline/ingest",
    "worker/pipeline/bus",
    "worker/tools/deepstream_canary",
    "worker/pipeline/perception/tracker.py",
)

#: Files that may legitimately name a deleted path: lockfiles nobody edits by
#: hand, immutable recorded measurements, vendored design output, and this
#: scan's own declaration of the trees.
EXCLUDED: tuple[str, ...] = (
    "front/pnpm-lock.yaml",
    "uv.lock",
    "front/design-handoff/",
    "scripts/qa/pyservicemaker-spike/receipts/",
    "scripts/deletion_closure_scan.py",
    ".gjc/",
)

_BINARY_SNIFF = 4096


def tracked_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=False
    ).stdout
    return [Path(name.decode()) for name in listing.split(b"\0") if name]


def _is_excluded(path: Path) -> bool:
    text = path.as_posix()
    return any(text == entry or text.startswith(entry) for entry in EXCLUDED)


def _is_text(path: Path) -> bool:
    try:
        head = path.open("rb").read(_BINARY_SNIFF)
    except OSError:
        return False
    return b"\0" not in head


def _patterns(trees: Sequence[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Forms a surviving reference can take.

    Three, because a leak hides in whichever one you forgot: the repo-relative
    path, the dotted module, and the *relative* path a file inside the tree's
    parent would use - a Dockerfile under `worker/` naming `native/deepstream`,
    or a CMake file naming `../native`. The relative form is built from the last
    two segments rather than the bare leaf, because a leaf like `decode` or
    `bus` appears everywhere in ordinary prose and would drown the report.
    """
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for tree in trees:
        forms = {re.escape(tree), re.escape(tree.removesuffix(".py").replace("/", "."))}
        segments = tree.split("/")
        if len(segments) > 2:
            forms.add(re.escape("/".join(segments[-2:])))
        patterns.append((tree, re.compile("|".join(sorted(forms)))))
    return patterns


def scan(trees: Sequence[str], files: Iterable[Path]) -> list[tuple[str, Path, int, str]]:
    compiled = _patterns(trees)
    hits: list[tuple[str, Path, int, str]] = []
    for path in files:
        if _is_excluded(path) or not path.is_file() or not _is_text(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            for tree, pattern in compiled:
                if pattern.search(line):
                    hits.append((tree, path, number, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(prog="deletion-closure-scan")
    _ = parser.add_argument(
        "--tree",
        action="append",
        help="scan only this tree; repeatable. Defaults to every tree P2 deletes.",
    )
    args = parser.parse_args()
    selected: list[str] | None = args.tree
    trees: tuple[str, ...] = tuple(selected) if selected else DELETED_TREES

    hits = scan(trees, tracked_files())
    for tree, path, number, line in hits:
        print(f"{tree}\t{path}:{number}\t{line[:160]}")
    if hits:
        print(f"\n{len(hits)} surviving reference(s) across {len({h[0] for h in hits})} tree(s)")
        return 1
    print(f"no surviving references to {len(trees)} deleted tree(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
