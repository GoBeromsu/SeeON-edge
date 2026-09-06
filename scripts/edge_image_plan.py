"""Decide, per edge image, whether that image has to be rebuilt for a release.

Both Dockerfiles take ``SOURCE_REVISION`` and stamp it into
``org.opencontainers.image.revision`` (``Dockerfile.backend`` and
``Dockerfile.edge``), so building an unchanged tree at a new commit produces a
*new* digest. Per-image isolation therefore cannot come from "the rebuild is
reproducible" -- it never is. It has to come from **not rebuilding**: when an
image's inputs did not change, the digest that is already published is given the
new release's tag and emitted unchanged into the release manifest.

Two things have to be right for that to be safe.

**What counts as an input.** Not a matter of taste: it is exactly what each
Dockerfile ``COPY``s out of the build context, plus the Dockerfile itself, plus
the files that decide what the context *contains*
(``.dockerignore``) or how it is built (the workflow). A path that belongs to an
image but is missing from its set means the image is not rebuilt when it should
be, and the release ships a stale digest under a new version.
``tests/test_edge_image_isolation.py`` re-derives the ``COPY`` sources from both
Dockerfiles and fails if anything here drifts from them.

**What to compare against.** Not the previous release's commit -- the commit
that actually built the digest being reused. Those differ as soon as an image is
reused twice: if ``ml-api`` was built at C1 for v0.1.0 and reused for v0.2.0,
then at v0.3.0 the published digest is still the one built at C1, so the
comparison has to run C1..C3. Comparing against v0.2.0's commit would skip
everything that changed between C1 and C2 and reuse a genuinely stale image.
The build commit is read back from the image's own
``org.opencontainers.image.revision`` label, which makes the registry
self-describing and the reuse chain self-correcting.

Unknown paths **fail closed**: a path this module cannot classify is treated as
affecting *both* images, so a new top-level directory costs a redundant rebuild
rather than a silently skipped one.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ML_API = "ml-api"
ML_WORKER = "ml-worker"
IMAGES = (ML_API, ML_WORKER)
BOTH = frozenset(IMAGES)
NEITHER: frozenset[str] = frozenset()

#: image -> the Dockerfile that builds it.
DOCKERFILES = {ML_API: "Dockerfile.backend", ML_WORKER: "Dockerfile.edge"}

#: Release tags this repository cuts, as a `git tag --list` glob.
TAG_GLOB = "seeon-edge-v*"

#: Paths that reach the ``ml-api`` build: the Dockerfile itself and every
#: ``COPY`` source in it. ``scripts/ops`` is the one that is easy to miss --
#: ``Dockerfile.backend`` copies it in, and a hand-written input set that omits
#: it would let an operator tool change ship without a rebuild. ``front/``
#: already covers the two lockfiles the front stage copies by name.
_ML_API_INPUTS = (
    "Dockerfile.backend",
    "backend/",
    "contracts/",
    "front/",
    "pyproject.toml",
    "scripts/ops/",
    "shared/",
    "uv.lock",
)

#: Paths that reach the ``ml-worker`` build (``COPY`` sources in
#: ``Dockerfile.edge``). ``worker/`` covers both the DeepStream native sources
#: and the pinned model manifest ``worker/tools/fetch_models/manifest.json``,
#: which is why neither is listed separately.
_ML_WORKER_INPUTS = (
    "Dockerfile.edge",
    "contracts/",
    "pyproject.toml",
    "shared/",
    "uv.lock",
    "worker/",
)

#: Paths that can change *either* image without either Dockerfile naming them.
#: ``.dockerignore`` decides what the build context contains for both builds;
#: the image workflow carries the build args, the base-image build flags and the
#: tag scheme, so a change to it can change what gets built.
_SHARED_INPUTS = (
    ".dockerignore",
    ".github/workflows/edge-images.yml",
)

#: Paths that provably never reach either build: nothing ``COPY``s them, and/or
#: ``.dockerignore`` strips them from the context. Listing them explicitly is
#: what stops a docs or CI edit from rebuilding both images. Anything NOT listed
#: here and not claimed above still falls through to the fail-closed default.
_NEUTRAL_INPUTS = (
    ".claude/",
    ".env.edge.prod.example",
    ".env.example",
    ".github/",
    ".gitignore",
    ".gitleaksignore",
    ".omo/",
    ".pre-commit-config.yaml",
    ".python-version",
    "AGENTS.md",
    "DESIGN.md",
    "LICENSE",
    "README.md",
    "artifacts/",
    "compose.edge.dev.yaml",
    "compose.edge.yaml",
    "docs/",
    "edge-env-inventory.json",
    "models/",
    # Only `scripts/ops` is COPYed (into ml-api); everything else under
    # `scripts/` is host-side tooling. The longer `scripts/ops/` rule wins.
    "scripts/",
    "tests/",
    "tests_support/",
)


def _build_rules() -> tuple[tuple[str, frozenset[str]], ...]:
    """Every classification rule, longest prefix first (most specific wins)."""
    collected: list[tuple[str, frozenset[str]]] = []
    collected.extend((prefix, BOTH) for prefix in _SHARED_INPUTS)
    collected.extend((prefix, frozenset({ML_API})) for prefix in _ML_API_INPUTS)
    collected.extend((prefix, frozenset({ML_WORKER})) for prefix in _ML_WORKER_INPUTS)
    collected.extend((prefix, NEITHER) for prefix in _NEUTRAL_INPUTS)

    merged: dict[str, frozenset[str]] = {}
    for prefix, images in collected:
        merged[prefix] = merged.get(prefix, NEITHER) | images
    return tuple(sorted(merged.items(), key=lambda item: -len(item[0])))


#: Public so the tests can prove the classification is a closed world.
RULES = _build_rules()


def _matches(path: str, prefix: str) -> bool:
    if prefix.endswith("/"):
        return path == prefix.rstrip("/") or path.startswith(prefix)
    return path == prefix


def is_classified(path: str) -> bool:
    """Whether any explicit rule claims ``path`` (as opposed to failing closed)."""
    return any(_matches(path, prefix) for prefix, _ in RULES)


def affected_images(path: str) -> frozenset[str]:
    """Which images a change to ``path`` can affect.

    Fail-closed: an unclassified path affects both images. That is the point --
    a new top-level directory must cost a redundant rebuild, never a skipped one.
    """
    for prefix, images in RULES:
        if _matches(path, prefix):
            return images
    return BOTH


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _short(reference: str) -> str:
    """Abbreviate a 40-hex commit; leave tags and refs like HEAD intact."""
    if len(reference) == 40 and all(c in "0123456789abcdef" for c in reference):
        return reference[:12]
    return reference


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def previous_tag(tag: str) -> str | None:
    """The highest existing release tag that is not ``tag`` itself."""
    listed = _git("tag", "--list", TAG_GLOB, "--sort=-version:refname").splitlines()
    for candidate in listed:
        if candidate and candidate != tag:
            return candidate
    return None


def changed_paths(base: str, head: str) -> list[str]:
    return [line for line in _git("diff", "--name-only", f"{base}..{head}").splitlines() if line]


def decide(image: str, base: str, head: str) -> dict[str, object]:
    """Whether ``image`` must be rebuilt for ``head``, given it was built at ``base``."""
    paths = changed_paths(base, head)
    triggers = [path for path in paths if image in affected_images(path)]
    unclassified = [path for path in triggers if not is_classified(path)]
    return {
        "image": image,
        "base": base,
        "head": head,
        "build": bool(triggers),
        "total_changed": len(paths),
        "input_changed": len(triggers),
        "triggers": triggers,
        "unclassified": unclassified,
    }


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


class DigestNotPreserved(RuntimeError):
    """`imagetools create` produced a different digest than the source manifest."""


def _run(argv: Iterable[str]) -> str:
    argv = list(argv)
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


Runner = Callable[[list[str]], str]


def published_digest(repo: str, tag: str, run: Runner | None = None) -> str:
    """The digest the registry currently resolves ``repo:tag`` to."""
    run = run or _run
    return run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            f"{repo}:{tag}",
            "--format",
            "{{.Manifest.Digest}}",
        ]
    ).strip()


def published_revision(repo: str, reference: str, run: Runner | None = None) -> str:
    """The commit that built ``repo@reference``, from its own OCI revision label.

    This is what makes the registry self-describing: a reused image keeps the
    label of the release that actually *built* it, so a reuse chain always
    compares against the true build commit instead of drifting forward one
    release at a time.
    """
    run = run or _run
    payload = run(
        ["docker", "buildx", "imagetools", "inspect", reference, "--format", "{{json .Image}}"]
    )
    revisions = {
        labels.get("org.opencontainers.image.revision")
        for labels in _iter_label_maps(json.loads(payload))
        if labels
    }
    revisions.discard(None)
    if len(revisions) != 1:
        raise RuntimeError(
            f"{reference} carries {len(revisions)} distinct "
            f"org.opencontainers.image.revision labels ({sorted(revisions)}); "
            "cannot determine the commit that built it"
        )
    return str(revisions.pop())


def _iter_label_maps(node: object) -> Iterable[dict[str, str]]:
    """Every ``config.Labels`` map in an image config, single- or multi-platform.

    `imagetools inspect --format '{{json .Image}}'` returns one config object for
    a single-platform image and a platform-keyed map for a multi-platform one.
    """
    if isinstance(node, dict):
        config = node.get("config")
        if isinstance(config, dict) and isinstance(config.get("Labels"), dict):
            yield config["Labels"]
        for value in node.values():
            yield from _iter_label_maps(value)


def retag_preserving_digest(repo: str, digest: str, tag: str, run: Runner | None = None) -> str:
    """Add ``tag`` to the manifest already published at ``repo@digest``.

    This is the whole reuse mechanism: it copies a manifest to a new tag without
    rebuilding, which is the only way to give a new release's tag to an image
    whose inputs did not change. Rebuilding would restamp
    ``org.opencontainers.image.revision`` and mint a different digest.

    It preserves the digest when the source is an OCI **index** -- which is what
    ``docker/build-push-action`` pushes, because buildx attaches a provenance
    attestation and therefore an index. It does **not** preserve the digest when
    the source is a plain single manifest: buildx wraps that in a fresh index and
    the digest changes. Both cases were measured against a real registry, so
    this is an observed property rather than an assumption -- which is exactly
    why the result is re-inspected here every time. A changed digest is a hard
    failure: emitting it into the seal would republish a different artefact under
    a "reused" label, and the runbook's whole premise is that the seal's digests
    are the ones an operator can pull.
    """
    run = run or _run
    source = f"{repo}@{digest}"
    run(["docker", "buildx", "imagetools", "create", "-t", f"{repo}:{tag}", source])
    observed = published_digest(repo, tag, run=run)
    if observed != digest:
        raise DigestNotPreserved(
            f"re-tagging {source} as {repo}:{tag} changed the digest to {observed}. "
            "The source manifest was probably not an index, so buildx re-wrapped it. "
            "Refusing to record a 'reused' image under a digest that nothing built."
        )
    return observed


def commit_exists(commit: str) -> bool:
    try:
        _git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        return False
    return True


def plan(
    image: str,
    head: str,
    release_tag: str,
    repo: str,
    reuse_eligible: bool,
    run: Runner | None = None,
    previous: str | None = None,
) -> dict[str, object]:
    """The full per-image decision, registry lookup and all.

    Every way of *not* knowing resolves to "build". Reuse requires positive
    evidence: a previous release tag, a digest the registry still resolves it
    to, a revision label on that digest, a commit that label names which this
    repository actually has, and no input change since it. Anything missing and
    the image is rebuilt -- the failure mode of a needless rebuild is a slow
    release, and the failure mode of a wrong reuse is shipping a stale image
    under a new version.
    """

    def build(reason: str, **extra: object) -> dict[str, object]:
        return {
            "image": image,
            "build": True,
            "reuse": False,
            "reason": reason,
            "head": head,
            **extra,
        }

    if not reuse_eligible:
        return build(
            "this event always builds (reuse is a release-time decision, and the "
            "boot-smoke gate must exercise a freshly built image)"
        )

    # Resolved from the repository's tags unless the caller names one. Tests
    # name it so they do not depend on the checkout having fetched tags; the
    # workflow lets it resolve (its checkout uses `fetch-depth: 0`).
    if previous is None:
        previous = previous_tag(release_tag)
    if not previous:
        return build(f"no previous {TAG_GLOB} tag exists, so there is no digest to reuse")

    # Two references can name the previous release's image, and both are tried.
    # The release tag is the legible one and is pushed from here on. The
    # full-commit-SHA tag is the one this workflow has pushed since it existed,
    # so it is what resolves for releases cut before per-image isolation landed
    # (v0.1.0 among them). Whichever resolves, the digest is the same artefact.
    candidates = [previous]
    with contextlib.suppress(subprocess.CalledProcessError):
        candidates.append(_git("rev-list", "-n", "1", previous))

    digest = ""
    for reference in candidates:
        try:
            digest = published_digest(repo, reference, run=run)
        except (subprocess.CalledProcessError, OSError):
            continue
        if digest.startswith("sha256:"):
            break
        digest = ""
    if not digest:
        return build(
            f"the registry resolves none of {', '.join(f'{repo}:{ref}' for ref in candidates)}"
        )

    try:
        base = published_revision(repo, f"{repo}@{digest}", run=run)
    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as error:
        return build(f"{repo}@{digest} has no usable revision label ({error})")

    if not commit_exists(base):
        return build(
            f"{repo}@{digest} names build commit {base}, which is not in this repository's history"
        )

    decision = decide(image, base, head)
    decision.update(
        {
            "reuse": not decision["build"],
            "previous_tag": previous,
            "previous_digest": digest,
            "reason": (
                f"{decision['input_changed']} input path(s) changed since {_short(base)}, "
                f"which built {previous}"
            )
            if decision["build"]
            else (
                f"no input path changed since {_short(base)}, which built {previous}; "
                "re-tagging that digest instead of rebuilding"
            ),
        }
    )
    return decision


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit(name: str, value: str) -> None:
    destination = os.environ.get("GITHUB_OUTPUT")
    if not destination:
        return
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def render_decision(decision: dict[str, object]) -> str:
    """Human-readable decision. The workflow must never decide silently."""
    verdict = "BUILD" if decision["build"] else "REUSE"
    lines = [f"{decision['image']}: {verdict} -- {decision['reason']}"]
    if decision.get("previous_tag"):
        lines.append(f"  previous release : {decision['previous_tag']}")
        lines.append(f"  published digest : {decision['previous_digest']}")
    if decision.get("base"):
        lines.append(f"  built at (base)  : {decision['base']}")
    lines.append(f"  releasing (head) : {decision['head']}")
    triggers = decision.get("triggers") or []
    if triggers:
        lines.append(
            f"  {decision['input_changed']} of {decision['total_changed']} changed "
            f"path(s) are inputs to {decision['image']}:"
        )
        lines.extend(f"    - {path}" for path in triggers[:25])  # type: ignore[index]
        if len(triggers) > 25:  # type: ignore[arg-type]
            lines.append(f"    ... and {len(triggers) - 25} more")  # type: ignore[arg-type]
    if decision.get("unclassified"):
        lines.append("  UNCLASSIFIED (fail-closed -> treated as affecting both images):")
        lines.extend(f"    ! {path}" for path in decision["unclassified"][:25])  # type: ignore[index]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-image edge release build plan.")
    sub = parser.add_subparsers(dest="command", required=True)

    paths = sub.add_parser("paths", help="print the input path set for an image")
    paths.add_argument("--image", choices=IMAGES, required=True)

    classify = sub.add_parser("classify", help="print which images a path affects")
    classify.add_argument("path")

    previous = sub.add_parser("previous-tag", help="print the previous release tag")
    previous.add_argument("--tag", required=True)

    decide_parser = sub.add_parser("decide", help="decide whether an image must be rebuilt")
    decide_parser.add_argument("--image", choices=IMAGES, required=True)
    decide_parser.add_argument(
        "--base", required=True, help="commit that built the currently published digest"
    )
    decide_parser.add_argument("--head", required=True, help="commit being released")

    plan_parser = sub.add_parser("plan", help="full per-image decision, registry lookup included")
    plan_parser.add_argument("--image", choices=IMAGES, required=True)
    plan_parser.add_argument("--head", required=True, help="commit being released")
    plan_parser.add_argument("--release-tag", required=True, help="the tag being released")
    plan_parser.add_argument("--repo", required=True, help="e.g. ghcr.io/<ns>/ml-api")
    plan_parser.add_argument(
        "--reuse-eligible",
        choices=("true", "false"),
        required=True,
        help="whether this event is allowed to reuse a published digest at all",
    )
    plan_parser.add_argument(
        "--json-out",
        help="write the decision as JSON here, for later steps to read verbatim",
    )

    retag = sub.add_parser("retag", help="give a published digest a new tag, digest preserved")
    retag.add_argument("--repo", required=True)
    retag.add_argument("--digest", required=True)
    retag.add_argument("--tag", required=True)

    args = parser.parse_args(argv)

    if args.command == "paths":
        for prefix, images in sorted(RULES):
            if args.image in images:
                print(prefix)
        return 0

    if args.command == "classify":
        images = affected_images(args.path)
        print(" ".join(sorted(images)) if images else "(neither)")
        return 0

    if args.command == "previous-tag":
        print(previous_tag(args.tag) or "")
        return 0

    if args.command == "retag":
        print(retag_preserving_digest(args.repo, args.digest, args.tag))
        return 0

    if args.command == "decide":
        decision = decide(args.image, args.base, args.head)
        decision["reason"] = f"inputs compared against {_short(args.base)}"
        print(render_decision(decision))
        return 0

    decision = plan(
        args.image,
        args.head,
        args.release_tag,
        args.repo,
        reuse_eligible=args.reuse_eligible == "true",
    )
    print(render_decision(decision))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _emit("build", "true" if decision["build"] else "false")
    _emit("reason", str(decision["reason"]))
    _emit("previous-tag", str(decision.get("previous_tag") or ""))
    _emit("previous-digest", str(decision.get("previous_digest") or ""))
    _emit("base", str(decision.get("base") or ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
