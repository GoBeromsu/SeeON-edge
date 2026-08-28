"""The per-image release isolation contract.

`scripts/edge_image_plan.py` decides, per image, whether that image has to be
rebuilt for a release. Getting an input set wrong is not cosmetic: a path that
belongs to an image but is missing from its set means the image is NOT rebuilt
when it should be, and the release ships a stale digest under a new version. So
the sets are not spot-checked here -- they are re-derived from the `COPY`
instructions in both Dockerfiles and compared against what the module claims.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.edge_image_plan import (
    BOTH,
    DOCKERFILES,
    ML_API,
    ML_WORKER,
    RULES,
    DigestNotPreserved,
    affected_images,
    decide,
    is_classified,
    plan,
    retag_preserving_digest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_COPY = re.compile(r"^COPY\s+(?P<rest>.+)$", re.MULTILINE)


def _context_copy_sources(dockerfile: str) -> set[str]:
    """Every build-context path a Dockerfile COPYs, read from the file itself.

    `--from=` copies take their source from an earlier build stage, not from the
    repository, so they are not inputs.
    """
    text = (REPO_ROOT / dockerfile).read_text(encoding="utf-8")
    sources: set[str] = set()
    for match in _COPY.finditer(text):
        parts = match.group("rest").split()
        if any(part.startswith("--from=") for part in parts):
            continue
        operands = [part for part in parts if not part.startswith("--")]
        # The last operand is the destination inside the image.
        sources.update(operands[:-1])
    assert sources, dockerfile
    return sources


def _tracked_paths() -> list[str]:
    # -z: at least one tracked path contains a space, and splitting on
    # whitespace invents paths that do not exist.
    out = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [path for path in out.split("\0") if path]


# ---------------------------------------------------------------------------
# The input sets
# ---------------------------------------------------------------------------


def test_input_sets_cover_every_context_copy_in_both_dockerfiles() -> None:
    """No `COPY` source may fall outside its image's input set.

    This is the assertion that catches the easy miss: `Dockerfile.backend`
    copies `scripts/ops` into ml-api, and an input set written from memory
    leaves it out -- which would let an operator-tool change ship inside a
    "reused" image.
    """
    for image, dockerfile in DOCKERFILES.items():
        for source in sorted(_context_copy_sources(dockerfile)):
            assert image in affected_images(source), (image, dockerfile, source)


def test_each_dockerfile_is_an_input_to_the_image_it_builds() -> None:
    for image, dockerfile in DOCKERFILES.items():
        assert affected_images(dockerfile) == frozenset({image}), dockerfile


def test_the_context_shaping_files_affect_both_images() -> None:
    # `.dockerignore` decides what the build context contains for both builds,
    # and the workflow carries the build args and tag scheme. Neither is COPYed
    # by either Dockerfile, so neither is discoverable from `COPY` alone.
    assert affected_images(".dockerignore") == BOTH
    assert affected_images(".github/workflows/edge-images.yml") == BOTH


@pytest.mark.parametrize(
    "path",
    ["pyproject.toml", "uv.lock", "contracts/anything.py", "shared/anything.py"],
)
def test_shared_sources_affect_both_images(path: str) -> None:
    assert affected_images(path) == BOTH, path


@pytest.mark.parametrize(
    "path",
    [
        "Dockerfile.backend",
        "backend/app/main.py",
        "front/package.json",
        "front/pnpm-lock.yaml",
        "front/src/main.tsx",
        # Dockerfile.backend: `COPY scripts/ops ./scripts/ops`.
        "scripts/ops/repair-clip-consistency.py",
    ],
)
def test_ml_api_only_inputs(path: str) -> None:
    assert affected_images(path) == frozenset({ML_API}), path


@pytest.mark.parametrize(
    "path",
    [
        "Dockerfile.edge",
        "worker/__main__.py",
        "worker/native/deepstream/src/child.cpp",
        # The pinned model manifest lives inside `worker/`, so it is covered
        # without a rule of its own -- but it must genuinely be covered.
        "worker/tools/fetch_models/manifest.json",
    ],
)
def test_ml_worker_only_inputs(path: str) -> None:
    assert affected_images(path) == frozenset({ML_WORKER}), path


def test_host_side_scripts_are_not_an_input_to_either_image() -> None:
    # Only `scripts/ops` is COPYed anywhere; the rest of `scripts/` is host-side
    # tooling that never enters a build context.
    assert affected_images("scripts/release_guard.py") == frozenset()
    assert affected_images("scripts/edge_image_plan.py") == frozenset()


@pytest.mark.parametrize("path", ["docs/runbooks/edge-image-publish.md", "tests/test_x.py"])
def test_documentation_and_tests_rebuild_nothing(path: str) -> None:
    assert affected_images(path) == frozenset(), path


# ---------------------------------------------------------------------------
# Failing closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "brand_new_top_level/thing.py",
        "brand_new_top_level",
        "unknown_root_file.toml",
        "vendor/lib/x.c",
    ],
)
def test_an_unclassified_path_fails_closed_onto_both_images(path: str) -> None:
    """A path nobody classified must cost a redundant rebuild, never a skip.

    The dangerous direction is one-way: rebuilding an image that did not need it
    wastes ~14 minutes, while skipping one that did need it ships a stale image
    under a new version.
    """
    assert not is_classified(path)
    assert affected_images(path) == BOTH, path


def test_every_tracked_path_is_deliberately_classified() -> None:
    """No path that exists today may reach the fail-closed default.

    The default exists for paths that do not exist yet. If a tracked path hits
    it, someone added a directory without deciding which images it feeds -- so
    this turns it into a visible decision instead of a permanent both-images
    rebuild that nobody notices.
    """
    unclassified = [path for path in _tracked_paths() if not is_classified(path)]
    assert not unclassified, unclassified


def test_a_new_top_level_directory_is_not_silently_outside_both_sets() -> None:
    """The property the fail-closed default exists for, stated directly."""
    for image in DOCKERFILES:
        assert image in affected_images("some_future_service/app.py")


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def test_an_empty_range_reuses_both_images() -> None:
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    for image in DOCKERFILES:
        decision = decide(image, head, head)
        assert decision["build"] is False, decision
        assert decision["input_changed"] == 0


def test_a_docs_only_change_reuses_both_images(tmp_path: Path) -> None:
    """The isolation claim, end to end, on a real commit range."""
    repo = tmp_path / "repo"
    _seed_repo(repo)
    _commit(repo, {"docs/note.md": "hello"}, "docs only")
    base, head = _range(repo)
    for image in DOCKERFILES:
        decision = _decide_in(repo, image, base, head)
        assert decision["build"] is False, (image, decision)


def test_a_worker_only_change_rebuilds_only_ml_worker(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _seed_repo(repo)
    _commit(repo, {"worker/runtime/x.py": "x = 1"}, "worker only")
    base, head = _range(repo)
    assert _decide_in(repo, ML_WORKER, base, head)["build"] is True
    assert _decide_in(repo, ML_API, base, head)["build"] is False


def test_a_backend_only_change_rebuilds_only_ml_api(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _seed_repo(repo)
    _commit(repo, {"backend/app/x.py": "x = 1"}, "backend only")
    base, head = _range(repo)
    assert _decide_in(repo, ML_API, base, head)["build"] is True
    assert _decide_in(repo, ML_WORKER, base, head)["build"] is False


def test_a_shared_change_rebuilds_both(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _seed_repo(repo)
    _commit(repo, {"shared/x.py": "x = 1"}, "shared")
    base, head = _range(repo)
    for image in DOCKERFILES:
        assert _decide_in(repo, image, base, head)["build"] is True, image


def test_an_unknown_new_directory_rebuilds_both(tmp_path: Path) -> None:
    """Fail-closed, proven over a real diff rather than a single path lookup."""
    repo = tmp_path / "repo"
    _seed_repo(repo)
    _commit(repo, {"newthing/app.py": "x = 1"}, "a directory nobody classified")
    base, head = _range(repo)
    for image in DOCKERFILES:
        decision = _decide_in(repo, image, base, head)
        assert decision["build"] is True, image
        assert decision["unclassified"] == ["newthing/app.py"], decision


def _git_in(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _seed_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _git_in(repo, "init", "-q", "-b", "main")
    _git_in(repo, "config", "user.email", "t@example.invalid")
    _git_in(repo, "config", "user.name", "t")
    _commit(repo, {"README.md": "seed"}, "seed")


def _commit(repo: Path, files: dict[str, str], message: str) -> None:
    for name, body in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", message)


def _range(repo: Path) -> tuple[str, str]:
    return _git_in(repo, "rev-parse", "HEAD~1"), _git_in(repo, "rev-parse", "HEAD")


def _decide_in(repo: Path, image: str, base: str, head: str) -> dict[str, object]:
    """Run the real decision against a throwaway repository's history."""
    import scripts.edge_image_plan as module

    original = module.REPO_ROOT
    module.REPO_ROOT = repo
    try:
        return decide(image, base, head)
    finally:
        module.REPO_ROOT = original


# ---------------------------------------------------------------------------
# The reuse path
# ---------------------------------------------------------------------------


_SOURCE_DIGEST = "sha256:" + "a" * 64
_WRAPPED_DIGEST = "sha256:" + "b" * 64
_REPO = "ghcr.io/seniorailab/eldercare-fall-ml/ml-api"
#: Named explicitly so these tests do not depend on the checkout having fetched
#: release tags -- the CI test job does not, and an unfetched tag would make the
#: plan short-circuit before reaching the branch under test.
_PREVIOUS = "seeon-edge-v0.1.0"


def test_reuse_emits_a_digest_identical_to_the_source_digest() -> None:
    """The core promise: reuse republishes the SAME artefact, not a rebuild."""
    calls: list[list[str]] = []

    def run(argv: list[str]) -> str:
        calls.append(argv)
        return _SOURCE_DIGEST + "\n" if "inspect" in argv else ""

    emitted = retag_preserving_digest(_REPO, _SOURCE_DIGEST, "seeon-edge-v0.2.0", run=run)

    assert emitted == _SOURCE_DIGEST
    create = calls[0]
    assert create[:4] == ["docker", "buildx", "imagetools", "create"]
    # The source is a digest reference, so no build can be involved.
    assert create[-1] == f"{_REPO}@{_SOURCE_DIGEST}"
    assert f"{_REPO}:seeon-edge-v0.2.0" in create
    # And the result is verified against the registry, not assumed.
    assert any("inspect" in call for call in calls), calls


def test_reuse_refuses_a_retag_that_changed_the_digest() -> None:
    """`imagetools create` does NOT always preserve the digest.

    Measured against a real registry: preserved when the source is an OCI index
    (what `docker/build-push-action` pushes, because it attaches a provenance
    attestation), NOT preserved when the source is a plain single manifest --
    buildx wraps that in a fresh index under a new digest. A changed digest must
    fail the release rather than land in the seal labelled "reused from ...".
    """

    def run(argv: list[str]) -> str:
        return _WRAPPED_DIGEST + "\n" if "inspect" in argv else ""

    with pytest.raises(DigestNotPreserved) as raised:
        retag_preserving_digest(_REPO, _SOURCE_DIGEST, "seeon-edge-v0.2.0", run=run)
    assert _WRAPPED_DIGEST in str(raised.value)


# ---------------------------------------------------------------------------
# Every way of not knowing means "build"
# ---------------------------------------------------------------------------


def _config_with_revision(revision: str) -> str:
    """An `imagetools inspect --format '{{json .Image}}'` payload, as the CLI emits it."""
    return json.dumps(
        {"config": {"Labels": {"org.opencontainers.image.revision": revision}}}
    )


def test_an_ineligible_event_always_builds() -> None:
    decision = plan(ML_API, "HEAD", "seeon-edge-v9.9.9", _REPO, reuse_eligible=False)
    assert decision["build"] is True
    assert "always builds" in str(decision["reason"])


def test_no_previous_release_builds() -> None:
    decision = plan(ML_API, "HEAD", "seeon-edge-v9.9.9", _REPO, reuse_eligible=True, previous="")
    assert decision["build"] is True
    assert "no previous" in str(decision["reason"])


def test_an_unresolvable_registry_reference_builds() -> None:
    def run(argv: list[str]) -> str:
        raise OSError("no registry here")

    decision = plan(
        ML_API, "HEAD", "seeon-edge-v9.9.9", _REPO,
        reuse_eligible=True, run=run, previous=_PREVIOUS,
    )
    assert decision["build"] is True
    assert "resolves none of" in str(decision["reason"])


def test_a_missing_revision_label_builds() -> None:
    def run(argv: list[str]) -> str:
        if "{{.Manifest.Digest}}" in argv:
            return _SOURCE_DIGEST + "\n"
        return "{}"  # an image config carrying no labels at all

    decision = plan(
        ML_API, "HEAD", "seeon-edge-v9.9.9", _REPO,
        reuse_eligible=True, run=run, previous=_PREVIOUS,
    )
    assert decision["build"] is True
    assert "revision label" in str(decision["reason"])


def test_a_revision_naming_an_unknown_commit_builds() -> None:
    """A label pointing at a commit this repository does not have is not a base."""
    unknown = "0" * 40

    def run(argv: list[str]) -> str:
        if "{{.Manifest.Digest}}" in argv:
            return _SOURCE_DIGEST + "\n"
        return _config_with_revision(unknown)

    decision = plan(
        ML_API, "HEAD", "seeon-edge-v9.9.9", _REPO,
        reuse_eligible=True, run=run, previous=_PREVIOUS,
    )
    assert decision["build"] is True
    assert "not in this repository's history" in str(decision["reason"])


def test_the_comparison_base_is_the_build_commit_not_the_release_commit() -> None:
    """A reused image must be compared against the commit that BUILT it.

    If ml-api is built at C1 for v0.1.0 and reused for v0.2.0, then at v0.3.0
    the published digest is still the one built at C1. Comparing against
    v0.2.0's commit would skip everything that changed between C1 and C2 and
    reuse a genuinely stale image. The base therefore comes from the image's own
    revision label, which a re-tag carries forward unchanged.
    """
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()

    def run(argv: list[str]) -> str:
        if "{{.Manifest.Digest}}" in argv:
            return _SOURCE_DIGEST + "\n"
        return _config_with_revision(head)

    decision = plan(
        ML_API, head, "seeon-edge-v9.9.9", _REPO,
        reuse_eligible=True, run=run, previous=_PREVIOUS,
    )
    # The label named HEAD, so the range is empty and the image is reused --
    # regardless of what the previous release's own commit was.
    assert decision["base"] == head
    assert decision["build"] is False
    assert decision["previous_digest"] == _SOURCE_DIGEST


def test_rules_are_ordered_most_specific_first() -> None:
    """`scripts/ops/` must win over `scripts/`, or ml-api loses an input."""
    lengths = [len(prefix) for prefix, _ in RULES]
    assert lengths == sorted(lengths, reverse=True)
