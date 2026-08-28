"""Compose the release notes for a ``seeon-edge-v<semver>`` tag.

The body has three parts:

1. **Highlights** — the hand-written ``docs/releases/<tag>.md``, when one
   exists. Prose a human wrote about what shipped; nothing derives it.
2. **Changes** — generated from the commit range since the previous
   ``seeon-edge-v*`` tag. The very first release has no previous tag to diff
   against, so it says so instead of dumping the whole history.
3. **Images** — where the digest-pinned GHCR references live. They cannot be
   inlined here: ``.github/workflows/edge-images.yml`` triggers on
   ``release: published``, so the images are built *after* these notes exist.

Both the rehearsal path and the real release path run this same code, so a
rehearsal proves the notes compose before a tag is ever pushed.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TAG_GLOB = "seeon-edge-v*"
IMAGE_NAMESPACE = "ghcr.io/seniorailab/eldercare-fall-ml"


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, repo-local git
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def previous_tag(tag: str) -> str | None:
    """The highest existing release tag that is not ``tag`` itself."""
    listed = _git("tag", "--list", TAG_GLOB, "--sort=-version:refname").splitlines()
    for candidate in (line.strip() for line in listed):
        if candidate and candidate != tag:
            return candidate
    return None


def _commit_lines(revision_range: str) -> list[str]:
    log = _git("log", "--first-parent", "--no-merges", "--pretty=format:%h %s", revision_range)
    return [line for line in log.splitlines() if line.strip()]


def compose(tag: str, head: str) -> str:
    sections: list[str] = []

    highlights = REPO_ROOT / "docs" / "releases" / f"{tag}.md"
    if highlights.is_file():
        sections.append(highlights.read_text(encoding="utf-8").strip())

    previous = previous_tag(tag)
    if previous is None:
        count = len(_commit_lines(head))
        sections.append(
            f"## Changes\n\nFirst tagged release — there is no previous `{TAG_GLOB}` tag to "
            f"diff against, so the whole history ({count} commits on the first-parent line) "
            "is what ships. See the highlights above for what is notable."
        )
    else:
        body = "\n".join(f"- {line}" for line in _commit_lines(f"{previous}..{head}"))
        sections.append(f"## Changes since {previous}\n\n{body or '- (no commits)'}")

    sections.append(
        "## Images\n\n"
        f"Publishing this release runs `.github/workflows/edge-images.yml` (`release: "
        f"published`), which pushes both images to `{IMAGE_NAMESPACE}` tagged with the full "
        "commit SHA and uploads the `edge-ml-image-refs-<sha>` artifact holding the two "
        "`@sha256:` digests. Pin deployments to those digests, never to a tag — see "
        "`docs/runbooks/edge-image-publish.md`."
    )

    return "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="the release tag, e.g. seeon-edge-v0.1.0")
    parser.add_argument(
        "--head",
        default=None,
        help="commit to end the range at (default: the tag itself, or HEAD when it does not exist)",
    )
    args = parser.parse_args(argv)

    head = args.head
    if head is None:
        commit = f"{args.tag}^{{commit}}"
        exists = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", "--quiet", commit],
            check=False,
            capture_output=True,
            text=True,
        )
        head = args.tag if exists.returncode == 0 else "HEAD"

    print(compose(args.tag, head), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
