"""Guard the class of bug behind issue #9, not just the instances fixed for it.

Homebrew bash 5.3.15 writes a heredoc body into a pipe *before* exec'ing the
command that reads it. A write up to ``PIPE_BUF`` (512 bytes on macOS) is
atomic and returns immediately; anything larger blocks until someone drains the
pipe -- and the only would-be reader has not been exec'd yet. Bash deadlocks
against itself: no output, no exit, forever.

That is not hypothetical. It hung a since-removed real-RTSP e2e harness for
operators and, through a since-removed contract test that ran it with no
timeout, hung the entire pytest run. bash 3.2.57 stages heredocs in a temp
file and is unaffected at any size.

The instances were fixed by pinning `#!/bin/bash`. This test stops the next one
from being added: a script that grows a heredoc past 512 bytes while still
using `#!/usr/bin/env bash` fails here instead of hanging someone's terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPTS_DIR: Final = REPO_ROOT / "scripts"

# PIPE_BUF on macOS. A heredoc body at or below this is written atomically and
# never blocks; above it, bash 5.3.15 deadlocks before exec.
PIPE_BUF: Final = 512

# `#!/usr/bin/env bash` picks whatever bash is first on PATH, which is how the
# affected build gets selected on developer machines.
UNPINNED_SHEBANG: Final = "#!/usr/bin/env bash"


def _shell_scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.rglob("*.sh"))


def _heredoc_bodies(text: str) -> list[tuple[int, int]]:
    """Return ``(start_line, body_bytes)`` for each heredoc in ``text``.

    Deliberately simple: find `<<TAG`, `<<'TAG'`, `<<"TAG"` or `<<-TAG` at end
    of line, then measure until a line whose stripped content is the tag. That
    is the shape every heredoc in this repository uses; a parser would be more
    than this guard needs.
    """
    bodies: list[tuple[int, int]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        marker = line.rfind("<<")
        if marker == -1:
            index += 1
            continue
        tag = line[marker + 2 :].strip()
        if tag.startswith("-"):
            tag = tag[1:].strip()
        tag = tag.strip("'\"")
        # A bare `<<` with no tag, or a here-string `<<<`, is not a heredoc.
        if not tag or not tag.replace("_", "").isalnum() or line[marker + 2 : marker + 3] == "<":
            index += 1
            continue
        start = index + 1
        size = 0
        index += 1
        while index < len(lines) and lines[index].strip() != tag:
            size += len(lines[index].encode("utf-8")) + 1
            index += 1
        bodies.append((start, size))
        index += 1
    return bodies


def test_scripts_with_large_heredocs_pin_their_interpreter() -> None:
    """A heredoc over PIPE_BUF must not be left to whatever bash is on PATH."""
    offenders: list[str] = []

    for script in _shell_scripts():
        text = script.read_text(encoding="utf-8")
        if not text.startswith(UNPINNED_SHEBANG):
            continue
        for start_line, size in _heredoc_bodies(text):
            if size > PIPE_BUF:
                offenders.append(
                    f"{script.relative_to(REPO_ROOT)}:{start_line} "
                    f"has a {size}-byte heredoc but uses {UNPINNED_SHEBANG}"
                )

    assert not offenders, (
        "These scripts deadlock under Homebrew bash 5.3.15, which writes heredoc "
        f"bodies into a pipe before exec'ing the reader (>{PIPE_BUF} bytes blocks "
        "forever). Pin the shebang to #!/bin/bash, or move the body into a file. "
        "See issue #9.\n  " + "\n  ".join(offenders)
    )


def test_heredoc_measurement_finds_the_known_sizes() -> None:
    """The measurement itself must be trustworthy, or the guard proves nothing.

    Pinned against a body of a known size rather than against the repository,
    so this keeps working when the scripts change.
    """
    body = "\n".join(f"line {n}" for n in range(10))
    script = f"f() {{\n  cat <<'PY'\n{body}\nPY\n}}\n"

    found = _heredoc_bodies(script)

    assert len(found) == 1
    _start, size = found[0]
    assert size == len(body.encode("utf-8")) + 1


def test_guard_would_have_caught_the_original_defect() -> None:
    """A script in the shape that actually broke must be reported."""
    oversized = "\n".join("x" * 60 for _ in range(12))  # comfortably over 512 B
    text = f"{UNPINNED_SHEBANG}\nf() {{\n  python3 - <<'PY'\n{oversized}\nPY\n}}\n"

    sizes = [size for _start, size in _heredoc_bodies(text)]

    assert sizes and max(sizes) > PIPE_BUF
