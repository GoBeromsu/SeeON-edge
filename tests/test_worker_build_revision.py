from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from worker.runtime.provenance.environment import resolve_worker_build_revision

_VALID_SOURCE_REVISION = "1" * 40
_OTHER_SOURCE_REVISION = "2" * 40


class _FakeGit:
    def __init__(
        self,
        *,
        status: str = "",
        head: str = _VALID_SOURCE_REVISION,
        status_returncode: int = 0,
        revision_returncode: int = 0,
    ) -> None:
        self.status: str = status
        self.head: str = head
        self.status_returncode: int = status_returncode
        self.revision_returncode: int = revision_returncode
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if "status" in command:
            return subprocess.CompletedProcess(
                command,
                self.status_returncode,
                stdout=self.status,
                stderr="",
            )
        if "rev-parse" in command:
            return subprocess.CompletedProcess(
                command,
                self.revision_returncode,
                stdout=f"{self.head}\n",
                stderr="",
            )
        raise AssertionError(f"unexpected Git command: {command}")


def _missing_image_marker(tmp_path: Path) -> Path:
    return tmp_path / "absent-image-revision"


def _baked_image_marker(tmp_path: Path, revision: str = _VALID_SOURCE_REVISION) -> Path:
    marker = tmp_path / "ml-worker-image-revision"
    _ = marker.write_text(f"{revision}\n", encoding="ascii")
    return marker


def test_matching_explicit_revision_is_authoritative_in_packaged_image(
    tmp_path: Path,
) -> None:
    git = _FakeGit(status=" M worker.py\n")

    resolved = resolve_worker_build_revision(
        _VALID_SOURCE_REVISION,
        repository_root=tmp_path,
        packaged_image_revision_path=_baked_image_marker(tmp_path),
        git_runner=git,
    )

    assert resolved == _VALID_SOURCE_REVISION
    assert git.commands == []


def test_dirty_local_source_rejects_injected_revision_without_image_marker(
    tmp_path: Path,
) -> None:
    git = _FakeGit(status=" M worker.py\n", head=_VALID_SOURCE_REVISION)

    resolved = resolve_worker_build_revision(
        _VALID_SOURCE_REVISION,
        repository_root=tmp_path,
        packaged_image_revision_path=_missing_image_marker(tmp_path),
        git_runner=git,
    )

    assert resolved is None
    assert git.commands == []


@pytest.mark.parametrize(
    "explicit,baked",
    (
        (None, _VALID_SOURCE_REVISION),
        (_OTHER_SOURCE_REVISION, _VALID_SOURCE_REVISION),
        ("", _VALID_SOURCE_REVISION),
        ("1" * 39, _VALID_SOURCE_REVISION),
        ("A" * 40, _VALID_SOURCE_REVISION),
        ("0" * 40, _VALID_SOURCE_REVISION),
        (_VALID_SOURCE_REVISION, "0" * 40),
    ),
)
def test_packaged_image_identity_fails_closed_unless_marker_and_runtime_match(
    explicit: str | None,
    baked: str,
    tmp_path: Path,
) -> None:
    git = _FakeGit()

    resolved = resolve_worker_build_revision(
        explicit,
        repository_root=tmp_path,
        packaged_image_revision_path=_baked_image_marker(tmp_path, baked),
        git_runner=git,
    )

    assert resolved is None
    assert git.commands == []


@pytest.mark.parametrize(
    "invalid",
    (
        "",
        "1" * 39,
        "1" * 41,
        "A" * 40,
        f" {_VALID_SOURCE_REVISION}",
        f"{_VALID_SOURCE_REVISION}\n",
        "0" * 40,
    ),
)
def test_untrusted_explicit_source_revision_fails_closed(
    invalid: str,
    tmp_path: Path,
) -> None:
    git = _FakeGit()

    resolved = resolve_worker_build_revision(
        invalid,
        repository_root=tmp_path,
        packaged_image_revision_path=_missing_image_marker(tmp_path),
        git_runner=git,
    )

    assert resolved is None
    assert git.commands == []


def test_clean_source_checkout_resolves_exact_git_head(tmp_path: Path) -> None:
    git = _FakeGit(head=_VALID_SOURCE_REVISION)

    resolved = resolve_worker_build_revision(
        None,
        repository_root=tmp_path,
        packaged_image_revision_path=_missing_image_marker(tmp_path),
        git_runner=git,
    )

    assert resolved == _VALID_SOURCE_REVISION
    assert ["status" in command for command in git.commands] == [True, False]


@pytest.mark.parametrize(
    "status",
    (
        " M worker.py\n",
        "M  worker.py\n",
        "?? new_worker.py\n",
    ),
)
def test_dirty_local_source_never_claims_bare_git_head(
    status: str,
    tmp_path: Path,
) -> None:
    git = _FakeGit(status=status)

    resolved = resolve_worker_build_revision(
        None,
        repository_root=tmp_path,
        packaged_image_revision_path=_missing_image_marker(tmp_path),
        git_runner=git,
    )

    assert resolved is None
    assert len(git.commands) == 1
    assert "status" in git.commands[0]


def test_ignored_build_artifact_preserves_clean_checkout_fallback(tmp_path: Path) -> None:
    git = _FakeGit(status="", head=_VALID_SOURCE_REVISION)

    resolved = resolve_worker_build_revision(
        None,
        repository_root=tmp_path,
        packaged_image_revision_path=_missing_image_marker(tmp_path),
        git_runner=git,
    )

    assert resolved == _VALID_SOURCE_REVISION


def test_non_repository_cannot_resolve_build_revision(tmp_path: Path) -> None:
    git = _FakeGit(status_returncode=128)

    resolved = resolve_worker_build_revision(
        None,
        repository_root=tmp_path,
        packaged_image_revision_path=_missing_image_marker(tmp_path),
        git_runner=git,
    )

    assert resolved is None
    assert len(git.commands) == 1


def test_git_execution_failure_fails_closed(tmp_path: Path) -> None:
    def fail(_command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        raise OSError("git unavailable")

    assert (
        resolve_worker_build_revision(
            None,
            repository_root=tmp_path,
            packaged_image_revision_path=_missing_image_marker(tmp_path),
            git_runner=fail,
        )
        is None
    )
