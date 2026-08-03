"""Unit coverage for ``e2e_worker_relay_fixtures.py``'s process fixtures.

Pure-unit, no external binaries required: ``MediaMtxProcess.__init__`` would
normally spawn a real ``subprocess.Popen`` for the ``mediamtx`` binary, so
this test replaces both ``shutil.which`` and ``subprocess.Popen`` to assert
the exact ``cwd`` argument passed to that invocation, without ever executing
it. Runs in CI (no ``real_stack`` marker needed) since nothing here touches a
real mediamtx/ffmpeg process.

Regression test for #84: mediamtx auto-generates a self-signed
``auto.crt``/``auto.key`` pair in the invoked process's cwd on startup (even
with its own TLS listeners disabled). Without an explicit ``cwd=`` override,
that cwd was whatever directory the test/demo process happened to be run
from -- the repo root for a normal local ``pytest``/demo invocation --
littering the working tree with untracked cert files.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import e2e_worker_relay_fixtures as fixtures
import pytest


class _FakePopenResult:
    """Stand-in for a ``Popen`` handle that reports an immediate exit, so
    ``MediaMtxProcess``'s readiness poll fails fast instead of waiting out
    its 10s connect timeout."""

    def __init__(self, cwd: str | None) -> None:
        self.cwd = cwd
        self.returncode = 1

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


def test_mediamtx_process_spawns_with_isolated_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MediaMtxProcess`` must pass an explicit, disposable ``cwd=`` to its
    ``subprocess.Popen`` call -- not inherit the caller's cwd -- so mediamtx's
    auto-generated ``auto.crt``/``auto.key`` never land in the repo root."""
    monkeypatch.setattr(fixtures.shutil, "which", lambda _name: "/usr/local/bin/mediamtx")
    monkeypatch.chdir(tmp_path)  # simulate a repo-root-like invocation cwd

    captured: dict[str, Any] = {}

    def fake_popen(*_args: object, cwd: str | None = None, **_kwargs: object) -> _FakePopenResult:
        assert cwd is not None, "MediaMtxProcess must pass an explicit cwd"
        assert Path(cwd).is_dir(), "the passed cwd must already exist"
        captured["cwd"] = cwd
        return _FakePopenResult(cwd)

    monkeypatch.setattr(fixtures.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="exited early"):
        fixtures.MediaMtxProcess(rtsp_port=1, path_names=("nominal",))

    assert captured, "subprocess.Popen was never called"
    spawned_cwd = Path(captured["cwd"])
    assert spawned_cwd != Path.cwd(), "must not inherit the test process cwd (repo root here)"
    assert spawned_cwd != tmp_path, "must not reuse the caller's cwd verbatim"
    assert "mediamtx-e2e-" in spawned_cwd.name


def test_mediamtx_process_cleans_up_its_cwd_on_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The disposable cwd handed to ``subprocess.Popen`` must be removed once
    the process is stopped, so repeated fixture runs don't accumulate temp
    directories."""
    monkeypatch.setattr(fixtures.shutil, "which", lambda _name: "/usr/local/bin/mediamtx")

    spawned_dirs: list[Path] = []

    def fake_popen(*_args: object, cwd: str | None = None, **_kwargs: object) -> _FakePopenResult:
        assert cwd is not None
        spawned_dirs.append(Path(cwd))
        return _FakePopenResult(cwd)

    monkeypatch.setattr(fixtures.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="exited early"):
        fixtures.MediaMtxProcess(rtsp_port=1, path_names=("nominal",))

    assert len(spawned_dirs) == 1
    assert not spawned_dirs[0].exists(), "cwd must be removed by MediaMtxProcess.stop()"


def test_no_stray_auto_cert_files_in_repo_root() -> None:
    """Regression guard for #84: no auto.crt/auto.key should ever be
    committed at the repo root (mediamtx's auto-generated TLS artifacts)."""
    repo_root = Path(__file__).resolve().parent.parent
    assert not (repo_root / "auto.crt").exists()
    assert not (repo_root / "auto.key").exists()


def test_subprocess_module_reference_is_the_stdlib_module() -> None:
    """Sanity check that the fixture module's ``subprocess`` reference is the
    real stdlib module (i.e. the monkeypatches above are patching what the
    fixture actually calls, not a re-export)."""
    assert fixtures.subprocess is subprocess
