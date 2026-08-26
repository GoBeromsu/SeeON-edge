from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import worker.tools.deepstream_canary.cli as cli

DIGEST = "b" * 64
REVISION = "1" * 40
IMAGE = f"registry.example/seeon-edge@sha256:{DIGEST}"


def _inspect_result(
    *, digest: str = DIGEST, revision: str = REVISION
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=("docker", "image", "inspect"),
        returncode=0,
        stdout=json.dumps(
            [
                {
                    "RepoDigests": [f"registry.example/seeon-edge@sha256:{digest}"],
                    "Config": {
                        "Labels": {"org.opencontainers.image.revision": revision}
                    },
                }
            ]
        ),
        stderr="",
    )


def test_real_canary_image_admission_rejects_omitted_and_tag_only_identity() -> None:
    # Given: real execution rather than render-only fixture generation.
    omitted = cli.WorkerImageAdmission(image=None, expected_revision=None, render_only=False)
    tagged = cli.WorkerImageAdmission(
        image="registry.example/seeon-edge:latest",
        expected_revision=REVISION,
        render_only=False,
    )

    # When / Then: both ambiguous identities fail before Docker inspection.
    with pytest.raises(cli.CanaryArgumentError, match="worker image is required"):
        cli._admit_worker_image(omitted)
    with pytest.raises(cli.CanaryArgumentError, match="repository@sha256"):
        cli._admit_worker_image(tagged)


def test_real_canary_image_admission_rejects_digest_or_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: inspect resolves a different immutable digest, then a different OCI revision.
    request = cli.WorkerImageAdmission(
        image=IMAGE,
        expected_revision=REVISION,
        render_only=False,
    )
    monkeypatch.setattr(
        cli.image_admission.subprocess,
        "run",
        lambda *_args, **_kwargs: _inspect_result(digest="a" * 64),
    )

    # When / Then: a digest mismatch fails closed.
    with pytest.raises(cli.CanaryArgumentError, match="digest mismatch"):
        cli._admit_worker_image(request)

    monkeypatch.setattr(
        cli.image_admission.subprocess,
        "run",
        lambda *_args, **_kwargs: _inspect_result(revision="2" * 40),
    )
    with pytest.raises(cli.CanaryArgumentError, match="revision mismatch"):
        cli._admit_worker_image(request)


def test_real_canary_image_admission_returns_inspected_immutable_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Docker resolves the requested repository digest with the requested OCI revision.
    request = cli.WorkerImageAdmission(
        image=IMAGE,
        expected_revision=REVISION,
        render_only=False,
    )
    monkeypatch.setattr(
        cli.image_admission.subprocess,
        "run",
        lambda *_args, **_kwargs: _inspect_result(),
    )

    # When: admission inspects the local image identity.
    binding = cli._admit_worker_image(request)

    # Then: the exact requested digest and revision are returned for the immutable receipt.
    assert binding.image == IMAGE
    assert binding.digest == f"sha256:{DIGEST}"
    assert binding.revision == REVISION


def test_render_only_allows_explicit_fixture_digest_without_revision_or_inspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: deterministic render-only fixture generation.
    request = cli.WorkerImageAdmission(
        image=IMAGE,
        expected_revision=None,
        render_only=True,
    )
    monkeypatch.setattr(
        cli.image_admission.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("render-only must not inspect Docker"),
    )

    # When: the fixture digest is admitted.
    binding = cli._admit_worker_image(request)

    # Then: no unobserved revision is invented.
    assert binding.image == IMAGE
    assert binding.digest == f"sha256:{DIGEST}"
    assert binding.revision is None


def test_render_only_does_not_collide_with_an_active_real_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: existing-project admission would report an active real canary.
    arguments = cli.RunArguments()
    _ = cli._parser().parse_args(
        [
            "run",
            "--rungs",
            "zero,loopback",
            "--evidence-dir",
            str(tmp_path / "render-concurrent"),
            "--render-only",
            "--worker-image",
            IMAGE,
        ],
        namespace=arguments,
    )
    monkeypatch.setattr(
        cli,
        "refuse_existing_project",
        lambda: pytest.fail("render-only must not inspect the active Compose project"),
    )

    # When: render-only prepares a separate immutable fixture.
    result = cli._run(arguments)

    # Then: no project admission or Compose lifecycle is touched.
    assert result == 0
    assert arguments.evidence_dir.joinpath("compose.rendered.yaml").is_file()


def test_run_request_records_selected_digest_and_expected_revision(tmp_path: Path) -> None:
    # Given: an explicit render-only fixture digest and expected revision.
    evidence = tmp_path / "render"

    # When: the CLI renders immutable evidence.
    completed = subprocess.run(
        [
            cli.sys.executable,
            "-m",
            "worker.tools.deepstream_canary",
            "run",
            "--rungs",
            "zero,loopback",
            "--evidence-dir",
            str(evidence),
            "--render-only",
            "--worker-image",
            IMAGE,
            "--expected-revision",
            REVISION,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: the verifier-bound request records both immutable identities.
    assert completed.returncode == 0, completed.stderr
    request = json.loads((evidence / "run-request.json").read_text())
    assert request["worker_image"] == IMAGE
    assert request["worker_image_digest"] == f"sha256:{DIGEST}"
    assert request["expected_revision"] == REVISION
