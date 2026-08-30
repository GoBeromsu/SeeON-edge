"""Tests for ``worker.tools.fetch_models`` -- the ``edge-model-fetch`` one-shot.

Owner decision 2026-08-28: models stay out of the images and are fetched from
pinned upstreams into the ``worker-models`` volume. These tests drive the
fetcher against a fake byte source so hash verification, idempotency, partial
file recovery, retry policy, and token handling are proven without a network.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from worker.tools.fetch_models import cli
from worker.tools.fetch_models.fetcher import (
    PART_SUFFIX,
    VerificationError,
    fetch_all,
    fetch_artifact,
    fetch_bundle,
    sha256_of,
)
from worker.tools.fetch_models.http_source import (
    RetryableSourceError,
    RetryPolicy,
    SourceError,
    attempts_from_env,
)
from worker.tools.fetch_models.manifest import (
    MANIFEST_PATH,
    SIDECAR_ROOT,
    Bundle,
    Manifest,
    ManifestError,
    canonical_json,
    load_manifest,
    parse_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_SIDECAR_DIR = REPO_ROOT / "models" / "fall" / "lstm"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


WEIGHT = b"\x00weights" * 700
UPSTREAM = b'{"dataset": "le2i"}\n'


def _manifest_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1,
        "sources": {
            "hf": {
                "kind": "huggingface",
                "repo": "owner/models",
                "revision": "d67887844bfd2e4b1ca3f3275f770b0b05e23aba",
            },
            "gh": {"kind": "github-release", "repo": "ultralytics/assets", "tag": "v8.4.0"},
        },
        "artifacts": [
            {
                "path": "fall/lstm/model.pt",
                "source": "hf",
                "remote_path": "lstm/model.pt",
                "size": len(WEIGHT),
                "sha256": _sha(WEIGHT),
            },
            {
                "path": "fall/lstm/metadata.upstream.json",
                "source": "gh",
                "remote_path": "metadata.json",
                "size": len(UPSTREAM),
                "sha256": _sha(UPSTREAM),
            },
        ],
        "sidecars": [],
    }
    base.update(overrides)
    return base


class FakeSource:
    """Scripted byte source: per-URL bodies, optional failures, call log."""

    def __init__(self, bodies: Mapping[str, bytes]) -> None:
        self.bodies = dict(bodies)
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.failures: dict[str, list[Exception]] = {}
        self.truncate_after: dict[str, int] = {}

    def stream(self, url: str, headers: Mapping[str, str]) -> Iterator[bytes]:
        self.calls.append((url, dict(headers)))
        queued = self.failures.get(url)
        if queued:
            raise queued.pop(0)
        if url not in self.bodies:
            raise SourceError(f"{url} returned HTTP 404; not retryable")
        body = self.bodies[url]
        limit = self.truncate_after.pop(url, None)
        view = io.BytesIO(body)
        sent = 0
        while chunk := view.read(1024):
            if limit is not None and sent + len(chunk) > limit:
                yield chunk[: limit - sent]
                raise RetryableSourceError(f"{url} transport error: connection reset")
            sent += len(chunk)
            yield chunk


def _fake_for(manifest: Manifest) -> FakeSource:
    return FakeSource(
        {
            manifest.artifacts[0].url: WEIGHT,
            manifest.artifacts[1].url: UPSTREAM,
        }
    )


def _no_sleep_policy(attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(attempts=attempts, sleep=lambda _s: None, rng=random.Random(0))


def _bundle_manifest() -> Manifest:
    raw = _manifest_dict(sidecars=[])
    members = [
        {
            "path": "model.pt",
            "source": "hf",
            "remote_path": "bundle/model.bin",
            "size": len(WEIGHT),
            "sha256": _sha(WEIGHT),
        },
        *[
            {
                "path": path,
                "source": "gh",
                "remote_path": f"bundle/{path}",
                "size": len(WEIGHT),
                "sha256": _sha(WEIGHT),
            }
            for path in (
                "arch.json",
                "metadata.yaml",
                "input-contract.json",
                "fall-policy-v2.json",
                "calibration.json",
                "conformance.json",
            )
        ],
    ]
    payload = {"identities": {"dataset": "fixture"}}
    identity = _sha(
        canonical_json(
            {
                "members": [
                    {"path": member["path"], "sha256": member["sha256"], "size": member["size"]}
                    for member in members
                ],
                "payload": payload,
            }
        ).encode()
    )
    receipt = canonical_json({"bundle_payload_digest": identity}).encode()
    receipts = [
        {
            "path": "receipts/evaluation.json",
            "source": "gh",
            "remote_path": "bundle/evaluation.json",
            "size": len(receipt),
            "sha256": _sha(receipt),
        }
    ]
    raw["bundles"] = [
        {"sha256": identity, "members": members, "payload": payload, "receipts": receipts}
    ]
    return parse_manifest(raw)


def _bundle_source(bundle: Bundle) -> FakeSource:
    return FakeSource(
        {
            artifact.url: body
            for artifact, body in zip(
                (*bundle.members, *bundle.receipts),
                (
                    *((WEIGHT,) * len(bundle.members)),
                    canonical_json({"bundle_payload_digest": bundle.sha256}).encode(),
                ),
                strict=True,
            )
        }
    )


# --- manifest -------------------------------------------------------------


def test_committed_manifest_parses_and_pins_every_family_the_worker_loads() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    paths = {artifact.path for artifact in manifest.artifacts}
    assert {
        "fall/lstm/model.pt",
        "fall/lstm/metadata.upstream.json",
        "pose/yolo26n-pose.pt",
        "person/yolo26n.pt",
        "bed/yolo26m-seg.pt",
    } <= paths
    assert set(manifest.sidecars) == {"fall/lstm/arch.json", "fall/lstm/metadata.yaml"}
    hf = manifest.sources["eldercare-fall-models"]
    assert (hf.repo, hf.ref) == (
        "Berom0227/eldercare-fall-models",
        "d67887844bfd2e4b1ca3f3275f770b0b05e23aba",
    )


def test_committed_manifest_digests_agree_with_runtime_pins() -> None:
    """The registry and the tracked LSTM sidecar already pin these digests;
    the manifest must never drift from what the runtime will accept."""
    from worker.domains.registry import _COMPONENT_ARTIFACT_DIGESTS

    by_path = {artifact.path: artifact.sha256 for artifact in load_manifest().artifacts}
    assert by_path["pose/yolo26n-pose.pt"] == _COMPONENT_ARTIFACT_DIGESTS["pose"]
    assert by_path["person/yolo26n.pt"] == _COMPONENT_ARTIFACT_DIGESTS["person"]
    assert by_path["bed/yolo26m-seg.pt"] == _COMPONENT_ARTIFACT_DIGESTS["bed"]
    assert by_path["fall/lstm/model.pt"] == _COMPONENT_ARTIFACT_DIGESTS["fall-classifier"]
    sidecar = (TRACKED_SIDECAR_DIR / "metadata.yaml").read_text(encoding="utf-8")
    assert f'artifact_digest: "{by_path["fall/lstm/model.pt"]}"' in sidecar


def test_bundled_sidecars_are_byte_identical_to_tracked_models_dir() -> None:
    """`.dockerignore` drops `models/` from the image, so the tool carries
    copies; they must stay identical to the git-tracked originals."""
    for relative in load_manifest().sidecars:
        bundled = SIDECAR_ROOT / relative
        tracked = REPO_ROOT / "models" / relative
        assert bundled.read_bytes() == tracked.read_bytes(), relative


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d["sources"]["hf"].update(revision="main"), "40-hex"),
        (lambda d: d["artifacts"][0].update(sha256="abc"), "64 lowercase hex"),
        (lambda d: d["artifacts"][0].update(size=0), "positive integer"),
        (lambda d: d["artifacts"][0].update(source="nope"), "unknown source"),
        (lambda d: d["artifacts"][0].update(path="../escape"), "plain relative path"),
        (lambda d: d["artifacts"][0].update(path="/abs"), "plain relative path"),
        (lambda d: d["artifacts"][1].update(path="fall/lstm/model.pt"), "duplicate"),
    ],
)
def test_manifest_rejects_malformed_entries(mutation: object, message: str) -> None:
    raw = _manifest_dict()
    mutation(raw)  # type: ignore[operator]
    with pytest.raises(ManifestError, match=message):
        parse_manifest(raw)


def test_source_urls_are_pinned_to_revision_or_tag() -> None:
    manifest = parse_manifest(_manifest_dict())
    assert manifest.artifacts[0].url == (
        "https://huggingface.co/owner/models/resolve/"
        "d67887844bfd2e4b1ca3f3275f770b0b05e23aba/lstm/model.pt"
    )
    assert manifest.artifacts[1].url == (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/metadata.json"
    )


# --- fetch + verify -------------------------------------------------------


def test_fetch_all_downloads_verifies_and_is_idempotent(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    env: dict[str, str] = {}

    first = fetch_all(manifest, tmp_path, source, env=env, retry=_no_sleep_policy())
    assert [r.outcome for r in first.results] == ["fetched", "fetched"]
    assert (tmp_path / "fall/lstm/model.pt").read_bytes() == WEIGHT
    assert (tmp_path / "fall/lstm/metadata.upstream.json").read_bytes() == UPSTREAM
    assert not list(tmp_path.rglob(f"*{PART_SUFFIX}"))

    second = fetch_all(manifest, tmp_path, source, env=env, retry=_no_sleep_policy())
    assert [r.outcome for r in second.results] == ["present", "present"]
    assert second.is_noop
    assert len(source.calls) == 2, "second run must not touch the network"


def test_hash_mismatch_fails_and_leaves_nothing_at_the_final_path(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    source.bodies[manifest.artifacts[0].url] = b"\x00tamperd" * 700  # same size, wrong bytes

    with pytest.raises(VerificationError, match="sha256 mismatch"):
        fetch_all(manifest, tmp_path, source, env={}, retry=_no_sleep_policy())
    assert not (tmp_path / "fall/lstm/model.pt").exists()
    assert not list(tmp_path.rglob(f"*{PART_SUFFIX}"))


def test_size_mismatch_fails_before_hash(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    source.bodies[manifest.artifacts[0].url] = WEIGHT[:-1]

    with pytest.raises(VerificationError, match="size mismatch"):
        fetch_artifact(manifest.artifacts[0], tmp_path, source, env={}, retry=_no_sleep_policy())
    assert not (tmp_path / "fall/lstm/model.pt").exists()


def test_oversized_body_is_cut_off_without_hashing_the_whole_stream(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    source.bodies[manifest.artifacts[0].url] = WEIGHT + b"\x00" * 4096

    with pytest.raises(VerificationError, match="exceeds manifest size"):
        fetch_artifact(manifest.artifacts[0], tmp_path, source, env={}, retry=_no_sleep_policy())


def test_corrupted_existing_file_is_re_downloaded(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    dest = tmp_path / "fall/lstm/model.pt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"\x00garbage" * 700)  # right size (8*700), wrong hash

    result = fetch_artifact(
        manifest.artifacts[0], tmp_path, source, env={}, retry=_no_sleep_policy()
    )
    assert result.outcome == "fetched"
    assert sha256_of(dest) == manifest.artifacts[0].sha256


def test_stale_partial_file_from_interrupted_run_is_discarded(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    dest = tmp_path / "fall/lstm/model.pt"
    dest.parent.mkdir(parents=True)
    stale = dest.with_name(dest.name + PART_SUFFIX)
    stale.write_bytes(WEIGHT[:100])

    result = fetch_artifact(
        manifest.artifacts[0], tmp_path, source, env={}, retry=_no_sleep_policy()
    )
    assert result.outcome == "fetched"
    assert dest.read_bytes() == WEIGHT
    assert not stale.exists()


def test_mid_stream_transport_error_restarts_the_file_from_scratch(tmp_path: Path) -> None:
    """A retry must not append to a half-written body; the hash would then be
    right by accident only if the server resumed, which it never does here."""
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    url = manifest.artifacts[0].url
    source.truncate_after[url] = 1500
    waits: list[float] = []
    policy = RetryPolicy(attempts=3, sleep=waits.append, rng=random.Random(0))

    result = fetch_artifact(manifest.artifacts[0], tmp_path, source, env={}, retry=policy)
    assert result.outcome == "fetched"
    assert (tmp_path / "fall/lstm/model.pt").read_bytes() == WEIGHT
    assert [call[0] for call in source.calls] == [url, url]
    assert len(waits) == 1 and waits[0] >= 4.0


def test_retryable_status_backs_off_and_honours_retry_after(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    url = manifest.artifacts[0].url
    source.failures[url] = [
        RetryableSourceError(f"{url} returned HTTP 429", retry_after=7.0),
        RetryableSourceError(f"{url} returned HTTP 503"),
    ]
    waits: list[float] = []
    policy = RetryPolicy(attempts=4, sleep=waits.append, rng=random.Random(0))

    result = fetch_artifact(manifest.artifacts[0], tmp_path, source, env={}, retry=policy)
    assert result.outcome == "fetched"
    assert waits[0] == 7.0, "Retry-After beats local backoff"
    assert 8.0 <= waits[1] <= 13.0, "second failure: 2**3 plus up to 5s jitter"


def test_gives_up_after_configured_attempts(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    url = manifest.artifacts[0].url
    source.failures[url] = [RetryableSourceError("HTTP 503")] * 5

    with pytest.raises(SourceError, match="giving up .* after 2 attempts"):
        fetch_artifact(manifest.artifacts[0], tmp_path, source, env={}, retry=_no_sleep_policy(2))
    assert not (tmp_path / "fall/lstm/model.pt").exists()


def test_non_retryable_status_fails_fast(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = FakeSource({})  # every URL 404s
    waits: list[float] = []
    policy = RetryPolicy(attempts=6, sleep=waits.append)

    with pytest.raises(SourceError, match="404; not retryable"):
        fetch_artifact(manifest.artifacts[0], tmp_path, source, env={}, retry=policy)
    assert waits == [], "a wrong pin is not worth waiting on"
    assert len(source.calls) == 1


def test_hf_token_goes_only_to_huggingface_and_is_never_logged(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict())
    source = _fake_for(manifest)
    lines: list[str] = []
    token = "hf_supersecret_token_value"

    fetch_all(
        manifest,
        tmp_path,
        source,
        env={"HF_TOKEN": token},
        retry=_no_sleep_policy(),
        log=lines.append,
    )
    by_url = dict(source.calls)
    assert by_url[manifest.artifacts[0].url]["Authorization"] == f"Bearer {token}"
    assert "Authorization" not in by_url[manifest.artifacts[1].url]
    assert token not in "\n".join(lines)


def test_sidecars_are_placed_from_the_bundle_and_verified(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "fall/lstm").mkdir(parents=True)
    (bundle / "fall/lstm/arch.json").write_text('{"hidden": 4}\n', encoding="utf-8")
    manifest = parse_manifest(_manifest_dict(sidecars=["fall/lstm/arch.json"]))
    source = _fake_for(manifest)
    root = tmp_path / "models"

    first = fetch_all(manifest, root, source, env={}, retry=_no_sleep_policy(), sidecar_root=bundle)
    assert first.results[-1].outcome == "sidecar-written"
    assert (root / "fall/lstm/arch.json").read_text(encoding="utf-8") == '{"hidden": 4}\n'
    (root / "fall/lstm/arch.json").write_text("edited", encoding="utf-8")

    second = fetch_all(
        manifest, root, source, env={}, retry=_no_sleep_policy(), sidecar_root=bundle
    )
    assert second.results[-1].outcome == "sidecar-written", "drift is corrected"
    third = fetch_all(manifest, root, source, env={}, retry=_no_sleep_policy(), sidecar_root=bundle)
    assert third.results[-1].outcome == "sidecar-present"


def test_missing_bundled_sidecar_is_a_failure(tmp_path: Path) -> None:
    manifest = parse_manifest(_manifest_dict(sidecars=["fall/lstm/arch.json"]))
    with pytest.raises(VerificationError, match="bundled sidecar missing"):
        fetch_all(
            manifest,
            tmp_path / "models",
            _fake_for(manifest),
            env={},
            retry=_no_sleep_policy(),
            sidecar_root=tmp_path / "empty",
        )


def test_bundle_is_atomically_published_and_then_a_noop(tmp_path: Path) -> None:
    manifest = _bundle_manifest()
    bundle = manifest.bundles[0]
    source = _bundle_source(bundle)

    first = fetch_bundle(bundle, tmp_path, source, env={}, retry=_no_sleep_policy())
    published = tmp_path / "bundles" / bundle.sha256
    assert [result.outcome for result in first.results] == ["fetched"] * 8
    assert (published / "model.pt").read_bytes() == WEIGHT
    assert (published / "manifest.json").read_bytes() == bundle.manifest_bytes
    assert (published / "receipts/evaluation.json").read_bytes() == canonical_json(
        {"bundle_payload_digest": bundle.sha256}
    ).encode()
    assert not list((tmp_path / "bundles").glob(f".{bundle.sha256}.*"))

    second = fetch_bundle(bundle, tmp_path, source, env={}, retry=_no_sleep_policy())
    assert second.is_noop
    assert len(source.calls) == 8


def test_bundle_failure_cleans_staging_and_never_publishes(tmp_path: Path) -> None:
    manifest = _bundle_manifest()
    bundle = manifest.bundles[0]
    source = FakeSource(
        {artifact.url: b"\x00" * artifact.size for artifact in (*bundle.members, *bundle.receipts)}
    )

    with pytest.raises(VerificationError, match="sha256 mismatch"):
        fetch_bundle(bundle, tmp_path, source, env={}, retry=_no_sleep_policy())
    assert not (tmp_path / "bundles" / bundle.sha256).exists()
    assert not list((tmp_path / "bundles").glob(f".{bundle.sha256}.*"))


def test_existing_mismatched_bundle_is_never_overwritten(tmp_path: Path) -> None:
    bundle = _bundle_manifest().bundles[0]
    existing = tmp_path / "bundles" / bundle.sha256
    existing.mkdir(parents=True)
    (existing / "unexpected").write_text("no", encoding="utf-8")

    with pytest.raises(VerificationError, match="tree mismatch"):
        fetch_bundle(bundle, tmp_path, FakeSource({}), env={}, retry=_no_sleep_policy())


def test_existing_bundle_with_extra_empty_directory_is_rejected(tmp_path: Path) -> None:
    manifest = _bundle_manifest()
    bundle = manifest.bundles[0]
    source = _bundle_source(bundle)
    fetch_bundle(bundle, tmp_path, source, env={}, retry=_no_sleep_policy())
    (tmp_path / "bundles" / bundle.sha256 / "empty").mkdir()

    with pytest.raises(VerificationError, match="tree mismatch"):
        fetch_bundle(bundle, tmp_path, source, env={}, retry=_no_sleep_policy())


def test_receipts_are_non_recursive_and_part_of_exact_bundle_tree(tmp_path: Path) -> None:
    bundle = _bundle_manifest().bundles[0]
    assert bundle.sha256 == _sha(bundle.canonical_payload.encode())
    assert bundle.receipts[0].sha256 not in bundle.canonical_payload
    assert b'"receipts"' in bundle.manifest_bytes
    assert bundle.manifest_bytes == _bundle_manifest().bundles[0].manifest_bytes

    fetch_bundle(bundle, tmp_path, _bundle_source(bundle), env={}, retry=_no_sleep_policy())
    published = tmp_path / "bundles" / bundle.sha256
    receipt = published / "receipts/evaluation.json"
    receipt.write_bytes(b"\x00" * bundle.receipts[0].size)
    with pytest.raises(VerificationError, match="member mismatch"):
        fetch_bundle(bundle, tmp_path, FakeSource({}), env={}, retry=_no_sleep_policy())


def test_bundle_receipts_must_be_nonempty_unique_and_disjoint() -> None:
    bundle = _bundle_manifest().bundles[0]
    raw = _manifest_dict(
        bundles=[
            {
                "sha256": bundle.sha256,
                "members": [
                    {
                        "path": member.path,
                        "source": member.source.name,
                        "remote_path": member.remote_path,
                        "size": member.size,
                        "sha256": member.sha256,
                    }
                    for member in bundle.members
                ],
                "payload": dict(bundle.payload),
                "receipts": [],
            }
        ]
    )
    with pytest.raises(ManifestError, match="non-empty"):
        parse_manifest(raw)


def test_missing_receipt_or_cross_tree_receipt_fails_without_publish(tmp_path: Path) -> None:
    bundle = _bundle_manifest().bundles[0]
    missing = _bundle_source(bundle)
    missing.bodies.pop(bundle.receipts[0].url)
    with pytest.raises(SourceError):
        fetch_bundle(bundle, tmp_path, missing, env={}, retry=_no_sleep_policy())
    assert not (tmp_path / "bundles" / bundle.sha256).exists()
    assert not list((tmp_path / "bundles").glob(f".{bundle.sha256}.*"))

    raw = _manifest_dict(sidecars=[])
    member = {
        "path": "same.json",
        "source": "hf",
        "remote_path": "same.json",
        "size": len(WEIGHT),
        "sha256": _sha(WEIGHT),
    }
    raw["bundles"] = [
        {
            "sha256": _sha(canonical_json({"members": [member], "payload": {}}).encode()),
            "members": [member],
            "receipts": [member],
            "payload": {},
        }
    ]
    with pytest.raises(ManifestError, match="overlap"):
        parse_manifest(raw)


# --- attempts env + CLI ---------------------------------------------------


@pytest.mark.parametrize("raw", ["0", "-1", "six", "1.5"])
def test_attempts_env_rejects_non_positive_or_non_integer(raw: str) -> None:
    with pytest.raises(SourceError, match="positive integer"):
        attempts_from_env({"ML_WORKER_FETCH_MODELS_ATTEMPTS": raw})


def test_attempts_env_defaults_and_parses() -> None:
    assert attempts_from_env({}) == 6
    assert attempts_from_env({"ML_WORKER_FETCH_MODELS_ATTEMPTS": " 2 "}) == 2


def test_cli_check_mode_reports_missing_files_without_downloading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest_dict()), encoding="utf-8")

    code = cli.main(
        ["--dest", str(tmp_path / "models"), "--manifest", str(manifest_path), "--check"], env={}
    )
    err = capsys.readouterr().err
    assert code == 1
    assert "would need downloading" in err
    assert not any(path.is_file() for path in (tmp_path / "models").rglob("*"))


def test_cli_rejects_malformed_manifest_with_usage_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    code = cli.main(["--dest", str(tmp_path), "--manifest", str(manifest_path)], env={})
    assert code == 2
    assert "missing 'schema_version'" in capsys.readouterr().err


def test_cli_resolves_dest_from_env_then_default() -> None:
    assert cli._resolve_dest(None, {}) == Path("models")
    assert cli._resolve_dest(None, {"ML_WORKER_FETCH_MODELS_DEST": "/app/models"}) == Path(
        "/app/models"
    )
    assert cli._resolve_dest(Path("x"), {"ML_WORKER_FETCH_MODELS_DEST": "/y"}) == Path("x")


def test_module_entrypoint_matches_compose_command() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "worker.tools.fetch_models", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--dest" in proc.stdout


def test_fetch_models_imports_only_the_standard_library() -> None:
    """The tool runs in the slim worker image before anything else and must
    never pull the runtime graph (or torch) into the one-shot."""
    package = REPO_ROOT / "worker" / "tools" / "fetch_models"
    for module in package.glob("*.py"):
        for line in module.read_text(encoding="utf-8").splitlines():
            if line.startswith(("import ", "from ")) and "worker." in line:
                assert line.split()[1].startswith("worker.tools.fetch_models"), (
                    f"{module.name}: {line}"
                )


def test_dev_wrapper_delegates_to_the_module() -> None:
    script = (REPO_ROOT / "scripts" / "fetch-models.sh").read_text(encoding="utf-8")
    assert "python -m worker.tools.fetch_models" in script
    assert "curl" not in script, "no second download path to drift from the manifest"
