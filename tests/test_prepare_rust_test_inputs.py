from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

from scripts.prepare_rust_test_inputs import InputPreparationError, restore_test_inputs


def _archive(tmp_path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> tuple[Path, str]:
    archive = tmp_path / "inputs.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for member, payload in members:
            if member.isreg():
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))
            else:
                bundle.addfile(member)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def _file(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    return tarfile.TarInfo(name), payload


def test_legacy_archive_paths_restore_to_canonical_destinations_idempotently(
    tmp_path: Path,
) -> None:
    archive_payloads = {
        "tests/fixtures/worker-wire/r/sample.json": b'{"sample": 1}\n',
        "tests/fixtures/worker-wire/r/sample.json.sha256": b"sample digest\n",
        "tests/fixtures/synthetic-scene-v1.json": b"{}\n",
        "worker/rust/tests/fixtures/bed_input/sample.json": b"[]\n",
        "worker/runtime/rust/tests/fixtures/gpu/manifest.json": b'{"sample": 2}\n',
    }
    destination_names = {
        "worker/rust/tests/fixtures/bed_input/sample.json": (
            "worker/policy/tests/fixtures/bed_input/sample.json"
        ),
        "worker/runtime/rust/tests/fixtures/gpu/manifest.json": (
            "worker/runtime/inference/tests/fixtures/gpu/manifest.json"
        ),
    }
    archive, digest = _archive(
        tmp_path, [_file(name, data) for name, data in archive_payloads.items()]
    )
    destination = tmp_path / "cache"

    assert restore_test_inputs(archive, destination, digest) == len(archive_payloads)
    helper = destination / "worker/policy/tests/fixtures/support.rs"
    helper.write_bytes(b"// Existing Rust helper\n")
    fixed_mtime_ns = 1_000_000_000
    for archive_name, payload in archive_payloads.items():
        target_name = destination_names.get(archive_name, archive_name)
        target = destination / target_name
        assert target.read_bytes() == payload
        os.utime(target, ns=(fixed_mtime_ns, fixed_mtime_ns))
    assert restore_test_inputs(archive, destination, digest) == len(archive_payloads)

    for archive_name, payload in archive_payloads.items():
        target_name = destination_names.get(archive_name, archive_name)
        restored = destination / target_name
        assert restored.read_bytes() == payload
        assert restored.stat().st_mtime_ns == fixed_mtime_ns
    for archive_name in destination_names:
        assert not (destination / archive_name).exists()
    assert helper.read_bytes() == b"// Existing Rust helper\n"


def test_digest_mismatch_writes_nothing(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path, [_file("tests/fixtures/synthetic-scene-v1.json", b"{}\n")])
    destination = tmp_path / "cache"

    with pytest.raises(InputPreparationError, match="SHA256 mismatch"):
        restore_test_inputs(archive, destination, "0" * 64)

    assert not destination.exists()


@pytest.mark.parametrize(
    "name",
    [
        "../escape.json",
        "/escape.json",
        "tests/fixtures/worker-wire/../../escape.json",
        "tests/fixtures/worker-wire/./sample.json",
        "tests/fixtures/worker-wire//sample.json",
        r"tests/fixtures/worker-wire/\escape.json",
        "tests/fixtures/baseline.json",
        "worker/rust/tests/fixtures/support.rs",
        "worker/runtime/rust/tests/fixtures/gpu/other.json",
        "worker/policy/tests/fixtures/bed_input/sample.json",
        "worker/runtime/inference/tests/fixtures/gpu/manifest.json",
    ],
)
def test_invalid_archive_paths_fail_before_restoring_valid_member(
    tmp_path: Path, name: str
) -> None:
    archive, digest = _archive(
        tmp_path,
        [
            _file("tests/fixtures/synthetic-scene-v1.json", b"{}\n"),
            _file(name, b"untrusted\n"),
        ],
    )
    destination = tmp_path / "cache"

    with pytest.raises(InputPreparationError, match="archive path|Archive path"):
        restore_test_inputs(archive, destination, digest)

    assert not destination.exists()
    assert not (tmp_path / "escape.json").exists()


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE]
)
def test_nonregular_archive_members_are_rejected(tmp_path: Path, kind: bytes) -> None:
    member = tarfile.TarInfo("tests/fixtures/worker-wire/linked.json")
    member.type = kind
    member.linkname = "tests/fixtures/synthetic-scene-v1.json"
    archive, digest = _archive(
        tmp_path,
        [_file("tests/fixtures/synthetic-scene-v1.json", b"{}\n"), (member, b"")],
    )
    destination = tmp_path / "cache"

    with pytest.raises(InputPreparationError, match="regular file"):
        restore_test_inputs(archive, destination, digest)

    assert not destination.exists()


@pytest.mark.parametrize(
    ("archive_name", "target_name"),
    [
        (
            "tests/fixtures/worker-wire/existing.json",
            "tests/fixtures/worker-wire/existing.json",
        ),
        (
            "worker/rust/tests/fixtures/bed_input/existing.json",
            "worker/policy/tests/fixtures/bed_input/existing.json",
        ),
    ],
)
def test_conflicting_cache_is_preserved_without_partial_restore(
    tmp_path: Path, archive_name: str, target_name: str
) -> None:
    archive, digest = _archive(
        tmp_path,
        [
            _file("tests/fixtures/synthetic-scene-v1.json", b"{}\n"),
            _file(archive_name, b"new bytes\n"),
        ],
    )
    destination = tmp_path / "cache"
    existing = destination / target_name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"local bytes\n")

    with pytest.raises(InputPreparationError, match="Conflicting existing cache data"):
        restore_test_inputs(archive, destination, digest)

    assert existing.read_bytes() == b"local bytes\n"
    assert not (destination / "tests/fixtures/synthetic-scene-v1.json").exists()


@pytest.mark.parametrize(
    ("archive_name", "target_name"),
    [
        (
            "tests/fixtures/worker-wire/sample.json",
            "tests/fixtures/worker-wire/sample.json",
        ),
        (
            "worker/rust/tests/fixtures/bed_input/sample.json",
            "worker/policy/tests/fixtures/bed_input/sample.json",
        ),
    ],
)
@pytest.mark.parametrize("link_parent", [False, True])
def test_existing_symlink_cannot_redirect_cache_writes(
    tmp_path: Path, archive_name: str, target_name: str, link_parent: bool
) -> None:
    archive, digest = _archive(tmp_path, [_file(archive_name, b"new bytes\n")])
    destination = tmp_path / "cache"
    target = destination / target_name
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sample.json"
    sentinel.write_bytes(b"outside bytes\n")
    link = target.parent if link_parent else target
    link.parent.mkdir(parents=True)
    link.symlink_to(outside if link_parent else sentinel, target_is_directory=link_parent)

    with pytest.raises(InputPreparationError, match="unsafe or inaccessible"):
        restore_test_inputs(archive, destination, digest)

    assert sentinel.read_bytes() == b"outside bytes\n"
    assert link.is_symlink()


def test_missing_archive_has_explicit_error_and_writes_nothing(tmp_path: Path) -> None:
    destination = tmp_path / "cache"

    with pytest.raises(InputPreparationError, match="Missing input archive"):
        restore_test_inputs(tmp_path / "absent.tar.gz", destination, "0" * 64)

    assert not destination.exists()


@pytest.mark.parametrize("second_name", ["sample.json", "sample.json/nested.json"])
def test_duplicate_or_overlapping_archive_paths_write_nothing(
    tmp_path: Path, second_name: str
) -> None:
    archive, digest = _archive(
        tmp_path,
        [
            _file("tests/fixtures/worker-wire/sample.json", b"{}\n"),
            _file(f"tests/fixtures/worker-wire/{second_name}", b"[]\n"),
        ],
    )
    destination = tmp_path / "cache"

    with pytest.raises(
        InputPreparationError, match="Duplicate archive path|Conflicting archive paths"
    ):
        restore_test_inputs(archive, destination, digest)

    assert not destination.exists()


@pytest.mark.parametrize(
    ("archive_name", "target_name"),
    [
        (
            "tests/fixtures/worker-wire/sample.json",
            "tests/fixtures/worker-wire/sample.json",
        ),
        (
            "worker/rust/tests/fixtures/bed_input/sample.json",
            "worker/policy/tests/fixtures/bed_input/sample.json",
        ),
    ],
)
def test_existing_hardlink_is_preserved_and_refused(
    tmp_path: Path, archive_name: str, target_name: str
) -> None:
    archive, digest = _archive(tmp_path, [_file(archive_name, b"same bytes\n")])
    destination = tmp_path / "cache"
    target = destination / target_name
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"same bytes\n")
    target.hardlink_to(outside)

    with pytest.raises(InputPreparationError, match="single-link regular file"):
        restore_test_inputs(archive, destination, digest)

    assert outside.read_bytes() == b"same bytes\n"
    assert target.samefile(outside)


def test_digest_matching_invalid_archive_writes_nothing(tmp_path: Path) -> None:
    archive = tmp_path / "invalid.tar.gz"
    archive.write_bytes(b"not a tar archive\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "cache"

    with pytest.raises(InputPreparationError, match="Invalid input archive"):
        restore_test_inputs(archive, destination, digest)

    assert not destination.exists()
