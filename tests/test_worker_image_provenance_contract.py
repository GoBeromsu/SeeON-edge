from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from scripts.edge_image_plan import ML_API, ML_WORKER, affected_images

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.edge"
ZERO_REVISION = "0" * 40
IMAGE_REVISION_MARKER = "/opt/seeon/ml-worker-image-revision"
SCHEMA_IDENTITY_MARKER = "/opt/seeon/edge-database-schema-version"
DOCKERIGNORE = ROOT / ".dockerignore"
NATIVE_LIBRARIES = (
    "libseeon_gpu.so",
    "libseeon_media.so",
    "libseeon_clipdec.so",
    "libseeon_ort.so",
)
BUILD_ONLY_TREES = (
    "bin",
    "rust",
    "runtime/rust",
    "adapters/deepstream/rust",
    "adapters/model/onnxruntime",
    "adapters/model/native",
)
DEEPSTREAM_CONFIGS = (
    "config_tracker_NvDCF_perf.yml",
    "config_tracker_NvDCF_accuracy.yml",
    "labels.txt",
    "nvinfer-yolo26-pose.txt",
)
_STAGE_HEADER = re.compile(
    r"^FROM(?:\s+--\S+)*\s+\S+(?:\s+AS\s+(\S+))?\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _stages(source: str) -> dict[str, str]:
    matches = list(_STAGE_HEADER.finditer(source))
    stages: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        assert name is not None, "Every build stage must be named, including the final stage"
        assert name not in stages, f"Duplicate build stage: {name}"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        stages[name] = source[match.end() : end]
    return stages


def _commands(stage: str) -> list[str]:
    commands: list[str] = []
    pending = ""
    for line in stage.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not pending and not stripped.startswith("RUN "):
            continue
        fragment = stripped.removeprefix("RUN ").strip() if not pending else stripped
        pending = f"{pending} {fragment}".strip() if pending else fragment
        if stripped.endswith("\\"):
            pending = pending.removesuffix("\\").rstrip()
            continue
        commands.append(pending)
        pending = ""
    if pending:
        commands.append(pending)
    return commands


def test_edge_image_requires_source_revision_without_unsafe_default() -> None:
    source = _dockerfile()

    assert re.search(r"^ARG SOURCE_REVISION$", source, re.MULTILINE)
    assert not re.search(r"^ARG SOURCE_REVISION=", source, re.MULTILINE)
    assert source.count("ARG EDGE_DATABASE_SCHEMA_VERSION=19") == 1


def test_unnamed_final_stage_cannot_hide_from_packaging_assertions() -> None:
    with pytest.raises(AssertionError, match="Every build stage must be named"):
        _stages(_dockerfile() + "\nFROM cargo-verify\n")


@pytest.mark.parametrize("payload", [None, b"", b"test-library-bytes"])
def test_cpu_loader_path_has_an_executed_image_presence_guard(
    tmp_path: Path, payload: bytes | None
) -> None:
    source = (ROOT / "worker/bin/src/run/cpu_admission.rs").read_text(encoding="utf-8")
    declared = re.search(r'const RUNTIME_LIBRARY: &str =\s*"([^"]+)";', source)
    assert declared is not None
    guards = [
        shlex.split(part.strip())
        for command in _commands(_stages(_dockerfile())["runtime"])
        for part in command.split("&&")
        if part.strip().startswith("test -s ")
    ]
    assert ["test", "-s", declared.group(1)] in guards
    library = tmp_path / "libonnxruntime.so"
    if payload is not None:
        library.write_bytes(payload)
    result = subprocess.run(
        ["/bin/sh", "-c", 'test -s "$1"', "ort-image-guard", str(library)],
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == (0 if payload else 1)


@pytest.mark.parametrize(
    "revision",
    [
        None,
        "",
        ZERO_REVISION,
        "1" * 39,
        "1" * 41,
        "A" * 40,
        "g" * 40,
        "1" * 39 + " ",
        "1" * 20 + "\n" + "1" * 19,
        "\n" + "1" * 39,
        "1" * 39 + "\n",
        "1" * 40,
    ],
)
def test_revision_shell_guard_executes_without_writing_image_markers(
    revision: str | None,
) -> None:
    stages = _stages(_dockerfile())
    owner = stages["source-revision"]
    start = owner.index("RUN if ")
    end = owner.index("    fi;", start) + len("    fi;")
    guard = owner[start:end].removeprefix("RUN ").replace("\\\n", "")
    assert "install -d" not in guard
    assert owner.count("RUN if ") == 1
    assert "RUN if " not in stages["runtime"]
    environment = {"PATH": os.defpath}
    if revision is not None:
        environment["SOURCE_REVISION"] = revision
    result = subprocess.run(
        ["/bin/sh", "-c", guard],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if revision == "1" * 40:
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
    else:
        assert result.returncode == 1, result.stderr
        assert result.stderr == (
            "SOURCE_REVISION must be a non-zero 40-character lowercase hexadecimal revision\n"
        )


def test_stage_parser_separates_case_insensitive_keywords() -> None:
    stages = _stages("from base as builder\nRUN build\nFrOm base aS runtime\nRUN serve\n")
    assert set(stages) == {"builder", "runtime"}
    assert "RUN build" in stages["builder"]
    assert "RUN serve" not in stages["builder"]
    assert "RUN serve" in stages["runtime"]


def test_edge_image_build_rejects_zero_and_malformed_source_revisions() -> None:
    owner = _stages(_dockerfile())["source-revision"]
    validation = owner[owner.index("ARG SOURCE_REVISION") : owner.index("chmod 0444")]

    assert f'[ "$SOURCE_REVISION" = "{ZERO_REVISION}" ]' in validation
    assert '"${#SOURCE_REVISION}" -ne 40' in validation
    assert "*[!0123456789abcdef]*" in validation
    assert "exit 1" in validation
    assert "LABEL " not in owner


def test_edge_image_bakes_one_revision_into_label_environment_and_marker() -> None:
    source = _dockerfile()

    assert 'org.opencontainers.image.revision="${SOURCE_REVISION}"' in source
    assert 'ML_WORKER_BUILD_REVISION="${SOURCE_REVISION}"' in source
    assert f'"$SOURCE_REVISION" > {IMAGE_REVISION_MARKER}' in source
    assert SCHEMA_IDENTITY_MARKER in source
    assert 'seeon.edge.database.schema-version="${EDGE_DATABASE_SCHEMA_VERSION}"' in source
    assert f"chmod 0444 {IMAGE_REVISION_MARKER} {SCHEMA_IDENTITY_MARKER}" in source


def test_revision_stage_refuses_marker_write_failure(tmp_path: Path) -> None:
    marker_root = tmp_path / "seeon"
    (marker_root / Path(IMAGE_REVISION_MARKER).name).mkdir(parents=True)
    owner = _stages(_dockerfile())["source-revision"]
    command = next(command for command in _commands(owner) if command.startswith("if "))
    command = command.replace("/opt/seeon", shlex.quote(str(marker_root)))
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env={
            "PATH": os.defpath,
            "SOURCE_REVISION": "1" * 40,
            "EDGE_DATABASE_SCHEMA_VERSION": "19",
        },
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert not (marker_root / Path(SCHEMA_IDENTITY_MARKER).name).exists()


def test_release_compile_binds_validated_revision_only_on_its_command() -> None:
    stages = _stages(_dockerfile())
    release = stages["cargo-build"]
    compile_commands = [
        command
        for command in _commands(release)
        if "cargo build --locked --release --workspace" in command
    ]

    assert len(compile_commands) == 1
    assert 'build_revision="$(cat /tmp/.source-revision)"' in compile_commands[0]
    assert 'ML_WORKER_BUILD_REVISION="$build_revision" cargo build' in compile_commands[0]
    assert "test -s /tmp/.source-revision" in compile_commands[0]
    assert "cargo test" not in compile_commands[0]
    assert (
        "COPY --from=source-revision /opt/seeon/ml-worker-image-revision /tmp/.source-revision"
        in release
    )
    assert "ENV ML_WORKER_BUILD_REVISION" not in release
    assert "ML_WORKER_BUILD_REVISION" not in stages["cargo-verify"]
    assert stages["runtime"].count("ML_WORKER_BUILD_REVISION") == 1


def test_release_command_exports_revision_to_cargo_without_leaking_it(tmp_path: Path) -> None:
    """Execute the Docker shell binding with an environment-observing Cargo double."""
    release = _stages(_dockerfile())["cargo-build"]
    command = next(
        command
        for command in _commands(release)
        if "cargo build --locked --release --workspace" in command
    )
    start = command.index("build_revision=")
    end = command.index(" && test -x target/release/ml-worker", start)
    revision = "1" * 40
    marker = tmp_path / "revision"
    marker.write_text(revision + "\n", encoding="utf-8")
    cargo = tmp_path / "cargo"
    cargo.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"${ML_WORKER_BUILD_REVISION-unset}\"\n",
        encoding="utf-8",
    )
    cargo.chmod(0o700)
    shell = command[start:end].replace("/tmp/.source-revision", shlex.quote(str(marker)))
    result = subprocess.run(
        ["/bin/sh", "-c", shell + "; cargo"],
        env={"PATH": f"{tmp_path}:{os.defpath}"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [revision, "unset"]


def test_runtime_packages_release_binary_and_native_libraries_without_verification() -> None:
    stages = _stages(_dockerfile())
    runtime = stages["runtime"]

    assert list(stages)[-1] == "runtime"
    assert "COPY --from=cargo-verify" not in runtime
    assert "--mount=type=secret,id=rust-test-inputs" not in runtime
    assert "prepare_rust_test_inputs.py" not in runtime
    assert (
        "COPY --from=cargo-build /usr/src/myapp/target/release/ml-worker /usr/local/bin/ml-worker"
        in runtime
    )
    for library in NATIVE_LIBRARIES:
        assert (
            f"COPY --from=native-build /opt/seeon/lib/{library} /opt/seeon/lib/{library}" in runtime
        )
    assert (
        "LD_LIBRARY_PATH=/opt/seeon/lib:/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}"
        in runtime
    )
    assert 'CMD ["python", "-m", "worker"]' in runtime
    assert (
        "COPY --from=source-revision /opt/seeon/ml-worker-image-revision "
        "/opt/seeon/ml-worker-image-revision"
    ) in runtime
    assert (
        "COPY --from=source-revision /opt/seeon/edge-database-schema-version "
        "/opt/seeon/edge-database-schema-version"
    ) in runtime
    assert 'org.opencontainers.image.revision="${SOURCE_REVISION}"' in runtime
    assert 'ENV ML_WORKER_BUILD_REVISION="${SOURCE_REVISION}"' in runtime


def test_runtime_copies_filtered_worker_without_shipping_build_stage_layers() -> None:
    source = _dockerfile()
    stages = _stages(source)
    assert "FROM uv-bin AS worker-runtime-source" in source
    assert "COPY worker /worker" in stages["worker-runtime-source"]
    assert "COPY --from=worker-runtime-source /worker ./worker" in stages["runtime"]
    assert "COPY worker " not in stages["runtime"]
    runtime_header = next(
        match.group(0) for match in _STAGE_HEADER.finditer(source) if match.group(1) == "runtime"
    )
    assert runtime_header.startswith("FROM nvcr.io/nvidia/deepstream@sha256:")
    for stage in ("cargo-build", "runtime"):
        assert "ORT_DISABLE_TELEMETRY=1" in stages[stage]
    for tree in BUILD_ONLY_TREES:
        assert f"COPY worker/{tree} ./worker/{tree}" in stages["cargo-build"]


def test_worker_payload_filter_removes_only_build_trees_before_final_copy(tmp_path: Path) -> None:
    commands = _commands(_stages(_dockerfile())["worker-runtime-source"])
    assert len(commands) == 1
    tokens = shlex.split(commands[0])
    removed = BUILD_ONLY_TREES
    assert tokens[:2] == ["rm", "-rf"]
    assert tokens[2:] == [f"/worker/{tree}" for tree in removed]
    root = tmp_path / "worker payload"
    for tree in removed:
        fixture = root / tree / "tests" / "private-fixture.json"
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b"verification-only")
    kept = {
        "__main__.py": b"python entry",
        "runtime/worker.py": b"python composition",
        "adapters/deepstream/configs/labels.txt": b"person\n",
        "adapters/deepstream/metadata.py": b"python adapter",
        "adapters/model/ort_bed_seg.py": b"existing CPU model adapter",
        "tools/edge_engine_build.py": b"offline builder",
    }
    for name, content in kept.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    subprocess.run(
        ["/bin/sh", "-eu", "-c", commands[0].replace("/worker", shlex.quote(str(root)))],
        check=True,
        capture_output=True,
        timeout=5,
    )
    for tree in removed:
        assert not (root / tree).exists()
    assert {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == kept


def test_cargo_verify_requires_secret_and_keeps_cpu_tests_unchanged() -> None:
    verify = _stages(_dockerfile())["cargo-verify"]
    test_commands = [command for command in _commands(verify) if "cargo test" in command]

    assert len(test_commands) == 1
    assert test_commands[0].startswith("--mount=type=secret,id=rust-test-inputs,required=true")
    assert "python3 /usr/src/myapp/scripts/prepare_rust_test_inputs.py" in test_commands[0]
    assert "--archive /run/secrets/rust-test-inputs" in test_commands[0]
    assert "cargo test --locked --workspace" in test_commands[0]
    assert test_commands[0].endswith("cargo test --locked --workspace")
    assert "FROM cargo-build AS cargo-verify" in _dockerfile()
    assert "USER 65534:65534" in verify
    assert "chown -R 65534:65534 /usr/src/myapp" in verify
    assert "ENV CARGO_HOME=/tmp/seeon-cargo" in verify
    assert "id=rust-test-inputs,required=true,uid=65534,gid=65534" in test_commands[0]
    assert (
        "id=seeon-rust-verification-registry,target=/tmp/seeon-cargo/registry,uid=65534,gid=65534"
        in test_commands[0]
    )
    assert "ML_WORKER_BUILD_REVISION" not in test_commands[0]
    assert "-- --skip" not in test_commands[0]
    assert (
        "COPY scripts/prepare_rust_test_inputs.py "
        "/usr/src/myapp/scripts/prepare_rust_test_inputs.py"
    ) in verify
    for name in DEEPSTREAM_CONFIGS:
        assert (ROOT / "worker/adapters/deepstream/configs" / name).is_file()
    assert (
        "COPY worker/adapters/deepstream/configs /usr/src/myapp/worker/adapters/deepstream/configs"
    ) in verify


def test_dockerignore_excludes_cargo_targets_and_live_pose_scratch() -> None:
    ignored = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()

    assert "**/target" in ignored
    assert "**/.live-pose-*" in ignored
    assert "tests" in ignored
    assert "worker/rust/tests/fixtures/**/*.json" in ignored
    assert "worker/runtime/rust/tests/fixtures/gpu/*.json" in ignored


def test_worker_input_classifier_prefers_specific_rust_inputs_over_neutral_scripts() -> None:
    assert affected_images("Cargo.toml") == frozenset({ML_WORKER})
    assert affected_images("Cargo.lock") == frozenset({ML_WORKER})
    assert affected_images("rust-toolchain.toml") == frozenset({ML_WORKER})
    assert affected_images("scripts/prepare_rust_test_inputs.py") == frozenset({ML_WORKER})
    assert affected_images("scripts/edge_image_plan.py") == frozenset()
    assert affected_images("scripts/ops/cloud-enrollment-smoke.sh") == frozenset({ML_API})
    assert "test_edge_image_isolation.py" not in (ROOT / "scripts/edge_image_plan.py").read_text(
        encoding="utf-8"
    )
