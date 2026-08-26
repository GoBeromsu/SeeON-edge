from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_runtime_image_installs_relocatable_ctest_and_runs_literal_contract() -> None:
    # Given / When: the shipped image recipe is inspected as a machine contract.
    dockerfile = (REPOSITORY_ROOT / "Dockerfile.edge").read_text(encoding="utf-8")

    # Then: runtime CTest files are copied without build source and exercised literally.
    assert "/opt/seeon/native-build" in dockerfile
    assert "COPY --from=deepstream-native-build" in dockerfile
    assert "ctest --test-dir /opt/seeon/native-build --output-on-failure" in dockerfile
    runtime = dockerfile.split(" AS runtime", maxsplit=1)[1]
    assert "COPY worker/native/deepstream/src" not in runtime
    assert "apt-get install" in runtime
    assert "apt-get purge" in runtime
    assert "test ! -e /usr/bin/g++" in runtime
    assert "test ! -d /opt/nvidia/deepstream/deepstream/sources" in runtime


def test_native_build_has_fail_closed_sanitizer_ctest_lane() -> None:
    # Given / When: native CMake and Docker build-stage contracts are inspected.
    cmake = (REPOSITORY_ROOT / "worker/native/deepstream/CMakeLists.txt").read_text()
    dockerfile = (REPOSITORY_ROOT / "Dockerfile.edge").read_text()

    # Then: ASan+UBSan are explicit and their CTest lane cannot be suppressed.
    assert "SEEON_ENABLE_SANITIZERS" in cmake
    assert "-fsanitize=address,undefined" in cmake
    assert "build-sanitized" in dockerfile
    assert "ctest --test-dir build-sanitized --output-on-failure" in dockerfile


def test_runtime_image_exposes_and_inspects_deepstream_plugins() -> None:
    dockerfile = (REPOSITORY_ROOT / "Dockerfile.edge").read_text(encoding="utf-8")
    runtime = dockerfile.split(" AS runtime", maxsplit=1)[1]

    assert (
        "GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins"
        in runtime
    )
    assert "/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}" in runtime
    assert (
        "test -f /opt/nvidia/deepstream/deepstream/lib/gst-plugins/"
        "libnvdsgst_multistream.so"
        in runtime
    )
