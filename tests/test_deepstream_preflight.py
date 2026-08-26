from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from worker.native.deepstream.engine_cache import (
    PLAN_IDENTITY_FILENAME,
    build_plan_cache,
    resolve_plan,
)
from worker.native.deepstream.preflight import (
    DeepStreamPreflightError,
    prepare_engine_cache,
    run_deepstream_preflight,
)

_WARMUP_RECEIPT = (
    '{"status":"ok","frames":1,"source":"loopback",'
    '"engines":["bed","person","pose"],"inference":"ok"}\n'
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    plugin = tmp_path / "libnvdsgst_multistream.so"
    plugin.write_bytes(b"pinned-plugin")
    native = tmp_path / "seeon-deepstream-preflight"
    native.write_bytes(b"pinned-native")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    engine_source = tmp_path / "built.engine"
    engine_source.write_bytes(b"pinned-engine")
    engine_cache = tmp_path / "engine-cache"
    for weight_name in ("pose", "person", "bed"):
        weight_dir = model_dir / weight_name
        weight_dir.mkdir()
        (weight_dir / "weights.pt").write_bytes(f"weights-{weight_name}".encode())
    data = {
        "schema_version": 1,
        "runtime": {
            "deepstream": "9.1.0",
            "cuda": "13.2",
            "tensorrt": "10.13.3.9",
            "minimum_driver": "595.58.03",
            "minimum_compute_capability": "12.0",
        },
        "plugins": [
            {
                "element": "nvstreammux",
                "version": "9.1.0",
                "path": str(plugin),
                "sha256": _sha256(plugin),
            }
        ],
        "native": {"path": str(native), "sha256": _sha256(native)},
        "models": {"path": str(model_dir), "require_read_only": True},
        "engine_cache": {
            "path": str(engine_cache),
            "engines": [
                {
                    "name": "warmup-fixture",
                    "source": str(engine_source),
                    "sha256": _sha256(engine_source),
                }
            ],
        },
        "engine_plan": {
            "exporter": "ultralytics-8.4.61-onnx-opset17",
            "precision": "fp32",
            "builder": "trtexec-fp32-dynhw",
            "models": [
                {"name": "pose", "weight": str(model_dir / "pose" / "weights.pt")},
                {"name": "person", "weight": str(model_dir / "person" / "weights.pt")},
                {"name": "bed", "weight": str(model_dir / "bed" / "weights.pt")},
            ],
        },
        "warmup": {"timeout_seconds": 10},
    }
    path = tmp_path / "deepstream-manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _build_plan(manifest: Path) -> None:
    """Simulate the explicit deploy-host engine build with a fake builder."""
    data = json.loads(manifest.read_text(encoding="utf-8"))

    def fake_onnx(name: str, weight: Path, scratch: Path) -> Path:
        onnx = scratch / f"{name}.onnx"
        onnx.write_bytes(b"onnx-" + weight.read_bytes())
        return onnx

    def fake_builder(onnx_path: Path, engine_path: Path) -> None:
        engine_path.write_bytes(b"engine-" + onnx_path.read_bytes())

    _ = build_plan_cache(data, engine_builder=fake_builder, onnx_exporter=fake_onnx)


def _probe(command: tuple[str, ...], timeout: float) -> str:
    assert timeout > 0
    if command[0] == "nvidia-smi":
        return "595.84|12.0\n"
    if command == (
        "dpkg-query",
        "-W",
        "-f=${Version}\\n",
        "cuda-cudart-13-2",
    ):
        return "13.2.51-1\n"
    if command[0] == "dpkg-query":
        return "10.13.3.9-1+cuda13.2\n"
    if command[0] == "gst-inspect-1.0":
        return "Version                  9.1.0\n"
    if len(command) >= 2 and command[1] == "--warmup":
        assert len(command) == 3, "warmup must receive the plan cache directory"
        return _WARMUP_RECEIPT
    raise AssertionError(command)


def test_preflight_accepts_pinned_manifest_after_explicit_engine_cache_step(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    _build_plan(manifest)

    receipt = run_deepstream_preflight(
        manifest,
        command_runner=_probe,
        mount_is_read_only=lambda _path: True,
    )

    assert receipt["status"] == "ok"
    assert receipt["profile"] == "nvidia"
    assert receipt["warmup"] == {
        "status": "ok",
        "frames": 1,
        "source": "loopback",
        "engines": ["bed", "person", "pose"],
        "inference": "ok",
    }
    assert set(receipt["engines"]) == {"bed", "person", "pose"}


def test_preflight_refuses_missing_cuda_runtime_package(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    _build_plan(manifest)

    def missing_cuda_runtime(command: tuple[str, ...], timeout: float) -> str:
        if command == (
            "dpkg-query",
            "-W",
            "-f=${Version}\\n",
            "cuda-cudart-13-2",
        ):
            raise FileNotFoundError("cuda-cudart-13-2")
        return _probe(command, timeout)

    with pytest.raises(DeepStreamPreflightError) as excinfo:
        run_deepstream_preflight(
            manifest,
            command_runner=missing_cuda_runtime,
            mount_is_read_only=lambda _path: True,
        )

    assert excinfo.value.code == "cuda_version_mismatch"
    assert "cuda-cudart-13-2" in excinfo.value.detail


def test_preflight_refuses_unbuilt_plan_cache_before_warmup(tmp_path: Path) -> None:
    # Given: the image manifest is valid but the deploy host never ran the
    # explicit engine build step.
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    commands: list[tuple[str, ...]] = []

    def recording_probe(command: tuple[str, ...], timeout: float) -> str:
        commands.append(command)
        return _probe(command, timeout)

    # When / Then: boot refuses with a typed first fault and never warms up.
    with pytest.raises(DeepStreamPreflightError) as excinfo:
        run_deepstream_preflight(
            manifest,
            command_runner=recording_probe,
            mount_is_read_only=lambda _path: True,
        )
    assert excinfo.value.code == "engine_cache_unbuilt"
    assert not any("--warmup" in command for command in commands)


def test_preflight_rejects_stale_plan_engine_digest(tmp_path: Path) -> None:
    # Given: a built plan cache whose pose engine bytes were altered afterwards.
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    _build_plan(manifest)
    plan = resolve_plan(json.loads(manifest.read_text(encoding="utf-8")))
    stale = plan.cache_dir / "pose.engine"
    stale.chmod(0o644)
    stale.write_bytes(b"stale-engine")

    # When / Then
    with pytest.raises(DeepStreamPreflightError) as excinfo:
        run_deepstream_preflight(
            manifest,
            command_runner=_probe,
            mount_is_read_only=lambda _path: True,
        )
    assert excinfo.value.code == "engine_identity_mismatch"
    assert (plan.cache_dir / PLAN_IDENTITY_FILENAME).is_file()


def test_preflight_rejects_wrong_plugin_digest_with_typed_first_fault(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    _build_plan(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["plugins"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    fault = tmp_path / "first-fault.json"

    with pytest.raises(DeepStreamPreflightError) as excinfo:
        run_deepstream_preflight(
            manifest,
            command_runner=_probe,
            mount_is_read_only=lambda _path: True,
            first_fault_path=fault,
        )

    assert excinfo.value.code == "plugin_digest_mismatch"
    assert json.loads(fault.read_text(encoding="utf-8"))["code"] == "plugin_digest_mismatch"


def test_preflight_rejects_writable_model_target_before_warmup(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    _build_plan(manifest)
    commands: list[tuple[str, ...]] = []

    def recording_probe(command: tuple[str, ...], timeout: float) -> str:
        commands.append(command)
        return _probe(command, timeout)

    with pytest.raises(DeepStreamPreflightError) as excinfo:
        run_deepstream_preflight(
            manifest,
            command_runner=recording_probe,
            mount_is_read_only=lambda _path: False,
        )

    assert excinfo.value.code == "model_target_writable"
    assert not any("--warmup" in command for command in commands)


def test_preflight_rejects_absent_gpu_before_plugin_or_source_activation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    _build_plan(manifest)
    commands: list[tuple[str, ...]] = []

    def no_gpu(command: tuple[str, ...], timeout: float) -> str:
        commands.append(command)
        if command[0] == "nvidia-smi":
            raise RuntimeError("NVIDIA device unavailable")
        return _probe(command, timeout)

    with pytest.raises(DeepStreamPreflightError) as excinfo:
        run_deepstream_preflight(
            manifest,
            command_runner=no_gpu,
            mount_is_read_only=lambda _path: True,
        )

    assert excinfo.value.code == "gpu_absent"
    assert commands == [
        (
            "nvidia-smi",
            "--query-gpu=driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        )
    ]


def test_preflight_rejects_stale_engine_cache_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    _build_plan(manifest)
    cached = tmp_path / "engine-cache" / "warmup-fixture.engine"
    cached.chmod(0o644)
    cached.write_bytes(b"stale")

    with pytest.raises(DeepStreamPreflightError) as excinfo:
        run_deepstream_preflight(
            manifest,
            command_runner=_probe,
            mount_is_read_only=lambda _path: True,
        )

    assert excinfo.value.code == "engine_identity_mismatch"


def test_preflight_rejects_misleading_success_text_and_hung_warmup(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    prepare_engine_cache(manifest)
    _build_plan(manifest)

    def misleading(command: tuple[str, ...], timeout: float) -> str:
        if len(command) >= 2 and command[1] == "--warmup":
            return _WARMUP_RECEIPT + "not-a-receipt\n"
        return _probe(command, timeout)

    with pytest.raises(DeepStreamPreflightError) as misleading_error:
        run_deepstream_preflight(
            manifest,
            command_runner=misleading,
            mount_is_read_only=lambda _path: True,
        )
    assert misleading_error.value.code == "warmup_failed"

    def hung(command: tuple[str, ...], timeout: float) -> str:
        if len(command) >= 2 and command[1] == "--warmup":
            raise subprocess.TimeoutExpired(command, timeout)
        return _probe(command, timeout)

    with pytest.raises(DeepStreamPreflightError) as hung_error:
        run_deepstream_preflight(
            manifest,
            command_runner=hung,
            mount_is_read_only=lambda _path: True,
        )
    assert hung_error.value.code == "warmup_failed"
