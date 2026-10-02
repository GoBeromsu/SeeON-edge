from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from worker.tools.prepare_nvdsinfer_observer import (
    ANCHOR,
    COMMAND,
    CUDA_OBSERVATION_INCLUDE,
    INCLUDE_ANCHOR,
    OBSERVATION,
    PROFILE,
    build,
)

ORIGINAL = INCLUDE_ANCHOR + "int build() {\n" + ANCHOR + "return 0;\n}\n"
MAKEFILE = "all:\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _pin(root: Path, model: str = ORIGINAL, makefile: str = MAKEFILE) -> None:
    source = root / "sdk/sources/libs/nvdsinfer"
    _write(source / "nvdsinfer_model_builder.cpp", model)
    _write(source / "Makefile", makefile)
    _write(source / "ignored.o", "object")
    _write(source / "ignored.so", "library")
    _write(source / "ignored.so.1", "versioned")
    includes = root / "sdk/sources/includes"
    includes.mkdir(parents=True, exist_ok=True)
    (includes / "nvdsinfer.h").write_text("header")


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    compile_result: tuple[int, bytes, bytes | None],
    *,
    race: bool = False,
) -> None:
    import worker.tools.prepare_nvdsinfer_observer as helper

    sdk = root / "sdk"
    source = sdk / "sources/libs/nvdsinfer"
    monkeypatch.setattr(helper, "SDK", sdk)
    monkeypatch.setattr(helper, "SOURCE", source)
    monkeypatch.setattr(helper, "INCLUDES", sdk / "sources/includes")
    monkeypatch.setattr(helper, "MODEL_SHA256", _sha256(source / "nvdsinfer_model_builder.cpp"))
    monkeypatch.setattr(helper, "MAKEFILE_SHA256", _sha256(source / "Makefile"))
    if race:
        original = helper.require_pin
        calls = {"count": 0}

        def racing_pin(path: Path, expected: str) -> str:
            calls["count"] += 1
            if calls["count"] == 2:
                owned = root / "observer"
                owned.mkdir()
                (owned / "kept.txt").write_text("user-owned")
            return original(path, expected)

        monkeypatch.setattr(helper, "require_pin", racing_pin)

    def fake_compile(copied: Path, log_path: Path) -> int:
        code, log, library = compile_result
        log_path.write_bytes(log)
        if library is not None:
            (copied / "libnvds_infer.so").write_bytes(library)
        return code

    monkeypatch.setattr(helper, "compile_copy", fake_compile)


def _build(root: Path) -> dict[str, object]:
    return build(root / "observer")


@pytest.mark.parametrize("target", ["model", "makefile"])
def test_pins_reject_before_any_destination_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    _pin(tmp_path)
    _patch(monkeypatch, tmp_path, (0, b"log", b"library"))
    mutated = (
        tmp_path
        / "sdk/sources/libs/nvdsinfer"
        / ("nvdsinfer_model_builder.cpp" if target == "model" else "Makefile")
    )
    mutated.write_text("mutated-after-pin\n")

    with pytest.raises(SystemExit) as captured:
        build(tmp_path / "observer")

    assert captured.value.code == 2
    assert not (tmp_path / "observer").exists()


def test_existing_destination_race_leaves_user_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(tmp_path)
    _patch(monkeypatch, tmp_path, (0, b"log", b"library"), race=True)

    with pytest.raises(SystemExit) as captured:
        _build(tmp_path)

    destination = tmp_path / "observer"
    assert captured.value.code == 2
    assert (destination / "kept.txt").read_text() == "user-owned"
    assert not (destination / "sources").exists()
    assert not (destination / "build-result.json").exists()


@pytest.mark.parametrize("model", ["int build(){ return 0; }\n", ORIGINAL + ANCHOR])
def test_zero_or_duplicate_anchor_refuses_without_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    _pin(tmp_path, model=model)
    _patch(monkeypatch, tmp_path, (0, b"log", b"library"))

    with pytest.raises(SystemExit) as captured:
        _build(tmp_path)

    assert captured.value.code == 2
    assert not (tmp_path / "observer/observation.patch").exists()
    assert not (tmp_path / "observer/build-result.json").exists()


def test_source_tree_and_pins_remain_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(tmp_path)
    _patch(monkeypatch, tmp_path, (0, b"compile-ok", b"observer-library"))
    source = tmp_path / "sdk/sources/libs/nvdsinfer/nvdsinfer_model_builder.cpp"
    before = source.read_bytes()

    result = _build(tmp_path)

    assert source.read_bytes() == before
    assert result["model_sha256"] == _sha256(source)
    copied = tmp_path / "observer/sources/libs/nvdsinfer"
    assert not (copied / "ignored.o").exists()
    assert not (copied / "ignored.so").exists()
    assert not (copied / "ignored.so.1").exists()
    assert (tmp_path / "observer/sources/includes").is_symlink()
    recipe = json.loads((tmp_path / "observer/source-recipe.json").read_text())
    prepatch = recipe["copied_input_sha256"]["nvdsinfer_model_builder.cpp"]
    assert prepatch == _sha256(source)
    assert prepatch != recipe["patched_model_sha256"]
    assert recipe["patched_model_sha256"] == _sha256(copied / "nvdsinfer_model_builder.cpp")
    assert recipe["copied_input_sha256"]["Makefile"] == _sha256(copied / "Makefile")


@pytest.mark.parametrize(
    ("code", "log", "library"), [(7, b"failed", b"partial"), (0, b"no-output", None)]
)
def test_compile_failure_or_missing_output_keeps_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    log: bytes,
    library: bytes | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _pin(tmp_path)
    _patch(monkeypatch, tmp_path, (code, log, library))

    with pytest.raises(SystemExit) as captured:
        _build(tmp_path)

    assert captured.value.code == (code or 1)
    assert (tmp_path / "observer/compile.log").read_bytes() == log
    assert not (tmp_path / "observer/build-result.json").exists()
    message = capsys.readouterr().err
    if code:
        assert message == f"observer compiler failed: exit={code}\n"
    else:
        assert message == "observer output refused after successful compile: invalid library file\n"


def test_regular_output_insertion_is_getter_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(tmp_path)
    _patch(monkeypatch, tmp_path, (0, b"compile-ok", b"observer-library"))

    result = _build(tmp_path)

    library = tmp_path / "observer/sources/libs/nvdsinfer/libnvds_infer.so"
    assert library.is_file() and not library.is_symlink() and library.stat().st_size > 0
    assert result["output_sha256"] == _sha256(library)
    model = (tmp_path / "observer/sources/libs/nvdsinfer/nvdsinfer_model_builder.cpp").read_text()
    expected = ORIGINAL.replace(
        INCLUDE_ANCHOR, INCLUDE_ANCHOR + CUDA_OBSERVATION_INCLUDE, 1
    ).replace(ANCHOR, OBSERVATION + PROFILE + ANCHOR, 1)
    assert model == expected
    inserted = model[model.index(OBSERVATION) : model.index(ANCHOR)]
    assert inserted == OBSERVATION + PROFILE
    hardware = inserted[: inserted.index(PROFILE)]
    assert hardware == OBSERVATION
    assert "setFlag" not in hardware
    assert "clearFlag" not in hardware
    assert "cudaSetDevice" not in hardware
    assert "getFlags()" in hardware
    assert "getFlag(nvinfer1::BuilderFlag::kFP16)" in hardware
    assert "getFlag(nvinfer1::BuilderFlag::kTF32)" in hardware
    assert "getInferLibVersion()" in hardware
    assert hardware.count("cudaGetDevice(") == 1
    assert hardware.count("cudaGetDeviceProperties(") == 1
    assert "device == m_GpuId" in hardware
    assert "cudaSuccess" in hardware
    assert 'dsInferInfo("SEEON_BUILD_OBSERVATION_V3 flags=%u fp16=%u tf32=%u trt=%d "' in hardware
    assert '"device=%d sm_major=%d sm_minor=%d device_name=%.*s"' in hardware
    assert 'dsInferInfo("SEEON_BUILD_OBSERVATION_V3 unavailable")' in hardware
    assert hardware.count("SEEON_BUILD_OBSERVATION_V3") == 2
    assert "SEEON_BUILD_OBSERVATION_V2" not in hardware
    assert model.count(CUDA_OBSERVATION_INCLUDE) == 1
    assert (
        model.index(INCLUDE_ANCHOR)
        < model.index(CUDA_OBSERVATION_INCLUDE)
        < model.index(OBSERVATION)
    )
    assert hardware.startswith("    {\n") and hardware.endswith("    }\n")
    assert "int device = -1;" in hardware
    assert "cudaDeviceProp properties{};" in hardware
    assert "static_cast<int>(sizeof(properties.name))" in hardware
    assert "properties.name" in hardware
    assert json.loads((tmp_path / "observer/source-recipe.json").read_text())["command"] == COMMAND
    assert "SEEON_BUILD_OBSERVATION_V2" not in model


def test_profile_observation_is_separate_getter_only_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(tmp_path)
    _patch(monkeypatch, tmp_path, (0, b"compile-ok", b"observer-library"))

    _build(tmp_path)

    model = (tmp_path / "observer/sources/libs/nvdsinfer/nvdsinfer_model_builder.cpp").read_text()
    inserted = model[model.index(OBSERVATION) : model.index(ANCHOR)]
    profile = inserted[inserted.index(PROFILE) :]
    assert inserted.startswith(OBSERVATION)
    assert profile == PROFILE
    assert model.index(OBSERVATION) < model.index(PROFILE) < model.index(ANCHOR)
    assert inserted.count("SEEON_BUILD_OBSERVATION_V3") == 2
    assert profile.count("SEEON_BUILD_PROFILE_V1") == 2
    assert (
        'dsInferInfo("SEEON_BUILD_PROFILE_V1 profiles=%d input_float=%d output_float=%d "'
        in profile
    )
    assert '"input_dims=%lldx%lldx%lldx%lld min=%lldx%lldx%lldx%lld "' in profile
    assert '"opt=%lldx%lldx%lldx%lld max=%lldx%lldx%lldx%lld "' in profile
    assert '"output_dims=%lldx%lldx%lld"' in profile
    assert 'dsInferInfo("SEEON_BUILD_PROFILE_V1 unavailable")' in profile
    assert "engine->getNbOptimizationProfiles()" in profile
    assert "engine->getNbIOTensors()" in profile
    assert "engine->getIOTensorName(" in profile
    assert "engine->getTensorIOMode(" in profile
    assert "engine->getTensorDataType(" in profile
    assert "engine->getTensorShape(" in profile
    assert "engine->getProfileShape(" in profile
    assert "nvinfer1::OptProfileSelector::kMIN" in profile
    assert "nvinfer1::OptProfileSelector::kOPT" in profile
    assert "nvinfer1::OptProfileSelector::kMAX" in profile
    assert '::strcmp(name, "images")' in profile
    assert '::strcmp(name, "output0")' in profile
    assert "TensorIOMode::kINPUT" in profile
    assert "TensorIOMode::kOUTPUT" in profile
    assert "DataType::kFLOAT" in profile
    assert "input_dims.nbDims == 4" in profile
    assert "output_dims.nbDims == 3" in profile
    assert "min_dims.nbDims == 4" in profile
    assert "opt_dims.nbDims == 4" in profile
    assert "max_dims.nbDims == 4" in profile
    assert "profiles == 1" in profile
    assert "profiles == 1 && inputs == 1 && outputs == 1" in profile
    assert "inputs == 1" in profile
    assert "outputs == 1" in profile
    assert profile.count("static_cast<long long>(") == 19
    assert "m_InitParams" not in profile
    assert "options." not in profile
    assert "network." not in profile
    assert "FullDims" not in profile
    assert "setDimensions" not in profile
    assert "setInputShape" not in profile
    assert "setFlag" not in profile
    assert "clearFlag" not in profile
    assert "buildSerializedNetwork" not in profile
    assert "cudaSetDevice" not in profile
    assert "640" not in profile
    assert "300x57" not in profile


def _place_invalid_output(copied: Path, kind: str) -> None:
    library = copied / "libnvds_infer.so"
    if kind == "symlink":
        target = copied.parent / "foreign-library.so"
        target.write_bytes(b"must remain untouched")
        library.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(library)
    elif kind == "directory":
        library.mkdir()
    else:
        library.write_bytes(b"")


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "empty"])
def test_nonregular_or_empty_output_refuses_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _pin(tmp_path)
    _patch(monkeypatch, tmp_path, (0, b"compiled", None))
    import worker.tools.prepare_nvdsinfer_observer as helper

    def fake_compile(copied: Path, log_path: Path) -> int:
        log_path.write_bytes(b"compiled")
        _place_invalid_output(copied, kind)
        return 0

    monkeypatch.setattr(helper, "compile_copy", fake_compile)

    with pytest.raises(SystemExit) as captured:
        _build(tmp_path)

    assert captured.value.code == 1
    assert (tmp_path / "observer/compile.log").read_bytes() == b"compiled"
    assert not (tmp_path / "observer/build-result.json").exists()
    if kind == "symlink":
        target = tmp_path / "observer/sources/libs/foreign-library.so"
        assert target.read_bytes() == b"must remain untouched"
