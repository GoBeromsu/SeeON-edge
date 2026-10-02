"""Build an isolated observation-only nvdsinfer library. Not a runtime import."""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SDK = Path("/opt/nvidia/deepstream/deepstream")
SOURCE = SDK / "sources/libs/nvdsinfer"
INCLUDES = SDK / "sources/includes"
MODEL_SHA256 = "34ea136abd0ff22c6ec896725a81b30a1d85c975ca5e65c5cfe3e6998328966f"
MAKEFILE_SHA256 = "2ea6b4ca20ba6b5c21947ab79f15fd1e677a60eaca6b40cc4d14f5c338836aa2"
ANCHOR = (
    "    //return std::make_unique<TrtEngine>(std::move(engine), options.dlaCore);\n"
    "    //PASS dl-lib\n"
)
CUDA_OBSERVATION_INCLUDE = "#include <cuda_runtime_api.h>\n"
INCLUDE_ANCHOR = '#include "nvdsinfer_utils.h"\n'
OBSERVATION = (
    "    {\n"
    "        int device = -1;\n"
    "        cudaDeviceProp properties{};\n"
    "        if (cudaGetDevice(&device) == cudaSuccess && device == m_GpuId &&\n"
    "            cudaGetDeviceProperties(&properties, device) == cudaSuccess) {\n"
    '            dsInferInfo("SEEON_BUILD_OBSERVATION_V3 flags=%u fp16=%u tf32=%u trt=%d "\n'
    '                "device=%d sm_major=%d sm_minor=%d device_name=%.*s",\n'
    "                static_cast<unsigned>(m_BuilderConfig->getFlags()),\n"
    "                static_cast<unsigned>(\n"
    "                    m_BuilderConfig->getFlag(nvinfer1::BuilderFlag::kFP16)),\n"
    "                static_cast<unsigned>(\n"
    "                    m_BuilderConfig->getFlag(nvinfer1::BuilderFlag::kTF32)),\n"
    "                getInferLibVersion(),\n"
    "                device,\n"
    "                properties.major,\n"
    "                properties.minor,\n"
    "                static_cast<int>(sizeof(properties.name)),\n"
    "                properties.name);\n"
    "        } else {\n"
    '            dsInferInfo("SEEON_BUILD_OBSERVATION_V3 unavailable");\n'
    "        }\n"
    "    }\n"
)
PROFILE = (
    "    {\n"
    "        const char* input_name = nullptr;\n"
    "        const char* output_name = nullptr;\n"
    "        int inputs = 0;\n"
    "        int outputs = 0;\n"
    "        for (int index = 0; index < engine->getNbIOTensors(); ++index) {\n"
    "            const char* name = engine->getIOTensorName(index);\n"
    "            if (!name) {\n"
    "                inputs = 0;\n"
    "                outputs = 0;\n"
    "                break;\n"
    "            }\n"
    "            const auto mode = engine->getTensorIOMode(name);\n"
    "            if (mode == nvinfer1::TensorIOMode::kINPUT) {\n"
    "                ++inputs;\n"
    '                if (::strcmp(name, "images") == 0) {\n'
    "                    input_name = name;\n"
    "                }\n"
    "            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {\n"
    "                ++outputs;\n"
    '                if (::strcmp(name, "output0") == 0) {\n'
    "                    output_name = name;\n"
    "                }\n"
    "            } else {\n"
    "                inputs = 0;\n"
    "                outputs = 0;\n"
    "                break;\n"
    "            }\n"
    "        }\n"
    "        const int profiles = engine->getNbOptimizationProfiles();\n"
    "        const bool named = profiles == 1 && inputs == 1 && outputs == 1 &&\n"
    "            input_name && output_name;\n"
    "        const nvinfer1::Dims input_dims = named ? engine->getTensorShape(input_name)\n"
    "                                                : nvinfer1::Dims{};\n"
    "        const nvinfer1::Dims output_dims = named ? engine->getTensorShape(output_name)\n"
    "                                                 : nvinfer1::Dims{};\n"
    "        const nvinfer1::Dims min_dims = named\n"
    "            ? engine->getProfileShape(input_name, 0, nvinfer1::OptProfileSelector::kMIN)\n"
    "            : nvinfer1::Dims{};\n"
    "        const nvinfer1::Dims opt_dims = named\n"
    "            ? engine->getProfileShape(input_name, 0, nvinfer1::OptProfileSelector::kOPT)\n"
    "            : nvinfer1::Dims{};\n"
    "        const nvinfer1::Dims max_dims = named\n"
    "            ? engine->getProfileShape(input_name, 0, nvinfer1::OptProfileSelector::kMAX)\n"
    "            : nvinfer1::Dims{};\n"
    "        const bool input_float = named &&\n"
    "            engine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT;\n"
    "        const bool output_float = named &&\n"
    "            engine->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT;\n"
    "        if (profiles == 1 && inputs == 1 && outputs == 1 && input_name && output_name &&\n"
    "            input_float && output_float && input_dims.nbDims == 4"
    " && output_dims.nbDims == 3 &&\n"
    "            min_dims.nbDims == 4 && opt_dims.nbDims == 4 && max_dims.nbDims == 4) {\n"
    '            dsInferInfo("SEEON_BUILD_PROFILE_V1 profiles=%d input_float=%d output_float=%d "\n'
    '                "input_dims=%lldx%lldx%lldx%lld min=%lldx%lldx%lldx%lld "\n'
    '                "opt=%lldx%lldx%lldx%lld max=%lldx%lldx%lldx%lld "\n'
    '                "output_dims=%lldx%lldx%lld",\n'
    "                profiles,\n"
    "                static_cast<int>(input_float),\n"
    "                static_cast<int>(output_float),\n"
    "                static_cast<long long>(input_dims.d[0]),\n"
    "                static_cast<long long>(input_dims.d[1]),\n"
    "                static_cast<long long>(input_dims.d[2]),\n"
    "                static_cast<long long>(input_dims.d[3]),\n"
    "                static_cast<long long>(min_dims.d[0]),\n"
    "                static_cast<long long>(min_dims.d[1]),\n"
    "                static_cast<long long>(min_dims.d[2]),\n"
    "                static_cast<long long>(min_dims.d[3]),\n"
    "                static_cast<long long>(opt_dims.d[0]),\n"
    "                static_cast<long long>(opt_dims.d[1]),\n"
    "                static_cast<long long>(opt_dims.d[2]),\n"
    "                static_cast<long long>(opt_dims.d[3]),\n"
    "                static_cast<long long>(max_dims.d[0]),\n"
    "                static_cast<long long>(max_dims.d[1]),\n"
    "                static_cast<long long>(max_dims.d[2]),\n"
    "                static_cast<long long>(max_dims.d[3]),\n"
    "                static_cast<long long>(output_dims.d[0]),\n"
    "                static_cast<long long>(output_dims.d[1]),\n"
    "                static_cast<long long>(output_dims.d[2]));\n"
    "        } else {\n"
    '            dsInferInfo("SEEON_BUILD_PROFILE_V1 unavailable");\n'
    "        }\n"
    "    }\n"
)
COMMAND = ["make", "-B", "-j2", "CUDA_VER=13.2"]
KIND = "isolated-observer-compile-result-not-attestation"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def refuse(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def require_pin(path: Path, expected: str) -> str:
    if not _regular(path):
        refuse(f"pinned source missing or not a regular file: {path}")
    digest = sha256(path)
    if digest != expected:
        refuse(f"pinned source hash mismatch: {path}")
    return digest


def claim_destination(destination: Path) -> None:
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        refuse(f"destination parent is missing or not a directory: {parent}")
    try:
        destination.mkdir()
    except FileExistsError:
        refuse(f"destination already exists: {destination}")


def copy_sources(destination: Path) -> Path:
    copied = destination / "sources/libs/nvdsinfer"
    shutil.copytree(
        SOURCE,
        copied,
        symlinks=False,
        ignore=shutil.ignore_patterns("*.o", "*.so", "*.so.*"),
    )
    if (
        sha256(copied / "nvdsinfer_model_builder.cpp") != MODEL_SHA256
        or sha256(copied / "Makefile") != MAKEFILE_SHA256
    ):
        refuse("copied source does not match the pinned source")
    (destination / "sources/includes").symlink_to(INCLUDES, target_is_directory=True)
    return copied


def insert_observation(model: Path, before: str) -> str:
    if before.count(ANCHOR) != 1:
        refuse("observation anchor is missing or duplicated")
    if before.count(INCLUDE_ANCHOR) != 1 or CUDA_OBSERVATION_INCLUDE in before:
        refuse("cuda observation include anchor is missing or duplicated")
    after = before.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + CUDA_OBSERVATION_INCLUDE, 1)
    after = after.replace(ANCHOR, OBSERVATION + PROFILE + ANCHOR, 1)
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="nvdsinfer_model_builder.cpp",
            tofile="nvdsinfer_model_builder.cpp",
        )
    )
    model.write_text(after)
    return patch


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compile_copy(copied: Path, log_path: Path) -> int:
    with log_path.open("xb") as stream:
        completed = subprocess.run(
            COMMAND,
            cwd=copied,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def input_map(copied: Path) -> dict[str, str]:
    return {
        str(path.relative_to(copied)): sha256(path)
        for path in sorted(copied.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def build(destination: Path) -> dict[str, object]:
    source_model = SOURCE / "nvdsinfer_model_builder.cpp"
    source_makefile = SOURCE / "Makefile"
    model_digest = require_pin(source_model, MODEL_SHA256)
    makefile_digest = require_pin(source_makefile, MAKEFILE_SHA256)
    claim_destination(destination)
    copied = copy_sources(destination)
    copied_inputs = input_map(copied)
    model = copied / "nvdsinfer_model_builder.cpp"
    patch = insert_observation(model, model.read_text())
    patch_path = destination / "observation.patch"
    patch_path.write_text(patch)
    recipe = {
        "command": COMMAND,
        "copied_input_sha256": copied_inputs,
        "kind": KIND,
        "makefile_sha256": makefile_digest,
        "model_sha256": model_digest,
        "patched_model_sha256": sha256(model),
        "patch_sha256": sha256(patch_path),
        "purpose": "build-source evidence only; not engine identity or attestation",
    }
    recipe_path = destination / "source-recipe.json"
    write_json(recipe_path, recipe)
    log_path = destination / "compile.log"
    exit_code = compile_copy(copied, log_path)
    library = copied / "libnvds_infer.so"
    if exit_code != 0 or not _regular(library) or library.stat().st_size == 0:
        message = (
            f"observer compiler failed: exit={exit_code}"
            if exit_code
            else "observer output refused after successful compile: invalid library file"
        )
        print(message, file=sys.stderr)
        raise SystemExit(exit_code or 1)
    result = {
        "compile_exit": exit_code,
        "compile_log_sha256": sha256(log_path),
        "copied_input_sha256": copied_inputs,
        "kind": KIND,
        "makefile_sha256": makefile_digest,
        "model_sha256": model_digest,
        "output_sha256": sha256(library),
        "patched_model_sha256": recipe["patched_model_sha256"],
        "patch_sha256": recipe["patch_sha256"],
        "purpose": recipe["purpose"],
        "recipe_sha256": sha256(recipe_path),
    }
    write_json(destination / "build-result.json", result)
    return result


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        refuse("usage: prepare_nvdsinfer_observer.py DESTINATION")
    build(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
