from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import onnx
import pytest
from onnx import TensorProto, helper

from worker.tools.edge_engine_build import EngineBuildError, build_engine, sha256


def _write_onnx(path: Path, *, input_name: str = "frames") -> None:
    # A loadable graph, not just a signature: the build tool reads the input
    # through onnxruntime, which refuses a model with no nodes.
    graph = helper.make_graph(
        [helper.make_node("Identity", [input_name], ["output0"])],
        "pose",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 3, 640, 640])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


def test_build_writes_identity(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    _write_onnx(onnx_path)
    parser = tmp_path / "parser.so"
    infer = tmp_path / "infer.yml"
    tracker = tmp_path / "tracker.yml"
    tracker_library = tmp_path / "libnvds_nvmultiobjecttracker.so"
    for path in (parser, tracker, tracker_library):
        path.write_bytes(path.name.encode())
    infer.write_text("model-engine-file=stale.engine\nbatch-size=1\n", encoding="utf-8")
    engine = tmp_path / "model.engine"

    def run(command: list[str], **_: object) -> CompletedProcess[str]:
        engine.write_bytes(b"engine")
        return CompletedProcess(command, 0, "", "")

    identity = build_engine(
        onnx=onnx_path,
        engine=engine,
        identity_path=tmp_path / "identity.json",
        parser_lib=parser,
        infer_config=infer,
        tracker_library=tracker_library,
        tracker_config=tracker,
        image_digest="image",
        batch_size=14,
        served_infer_config=tmp_path / "served-infer.yml",
        run=run,
    )
    assert identity["engine_sha256"]
    assert identity["batch_size"] == "14"
    assert identity["tracker_library_sha256"] == sha256(tracker_library)
    assert (tmp_path / "identity.json").is_file()


def test_a_second_run_against_a_populated_cache_verifies_instead_of_building(
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    _write_onnx(onnx_path)
    parser = tmp_path / "parser.so"
    infer = tmp_path / "infer.yml"
    tracker = tmp_path / "tracker.yml"
    tracker_library = tmp_path / "libnvds_nvmultiobjecttracker.so"
    for path in (parser, tracker, tracker_library):
        path.write_bytes(path.name.encode())
    engine = tmp_path / "model.engine"
    infer.write_text(f"model-engine-file={engine}\nbatch-size=14\n", encoding="utf-8")
    identity_path = tmp_path / "engine-identity.json"
    builds: list[list[str]] = []

    def run(command: list[str], **_: object) -> CompletedProcess[str]:
        assert not engine.exists(), "nvinfer must rebuild rather than deserialize a stale engine"
        builds.append(command)
        engine.write_bytes(b"engine")
        return CompletedProcess(command, 0, "", "")

    def build() -> dict[str, str]:
        return build_engine(
            onnx=onnx_path,
            engine=engine,
            identity_path=identity_path,
            parser_lib=parser,
            infer_config=infer,
            tracker_library=tracker_library,
            tracker_config=tracker,
            image_digest="image",
            batch_size=14,
            run=run,
        )

    assert build() == build()
    assert len(builds) == 1, "the populated cache must not rebuild the engine"


def test_nvinfer_is_invoked_for_the_declared_batch_and_cached_by_batch(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    _write_onnx(onnx_path, input_name="images")
    parser = tmp_path / "parser.so"
    infer = tmp_path / "infer.yml"
    tracker = tmp_path / "tracker.yml"
    tracker_library = tmp_path / "libnvds_nvmultiobjecttracker.so"
    for path in (parser, tracker, tracker_library):
        path.write_bytes(path.name.encode())
    infer.write_text("model-engine-file=stale.engine\nbatch-size=1\n", encoding="utf-8")
    engine = tmp_path / "model.engine"
    identity_path = tmp_path / "engine-identity.json"
    served_infer = tmp_path / "cache" / "nvinfer.txt"
    builds: list[list[str]] = []

    def run(command: list[str], **_: object) -> CompletedProcess[str]:
        assert not engine.exists(), "nvinfer must rebuild rather than deserialize a stale engine"
        builds.append(command)
        engine.write_bytes(b"engine")
        return CompletedProcess(command, 0, "", "")

    kwargs = {
        "onnx": onnx_path,
        "engine": engine,
        "identity_path": identity_path,
        "parser_lib": parser,
        "infer_config": infer,
        "tracker_library": tracker_library,
        "tracker_config": tracker,
        "image_digest": "image",
        "served_infer_config": served_infer,
        "run": run,
    }
    build_engine(**kwargs, batch_size=14)
    assert builds[0][0] == "gst-launch-1.0"
    assert "nvinfer" in builds[0]
    assert f"config-file-path={served_infer}" in builds[0]
    assert "batch-size=14" in builds[0]
    assert builds[0].count("videotestsrc") == 14
    assert served_infer.read_text(encoding="utf-8") == (
        f"model-engine-file={engine}\nbatch-size=14\n"
    )

    build_engine(**kwargs, batch_size=14)
    assert len(builds) == 1
    build_engine(**kwargs, batch_size=13)
    assert len(builds) == 2


def _build_against(
    tmp_path: Path,
    onnx_path: Path,
    *,
    batch_size: int,
    run,
    engine: Path | None = None,
) -> dict[str, str]:
    parser = tmp_path / "parser.so"
    infer = tmp_path / "infer.yml"
    tracker = tmp_path / "tracker.yml"
    tracker_library = tmp_path / "libnvds_nvmultiobjecttracker.so"
    for path in (parser, tracker, tracker_library):
        path.write_bytes(path.name.encode())
    engine = engine if engine is not None else tmp_path / "model.engine"
    infer.write_text(
        f"model-engine-file={engine}\nbatch-size={batch_size}\n",
        encoding="utf-8",
    )
    return build_engine(
        onnx=onnx_path,
        engine=engine,
        identity_path=tmp_path / "identity.json",
        parser_lib=parser,
        infer_config=infer,
        tracker_config=tracker,
        tracker_library=tracker_library,
        image_digest="sha256:image",
        batch_size=batch_size,
        run=run,
    )


def test_builder_failure_names_the_nvinfer_pipeline(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    _write_onnx(onnx_path)

    def run(command, **_kwargs):
        return CompletedProcess(command, 9, "", "nvinfer failed")

    with pytest.raises(EngineBuildError, match="nvinfer build pipeline failed: nvinfer failed"):
        _build_against(tmp_path, onnx_path, batch_size=1, run=run)


def test_builder_must_create_the_engine_before_identity_is_written(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    _write_onnx(onnx_path)

    def run(command, **_kwargs):
        return CompletedProcess(command, 0, "", "")

    with pytest.raises(EngineBuildError, match="without creating an engine at either"):
        _build_against(tmp_path, onnx_path, batch_size=1, run=run)
    assert not (tmp_path / "identity.json").exists()


def test_the_engine_nvinfer_writes_beside_the_onnx_is_adopted(tmp_path: Path) -> None:
    """nvinfer ignores model-engine-file when it builds rather than deserialises.

    It writes `<onnx>_b<N>_gpu0_fp16.engine` next to the model instead, and that
    file is the one that actually serves, so the build must adopt it rather than
    report success with nothing at the configured path.
    """
    onnx_path = tmp_path / "model.onnx"
    _write_onnx(onnx_path)
    engine = tmp_path / "cache" / "model.engine"
    produced = onnx_path.with_name(f"{onnx_path.name}_b1_gpu0_fp16.engine")

    def run(command, **_kwargs):
        produced.write_bytes(b"nvinfer-built-engine")
        return CompletedProcess(command, 0, "", "")

    identity = _build_against(tmp_path, onnx_path, batch_size=1, run=run, engine=engine)

    assert engine.read_bytes() == b"nvinfer-built-engine"
    assert not produced.exists(), "the adopted engine must not be left beside the model"
    assert identity["engine_sha256"] == sha256(engine)
