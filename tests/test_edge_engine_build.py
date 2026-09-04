from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import onnx
from onnx import TensorProto, helper

import pytest

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
    for path in (parser, infer, tracker, tracker_library):
        path.write_bytes(path.name.encode())
    engine = tmp_path / "model.engine"
    identity_path = tmp_path / "engine-identity.json"
    builds: list[list[str]] = []

    def run(command: list[str], **_: object) -> CompletedProcess[str]:
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


def test_batch_shapes_are_derived_from_onnx_and_cached_by_batch(tmp_path: Path) -> None:
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
        builds.append(command)
        engine.write_bytes(b"engine")
        return CompletedProcess(command, 0, "", "")

    kwargs = dict(
        onnx=onnx_path,
        engine=engine,
        identity_path=identity_path,
        parser_lib=parser,
        infer_config=infer,
        tracker_library=tracker_library,
        tracker_config=tracker,
        image_digest="image",
        served_infer_config=served_infer,
        run=run,
    )
    build_engine(**kwargs, batch_size=14)
    assert "--minShapes=images:1x3x640x640" in builds[0]
    assert "--optShapes=images:14x3x640x640" in builds[0]
    assert "--maxShapes=images:14x3x640x640" in builds[0]
    assert served_infer.read_text(encoding="utf-8") == (
        f"model-engine-file={engine}\nbatch-size=14\n"
    )

    build_engine(**kwargs, batch_size=14)
    assert len(builds) == 1
    build_engine(**kwargs, batch_size=13)
    assert len(builds) == 2


def _write_static_batch_onnx(path: Path, *, batch: int) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["frames"], ["output0"])],
        "pose",
        [helper.make_tensor_value_info("frames", TensorProto.FLOAT, [batch, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [batch, 3, 640, 640])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


def _build_against(tmp_path: Path, onnx_path: Path, *, batch_size: int, run) -> dict[str, str]:
    parser = tmp_path / "parser.so"
    infer = tmp_path / "infer.yml"
    tracker = tmp_path / "tracker.yml"
    tracker_library = tmp_path / "libnvds_nvmultiobjecttracker.so"
    for path in (parser, tracker, tracker_library):
        path.write_bytes(path.name.encode())
    infer.write_text("model-engine-file=stale.engine\nbatch-size=1\n", encoding="utf-8")
    return build_engine(
        onnx=onnx_path,
        engine=tmp_path / "model.engine",
        identity_path=tmp_path / "identity.json",
        parser_lib=parser,
        infer_config=infer,
        tracker_config=tracker,
        tracker_library=tracker_library,
        image_digest="sha256:image",
        batch_size=batch_size,
        run=run,
    )


def test_a_static_batch_model_is_built_without_explicit_shapes(tmp_path: Path) -> None:
    # TensorRT rejects --minShapes/--optShapes for a model whose batch the graph
    # already fixes, so passing them made every build of the real pose model fail.
    onnx_path = tmp_path / "static.onnx"
    _write_static_batch_onnx(onnx_path, batch=1)
    seen: list[list[str]] = []

    def run(command, **_kwargs):
        seen.append(list(command))
        (tmp_path / "model.engine").write_bytes(b"engine")
        return CompletedProcess(command, 0, "", "")

    _build_against(tmp_path, onnx_path, batch_size=1, run=run)
    assert seen, "trtexec was never invoked"
    assert not [argument for argument in seen[0] if "Shapes=" in argument]


def test_a_static_batch_smaller_than_the_roster_is_refused(tmp_path: Path) -> None:
    # Serving more sources than the engine's fixed batch is exactly what made
    # nvinfer rebuild at runtime, so the build refuses rather than producing it.
    onnx_path = tmp_path / "static.onnx"
    _write_static_batch_onnx(onnx_path, batch=1)

    def run(command, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("trtexec must not run for a refused batch")

    with pytest.raises(EngineBuildError) as failure:
        _build_against(tmp_path, onnx_path, batch_size=13, run=run)
    assert "1" in str(failure.value) and "13" in str(failure.value)
