from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from worker.tools.edge_engine_build import build_engine, sha256


def test_build_writes_identity(tmp_path: Path) -> None:
    onnx = tmp_path / "model.onnx"
    parser = tmp_path / "parser.so"
    infer = tmp_path / "infer.yml"
    tracker = tmp_path / "tracker.yml"
    tracker_library = tmp_path / "libnvds_nvmultiobjecttracker.so"
    for path in (onnx, parser, infer, tracker, tracker_library):
        path.write_bytes(path.name.encode())
    engine = tmp_path / "model.engine"

    def run(command: list[str], **_: object) -> CompletedProcess[str]:
        engine.write_bytes(b"engine")
        return CompletedProcess(command, 0, "", "")

    identity = build_engine(
        onnx=onnx,
        engine=engine,
        identity_path=tmp_path / "identity.json",
        parser_lib=parser,
        infer_config=infer,
        tracker_library=tracker_library,
        tracker_config=tracker,
        image_digest="image",
        run=run,
    )
    assert identity["engine_sha256"]
    assert identity["tracker_library_sha256"] == sha256(tracker_library)
    assert (tmp_path / "identity.json").is_file()
