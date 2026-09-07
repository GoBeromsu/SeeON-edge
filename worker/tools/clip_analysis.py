"""Isolated command-line entrypoint for evidence clip re-analysis."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from worker.runtime.clip_analysis_subprocess import bootstrap_child


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-parent", required=True, type=int)
    parser.add_argument("--cpu", required=True, type=int)
    parser.add_argument("--control-fd", required=True, type=int)
    args = parser.parse_args(argv)
    bootstrap_child(
        expected_parent=args.expected_parent, cpu_index=args.cpu, control_fd=args.control_fd
    )
    # Imports below this line may initialize native decoder/ML thread pools.
    from shared.events.clip_analysis_wire import encode_clip_analysis
    from worker.adapters.model.clip_reanalysis import (
        ClipAnalysisFailed,
        ClipAnalysisRejected,
        analyze_clip,
    )

    try:
        request = _load_request(args.request)
        decoder_identity = _decoder_identity(request.clip_path)
        result = analyze_clip(request, decoder_identity=decoder_identity)
        _atomic_write(args.out, encode_clip_analysis(result))
    except ClipAnalysisRejected as exc:
        _error("ClipAnalysisRejected", str(exc))
        return 2
    except ClipAnalysisFailed as exc:
        _error("ClipAnalysisFailed", str(exc))
        return 3
    except Exception:  # noqa: BLE001 - tool boundary maps every failure to exit 3
        _error("ClipAnalysisFailed", "tool_failed")
        return 3
    return 0


def _load_request(path: Path):
    """Decode the shared supervisor/tool request schema without native imports."""
    from worker.adapters.model.clip_reanalysis import ClipAnalysisProfile, ClipAnalysisRequest

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "clip_id",
        "clip_path",
        "clip_sha256",
        "pose_model_path",
        "bed_model_path",
        "analysis_profile",
    }:
        raise ValueError("request_shape")
    profile = payload["analysis_profile"]
    if not isinstance(profile, dict) or set(profile) != {
        "person_threshold",
        "bed_confidence",
        "max_frames",
        "max_duration_s",
        "max_pixels",
        "max_input_bytes",
    }:
        raise ValueError("profile_shape")
    return ClipAnalysisRequest(
        clip_id=_string(payload["clip_id"], "clip_id"),
        clip_path=Path(_string(payload["clip_path"], "clip_path")),
        clip_sha256=_string(payload["clip_sha256"], "clip_sha256"),
        pose_model_path=Path(_string(payload["pose_model_path"], "pose_model_path")),
        bed_model_path=Path(_string(payload["bed_model_path"], "bed_model_path")),
        analysis_profile=ClipAnalysisProfile(**profile),
    )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(name)
    return value


def _decoder_identity(path: Path) -> str:
    import av

    try:
        with av.open(str(path)) as container:
            codec_name = container.streams.video[0].codec_context.name
    except Exception as exc:
        from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected

        raise ClipAnalysisRejected("container_open") from exc
    if not codec_name:
        from worker.adapters.model.clip_reanalysis import ClipAnalysisFailed

        raise ClipAnalysisFailed("codec_name")
    return f"pyav-{av.__version__}/{codec_name}"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary.exists():
            temporary.unlink()
        raise


def _error(error: str, reason: str) -> None:
    print(json.dumps({"error": error, "reason": reason}, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
