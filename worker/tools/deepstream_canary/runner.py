"""Real Docker execution with a host watchdog and canary-only teardown."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from worker.tools.deepstream_canary.collector import CollectionRequest, collect_recorded_telemetry
from worker.tools.deepstream_canary.execution_artifacts import (
    ExecutionArtifactSources,
    ReceiptEmissionRequest,
    emit_receipts,
    generate_corpus,
    gpu_sample,
    native_windows,
)
from worker.tools.deepstream_canary.models import CanaryMode
from worker.tools.deepstream_canary.safety import (
    CanarySafetyError,
    LiveSnapshot,
    SafetyLimits,
    compare_live_snapshot,
    persist_first_fault,
)
from worker.tools.deepstream_canary.telemetry import RuntimeGpuSample


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    compose_path: Path
    evidence_dir: Path
    baseline: LiveSnapshot
    rung_durations: tuple[tuple[str, int], ...]
    publisher_count: int
    relay_token: str
    safety_limits: SafetyLimits
    mode: CanaryMode
    rungs: tuple[str, ...]
    artifacts: ExecutionArtifactSources


def _compose(request: ExecutionRequest, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            "docker",
            "compose",
            "-p",
            "seeon-ds-canary",
            "-f",
            str(request.compose_path),
            *arguments,
        ),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CANARY_RELAY_TOKEN": request.relay_token},
    )


def _require_success(
    completed: subprocess.CompletedProcess[str],
    code: str,
    detail: str,
) -> None:
    if completed.returncode != 0:
        raise CanarySafetyError(code, detail)


def _require_sample(sample: RuntimeGpuSample | None, rung: str) -> RuntimeGpuSample:
    if sample is None:
        raise CanarySafetyError("native_child_not_ready", rung)
    return sample


def _require_corpus(corpus: Path | None) -> Path:
    if corpus is None:
        raise CanarySafetyError("corpus_missing", "execution")
    return corpus


def _wait_for_source_ready(path: Path, camera_id: str) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        threading.Event().wait(0.25)
    raise CanarySafetyError("native_source_ready_timeout", camera_id)


def _preview_url(request: ExecutionRequest, camera_id: str) -> str:
    inspect = subprocess.run(
        (
            "docker",
            "inspect",
            "seeon-ds-canary-ml-worker-1",
            "--format",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    _require_success(inspect, "preview_container_ip_missing", camera_id)
    address = inspect.stdout.strip()
    if not address:
        raise CanarySafetyError("preview_container_ip_missing", camera_id)
    return f"http://{address}:8090/stream/{camera_id}"


def _probe_preview_paths(request: ExecutionRequest, camera_id: str) -> str:
    internal = subprocess.run(
        (
            "docker",
            "exec",
            "seeon-ds-canary-relay-stub-1",
            "python",
            "-c",
            (
                "import urllib.request;"
                "q=urllib.request.Request("
                f"'http://ml-worker:8090/stream/{camera_id}',"
                f"headers={{'X-Edge-Relay-Token':'{request.relay_token}'}});"
                "r=urllib.request.urlopen(q,timeout=5);"
                "print(r.status,len(r.read(1024)))"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    host = subprocess.run(
        (
            "curl",
            "--silent",
            "--show-error",
            "--max-time",
            "3",
            "--header",
            f"X-Edge-Relay-Token: {request.relay_token}",
            "--output",
            "/dev/null",
            f"http://127.0.0.1:18090/stream/{camera_id}",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    (request.evidence_dir / "raw" / "preview-probes.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_id": camera_id,
                "host_exit": host.returncode,
                "host_stderr": host.stderr[-200:],
                "internal_exit": internal.returncode,
                "internal_stdout": internal.stdout[-200:],
                "internal_stderr": internal.stderr[-200:],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return _preview_url(request, camera_id)


def _capture_preview_evidence(
    request: ExecutionRequest,
    corpus: Path,
    camera_id: str,
    viewer_url: str,
) -> None:
    raw = request.evidence_dir / "raw"
    event = raw / "event-clip.mp4"
    derivative = raw / "event-derivative.mp4"
    shutil.copy2(corpus, event)
    shutil.copy2(corpus, derivative)
    event_digest = hashlib.sha256(event.read_bytes()).hexdigest()
    derivative_digest = hashlib.sha256(derivative.read_bytes()).hexdigest()
    (raw / "event-evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_sha256": event_digest,
                "derivative_sha256": derivative_digest,
                "single_render": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        browser = subprocess.run(
            (
                "node",
                "scripts/qa/deepstream_canary_browser.mjs",
                camera_id,
                str(raw),
                str(derivative),
                viewer_url,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
            env={**os.environ, "CANARY_RELAY_TOKEN": request.relay_token},
        )
    except subprocess.TimeoutExpired as error:
        raise CanarySafetyError("browser_evidence_timeout", camera_id) from error
    (raw / "browser-command.log").write_text(browser.stdout + browser.stderr, encoding="utf-8")
    _require_success(browser, "browser_evidence_failed", camera_id)


def execute_canary(request: ExecutionRequest) -> int:
    """Execute only the isolated project; any first fault tears down only it."""
    fault_path = request.evidence_dir / "first-fault.json"
    stop = threading.Event()
    stack_started = threading.Event()
    capacity_armed = threading.Event()
    active_rung: list[str] = ["prepare"]
    gpu_samples: dict[str, list[RuntimeGpuSample]] = {}
    sample_lock = threading.Lock()
    watchdog_fault: list[CanarySafetyError] = []
    gpu_loss_observed_at: dict[str, float] = {}
    native_path = request.evidence_dir / "raw" / "native-telemetry.jsonl"
    native_path.touch(mode=0o666, exist_ok=False)
    native_path.chmod(0o666)
    corpus: Path | None = None

    def flush_partial(error: CanarySafetyError) -> None:
        for rung, duration in request.rung_durations:
            telemetry_path = request.evidence_dir / "raw" / f"telemetry-{rung}.json"
            with sample_lock:
                phase_gpu = tuple(gpu_samples.get(rung, ()))
            if telemetry_path.is_file() or not phase_gpu:
                continue
            camera_ids = (
                tuple(f"loop-{index:02d}" for index in range(1, request.publisher_count + 1))
                if rung != "zero"
                else ()
            )
            _ = collect_recorded_telemetry(
                CollectionRequest(
                    evidence_dir=request.evidence_dir,
                    rung=rung,
                    mode=request.mode,
                    clean_steady_seconds=duration,
                    camera_ids=camera_ids,
                    gpu_samples=phase_gpu,
                    native_windows=native_windows(native_path),
                    allow_partial=True,
                    fault_windows=(error.code,),
                )
            )
        available = tuple(
            rung
            for rung, _duration in request.rung_durations
            if (request.evidence_dir / "raw" / f"telemetry-{rung}.json").is_file()
        )
        if corpus is not None and available:
            _ = emit_receipts(
                ReceiptEmissionRequest(
                    evidence_dir=request.evidence_dir,
                    rungs=available,
                    artifacts=request.artifacts,
                ),
                corpus,
            )
        (request.evidence_dir / "raw" / "partial-execution.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "aborted",
                    "first_fault": error.code,
                    "receipts": list(available),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    def abort(error: CanarySafetyError) -> None:
        watchdog_fault.append(error)
        stop.set()
        persist_first_fault(fault_path, error)

    def watchdog() -> None:
        while not stop.wait(10.0):
            if stack_started.is_set():
                exited = _compose(request, "ps", "--status", "exited", "--services")
                failed_services = tuple(
                    service
                    for service in exited.stdout.splitlines()
                    if service and service != "engine-builder"
                )
                if exited.returncode != 0 or failed_services:
                    abort(CanarySafetyError("canary_service_exited", ",".join(failed_services)))
                    return
            try:
                snapshot = compare_live_snapshot(
                    request.baseline,
                    replace(
                        request.safety_limits,
                        enforce_gpu_capacity=capacity_armed.is_set(),
                    ),
                    gpu_loss_observed_at,
                )
                if capacity_armed.is_set() and stack_started.is_set():
                    sample = gpu_sample(snapshot)
                    if sample is not None:
                        with sample_lock:
                            gpu_samples.setdefault(active_rung[0], []).append(sample)
            except CanarySafetyError as error:
                abort(error)
                return

    try:
        corpus = generate_corpus(request.evidence_dir)
        monitor = threading.Thread(target=watchdog, name="canary-live-watchdog", daemon=True)
        monitor.start()
        prepare = _compose(request, "run", "--rm", "engine-builder")
        _ = (request.evidence_dir / "engine-prepare.log").write_text(
            prepare.stdout + prepare.stderr, encoding="utf-8"
        )
        if watchdog_fault:
            raise watchdog_fault[0]
        _require_success(prepare, "engine_prepare_failed", prepare.stderr[-500:])
        capacity_armed.set()
        phase_logs: list[str] = []
        for rung, duration in request.rung_durations:
            stack_started.clear()
            active_rung[0] = rung
            selected = "zero" if rung == "zero" else "workload"
            (request.evidence_dir / "raw" / "active-config").write_text(
                f"{selected}\n", encoding="utf-8"
            )
            camera_ids = (
                tuple(f"loop-{index:02d}" for index in range(1, request.publisher_count + 1))
                if selected == "workload"
                else ()
            )
            (request.evidence_dir / "raw" / f"camera-registry-{rung}.json").write_text(
                json.dumps(
                    {"schema_version": 1, "rung": rung, "camera_ids": camera_ids},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            if selected == "workload":
                publishers = tuple(
                    f"publisher-{index:02d}" for index in range(1, request.publisher_count + 1)
                )
                publisher_up = _compose(request, "up", "-d", *publishers)
                phase_logs.append(publisher_up.stdout + publisher_up.stderr)
                _require_success(publisher_up, "publisher_start_failed", rung)
                up = _compose(request, "up", "-d", "--force-recreate", "ml-worker")
            else:
                up = _compose(request, "up", "-d", "relay-stub", "mediamtx", "ml-worker")
            phase_logs.append(up.stdout + up.stderr)
            stack_started.set()
            if watchdog_fault:
                raise watchdog_fault[0]
            _require_success(up, "compose_up_failed", up.stderr[-500:])
            _ = stop.wait(duration)
            if watchdog_fault:
                raise watchdog_fault[0]
            final_snapshot = compare_live_snapshot(
                request.baseline,
                request.safety_limits,
                gpu_loss_observed_at,
            )
            if camera_ids:
                viewer_url = _probe_preview_paths(request, camera_ids[0])
                _capture_preview_evidence(
                    request, _require_corpus(corpus), camera_ids[0], viewer_url
                )
                _wait_for_source_ready(native_path, camera_ids[0])
            final_sample = _require_sample(gpu_sample(final_snapshot), rung)
            with sample_lock:
                gpu_samples.setdefault(rung, []).append(final_sample)
                phase_gpu = tuple(gpu_samples[rung])
            _ = collect_recorded_telemetry(
                CollectionRequest(
                    evidence_dir=request.evidence_dir,
                    rung=rung,
                    mode=request.mode,
                    clean_steady_seconds=duration,
                    camera_ids=camera_ids,
                    gpu_samples=phase_gpu,
                    native_windows=(
                        ()
                        if rung == "zero"
                        else native_windows(request.evidence_dir / "raw" / "native-telemetry.jsonl")
                    ),
                )
            )
            stack_started.clear()
            stopped = _compose(request, "stop", "ml-worker")
            phase_logs.append(stopped.stdout + stopped.stderr)
        stop.set()
        monitor.join(timeout=15.0)
        _ = (request.evidence_dir / "compose-up.log").write_text(
            "".join(phase_logs), encoding="utf-8"
        )
        _ = emit_receipts(
            ReceiptEmissionRequest(
                evidence_dir=request.evidence_dir,
                rungs=request.rungs,
                artifacts=request.artifacts,
            ),
            _require_corpus(corpus),
        )
        logs = _compose(request, "logs", "--no-color")
        (request.evidence_dir / "compose.log").write_text(
            logs.stdout + logs.stderr, encoding="utf-8"
        )
        execution = {
            "schema_version": 1,
            "status": "collected",
            "note": "raw rung receipts emitted for independent verification",
        }
        (request.evidence_dir / "execution.json").write_text(
            json.dumps(execution, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except CanarySafetyError as error:
        try:
            flush_partial(error)
        except (OSError, ValueError) as flush_error:
            (request.evidence_dir / "raw" / "receipt-flush-error.json").write_text(
                json.dumps(
                    {"schema_version": 1, "detail": str(flush_error)},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
        logs = _compose(request, "logs", "--no-color")
        _ = (request.evidence_dir / "compose-failure.log").write_text(
            logs.stdout + logs.stderr, encoding="utf-8"
        )
        if not watchdog_fault:
            persist_first_fault(fault_path, error)
        return 1
    else:
        return 0
    finally:
        stop.set()
        down = _compose(request, "down", "--remove-orphans")
        (request.evidence_dir / "compose-down.log").write_text(
            down.stdout + down.stderr, encoding="utf-8"
        )
