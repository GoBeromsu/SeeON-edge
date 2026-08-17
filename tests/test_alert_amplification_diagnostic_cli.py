from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path("scripts/ops/alert-amplification-diagnostic.py")
_INSECURE_HTTP_ENV = "API_BACKEND_" "ALLOW_INSECURE_HTTP"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("alert_amplification_diagnostic", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(path: Path, **overrides: object) -> Path:
    body: dict[str, object] = {
        "schemaVersion": 1,
        "run_id": "run-1",
        "source_revision": "4" * 40,
        "evidence_sha256": "0" * 64,
        "diagnostic_compose_sha256": "0" * 64,
        "package_compose_sha256": "0" * 64,
        "image_inspect_sha256": ["0" * 64],
        "container_inspect_sha256": "0" * 64,
        "diagnostic_services": ["diagnostic-hub", "ml-api", "ml-worker"],
        "package_services": ["ml-api"],
        "image_ids": ["sha256:" + "1" * 64],
        "running_container_ids": {
            "diagnostic-hub": "container-hub",
            "ml-api": "container-api",
            "ml-worker": "container-worker",
        },
        "service_image_ids": {
            "diagnostic-hub": "sha256:" + "1" * 64,
            "ml-api": "sha256:" + "1" * 64,
            "ml-worker": "sha256:" + "1" * 64,
        },
    }
    body.update(overrides)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["receipt_id"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(body))
    return path


def test_fixture_preflight_is_explicitly_synthetic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "fixture-preflight"])

    assert module.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["evidence_scope"] == "synthetic_fixture_preflight_only"
    assert result["snapshot_bytes_retained"] == 0


def test_offline_classifier_never_claims_model_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    artifact = tmp_path / "rows.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "transition_id": "transition-1",
                    "edge_event_id": "edge-1",
                    "attempt_ordinals": [1, 2],
                    "backend_event_ids": ["backend-1"],
                    "incident_ids": ["incident-1"],
                    "terminal_state": "ACKED",
                    "clock_order_valid": True,
                    "order_evidence_valid": True,
                }
            ]
        )
    )
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "classify", str(artifact)])

    assert module.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "transport_retry"
    assert result["model_policy_cause"] == "판정 불가"
    assert result["evidence_scope"] == "offline_fixture_only"


def test_offline_classifier_fails_closed_without_order_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    artifact = tmp_path / "rows.json"
    row = {
        "transition_id": "transition-1",
        "edge_event_id": "edge-1",
        "attempt_ordinals": [1, 2],
        "backend_event_ids": ["backend-1"],
        "incident_ids": ["incident-1"],
        "terminal_state": "ACKED",
        "clock_order_valid": True,
    }
    artifact.write_text(json.dumps([row]))
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "classify", str(artifact)])

    assert module.main() == 0
    without_flag = json.loads(capsys.readouterr().out)

    artifact.write_text(json.dumps([{**row, "order_evidence_valid": True}]))
    assert module.main() == 0
    with_flag = json.loads(capsys.readouterr().out)

    assert without_flag["outcome"] == "판정 불가"
    assert with_flag["outcome"] == "transport_retry"


def test_semantic_http_reads_complete_compose_service_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    compose = tmp_path / "compose.json"
    package = tmp_path / "package.json"
    image = tmp_path / "image.json"
    container = tmp_path / "container.json"
    receipt = tmp_path / "run-receipt.json"
    compose.write_text(
        json.dumps(
            {
                "services": {
                    "ml-api": {
                        "environment": {_INSECURE_HTTP_ENV: "1"}
                    },
                    "ml-worker": {"environment": {}},
                    "diagnostic-hub": {"environment": []},
                }
            }
        )
    )
    package.write_text(json.dumps({"services": {"ml-api": {"environment": {}}}}))
    image_id = "sha256:" + "1" * 64
    image.write_text(
        json.dumps([{"Id": image_id, "Config": {"Env": ["HOST=127.0.0.1"]}}])
    )
    container.write_text(
        json.dumps(
            [
                {
                    "Id": f"container-{suffix}",
                    "Image": image_id,
                    "Config": {"Labels": {"com.docker.compose.service": service}},
                }
                for service, suffix in (
                    ("diagnostic-hub", "hub"),
                    ("ml-api", "api"),
                    ("ml-worker", "worker"),
                )
            ]
        )
    )
    _write_receipt(
        receipt,
        diagnostic_compose_sha256=_sha256(compose),
        package_compose_sha256=_sha256(package),
        image_inspect_sha256=[_sha256(image)],
        container_inspect_sha256=_sha256(container),
    )
    monkeypatch.delenv(_INSECURE_HTTP_ENV, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "semantic-http",
            str(compose),
            "--package-compose-json",
            str(package),
            "--image-inspect-json",
            str(image),
            "--container-inspect-json",
            str(container),
            "--run-receipt",
            str(receipt),
            "--repo-root",
            str(Path.cwd()),
        ],
    )

    exit_code = module.main()
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0, result
    assert result["effective_opt_in_service"] == "ml-api"
    assert result["diagnostic_service_count"] == 3
    assert result["package_service_count"] == 1
    assert result["companion_rejection"] is True
    assert result["runtime_identity_bound"] is True

    incomplete = _write_receipt(
        tmp_path / "incomplete-receipt.json",
        diagnostic_compose_sha256=_sha256(compose),
        package_compose_sha256=_sha256(package),
        image_inspect_sha256=[_sha256(image)],
        container_inspect_sha256=_sha256(container),
        running_container_ids={"ml-api": "container-api"},
        service_image_ids={"ml-api": image_id},
    )
    with pytest.raises(ValueError, match="do not cover the diagnostic topology"):
        module._semantic_http(
            compose,
            package_path=package,
            image_paths=[image],
            container_path=container,
            run_receipt_path=incomplete,
            repo_root=Path.cwd(),
        )


class _MeasuredResponse:
    status_code = 200

    def json(self) -> dict[str, object]:
        return {
            "incidents": [
                {
                    "incident_id": "incident-1",
                    "edge_event_id": "edge-1",
                    "detected_at": "2026-08-16T00:00:00Z",
                    "lifecycle_state": "OPEN",
                    "event_delivery_state": "ACKED",
                    "review": None,
                }
            ],
            "pagination": {"limit": 100, "next_cursor": None, "has_more": False},
        }


class _MeasuredClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> _MeasuredClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, *_args: object, **_kwargs: object) -> _MeasuredResponse:
        return _MeasuredResponse()


def test_measured_run_keeps_identity_findings_without_projection_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    evidence = tmp_path / "measured.json"
    evidence.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "run_id": "run-1",
                "source_revision": "4" * 40,
                "transitions": {"edge-1": "transition-1"},
                "attempts": {"edge-1": [1]},
                "backend_event_ids": {"edge-1": ["backend-1"]},
                "attempt_times": {"edge-1": [1.0]},
                "receipt_times": {"edge-1": [2.0]},
                "clock_samples": [
                    {"offset_ms": 0, "uncertainty_ms": 1, "rtt_ms": 2}
                    for _ in range(5)
                ],
            }
        )
    )
    receipt = _write_receipt(
        tmp_path / "measured-receipt.json",
        evidence_sha256=_sha256(evidence),
    )
    monkeypatch.setattr(module.httpx, "Client", _MeasuredClient)
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"dashboard-secret")
        os.close(write_fd)
        write_fd = -1
        result = module._measured_run(
            evidence,
            "http://127.0.0.1:8000",
            read_fd,
            run_receipt_path=receipt,
        )
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)

    assert result["outcome"] == "single_api_delivery"
    assert result["model_policy_cause"] == "판정 불가"
    assert result["clock_gate_passed"] is True
    assert result["projection_timestamp_complete"] is False
    assert result["order_dependent_claims_eligible"] is False
    assert result["temporal_order_valid"] is False
    assert result["promotion_eligible"] is False


class _ProbeResult:
    def __init__(self, returncode: int, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_host_probe_emits_only_privacy_safe_booleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(module, "probe_procfs_visibility", lambda: False)
    monkeypatch.setattr(
        module,
        "descriptor_scan_scope",
        lambda: module.DescriptorScanScope(0, 0, 0),
    )

    def fake_run(command: list[str], **_kwargs: object) -> _ProbeResult:
        if command[:2] == ["docker", "run"]:
            return _ProbeResult(1)
        if command[0] == "swapon":
            return _ProbeResult(0, "")
        return _ProbeResult(0, "local/fall-ml-api:test\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    clip_path = str(tmp_path / "missing.mp4")
    clip_digest = "4" * 64
    read_fd, write_fd = os.pipe()
    try:
        os.write(
            write_fd,
            json.dumps(
                {"clip_path": clip_path, "expected_clip_sha256": clip_digest}
            ).encode(),
        )
        os.close(write_fd)
        write_fd = -1
        argv = [str(_SCRIPT), "host-probe", "--probe-envelope-fd", str(read_fd)]
        monkeypatch.setattr(sys, "argv", argv)
        assert clip_path not in repr(argv)
        assert clip_digest not in repr(argv)
        assert module.main() == 0
        result = json.loads(capsys.readouterr().out)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)

    assert result["gateway_candidate_present"] is False
    assert result["gateway_image_present"] is False
    assert result["pinned_clip_present"] is False
    assert result["container_isolation_available"] is False
    assert result["host_swap_enabled"] is False
    assert result["procfs_visibility_usable"] is False
    assert result["descriptor_scan_skipped_protected"] == 0
    assert result["contains_paths"] is False
    assert result["contains_process_inventory"] is False
    assert result["contains_secrets"] is False
    assert result["contains_media_hashes"] is False
    assert "missing.mp4" not in repr(result)


def test_host_probe_treats_missing_tools_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    def missing(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(module.subprocess, "run", missing)
    result = module._host_probe(tmp_path / "missing.mp4", "4" * 64)

    assert result["container_isolation_available"] is False
    assert result["gateway_image_present"] is False


def test_secret_descriptor_is_read_without_using_standard_streams() -> None:
    module = _load_script()
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"dashboard-secret")
        os.close(write_fd)
        write_fd = -1
        assert module._read_secret_fd(read_fd) == "dashboard-secret"
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_secret_descriptor_rejects_standard_streams() -> None:
    module = _load_script()
    with pytest.raises(ValueError):
        module._read_secret_fd(0)


def test_incident_projection_rejects_public_origin_before_reading_token() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="isolated ml-api"):
        module._incident_projection("https://production.example", 0)
