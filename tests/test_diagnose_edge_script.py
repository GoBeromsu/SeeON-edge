"""Subprocess coverage for the read-only edge diagnostic script."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "edge-preflight" / "diagnose-edge.sh"


def _write_stub(path: Path, name: str, body: str) -> None:
    command = path / name
    command.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    command.chmod(command.stat().st_mode | stat.S_IXUSR)


def _run_diagnose(
    tmp_path: Path,
    *,
    kernel_log: str,
    status: object | str | None = None,
    command_stubs: dict[str, str] | None = None,
    arguments: tuple[str, ...] = (),
    timeout_sec: str = "20",
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fixture = tmp_path / "status.json"
    if status is None:
        status = {
            "cameras": {
                "camera-1": {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "status": "online",
                }
            },
            "runtime": {
                "facilities": {
                    "facility-1": {
                        "facility_id": "facility-1",
                        "cameras": [
                            {
                                "camera_id": "camera-1",
                                "measured_fps": 15.0,
                                "decode": {
                                    "requested": "nvdec",
                                    "selected": "nvdec",
                                    "fallback_count": 0,
                                },
                            }
                        ],
                        "stale": False,
                        "latency": {"max_sec": 0.2},
                        "gpu": {
                            "nvml_available": True,
                            "cuda_context_ok": True,
                        },
                        "worker": {"alive": True},
                    }
                }
            },
        }
    fixture.write_text(
        status if isinstance(status, str) else json.dumps(status),
        encoding="utf-8",
    )
    kernel_fixture = tmp_path / "kernel.log"
    kernel_fixture.write_text(kernel_log, encoding="utf-8")
    stubs = {
        "nvidia-smi": (
            'if [ "$1" = "--query-compute-apps=pid,process_name" ]; '
            'then echo "123, seeon-deepstream-child"; else echo "GPU 0"; fi'
        ),
        "docker": "exit 1",
        "sh": "exit 0",
        "curl": 'cat "$EDGE_STATUS_FIXTURE"',
        "journalctl": 'cat "$EDGE_KERNEL_FIXTURE"',
    }
    stubs.update(command_stubs or {})
    for name, body in stubs.items():
        _write_stub(bin_dir, name, body)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "EDGE_STATUS_FIXTURE": str(fixture),
        "EDGE_KERNEL_FIXTURE": str(kernel_fixture),
        "EDGE_PREFLIGHT_TIMEOUT_SEC": timeout_sec,
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments], env=env, text=True, capture_output=True, check=False
    )


def test_diagnose_parses_heartbeat_and_facility_runtime_schema(tmp_path: Path) -> None:
    result = _run_diagnose(
        tmp_path,
        kernel_log="NVRM: Xid (PCI:0000:01:00): 1430, benign unrelated value\n",
    )

    assert "1a. OK" in result.stdout
    assert "6. OK" in result.stdout
    assert "heartbeat.status=online" in result.stdout
    assert "measured_fps=15.0, facility.stale=False" in result.stdout


def test_diagnose_fails_for_fatal_current_boot_gpu_signature(tmp_path: Path) -> None:
    result = _run_diagnose(
        tmp_path,
        kernel_log="NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed\n",
    )

    assert result.returncode != 0
    assert "1a. FAIL" in result.stdout
    assert "Xid (PCI:0000:01:00): 154" in result.stdout


def test_diagnose_detects_gsp_boot_failure_but_not_near_miss(tmp_path: Path) -> None:
    fatal = _run_diagnose(tmp_path / "fatal", kernel_log="NVRM: GSP boot failed")
    near_miss = _run_diagnose(tmp_path / "near-miss", kernel_log="NVRM: GSP boot completed")

    assert fatal.returncode != 0
    assert "1a. FAIL" in fatal.stdout
    assert "1a. OK" in near_miss.stdout


def test_diagnose_reports_kernel_signature_scanner_failure(tmp_path: Path) -> None:
    result = _run_diagnose(
        tmp_path,
        kernel_log="NVRM: no fatal signature",
        command_stubs={"grep": 'echo "scanner unavailable" >&2; exit 2'},
    )

    assert result.returncode != 0
    assert "1a. FAIL" in result.stdout
    assert "검사 실패(rc=2): scanner unavailable" in result.stdout
    assert "1a. OK" not in result.stdout


def test_diagnose_fails_when_current_boot_log_is_inconclusive(tmp_path: Path) -> None:
    for index, journalctl_stub in enumerate(
        (
            "sleep 2",
            'echo "access denied" >&2; exit 1',
            "exit 0",
            'echo "-- No entries --"',
            'echo "Hint: You are currently not seeing messages from other users."',
            (
                'if [ "$LC_ALL" = C ]; then '
                'echo "Hint: You are currently not seeing messages from other users."; '
                'else echo "커널 로그 접근 제한"; fi'
            ),
        )
    ):
        result = _run_diagnose(
            tmp_path / str(index),
            kernel_log="NVRM: no fatal signature",
            command_stubs={"journalctl": journalctl_stub},
        )

        assert result.returncode != 0
        assert "1a. FAIL" in result.stdout
        assert "1a. SKIP" not in result.stdout
    overridden = _run_diagnose(
        tmp_path / "overridden",
        kernel_log="NVRM: no fatal signature",
        command_stubs={"journalctl": 'echo "access denied" >&2; exit 1'},
        arguments=("--allow-inconclusive-kernel-log",),
    )

    assert overridden.returncode == 0
    assert "1a. SKIP" in overridden.stdout
    assert "--allow-inconclusive-kernel-log override" in overridden.stdout


def test_diagnose_fails_for_unhealthy_joined_telemetry(tmp_path: Path) -> None:
    cases = (
        (
            "stale-heartbeat",
            {
                "cameras": {
                    "camera-1": {
                        "camera_id": "camera-1",
                        "facility_id": "facility-1",
                        "status": "stale",
                    }
                },
                "runtime": {
                    "facilities": {
                        "facility-1": {
                            "facility_id": "facility-1",
                            "stale": False,
                            "cameras": [
                                {
                                    "camera_id": "camera-1",
                                    "measured_fps": 15.0,
                                    "decode": {
                                        "requested": "nvdec",
                                        "selected": "nvdec",
                                        "fallback_count": 0,
                                    },
                                }
                            ],
                            "latency": {},
                            "gpu": {
                                "nvml_available": True,
                                "cuda_context_ok": True,
                            },
                            "worker": {"alive": True},
                        }
                    }
                },
            },
            "status='stale'",
        ),
        (
            "stale-facility",
            {
                "cameras": {
                    "camera-1": {
                        "camera_id": "camera-1",
                        "facility_id": "facility-1",
                        "status": "online",
                    }
                },
                "runtime": {
                    "facilities": {
                        "facility-1": {
                            "facility_id": "facility-1",
                            "stale": True,
                            "cameras": [],
                            "latency": {},
                            "gpu": {
                                "nvml_available": True,
                                "cuda_context_ok": True,
                            },
                            "worker": {"alive": True},
                        }
                    }
                },
            },
            "stale=True",
        ),
        (
            "worker-dead",
            {
                "cameras": {},
                "runtime": {
                    "facilities": {
                        "facility-1": {
                            "facility_id": "facility-1",
                            "stale": False,
                            "cameras": [],
                            "latency": {},
                            "gpu": {
                                "nvml_available": True,
                                "cuda_context_ok": True,
                            },
                            "worker": {"alive": False},
                        }
                    }
                },
            },
            "worker.alive=False",
        ),
        (
            "gpu-unavailable",
            {
                "cameras": {},
                "runtime": {
                    "facilities": {
                        "facility-1": {
                            "facility_id": "facility-1",
                            "stale": False,
                            "cameras": [],
                            "latency": {},
                            "gpu": {
                                "nvml_available": False,
                                "cuda_context_ok": True,
                            },
                            "worker": {"alive": True},
                        }
                    }
                },
            },
            "gpu.nvml_available=False",
        ),
        (
            "cuda-context-failed",
            {
                "cameras": {},
                "runtime": {
                    "facilities": {
                        "facility-1": {
                            "facility_id": "facility-1",
                            "stale": False,
                            "cameras": [],
                            "latency": {},
                            "gpu": {
                                "nvml_available": True,
                                "cuda_context_ok": False,
                            },
                            "worker": {"alive": True},
                        }
                    }
                },
            },
            "gpu.cuda_context_ok=False",
        ),
        (
            "no-fps",
            {
                "cameras": {
                    "camera-1": {
                        "camera_id": "camera-1",
                        "facility_id": "facility-1",
                        "status": "online",
                    }
                },
                "runtime": {
                    "facilities": {
                        "facility-1": {
                            "facility_id": "facility-1",
                            "stale": False,
                            "cameras": [
                                {
                                    "camera_id": "camera-1",
                                    "decode": {
                                        "requested": "nvdec",
                                        "selected": "nvdec",
                                        "fallback_count": 0,
                                    },
                                }
                            ],
                            "latency": {},
                            "gpu": {
                                "nvml_available": True,
                                "cuda_context_ok": True,
                            },
                            "worker": {"alive": True},
                        }
                    }
                },
            },
            "measured_fps 누락",
        ),
        (
            "nvdec-fallback",
            {
                "cameras": {
                    "camera-1": {
                        "camera_id": "camera-1",
                        "facility_id": "facility-1",
                        "status": "online",
                    }
                },
                "runtime": {
                    "facilities": {
                        "facility-1": {
                            "facility_id": "facility-1",
                            "stale": False,
                            "cameras": [
                                {
                                    "camera_id": "camera-1",
                                    "measured_fps": 0,
                                    "decode": {
                                        "requested": "nvdec",
                                        "selected": "opencv",
                                        "fallback_count": 1,
                                    },
                                }
                            ],
                            "latency": {},
                            "gpu": {
                                "nvml_available": True,
                                "cuda_context_ok": True,
                            },
                            "worker": {"alive": True},
                        }
                    }
                },
            },
            "requested NVDEC인데 selected='opencv'",
        ),
    )

    for name, status, expected in cases:
        result = _run_diagnose(
            tmp_path / name,
            kernel_log="NVRM: no fatal signature",
            status=status,
        )

        assert result.returncode != 0
        assert "6. FAIL" in result.stdout
        assert expected in result.stdout


def test_diagnose_reports_docker_probe_failure_evidence(tmp_path: Path) -> None:
    cases = (
        ("exec sleep 999", "Docker 데몬 조회가 1초를 초과했습니다", "1"),
        ('echo "permission denied" >&2; exit 1', "실패(rc=1): permission denied", "20"),
        ('echo "daemon unavailable" >&2; exit 42', "실패(rc=42): daemon unavailable", "20"),
    )

    for index, (docker_stub, expected, timeout_sec) in enumerate(cases):
        result = _run_diagnose(
            tmp_path / str(index),
            kernel_log="NVRM: no fatal signature",
            command_stubs={"docker": docker_stub},
            timeout_sec=timeout_sec,
        )

        assert "3. SKIP" in result.stdout
        assert expected in result.stdout


def test_diagnose_fails_when_no_runtime_cameras_join(tmp_path: Path) -> None:
    status = {
        "cameras": {},
        "runtime": {
            "facilities": {
                "facility-1": {
                    "facility_id": "facility-1",
                    "stale": False,
                    "cameras": [],
                    "latency": {},
                    "gpu": {"nvml_available": True, "cuda_context_ok": True},
                    "worker": {"alive": True},
                }
            }
        },
    }
    result = _run_diagnose(
        tmp_path,
        kernel_log="NVRM: no fatal signature",
        status=status,
    )

    assert result.returncode != 0
    assert "6. FAIL" in result.stdout
    assert "joined runtime camera diagnostics 없음" in result.stdout
    assert "6. OK" not in result.stdout


def test_diagnose_fails_for_malformed_and_unmatched_status(tmp_path: Path) -> None:
    malformed = _run_diagnose(
        tmp_path / "malformed",
        kernel_log="NVRM: no fatal signature",
        status="{not json",
    )
    unmatched = _run_diagnose(
        tmp_path / "unmatched",
        kernel_log="NVRM: no fatal signature",
        status={
            "cameras": {
                "camera-1": {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "status": "online",
                }
            },
            "runtime": {
                "facilities": {
                    "facility-1": {
                        "facility_id": "facility-1",
                        "stale": False,
                        "cameras": [],
                        "latency": {},
                        "gpu": {"nvml_available": True, "cuda_context_ok": True},
                        "worker": {"alive": True},
                    }
                }
            },
        },
    )

    assert malformed.returncode != 0
    assert "6. FAIL" in malformed.stdout
    assert unmatched.returncode != 0
    assert "runtime camera diagnostics 없음: camera-1" in unmatched.stdout
