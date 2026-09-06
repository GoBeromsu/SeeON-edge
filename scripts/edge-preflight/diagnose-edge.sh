#!/usr/bin/env bash
# Edge GPU 장애를 조회만으로 좁히는 6단계 진단. 기본 모드는 상태 변경 명령을 넣지 않는다.
set -u
set -o pipefail

readonly PREFIX='[edge-diagnose]'
readonly SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
readonly STEP_TIMEOUT="${EDGE_PREFLIGHT_TIMEOUT_SEC:-15}"
readonly CUDA_IMAGE="${EDGE_PREFLIGHT_CUDA_IMAGE:-nvidia/cuda:12.8.0-base-ubuntu24.04}"
readonly STATUS_URL="${EDGE_STATUS_URL:-http://127.0.0.1:${ML_SERVING_PORT:-8000}/api/v1/status}"
readonly WORKER_CONTAINER="${EDGE_WORKER_CONTAINER:-eldercare-fall-ml-ml-worker-1}"

container_probe=false
allow_inconclusive_kernel_log=false
for argument in "$@"; do
  case "$argument" in
    --with-container-probe) container_probe=true ;;
    --allow-inconclusive-kernel-log) allow_inconclusive_kernel_log=true ;;
    *) printf '%s 사용법: %s [--with-container-probe] [--allow-inconclusive-kernel-log]\n' "$PREFIX" "$0" >&2; exit 2 ;;
  esac
done

ok_count=0
fail_count=0
skip_count=0
fail_reasons=()

say() { printf '%s %s\n' "$PREFIX" "$*"; }
need_command() { command -v "$1" >/dev/null 2>&1; }
run_timeout() { timeout "${STEP_TIMEOUT}s" "$@"; }
brief() { printf '%s' "$1" | tr '\n' ' ' | cut -c1-180; }

record() {
  local number=$1 result=$2 detail=$3 action=${4:-}
  case "$result" in
    OK) ((ok_count += 1));;
    FAIL) ((fail_count += 1)); fail_reasons+=("$detail");;
    SKIP) ((skip_count += 1));;
  esac
  say "$number. $result — $detail"
  [[ -z "$action" ]] || say "   다음 조치: $action"
}

step_driver() {
  if ! need_command nvidia-smi; then
    record 1 SKIP 'nvidia-smi가 없어 드라이버를 조회할 수 없습니다.' '런북 2절에서 드라이버 증상을 고르세요.'
    return
  fi
  local output rc
  output=$(run_timeout nvidia-smi 2>&1); rc=$?
  if ((rc == 124)); then
    record 1 FAIL "nvidia-smi가 ${STEP_TIMEOUT}초 안에 끝나지 않습니다." '런북 4절의 warm reboot 분기를 준비하세요.'
  elif ((rc != 0)); then
    record 1 FAIL "nvidia-smi 실패: $(brief "$output")" '런북 2절에서 드라이버 상태를 확인하고, WPR2/fullchip이면 런북 4절로 가세요.'
  elif grep -q 'ERR!' <<<"$output"; then
    record 1 FAIL 'nvidia-smi에 ERR!가 있어 GPU 텔레메트리가 비정상입니다.' '런북 4절의 warm reboot 분기를 따르세요.'
  else
    record 1 OK 'nvidia-smi가 정상 응답했고 ERR!가 없습니다.'
  fi
}
step_kernel_log() {
  inconclusive_detail() {
    if "$allow_inconclusive_kernel_log"; then
      record '1a' SKIP "$1 (명시적 --allow-inconclusive-kernel-log override)" '런북 2절에서 현재 부팅의 커널 로그를 수동으로 확인하세요.'
    else
      record '1a' FAIL "$1" '치명적 GPU 서명을 배제할 수 없습니다. 런북 2절에서 현재 부팅의 커널 로그를 확인하거나, 검토 후에만 --allow-inconclusive-kernel-log을 사용하세요.'
    fi
  }

  if ! need_command journalctl; then
    inconclusive_detail 'journalctl이 없어 현재 부팅의 NVIDIA 커널 로그를 조회할 수 없습니다.'
    return
  fi
  local output rc signatures scanner_rc
  output=$(LC_ALL=C run_timeout journalctl --boot 0 --dmesg --no-pager 2>&1); rc=$?
  if ((rc == 124)); then
    inconclusive_detail "현재 부팅의 커널 로그 조회가 ${STEP_TIMEOUT}초를 초과했습니다."
    return
  elif ((rc != 0)); then
    inconclusive_detail "현재 부팅의 커널 로그를 읽을 수 없습니다(rc=$rc): $(brief "$output")"
    return
  fi
  if [[ -z "${output//[[:space:]]/}" ]] || [[ "$output" =~ (--[[:space:]]No[[:space:]]entries[[:space:]]--|[Pp]ermission[[:space:]]denied|[Nn]ot[[:space:]]seeing[[:space:]]messages|[Ii]nsufficient[[:space:]]permissions|[Ff]ailed[[:space:]]to[[:space:]]open[[:space:]]journal) ]]; then
    inconclusive_detail "현재 부팅의 커널 로그가 비어 있거나 권한에 의해 필터링되었습니다: $(brief "$output")"
    return
  fi

  signatures=$(printf '%s\n' "$output" | grep -E -i \
    'NVRM:.*Xid[[:space:]]*\([^)]*\):[[:space:]]*(143|154)([^[:digit:]]|$)|NVRM:.*RmInitAdapter.*(fail|error)|NVRM:.*((WPR|[Ff]ull[[:space:]-]?[Cc]hip).*(reset|Reset)|(reset|Reset).*(WPR|[Ff]ull[[:space:]-]?[Cc]hip))|NVRM:.*((GSP|FSP).*(boot|Boot).*(fail|error)|(fail|error).*(boot|Boot).*(GSP|FSP))' 2>&1)
  scanner_rc=$?
  if ((scanner_rc == 0)); then
    record '1a' FAIL "현재 부팅 NVIDIA 치명적 서명: $(brief "$signatures")" 'nvidia-smi가 정상이어도 런북 4절의 WPR/fullchip 또는 GSP/FSP 복구 분기를 따르세요.'
  elif ((scanner_rc == 1)); then
    record '1a' OK '현재 부팅의 NVIDIA 커널 로그에서 알려진 치명적 GPU 서명이 없습니다.'
  else
    inconclusive_detail "현재 부팅 NVIDIA 치명적 서명 검사 실패(rc=$scanner_rc): $(brief "$signatures")"
  fi
}

step_version() {
  if [[ ! -r /proc/driver/nvidia/version ]]; then
    record 2 SKIP '/proc/driver/nvidia/version을 읽을 수 없습니다(드라이버 미적재 또는 권한 제한).' '런북 2절에서 드라이버 증상을 고르세요.'
    return
  fi
  if ! need_command dpkg-query; then
    record 2 SKIP 'dpkg-query가 없어 설치된 NVIDIA 패키지 버전을 확인할 수 없습니다.' '런북 2절을 확인하세요.'
    return
  fi
  local kernel kernel_line kernel_rc packages packages_rc package package_version version
  kernel_line=$(run_timeout grep '^NVRM version:' /proc/driver/nvidia/version 2>&1); kernel_rc=$?
  packages=$(run_timeout dpkg-query -W -f='${Package} ${Version}\n' 'nvidia-driver-*' 'nvidia-utils-*' 2>&1); packages_rc=$?

  if ((kernel_rc == 124)); then
    record 2 SKIP "커널 드라이버 버전 조회가 ${STEP_TIMEOUT}초를 초과했습니다: $(brief "$kernel_line")" '런북 2절에서 드라이버 상태를 확인하세요.'
    return
  elif ((kernel_rc != 0)); then
    record 2 SKIP "커널 드라이버 버전 조회 실패(rc=$kernel_rc): $(brief "$kernel_line")" '런북 2절에서 드라이버 상태를 확인하세요.'
    return
  elif [[ "$kernel_line" =~ ([0-9]+\.[0-9]+(\.[0-9]+)?) ]]; then
    kernel=${BASH_REMATCH[1]}
  else
    record 2 SKIP "커널 드라이버 버전 조회는 성공했지만 버전을 해석할 수 없습니다: $(brief "$kernel_line")" '런북 2절에서 /proc/driver/nvidia/version 내용을 확인하세요.'
    return
  fi

  if ((packages_rc == 124)); then
    record 2 SKIP "설치된 NVIDIA 패키지 버전 조회가 ${STEP_TIMEOUT}초를 초과했습니다: $(brief "$packages")" '런북 2절에서 패키지 상태를 확인하세요.'
    return
  elif ((packages_rc != 0)); then
    record 2 SKIP "설치된 NVIDIA 패키지 버전 조회 실패(rc=$packages_rc): $(brief "$packages")" '런북 2절에서 패키지 상태를 확인하세요.'
    return
  fi

  package_version=''
  while read -r package version; do
    if [[ "$version" =~ ^([0-9]+\.[0-9]+(\.[0-9]+)?) ]]; then package_version=${BASH_REMATCH[1]}; break; fi
  done <<<"$packages"
  if [[ -z "$package_version" ]]; then
    record 2 SKIP "설치된 NVIDIA 패키지 버전 조회는 성공했지만 버전을 해석할 수 없습니다: $(brief "$packages")" '런북 2절에서 패키지 상태를 확인하세요.'
  elif [[ "$kernel" == "$package_version" ]]; then
    record 2 OK "커널 모듈($kernel)과 설치 패키지($package_version)가 일치합니다."
  else
    record 2 FAIL "커널 모듈($kernel)과 설치 패키지($package_version)가 불일치합니다." '런북 3절의 soft-reload 전제조건을 확인하세요.'
  fi
}

step_container_gpu() {
  if ! need_command docker; then
    record 3 SKIP 'docker가 없어 컨테이너 GPU 설정을 확인할 수 없습니다.' '런북 1-2절과 런북 3-3절을 확인하세요.'
    return
  fi
  local docker_output docker_rc runtime_output runtime_rc inspect_output inspect_rc cdi_output cdi_rc probe_output probe_rc
  docker_output=$(run_timeout docker info 2>&1); docker_rc=$?
  if ((docker_rc == 124)); then
    record 3 SKIP "Docker 데몬 조회가 ${STEP_TIMEOUT}초를 초과했습니다: $(brief "$docker_output")" '런북 1-2절에서 Docker 데몬 상태를 확인하세요.'
    return
  elif ((docker_rc != 0)); then
    record 3 SKIP "Docker 데몬 조회 실패(rc=$docker_rc): $(brief "$docker_output")" '런북 1-2절에서 Docker 접근 권한과 데몬 상태를 확인하세요.'
    return
  fi
  runtime_output=$(run_timeout sh "$SCRIPT_DIR/check-nvidia-runtime.sh" 2>&1); runtime_rc=$?
  inspect_output=$(run_timeout docker inspect -f '{{.State.Status}} gpu={{json .HostConfig.DeviceRequests}}' "$WORKER_CONTAINER" 2>&1); inspect_rc=$?
  if need_command nvidia-ctk; then
    cdi_output=$(run_timeout nvidia-ctk cdi list 2>&1); cdi_rc=$?
  else
    cdi_output='nvidia-ctk 없음'; cdi_rc=127
  fi
  if ((runtime_rc == 124)); then
    record 3 FAIL "NVIDIA runtime 확인이 ${STEP_TIMEOUT}초를 초과했습니다: $(brief "$runtime_output")" '런북 3-3절에서 CDI/runtime을 복구하세요.'
  elif ((runtime_rc != 0)); then
    record 3 FAIL "NVIDIA runtime 확인 실패(rc=$runtime_rc): $(brief "$runtime_output")" '런북 3-3절에서 실패 출력을 기준으로 CDI/runtime을 복구하세요.'
  elif ((inspect_rc == 124)); then
    record 3 SKIP "기존 워커 컨테이너 inspect가 ${STEP_TIMEOUT}초를 초과했습니다: $(brief "$inspect_output")" '런북 1-3절에서 워커 상태를 확인하세요.'
  elif ((inspect_rc != 0)); then
    record 3 SKIP "기존 워커 컨테이너 inspect 실패(rc=$inspect_rc): $(brief "$inspect_output")" '런북 1-3절에서 워커 이름과 상태를 확인하세요.'
  elif ((cdi_rc == 124)); then
    record 3 FAIL "CDI 목록 조회가 ${STEP_TIMEOUT}초를 초과했습니다: $(brief "$cdi_output")" '런북 3-3절의 CDI 재생성을 확인하세요.'
  elif ((cdi_rc != 0)); then
    record 3 FAIL "CDI 목록 조회 실패(rc=$cdi_rc): $(brief "$cdi_output")" '런북 3-3절의 CDI 재생성을 확인하세요.'
  else
    record 3 OK "기존 워커 설정: $inspect_output; CDI: $(brief "$cdi_output")"
  fi
  if "$container_probe"; then
    probe_output=$(run_timeout docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi 2>&1); probe_rc=$?
    if ((probe_rc == 0)); then
      say "3. opt-in 컨테이너 probe OK — $CUDA_IMAGE에서 GPU가 보입니다."
    else
      record 3 FAIL "opt-in 컨테이너 probe 실패(rc=$probe_rc): $(brief "$probe_output")" '런북 3-3절의 CDI 재생성을 확인하세요.'
    fi
  fi
}

step_cuda_context() {
  local output rc
  output=$(run_timeout sh "$SCRIPT_DIR/check-cuda-context.sh" 2>&1); rc=$?
  if ((rc == 0)); then
    record 4 OK '기존 check-cuda-context.sh의 CUDA 컨텍스트 검사가 통과했습니다.'
  elif ((rc == 124)); then
    record 4 FAIL "기존 CUDA 컨텍스트 검사가 ${STEP_TIMEOUT}초 초과했습니다." '런북 4절의 warm reboot 분기를 확인하세요.'
  else
    record 4 FAIL "기존 CUDA 컨텍스트 검사 실패: $(brief "$output")" '런북 2절을 확인하세요.'
  fi
}

step_service() {
  if ! need_command curl; then
    record 6 SKIP 'curl이 없어 ml-api 상태를 조회할 수 없습니다.' '런북 1-4절에서 ml-api와 포트 바인딩을 확인하세요.'
    return
  fi
  local body rc summary
  body=$(run_timeout curl --silent --show-error --fail "$STATUS_URL" 2>&1); rc=$?
  if ((rc != 0)); then
    record 6 FAIL "ml-api 상태 조회 실패($STATUS_URL, rc=$rc)." '런북 1-4절에서 ml-api와 포트 바인딩을 확인하세요.'
    return
  fi
  if ! need_command python3; then
    record 6 SKIP 'python3가 없어 상태 JSON 필드를 검증할 수 없습니다.' '런북 1-4절을 확인하세요.'
    return
  fi
  summary=$(printf '%s' "$body" | run_timeout python3 -c '
import json, sys

try:
    data = json.load(sys.stdin)
    heartbeats = data["cameras"]
    runtime = data["runtime"]
    facilities = runtime["facilities"]
    if not isinstance(heartbeats, dict):
        raise ValueError("top-level cameras가 객체가 아님")
    if not isinstance(facilities, dict) or not facilities:
        raise ValueError("runtime.facilities가 없거나 비어 있음")

    heartbeat_by_camera = {}
    for camera_id, heartbeat in heartbeats.items():
        if not isinstance(camera_id, str) or not isinstance(heartbeat, dict):
            raise ValueError("top-level cameras 항목 형식 오류")
        for key in ("camera_id", "status", "facility_id"):
            if key not in heartbeat:
                raise ValueError(f"heartbeat {camera_id}: {key} 누락")
        if heartbeat["camera_id"] != camera_id:
            raise ValueError(f"heartbeat {camera_id}: camera_id 불일치")
        if heartbeat["status"] != "online":
            raise ValueError("heartbeat {}: status={!r} (online이어야 함)".format(camera_id, heartbeat["status"]))
        heartbeat_by_camera[camera_id] = heartbeat

    seen = set()
    lines = []
    for facility_id, facility in facilities.items():
        if not isinstance(facility_id, str) or not isinstance(facility, dict):
            raise ValueError("runtime.facilities 항목 형식 오류")
        for key in ("facility_id", "stale", "cameras", "latency", "gpu", "worker"):
            if key not in facility:
                raise ValueError(f"{facility_id}: {key} 누락")
        if facility["facility_id"] != facility_id:
            raise ValueError(f"{facility_id}: facility_id 불일치")
        if facility["stale"] is not False:
            raise ValueError("{}: stale={!r} (False여야 함)".format(facility_id, facility["stale"]))
        cameras = facility["cameras"]
        if not isinstance(cameras, list):
            raise ValueError(f"{facility_id}: cameras가 목록이 아님")
        gpu = facility["gpu"]
        if not isinstance(gpu, dict):
            raise ValueError(f"{facility_id}: gpu가 객체가 아님")
        for key in ("nvml_available", "cuda_context_ok"):
            if key not in gpu:
                raise ValueError(f"{facility_id}: gpu.{key} 누락")
            if gpu[key] is not True:
                raise ValueError(f"{facility_id}: gpu.{key}={gpu[key]!r} (True여야 함)")
        worker = facility["worker"]
        if not isinstance(worker, dict) or "alive" not in worker:
            raise ValueError(f"{facility_id}: worker.alive 누락")
        if worker["alive"] is not True:
            raise ValueError("{}: worker.alive={!r} (True여야 함)".format(facility_id, worker["alive"]))
        for camera in cameras:
            if not isinstance(camera, dict) or "camera_id" not in camera:
                raise ValueError(f"{facility_id}: runtime camera_id 누락")
            camera_id = camera["camera_id"]
            if not isinstance(camera_id, str) or camera_id not in heartbeat_by_camera:
                raise ValueError(f"{facility_id}/{camera_id}: 대응하는 heartbeat 없음")
            heartbeat = heartbeat_by_camera[camera_id]
            if heartbeat["facility_id"] != facility_id:
                raise ValueError(f"{facility_id}/{camera_id}: heartbeat facility_id 불일치")
            if "decode" not in camera or not isinstance(camera["decode"], dict):
                raise ValueError(f"{facility_id}/{camera_id}: decode 누락")
            decode = camera["decode"]
            for key in ("requested", "selected", "fallback_count"):
                if key not in decode:
                    raise ValueError(f"{facility_id}/{camera_id}: decode.{key} 누락")
            if decode["requested"] == "nvdec":
                if decode["selected"] != "nvdec":
                    raise ValueError(
                        "{}/{}: requested NVDEC인데 selected={!r}".format(
                            facility_id, camera_id, decode["selected"]
                        )
                    )
                if decode["fallback_count"] != 0:
                    raise ValueError(
                        "{}/{}: requested NVDEC인데 fallback_count={!r}".format(
                            facility_id, camera_id, decode["fallback_count"]
                        )
                    )
            if "measured_fps" not in camera:
                raise ValueError(f"{facility_id}/{camera_id}: measured_fps 누락")
            measured_fps = camera["measured_fps"]
            if (
                isinstance(measured_fps, bool)
                or not isinstance(measured_fps, (int, float))
                or measured_fps <= 0
            ):
                raise ValueError(
                    f"{facility_id}/{camera_id}: measured_fps={measured_fps!r} (0보다 커야 함)"
                )
            lines.append(
                "{}/{}: heartbeat.status={}, decode.requested={}, decode.selected={}, "
                "fallback_count={}, measured_fps={}, facility.stale={}, gpu.nvml_available={}, "
                "gpu.cuda_context_ok={}, worker.alive={}".format(
                    facility_id, camera_id, heartbeat["status"], decode["requested"],
                    decode["selected"], decode["fallback_count"], measured_fps,
                    facility["stale"], gpu["nvml_available"], gpu["cuda_context_ok"], worker["alive"],
                )
            )
            seen.add(camera_id)

    unmatched = sorted(set(heartbeat_by_camera) - seen)
    if unmatched:
        raise ValueError("runtime camera diagnostics 없음: " + ", ".join(unmatched))
    if not lines:
        raise ValueError("joined runtime camera diagnostics 없음")
    print(" | ".join(lines))
except (KeyError, TypeError, ValueError) as exc:
    print(f"INVALID: {exc}")
    sys.exit(1)
')
  rc=$?
  if ((rc == 0)); then record 6 OK "서비스 상태: $summary"; else record 6 FAIL "상태 응답에 필수 필드가 없거나 비정상입니다: $summary" '런북 1-4절에서 API heartbeat와 런타임 상태를 확인하세요.'; fi
}

say '조회 전용 5단계 진단을 시작합니다. 기본 모드는 컨테이너를 생성하지 않습니다.'
if "$container_probe"; then say '경고: --with-container-probe는 컨테이너를 생성·삭제하고 이미지를 내려받을 수 있습니다.'; fi
step_driver
step_kernel_log
step_version
step_container_gpu
step_cuda_context
step_service
say "요약: OK=${ok_count}, FAIL=${fail_count}, SKIP=${skip_count}"
if ((fail_count > 0)); then say "가장 가능성 높은 원인: ${fail_reasons[0]}"; exit 1; fi
if ((skip_count > 0)); then say '진단 범위에 SKIP만 포함되면 exit 0은 장애 없음이 아니라 불완전 진단을 뜻합니다.'; fi
say '가장 가능성 높은 원인: 진단 범위에서 장애가 발견되지 않았습니다.'
exit 0
