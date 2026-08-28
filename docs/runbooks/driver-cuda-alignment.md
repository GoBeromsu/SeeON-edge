# 엣지 GPU 워커 장애 복구 런북: NVIDIA 드라이버 / CUDA / 이벤트 미수신

새벽에 **이벤트가 안 오면 먼저 1절의 읽기 전용 진단부터** 실행한다. `WPR2` 또는
`fullchip reset` 로그가 있으면 3절의 모듈 재적재를 건너뛰고 즉시 4절의 재부팅으로
간다. 단순 사용자 공간/커널 모듈 버전 불일치는 3절 soft-reload로 복구할 수 있지만,
그 GPU 하드웨어 상태는 soft-reload로 복구되지 않는다.

운영 작업 디렉터리와 컨테이너 이름은 다음과 같다.

```sh
cd "/home/seniorsailab/Senior AI Lab/eldercare-fall-ml"
DC='docker compose --env-file .env.edge.deploy -f compose.edge.yaml -f compose.edge.local.yaml'
```

> [!WARNING]
> `.env.edge.deploy`에는 `ML_WORKER_PROFILE`이 없다. 따라서 그대로는 `$DC ps`조차
> `${ML_WORKER_PROFILE:?set cuda|mps|cpu}` 오류로 실패한다. 이 상태에서 compose로
> 워커를 재생성하지 말 것. compose 재생성은 현재 컨테이너의 정책과 달리
> `restart: unless-stopped`를 붙인다. 이는 알려진 배포 재현성 결함이다. 아래의
> 장애 진단과 기존 컨테이너 조작에는 이름/ID를 직접 사용한다.

- API: `eldercare-fall-ml-ml-api-1` (정상일 때 healthy)
- 워커: `eldercare-fall-ml-ml-worker-1` (확인 당시 ID `c7fe502f89fd`)
- 클립 저장소: `/home/seniorsailab/.local/share/eldercare-fall-ml/clip-store`

## 1. “이벤트가 안 온다” — 먼저 원인을 좁히기 (상태 변경 없음)

아래 명령은 조회만 한다. 각 블록을 위에서 아래로 실행하고, 출력에 맞는 화살표를
따른다.
빠른 읽기 전용 진단은 다음 명령으로 실행한다. 기본 모드는 컨테이너를 만들거나 지우지
않으며, 6단계를 모두 실행한다. `--with-container-probe`는 새 컨테이너 생성·삭제 및
이미지 내려받기를 할 수 있는 상태 변경 opt-in이다.
```sh
bash scripts/edge-preflight/diagnose-edge.sh
# 상태 변경을 이해하고 컨테이너 내부 GPU probe가 꼭 필요할 때만:
bash scripts/edge-preflight/diagnose-edge.sh --with-container-probe
```
종료 `0`은 이 진단에서 `FAIL`이 없다는 뜻이다. `SKIP`만 있거나 `SKIP`이 섞인
종료 `0`은 권한·명령 부재 등으로 일부를 확인하지 못한 **불완전 진단**일 수 있으므로,
SKIP 출력을 해결하거나 아래 수동 조회를 계속한다. 종료 `1`은 하나 이상의 FAIL이며
출력의 “다음 조치”가 가리키는 실제 절을 따른다.


### 1-1. GPU와 호스트 드라이버

```sh
nvidia-smi
cat /proc/driver/nvidia/version
journalctl -k -b | grep -E 'NVRM|Xid|WPR2|fullchip|RmInitAdapter'
```

- `nvidia-smi`가 GPU 이름, 드라이버 버전, 온도/전력을 표시하고 `/proc` 버전과
  일치하면 → 1-2로 간다.
- `nvidia-smi`가 실패하거나 `/proc`의 커널 모듈 버전과 `nvidia-smi`의 드라이버
  버전이 다르면 → 2절에서 증상을 고른다.
- 커널 로그에
  `GPU0 _kgspBootGspRm: unexpected WPR2 already up, cannot proceed with booting GSP`,
  `RmInitAdapter failed! (0x62:0x40:2168)`, 또는
  `NV_ERR_GPU_IN_FULLCHIP_RESET`가 있으면 → **soft-reload를 시도하지 말고 4절로
  즉시 간다.**
- `nvidia-smi`의 팬/온도/전력이 `ERR!`이면 → **4절로 즉시 간다.**
  `nvidia-smi -r -i 0`가 `Not Supported`인 GeForce에서는 소프트웨어 GPU reset이
  해법이 아니다.

### 1-2. 컨테이너가 GPU를 실제로 받는지

기본 진단은 기존 워커의 `docker inspect`, `nvidia-ctk cdi list`, Docker runtime
설정만 조회한다. 새 컨테이너 probe가 필요한 경우에만 명시적으로 실행한다.

```sh
bash scripts/edge-preflight/diagnose-edge.sh
# 아래는 컨테이너를 생성·삭제할 수 있으므로 필요할 때만 실행
bash scripts/edge-preflight/diagnose-edge.sh --with-container-probe
sh scripts/edge-preflight/check-nvidia-runtime.sh
sh scripts/edge-preflight/check-cuda-context.sh
```

- 기본 진단의 기존 워커 설정·CDI 목록과 두 스크립트가 성공(exit 0)하면 → 1-3으로.
- runtime 검사 실패 시에는 출력 전문을 먼저 보고 누락된 runtime/CDI 원인을 고른다.
  CDI가 원인이면 3-3절의 재생성 뒤 이 블록을 다시 실행한다.
- opt-in 컨테이너 probe 또는 `check-nvidia-runtime.sh`가 실패하면 → 3-3절의 CDI/runtime
  상태와 실패 출력을 확인한다.
- `check-cuda-context.sh`가 `cuInit=100` 또는 비영(非零) exit로 실패하지만
  `nvidia-smi`는 정상이라면 → 2-2 (기존 580.x/CUDA 13.0 분기)로 간다.
- 호스트 `nvidia-smi`도 실패하면 → 2-1 또는 2-3을 따른다.

### 1-3. 워커가 살아 있고 GPU를 쓰는지

```sh
docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} restart={{.HostConfig.RestartPolicy.Name}}' eldercare-fall-ml-ml-worker-1
docker logs --tail 200 eldercare-fall-ml-ml-worker-1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

- `running`, 로그에 처리 오류가 없고 compute-apps에 워커 PID가 보이면 → GPU/워커는
  살아 있다. 1-4의 API/네트워크를 확인한다.
- `exited exit=137 restart=no`이면 → 외부 SIGKILL 뒤 자동 재시작되지 않은 것이다.
  5절의 워커 복구 및 재발 방지를 따른다.
- `exited` 또는 로그에 CUDA/NVML 오류가 있으면 → 2절로 돌아가 GPU 증상을 고른다.
- `nvidia-smi --query-compute-apps`가 빈 출력이면서 워커가 `running`이면 → 워커가
  GPU를 사용하지 못하거나 입력이 없다. `docker logs`의 CUDA 초기화/카메라 오류를
  보고 1-5 또는 2절로 간다. 이 명령 자체의 exit 0은 “GPU 프로세스 없음”도 뜻한다.

### 1-4. API와 엣지-API 네트워크

```sh
curl -s http://127.0.0.1:8000/api/v1/status
docker logs --tail 200 eldercare-fall-ml-ml-api-1
```

- 첫 명령이 상태 JSON을 출력하고 exit 0이면 → 로컬 API는 살아 있다. 워커 로그의
  relay/HTTP 전송 오류를 확인하고, 원격 API 주소·네트워크를 점검한다.
- 상태 JSON이 HTTP 200/exit 0이지만 `runtime.facilities[].gpu`, `worker`, 카메라의
  `measured_fps`, `latency` 중 하나가 없으면 → 배포된 `ml-api` 이미지가 G004의 관측
  확장 이전 버전이다. 이는 GPU 장애가 아니라 **이미지 갱신 사안**이다. 우선 실제
  이미지 tag/digest를 식별하고 별도 배포 작업으로 갱신한다(이 런북에서 이미지를 교체하지
  않는다).
  ```sh
  docker inspect -f '{{.Config.Image}} image-id={{.Image}}' eldercare-fall-ml-ml-api-1
  docker image inspect -f '{{index .RepoDigests 0}}' "$(docker inspect -f '{{.Image}}' eldercare-fall-ml-ml-api-1)"
  ```
- 연결 거부, 빈 출력, 또는 curl 비영 exit이면 → API 컨테이너/포트 문제다.
  API 로그를 확인한다. 컨테이너가 `healthy`가 아니면 GPU 문제가 아니라 API 복구
  절차를 우선한다.
- API가 정상이고 워커 로그에 전송 성공도 있는데 이벤트가 없다면 → 수신 측 네트워크
  또는 수신 서비스를 점검한다.

### 1-5. 카메라/입력인지 구분

```sh
docker logs --tail 200 eldercare-fall-ml-ml-worker-1
```

- 로그에 RTSP 연결, 인증, timeout, decoder 오류가 있으면 → 카메라 또는 카메라
  네트워크 문제다. GPU 복구를 반복하지 말고 카메라 전원·RTSP 주소·망을 점검한다.
- 로그에 프레임/추론은 진행되는데 이벤트 전송 오류가 있으면 → 1-4의 네트워크
  분기다.
- 로그에 CUDA/NVML/cuInit 오류가 있으면 → 2절 GPU 분기다.

## 2. GPU 증상별로 갈라서 판단

### 2-1. 단순 드라이버 사용자 공간/커널 모듈 버전 불일치 (595.84)

다음처럼 사용자 공간은 `595.84`인데 커널 모듈은 `595.71.05`인 경우다.

```sh
nvidia-smi
cat /proc/driver/nvidia/version
```

- 두 버전이 다르면 → 3절 soft-reload를 수행한다.
- 두 버전이 같고 `nvidia-smi`가 정상인데 `check-cuda-context.sh`만 실패하면 →
  2-2를 따른다.
- 1-1의 WPR2/fullchip 로그 또는 `ERR!`가 하나라도 있으면 → 버전과 무관하게
  4절 재부팅으로 간다.

### 2-2. 기존 유효 분기: `cuInit(0)=100` / 580.x + CUDA 13.0

다음은 **호스트 GPU가 열거되고 `nvidia-smi`도 정상인데**, CUDA 컨텍스트만 실패하는
별도 문제다. 595.84 모듈 불일치나 WPR2 상태와 혼동하지 않는다.

```sh
python3 -c "import ctypes; print('cuInit', ctypes.CDLL('libcuda.so.1').cuInit(0))"
sh scripts/edge-preflight/check-cuda-context.sh
nvidia-smi
```

- `cuInit 100` 및 스크립트 비영 exit, `nvidia-smi` 정상, 580.x/CUDA 13.0이면 →
  유지보수 창에 580.x/CUDA 13.0을 CUDA 12.x와 Blackwell 지원 `*-open` 드라이버의
  알려진 정상 조합으로 정렬한다. Blackwell에는 반드시 open 커널 모듈을 쓴다.
  먼저 기존 워커를 멈추고, `apt` 경로 또는 NVIDIA CUDA 12.9 local installer 경로 중
  현장 표준을 하나만 사용한 뒤 재부팅한다.
  ```sh
  docker stop eldercare-fall-ml-ml-worker-1
  sudo apt-get update
  sudo apt-get install -y nvidia-driver-575-open
  # 또는 CUDA 12.9.1 Ubuntu 24.04 deb local installer를 설치한 뒤 nvidia-open 설치
  sudo reboot
  # 부팅 후: cuInit 0, 새 드라이버/CUDA 버전을 모두 확인
  python3 -c "import ctypes; print('cuInit', ctypes.CDLL('libcuda.so.1').cuInit(0))"
  nvidia-smi
  sh scripts/edge-preflight/check-cuda-context.sh
  ```
  CUDA 12.x로 바꾸면 별도 배포 변경에서 `pyproject.toml`의 `[[tool.uv.index]]` URL과
  두 `[tool.uv.sources]`를 해당 cu12x wheel index(예: `https://download.pytorch.org/whl/cu128`)로
  함께 정렬한 뒤 lock/image를 갱신한다. `cu128` wheel을 쓸 경우에도 실제 Blackwell
  `sm_120` 포함 여부를 검증 전제로 하며, 다음 결과에 `sm_120`이 반드시 있어야 한다.
  ```sh
  uv lock && uv sync --frozen
  uv run python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
  ```
  현재 잠금/이미지를 임의로 바꾸지 말고, 6절의 이미지 검증을 통과한 뒤에만 배포
  결함을 고친 compose 경로로 워커 cutover를 한다.
- `cuInit 0`이면 → CUDA 컨텍스트는 정상이다. 1-3으로 돌아가 워커/카메라/네트워크를
  좁힌다.
- WPR2/fullchip 로그가 있으면 → 드라이버 조합 변경보다 4절 전원 사이클이 먼저다.

### 2-3. WPR2 또는 fullchip reset

이 상태는 PCI FLR와 모듈 재적재가 일시적으로 `nvidia-smi`를 살려도 약 10분 뒤
`NV_ERR_GPU_IN_FULLCHIP_RESET`으로 다시 떨어질 수 있다. GeForce의
`nvidia-smi -r -i 0`는 `Not Supported`다.

```sh
journalctl -k -b | grep -E 'unexpected WPR2|RmInitAdapter|NV_ERR_GPU_IN_FULLCHIP_RESET'
```

- 위 문자열 중 하나가 출력되면 → soft-reload를 건너뛰고 4절의 **warm reboot**
  단계로 간다. warm reboot가 WPR2/fullchip을 해소하는지는 이 호스트에서 아직
  검증되지 않았으며, 남으면 4절의 cold 전원 차단이 필요하다.
- 출력이 없고 595.84/595.71.05처럼 버전만 다르면 → 3절로 간다.

## 3. 595.84 버전 불일치 soft-reload (위험: GPU 작업 중단)

**전제조건 게이트(모두 통과해야 실행):** 유지보수 창과 SSH 콘솔을 확보하고,
1-1의 WPR2/fullchip/`ERR!` 조건이 없어야 한다. 다음 명령에서 워커를 먼저 멈추고,
GPU 점유 프로세스가 0개이며 대상 BDF가 맞는지 확인한다. `<워커-컨테이너>`은
실제 `docker ps -a --format '{{.Names}}'` 결과로 바꾼다.

```sh
docker stop <워커-컨테이너>
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
lspci -Dnn | grep -i 'NVIDIA'
```

- `docker stop`이 성공하고 compute-apps 출력이 비어 있으며 `lspci`에
  `0000:02:00.0` NVIDIA BDF가 보이면 → 3-1로 간다.
- 하나라도 실패하거나 BDF가 다르면 → 명령을 진행하지 말고 **먼저 3-4절의 서비스 복귀 절차로 gdm과 워커를 즉시 복귀**한다. 그 뒤 실제 점유 프로세스/BDF를 재판단하여 3-1을 재시도하거나, 안전하게 멈출 수 없으면 4절로 간다.

### 3-1. gdm과 seat0가 잡고 있는 GPU를 해제

```sh
sudo systemctl stop gdm3
loginctl list-sessions
sudo loginctl terminate-session <seat0-세션-ID>
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

- `loginctl list-sessions`에서 seat0/tty2 사용자 세션의 ID를 찾아 `<...>`에 넣는다.
- gdm3 중지와 seat0 종료 뒤 compute-apps 출력이 비어 있으면 → 3-2로.
- `sudo systemctl stop gdm3` 자체가 실패하면 → 이 시점에는 이미 워커가 멈춰 있다.
  더 진행하지 말고 **먼저 3-4절의 서비스 복귀 절차로 gdm과 워커를 즉시 복귀**한 뒤,
  `systemctl status gdm3`로 실패 원인을 확인한다. 해소되지 않으면 4절로 간다.
- `systemctl isolate multi-user.target`만으로는 Xorg(PID 1913처럼 gdm이 띄운
  seat0/tty2 세션)이 남는다. 이것은 GPU 해제 방법이 아니다.
- 세션을 종료할 수 없거나 GPU 사용 프로세스가 남으면 → 원격 접속을 보존한 채
  사용 프로세스를 정리한다. 파괴적 단계를 포기하면 **먼저 3-4절의 서비스 복귀 절차로
  gdm과 워커를 즉시 복귀**한 뒤 원인을 재판단하여 3-1을 재시도하거나, 안전하게 해제할 수
  없으면 4절로 간다.

### 3-2. 전체 모듈 언로드 → 조건부 unbind → FLR → PCI 재탐색

> [!WARNING]
> 이 실제 성공 경로는 `nvidia-smi`를 약 10분만 살린 뒤 fullchip reset으로 다시
> 떨어진 실측이 있다. WPR2/fullchip/`ERR!`에는 이 절을 반복하지 말고 **먼저 3-4절의
> 서비스 복귀 절차로 gdm과 워커를 즉시 복귀**한 뒤 4절로 간다.

```sh
# 중단 경로에서도 3-1에서 멈춘 gdm/seat0와 워커를 방치하지 않는다.
abort_soft_reload() {
  echo 'soft-reload 중단: PCI를 더 건드리지 말고 먼저 런북 3-4절로 gdm과 워커를 복귀한다. 그 뒤 원인을 재판단해 3-1을 재시도하거나 4절로 간다.' >&2
  exit 1
}

# 1단계: rmmod 직후 잔존 모듈이 0개인지 확인한다. 하나라도 남으면 여기서 중단한다.
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia || abort_soft_reload
if lsmod | grep -q '^nvidia'; then
  abort_soft_reload
fi
lsmod | grep '^nvidia' || true
```

잔존 모듈이 **0개임을 위에서 확인한 뒤에만** 아래 2단계를 별도 실행한다. 각 PCI
변경 직전에도 같은 fail-closed 게이트를 통과해야 한다.

```sh
abort_soft_reload() {
  echo 'soft-reload 중단: PCI를 더 건드리지 말고 먼저 런북 3-4절로 gdm과 워커를 복귀한다. 그 뒤 원인을 재판단해 3-1을 재시도하거나 4절로 간다.' >&2
  exit 1
}
require_no_nvidia_modules() {
  if lsmod | grep -q '^nvidia'; then
    echo 'NVIDIA 모듈이 남아 있으므로 PCI 변경을 중단한다. 먼저 런북 3-4절로 gdm과 워커를 복귀한 뒤 원인을 재판단해 3-1을 재시도하거나 4절로 간다.' >&2
    return 1
  fi
}
require_no_nvidia_modules || abort_soft_reload
# core 모듈을 내렸으므로 unbind 경로가 아직 있을 때만 시도한다.
if [ -e /sys/bus/pci/drivers/nvidia/unbind ]; then
  require_no_nvidia_modules || abort_soft_reload
  echo 0000:02:00.0 | sudo tee /sys/bus/pci/drivers/nvidia/unbind || abort_soft_reload
fi
require_no_nvidia_modules || abort_soft_reload
test -e /sys/bus/pci/devices/0000:02:00.0/reset || abort_soft_reload
# FLR 실패는 기록하되, 모듈 0개 게이트를 다시 통과한 경우에만 remove로 진행한다.
echo 1 | sudo tee /sys/bus/pci/devices/0000:02:00.0/reset || true
require_no_nvidia_modules || abort_soft_reload
test -e /sys/bus/pci/devices/0000:02:00.0/remove || abort_soft_reload
echo 1 | sudo tee /sys/bus/pci/devices/0000:02:00.0/remove || abort_soft_reload
require_no_nvidia_modules || abort_soft_reload
test -e /sys/bus/pci/rescan || abort_soft_reload
echo 1 | sudo tee /sys/bus/pci/rescan || abort_soft_reload
sudo modprobe nvidia || abort_soft_reload
sudo modprobe nvidia_uvm nvidia_modeset nvidia_drm || abort_soft_reload
nvidia-smi
cat /proc/driver/nvidia/version
```

- `rmmod` 직후 `lsmod | grep '^nvidia'`가 비어 있지 않거나 `rmmod`가 실패하면 →
  **PCI unbind/reset/remove/rescan을 절대 실행하지 말고 먼저 3-4절의 서비스 복귀 절차로
  gdm과 워커를 즉시 복귀**한다. 그 뒤 원인을 재판단하여 3-1을 재시도하거나 4절로 간다.
- unbind 파일이 없다는 것은 core 모듈 언로드 뒤 정상적인 경우이며 건너뛴다.
- FLR `reset`의 실패는 기록만 하고, 이후에도 모듈 0개 게이트를 통과할 때만
  `remove`/`rescan`으로 진행한다. `remove` 뒤 `lspci -Dnn | grep -i NVIDIA`가 사라지고
  `rescan` 뒤 다시 보이면 → `modprobe`로 간다.
- `modprobe` 뒤 `nvidia-smi`가 GPU를 표시하고 `/proc` 버전이 `595.84`이면 → 3-3으로.
- `No devices were found`, `RmInitAdapter failed`, WPR2, `ERR!`, 또는 10분 내 재발이면 →
  **먼저 3-4절의 서비스 복귀 절차로 gdm과 워커를 즉시 복귀**한 뒤, 더 이상 반복하지 말고
  4절 warm reboot 분기로 간다.
- 롤백: 이 절의 어느 단계가 실패해도 모듈/PCI 명령을 역순으로 반복하지 않는다. **먼저
  3-4절의 서비스 복귀 절차로 gdm과 워커를 즉시 복귀**한 뒤 원인을 재판단하여 3-1을
  재시도하거나 4절로 간다.

### 3-3. 컨테이너 CDI 스펙을 현재 드라이버로 재생성

호스트가 살아도 이전 부팅 시각의 `/var/run/cdi/nvidia.yaml`이
`libEGL_nvidia.so.595.71.05`를 가리키면 컨테이너 GPU는 계속 실패한다.

```sh
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
nvidia-ctk cdi list
sh scripts/edge-preflight/check-nvidia-runtime.sh
sh scripts/edge-preflight/check-cuda-context.sh
# 새 컨테이너 생성·삭제를 허용할 때만:
bash scripts/edge-preflight/diagnose-edge.sh --with-container-probe
```

- `nvidia-ctk cdi list`와 runtime/context 스크립트가 성공(exit 0)하면 → 3-4의 서비스
  복귀와 검증으로 간다.
- 실패하면 → CDI 생성 출력과 `/var/run/cdi/nvidia.yaml`, 특히
  `check-nvidia-runtime.sh`의 실패 출력을 확인한다. 그래도 실패하거나 2-3 로그가 나타나면
  **먼저 3-4절의 서비스 복귀 절차로 gdm과 워커를 즉시 복귀**한 뒤 4절로 간다.
- CDI 생성은 `/var/run`의 런타임 파일을 현재 드라이버에 맞추는 작업이다.
  `nvidia-cdi-refresh`가 enabled여도 active/성공을 보장하지 않으므로 자동 재생성을
  전제하지 않는다.

### 3-4. soft-reload 뒤 서비스 복귀와 낙상 탐지 검증

soft-reload를 중단했더라도 3-1에서 멈춘 gdm/seat0와 워커는 **반드시 이 절에서 복귀**한다.
현재 배포는 `ML_WORKER_PROFILE` 부재와 restart policy 차이로 compose 재생성이 위험하므로,
**기존 컨테이너를 직접 시작하는 것이 1순위**다.

```sh
sudo systemctl start gdm3
systemctl is-active gdm3
docker start eldercare-fall-ml-ml-worker-1
docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} restart={{.HostConfig.RestartPolicy.Name}}' eldercare-fall-ml-ml-worker-1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
curl -sf http://127.0.0.1:8000/api/v1/status
bash scripts/edge-preflight/diagnose-edge.sh
```

- `gdm3`가 `active`, 워커가 `running`, compute-apps에 `python` 또는 `ffmpeg`가 보이고,
  status의 각 카메라가 `online`, `decode.selected=nvdec`, `fallback_count=0`이며
  진단 6단계가 통과하면 → 낙상 탐지가 복귀했다.
- 이 상태 확인 전에는 GPU만 살아난 것을 복구 완료로 선언하지 않는다. status 필드가
  없으면 1-4의 이전 `ml-api` 이미지 분기를 따른다.
- compose 경로는 `ML_WORKER_PROFILE`과 restart policy 배포 결함을 먼저 고친 뒤에만
  사용한다. 그 전에는 `docker compose up`/재생성으로 워커를 cutover하지 않는다.

## 4. WPR2/fullchip 또는 soft-reload 실패: 단계적 재부팅

**전제조건:** 1절의 읽기 전용 진단 결과를 남기고 가능한 경우 워커를 중지한다.
`nvidia-cdi-refresh.service`와 `.path`가 `enabled`여도 실제 실행 성공을 뜻하지
않는다. 특히 service가 `failed`일 수 있으므로 재부팅 뒤 active 상태를 별도로
확인해야 한다.

```sh
systemctl is-enabled ssh docker nvidia-cdi-refresh.service nvidia-cdi-refresh.path
docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' eldercare-fall-ml-ml-worker-1
```

- 각 유닛이 `enabled`, restart policy가 `unless-stopped`이면 → 원격 접속과 Docker
  자동 시작의 전제만 확인된 것이다. CDI 자동 생성 성공까지 뜻하지는 않는다.
- 어떤 유닛이 `disabled` 또는 restart policy가 `no`면 → 재부팅 전에 현장/원격 관리
  복구 수단을 준비한다. 현재 컨테이너를 compose로 성급히 재생성하지 않는다
  (상단 경고 참조).

### 4-1. 먼저 warm reboot (원격에서 가능한 저비용 단계)

```sh
sudo reboot
```

부팅 후 다음을 확인한다.

```sh
nvidia-smi
cat /proc/driver/nvidia/version
systemctl is-active nvidia-cdi-refresh.service nvidia-cdi-refresh.path
nvidia-ctk cdi list
sh scripts/edge-preflight/check-nvidia-runtime.sh
sh scripts/edge-preflight/check-cuda-context.sh
docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} restart={{.HostConfig.RestartPolicy.Name}}' eldercare-fall-ml-ml-worker-1
docker logs --tail 200 eldercare-fall-ml-ml-worker-1
curl -s http://127.0.0.1:8000/api/v1/status
```

- `nvidia-smi`가 정상 GPU 표를 보이고 `ERR!`/WPR2가 없으며 CDI service가 `active`,
  runtime/context가 성공하면 → **3-4의 서비스 복귀와 낙상 탐지 검증을 반드시
  수행한다.** 워커가 자동으로 `running`이어도 gdm, 카메라 status, NVDEC, 진단 6단계를
  확인하기 전에는 warm reboot 성공으로 끝내지 않는다.
- `nvidia-cdi-refresh.service`가 `failed` 또는 `inactive`이면 → 자동 재생성을
  가정하지 말고 아래 수동 폴백을 실행한 뒤 `nvidia-ctk cdi list`와 runtime 검사를
  다시 한다.
  ```sh
  sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
  ```
- warm reboot 뒤에도 `nvidia-smi`의 `ERR!`, WPR2/fullchip 로그,
  `NV_ERR_GPU_IN_FULLCHIP_RESET`, 또는 GPU 초기화 실패가 남으면 → 4-2로 간다.
  warm reboot는 PCIe 보조전원 제거를 보장하지 않으며 이 증상을 해소한다고 검증되지
  않았다.
- **실측된 warm-reboot 실패 서명(2026-07-29).** warm reboot 뒤 `nvidia-smi`가
  `ERR!`가 아니라 **`No devices were found`**를 내고, 커널 로그에 다음이 남았다.
  ```
  NVRM: GPU0 gpuHandleSanityCheckRegReadError_GH100: Possible bad register read:
        addr: 0x110810, regvalue: 0xbadf4100
  NVRM: GPU0 nvCheckOkFailedNoLog: Call timed out [NV_ERR_TIMEOUT] (0x00000065)
        returned from kgspWaitForGfwBootOk_HAL(...) @ kernel_gsp.c:4945
  NVRM: GPU0 RmInitAdapter: Cannot initialize GSP firmware RM
  NVRM: GPU 0000:02:00.0: RmInitAdapter failed! (0x62:0x65:2168)
  ```
  `0xbadf____`는 GPU가 PRI 버스에 응답조차 하지 않는 값이고, `kgspWaitForGfwBootOk`
  타임아웃은 **GPU 자체 펌웨어(GFW) 부팅이 끝나지 않았다**는 뜻이다. 직전 WPR2
  오류(`0x62:0x40:2168`)보다 더 낮은 계층이며 PCI 장치는 여전히 `nvidia`에
  바인딩되어 있다. 이 서명이 보이면 **soft-reload도 warm reboot도 반복하지 말고
  즉시 4-2(완전 전원 차단)로 간다.**
- **실측 확정 진단(2026-07-29 10:57) — Xid 143 / FSP boot 실패는 하드웨어다.**
  실제 전원 차단(10:51~10:53, 약 2분) 뒤에도, 그리고 cold boot 직후 PCI
  remove/rescan을 다시 해도 아래가 남으면 **소프트웨어로 고칠 수 없다.**
  ```
  NVRM: Xid (PCI:0000:02:00): 143, Error status 0x65 while polling for FSP boot complete
  NVRM: GPU0 nvCheckOkFailedNoLog: Call timed out [NV_ERR_TIMEOUT] (0x00000065)
        returned from kgspWaitForGfwBootOk_HAL(...)
  NVRM: GPU 0000:02:00.0: RmInitAdapter failed! (0x62:0x65:2168)
  ```
  FSP는 GSP보다 **먼저** 부팅해야 하는 보드 내장 보안 프로세서다. FSP boot가
  완료되지 않으면 그 위 어떤 초기화도 불가능하다. 이때 PCIe 링크는 정상일 수
  있으니(`power_state D0`, `Speed 32GT/s`, `Width x16`, AER 오류 없음, BAR 정상 매핑)
  링크가 멀쩡하다고 소프트웨어 문제로 오판하지 마라.
  **다음은 전부 물리 작업이며 이 순서로 한다:**
  1. 보조전원 커넥터 재체결 — 12VHPWR/16핀은 **완전히 딸깍 소리가 날 때까지** 밀어
     넣는다. 부분 체결이 이 증상의 가장 흔한 원인이다.
  2. 카드 재장착(reseat). 슬롯 접점을 확인한다.
  3. 다른 PCIe 슬롯으로 옮겨 본다.
  4. 다른 메인보드/PC에서 부팅해 본다. 거기서도 Xid 143이면 카드 고장이 확정된다.
  5. 확정되면 RMA. 그때까지 탐지는 복구할 수 없다.

### 4-2. warm reboot 뒤에도 WPR2/fullchip이 남을 때: cold 전원 차단

완전 전원 차단(cold power cycle)은 호스트 전원을 끄고 PCIe 보조전원이 실제로
제거된 뒤 다시 켜는 작업이다. 원격 `sudo reboot`로 대체할 수 없다. **물리적 접근
또는 BMC/원격 전원 관리가 필요하다.** 현장 담당자 또는 BMC 절차로 cold power
cycle을 수행한 뒤 4-1의 부팅 후 검사를 처음부터 반복한다.

- cold power cycle 뒤 `nvidia-smi` 정상·`ERR!`/WPR2 없음이면 → CDI service 상태와
  runtime/context를 4-1 순서로 확인한 뒤, **3-4의 서비스 복귀와 낙상 탐지 검증**을
  수행한다.
- 그래도 WPR2/fullchip 또는 GPU 초기화 실패가 남으면 → 하드웨어/전원 경로 장애로
  보고하고 모듈 재적재·재부팅을 반복하지 않는다.

## 5. 6일 동안 워커가 멈춘 이유와 재발 방지

2026-07-22에 워커는 외부 SIGKILL로 종료(exit 137, `OOMKilled=false`)했고, 당시
`RestartPolicy=no`라 자동 재시작되지 않아 6일 동안 방치됐다. 같은 날 클립 3건도
`state: UNAVAILABLE`로 남았다. 현재 워커의 정책은 `unless-stopped`로 바꿔 두었다.

```sh
docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} restart={{.HostConfig.RestartPolicy.Name}}' eldercare-fall-ml-ml-worker-1
docker logs --tail 200 eldercare-fall-ml-ml-worker-1
```

- `restart=unless-stopped`이면 → 외부 SIGKILL 뒤 Docker가 워커를 재시작할 수 있다.
- `restart=no`이면 → 자동 복구가 없다. 상단의 compose 재현성 경고를 해결하기 전
  compose 재생성으로 정책만 바꾸지 말고, 배포 설정을 별도로 바로잡는다.
- `exit=137 oom=false`이면 → OOM으로 단정하지 않는다. 외부 kill, 호스트 로그,
  Docker 이벤트를 함께 조사한다.

## 6. 기존 580.x/CUDA 13.0 정렬 후 이미지 검증

2-2의 드라이버/CUDA 조합 변경을 실제로 수행한 경우에만 아래를 쓴다. 이 절은
595.84 soft-reload 또는 WPR2 재부팅의 대체 절차가 아니다.

```sh
docker build -f Dockerfile.edge --target runtime -t local/fall-ml-worker:failfast .
docker run --rm --gpus all local/fall-ml-worker:failfast \
  python -c "import torch; assert torch.cuda.is_available(); print('OK', torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))"

```

- 컨테이너 명령 exit 0이고 `OK` 및 `sm_120`이 출력되면 → GPU 이미지 검증 성공이다.
- 실패하면 → worker를 cutover하지 말고 `check-cuda-context.sh`와 드라이버 조합을
  다시 확인한다.
- 검증 성공 후에도 현재 배포 재현성 결함(`ML_WORKER_PROFILE` 부재와 restart policy)을
  먼저 별도 배포 작업으로 고친다. 그 작업에서 현재 이미지에 rollback tag를 붙이고,
  수정된 compose 설정으로만 워커를 재생성·cutover한 뒤 3-4의 검증과 실제 스트림
  smoke를 수행한다.
  ```sh
  docker tag "$(docker inspect -f '{{.Config.Image}}' eldercare-fall-ml-ml-worker-1)" local/fall-ml-worker:rollback
  # ML_WORKER_PROFILE 및 restart policy를 고친 배포 설정에서만 실행:
  docker compose --env-file .env.edge.deploy -f compose.edge.yaml -f compose.edge.local.yaml up -d --force-recreate ml-worker
  bash scripts/edge-preflight/diagnose-edge.sh
  ```
- smoke 또는 진단이 실패하면 rollback tag와 **수정된** 배포 설정으로 되돌린 뒤 원인을
  조사한다. 결함이 남은 현재 compose 설정으로는 cutover/rollback하지 않는다.
