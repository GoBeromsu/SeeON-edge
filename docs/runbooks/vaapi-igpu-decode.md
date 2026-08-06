# Intel iGPU(VAAPI) RTSP 디코드 — 선택/검증/폴백 확인

`igpu` 프로필은 RTSP **디코드만** Intel iGPU(VAAPI)로 오프로드한다. 추론은
이 PR 기준으로 여전히 CPU에서 돈다 — OpenVINO GPU/NPU 추론은 별도 후속
작업이며, 이 문서가 다루는 범위가 아니다
(`worker/runtime/profile/registry.py`의 `"igpu"` `ProfileSpec`: `device="cpu"`).

## 프로필 선택

```bash
docker compose --env-file .env.edge.prod \
  -f compose.edge.yaml -f compose.edge.igpu.yaml up -d
```

`.env.edge.prod`에는 최소한 다음이 필요하다:

```bash
ML_WORKER_PROFILE=igpu
```

`compose.edge.igpu.yaml`이 `deploy: !reset null`로 `compose.edge.yaml`의
무조건적 NVIDIA GPU 예약을 지운다(Intel iGPU 노드에는 NVIDIA 디바이스가
없다 — `compose.edge.cpu.yaml`과 동일한 처리). 그리고 `/dev/dri`를
컨테이너에 통과시키고, `render`/`video` 그룹을 `group_add`로 추가하며,
`LIBVA_DRIVER_NAME=iHD`를 설정한다.

카메라별 `decode_backend`는 지정하지 않거나(`auto`) `opencv`/`cpu`로 둔다.
`vaapi`는 카메라 단위로 오버라이드할 수 없다 — 프로필을 통해서만 선택된다
(`worker/runtime/config/camera_models.py`의 `decode_backend` 검증기가
`auto, nvdec, opencv, cpu`만 허용한다. 이는 의도된 것이다: iGPU 오프로드는
호스트 전체의 하드웨어 가용성 문제이지, 카메라별로 켜고 끌 대상이 아니다).

## 호스트 사전 준비물

- `/dev/dri/renderD128` (또는 동등한 렌더 노드)가 호스트에 존재해야 한다.
- iHD VAAPI 드라이버(`intel-media-va-driver-non-free`)가 **이미지 안에**
  설치돼 있어야 한다 — `Dockerfile.edge`가 빌드 시점에 설치한다. 별도로
  호스트에 설치할 필요는 없다(디바이스 노드만 통과시키면 된다).
- 컨테이너를 구동하는 사용자가 `/dev/dri/renderD128`의 그룹(대개
  `render`, 배포에 따라 `video`)에 접근 가능해야 한다. 정확한 GID는 호스트
  마다 다를 수 있으므로 배포 전에 확인한다:

  ```bash
  getent group render video
  ```

  `compose.edge.igpu.yaml`의 `group_add` 값(기본 `104`/`44`)이 실제 호스트
  값과 다르면 그에 맞게 오버레이를 수정한다.

## iGPU를 실제로 쓰고 있는지 검증

### 1) 컨테이너 안에서 `vainfo`

```bash
docker compose -f compose.edge.yaml -f compose.edge.igpu.yaml \
  exec ml-worker vainfo --display drm --device /dev/dri/renderD128
```

`iHD` 드라이버 이름과 지원 프로파일(`VAProfileH264...`, `VAProfileHEVC...`
등) 목록이 출력되면 드라이버가 정상적으로 로드된 것이다. 이 명령이
실패하면(드라이버 초기화 오류) VAAPI 자체가 컨테이너 안에서 동작하지
않는다는 뜻이므로, 워커가 부팅 시 CPU 디코드로 폴백했을 것이다 — 아래
"폴백 시 확인할 것" 참고.

### 2) 호스트에서 `intel_gpu_top`

```bash
intel_gpu_top
```

(호스트에 `intel-gpu-tools` 설치 필요, 컨테이너 밖에서 실행). 카메라가
실제로 iGPU 디코드 중이면 `Video` 엔진 사용률이 0%보다 유의미하게 높게
나타난다. 워커가 조용히 CPU 폴백했다면 `Video` 엔진은 계속 유휴 상태이고
CPU 사용률만 올라간다 — 이 두 지표를 함께 보는 것이 "정말 iGPU를 쓰고
있는가"에 대한 가장 직접적인 확인이다.

### 3) 워커 로그 / 진단 API

부팅 시 VAAPI 프리플라이트가 통과하면 별도 WARNING 없이 `igpu` 프로필로
정상 부팅한다. 실패 시 다음 형태의 WARNING 한 줄이 남는다(자세한 내용은
아래 "폴백 시 확인할 것" 참고):

```text
profile 'igpu' decode preflight failed (<reason>); falling back to opencv (CPU) decode
```

런타임 진단(`WorkerDiagnostics`/relay 페이로드)의 카메라별
`decode.selected` 필드도 실제 사용 중인 백엔드(`vaapi` 또는 `opencv`)를
그대로 반영한다.

## 폴백 시 확인할 것

VAAPI를 쓸 수 없으면(드라이버 없음, `/dev/dri` 없음, 지원하지 않는 코덱)
워커는 크래시하거나 프레임 없이 조용히 죽지 않는다 — 부팅 시 위 WARNING을
남기고 `opencv`(CPU) 디코드로 자동 전환한다
(`worker/runtime/profile/boot.py`의 `resolve_decode_or_fallback`). 이슈
#191/#194가 지적한 "조용한 무-프레임 실패"를 반복하지 않기 위한
설계다 — 하지만 폴백 자체가 정상 동작이라는 뜻은 아니다. iGPU 오프로드가
실제로 필요한 배포라면 다음을 순서대로 확인한다:

1. **`/dev/dri` 존재 여부** — 호스트에서 `ls -la /dev/dri`. 렌더 노드가
   없으면 커널 i915 드라이버가 로드되지 않았거나 iGPU가 비활성화된
   것이다.
2. **드라이버 설치 여부** — 컨테이너 안에서 `vainfo` 실행(위 1번 검증
   절차). "Failed to initialise VAAPI connection" 류의 오류가 나오면
   `intel-media-va-driver-non-free`가 이미지에 실제로 설치돼 있는지
   `Dockerfile.edge` 빌드 로그를 확인한다.
3. **그룹 권한** — `group_add`에 넣은 GID가 실제 `/dev/dri/renderD128`의
   그룹 GID와 일치하는지 호스트에서 `stat /dev/dri/renderD128` 및
   `getent group render video`로 대조한다.
4. **코덱 미지원** — 특정 카메라의 코덱을 VAAPI가 지원하지 않는 경우는
   부팅 시점의 호스트-레벨 프리플라이트로는 잡히지 않는다(프리플라이트는
   더미 디바이스 초기화만 확인하지, 카메라별 스트림 코덱까지는 모른다).
   이 카메라는 개별적으로 열기에 실패할 수 있다 — 알려진 한계이며, 카메라
   단위 폴백은 이번 PR의 범위 밖이다(별도 이슈로 추적).

## 하드웨어 검증 상태

이 PR은 macOS 개발 환경에서 작성됐고, VAAPI/iGPU 하드웨어에 접근할 수
없다. 위 절차(디코드 성공, `vainfo` 출력, `intel_gpu_top` 엔진 사용률,
폴백 경로의 실제 트리거)는 **실제 Intel Arrow Lake 엣지 노드에서
검증되지 않았다.** 배포 전 실제 하드웨어에서 반드시 확인해야 한다.
