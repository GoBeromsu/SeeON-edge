# 엣지 브링업 인계 (2026-08-17)

과다 알림 진단에서 시작해 운영 스택 복구까지 진행한 작업의 인계 문서다.
이 저장소(`SeeON-edge`) 체크아웃에서 이어서 작업할 수 있도록 현재 상태, 미해결
문제, 다음 순서를 남긴다.

## 지금 돌아가는 것

운영 스택은 **compose 로 기동되어 실제 시설에 연결돼 있다.**

```
프로젝트   seeon-edge-wt-alert-api
디렉터리   ../SeeON-edge-wt-alert-api   (진단용 worktree — 정리 대상)
서비스     ml-api (healthy), ml-worker
대시보드   http://100.96.162.127:8000/   (tailscale, 평문 HTTP)
카메라     13대 online, registry_version 94
Hub        heartbeat 202, reachable
임계값     13대 fall operating_threshold 0.8 (camera-override)
```

### 즉시 알아야 할 두 가지

**1. 실행 중인 이미지가 `main` 보다 낡았다.** `b14420a`(#303/#304 보안 수정)
**이전** 코드로 빌드됐다. 대시보드 표면의 릴레이 자격증명 노출과 조용한 레거시
인증 폴백이 아직 남아 있는 상태로 운영 중이다. 재빌드 전까지 이 사실을 전제로 다뤄야
한다.

**2. 지금 Hub 연결이 살아 있는 이유는 커밋에 없다.** 레거시 카메라 레지스트리
(13대 + `backend_camera_id` 매핑)를 `edge-state` 볼륨에 **수작업으로 복원**해서
동작한다. 저장소에서 재현되지 않는다. 백업:

```
../seeon-edge-state-backup-20260817/edge-state.tar.gz
../seeon-edge-state-backup-20260817/compose.env
```

절차는 `docs/runbooks/edge-redeploy-identity-continuity.md` 참조.

## 해결된 것

| 항목 | 결과 |
| --- | --- |
| 아티팩트 핀이 조달 불가 값을 가리켜 워커 부팅 불가(exit 3) | #307 병합 |
| 평문 HTTP 접속 시 설정 페이지 크래시(`crypto.randomUUID`) | #307 병합 |
| 재배포 시 카메라 신원 상실 절차 부재 | #309 (PR) |

## 미해결

우선순위 순.

### #308 클린 상태에서 카메라 sync 불가 — **최우선**

깨끗한 엣지에서 카메라를 등록하면 `backend_camera_id` 가 비고, 이를 채우는 유일한
경로인 topology sync 가 `CONFLICT` 로 멈춘 뒤 해제되지 않는다. 신규 시설 설치와
재배포가 모두 막힌다.

이게 풀리기 전에는:

- 재배포 때마다 볼륨 보존에 의존해야 한다
- Hub 주소 이전(아래 sslip.io 문제)을 진행할 수 없다

### 처리량 — 낙상 탐지가 실질적으로 불가능

측정값:

```
추론 버스   published 20,544 / taken 295 / dropped 20,248   (드롭 98.6%)
pose        프레임당 14.6초  (카메라 1~2대일 때는 21~23ms)
워커 CPU    132% = 24코어 중 1.3개
torch       24 스레드 정상 설정
```

fall 모델은 `window: 30, stride: 5` 로 **연속 30프레임**을 요구한다. 프레임당 14.6초면
윈도 하나가 **7분 18초**를 덮는다. 낙상은 1~2초다. 즉 현재 입력은 학습 시 시간 구조와
무관하며, **어떤 임계값이나 가중치를 써도 낙상을 탐지할 수 없다.**

CPU 포화가 아니므로(24코어 중 1.3개) 병목은 추론이 아니라 그 앞단으로 보인다. 카메라는
HEVC 640x360@30fps × 13대 = 초당 390프레임 H.265 소프트웨어 디코드다. 이 호스트에는
Intel Arrow Lake iGPU 와 `compose.edge.igpu.yaml`(`LIBVA_DRIVER_NAME: iHD`)이 있는데
`cpu` 프로파일로 돌고 있다.

GPU 를 올리더라도 **디코드를 어디서 할지**를 함께 정해야 효과가 난다. 프로파일만
바꾸면 디코드는 여전히 CPU 일 수 있다.

### #306 fall `operating_threshold` 죽은 배관

네 곳이 소유권을 주장하고 셋은 판정에 쓰이지 않는다. 로그가 실제 게이트와 다른 값을
찍어 진단을 오도한다. ralplan 합의에서 코드 변경 설계는 원칙적 승인, 배포는 별도
결함으로 미승인.

### #310 릴레이 표면 바인딩 분리

릴레이와 대시보드가 같은 published 포트에 섞여 있어 공유 비밀로만 구분된다. 내부
네트워크 바인딩으로 경계를 강제하면 relay token 이 불필요해진다. `worker-config` 가
RTSP 자격증명을 평문 반환하는 문제도 함께 처리해야 한다.

### Hub 주소가 `sslip.io`

```
https://49-247-204-81.sslip.io   (Let's Encrypt, SAN 1개)
```

`sslip.io` 는 IP 를 호스트명으로 바꿔주는 공개 와일드카드 DNS 다. 서드파티 DNS 가 전
시설 안전 데이터 경로의 단일 장애점이고, Hub IP 가 바뀌면 호스트명도 바뀌어 모든 엣지의
재등록이 필요하다. #308 이 선행되어야 이전할 수 있다.

### 배포 자동화 결함

edge updater 가 updater 완료 전에 스케줄러를 복구하고, CHECK/PULL 실패 시 이미지 참조를
롤백하지 않는다. 승인되지 않은 이미지 쌍이 나중에 스케줄로 배포될 수 있다. ralplan 이
프로덕션 배포를 승인하지 않은 사유이며 아직 이슈로 등록되지 않았다.

## 정리 대상

진단 과정에서 만든 것들이다. 운영 스택이 worktree 에서 돌고 있으므로 순서가 있다.

1. 이 저장소(머지된 `main`)에서 재빌드
2. **볼륨을 유지한 채** 프로젝트 이관 (카메라·매핑 보존 — 런북 참조)
3. worktree `../SeeON-edge-wt-alert-api` 제거
4. 진단 이미지·볼륨 정리:
   `local/fall-ml-{api,worker}:{diag-alert-api,diag-pinfix,41d2a0d,settingsfix}`,
   `seeon-prod-edge-state`, `seeon-prod-legacy-*`, `edge_edge-state`
5. 백업은 #308 해결 전까지 유지

2번이 위험 구간이다.

## 운영자 조치 (코드 아님)

- **카메라 계정 자격증명 로테이션.** 진단 중 RTSP URL 이 자격증명째로 세션 기록에
  출력됐다. 디스크·보고서·저장소에는 기록되지 않았으나 자격증명은 교체해야 한다.

## 진단 자료

- 과다 알림 분석 보고서: `../alert-amplification-diagnosis-report.md` (저장소 외부)
- 진단 하네스: `tests_support/alert_amplification_*.py`,
  `scripts/ops/alert-amplification-diagnostic.py` (#307 로 병합됨)
- 오프라인 Hub fixture: `tests_support/local_backend_fixture.py` — 라이브 시설을 건드리지
  않고 enrollment/토폴로지 흐름을 재현할 때 사용한다. 같은 시설에 두 번째 엣지를
  enroll 하면 소유권 이전으로 운영 엣지가 밀려날 수 있다.
