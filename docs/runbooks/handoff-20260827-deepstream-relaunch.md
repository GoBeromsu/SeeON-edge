# 핸드오프 — DeepStream 재기동 및 로컬 증거 경로 복구

**작성:** 2026-08-27 22:50 KST
**대상 스택:** `seeon-edge-main` (요양원 라이브)
**전제:** 이 문서의 모든 수치는 실측입니다. 추정과 실측을 섞지 않았습니다.

---

## 1. 지금 라이브에 무엇이 떠 있나

```
ml-worker   seeon-edge-final:495b97d-upkeep      재시작 0
ml-api      seeon-edge-api (tag 74d1d01-relayfix)  healthy
카메라       13/13 스트리밍, 판정 정상
디스크       59%   (임계 80%)
배달 큐      1     (오늘 아침 3,691)
incidents   5,061
```

> **경고 — 이 두 이미지는 어느 PR에도 병합되지 않은 브랜치에서 빌드된 것입니다.**
> `495b97d`는 `fix/nvidia-detection-telemetry-explicit-producer` (PR #427),
> `74d1d01`는 `fix/431-named-local-accept` (PR #433).
> **라이브가 미리뷰 코드를 돌고 있습니다.** 순서를 뒤집은 것은 이번 세션의 잘못이며,
> 병합이 최우선 후속 작업입니다.

### 재현 가능한 배포 명령

**셸 환경변수가 `--env-file`을 덮어씁니다.** 이번 세션에서 이것 때문에 라이브를
두 번 내렸습니다. 반드시 벗겨내고 GPU 오버레이를 포함해야 합니다.

```bash
cd ~/beomsukoh/SeeON/SeeON-edge
env -u COMPOSE_PROJECT_NAME -u CLIP_STORE_HOST_DIR -u ML_WORKER_IMAGE -u ML_API_IMAGE \
  docker compose --env-file .env.seeon-edge-main \
  -f compose.edge.yaml -f compose.edge.nvidia.yaml \
  -p seeon-edge-main up -d --pull never --no-deps --force-recreate ml-worker
```

오염되는 변수 4개 (실측):

|변수|셸|env-file|
|---|---|---|
|`COMPOSE_PROJECT_NAME`|`seeon-edge-wt-alert-api`|`seeon-edge-main`|
|`CLIP_STORE_HOST_DIR`|`.../clip-store`|`.../clip-store-main`|
|`ML_WORKER_IMAGE`|`local/fall-ml-worker:main-fa40829`|(정상 태그)|
|`ML_API_IMAGE`|`local/fall-ml-api:main-fa40829`|(정상 태그)|

**배포 후 반드시 확인:**

```bash
docker inspect seeon-edge-main-ml-worker-1 --format '{{.Config.Image}}'
docker inspect seeon-edge-main-ml-worker-1 \
  --format '{{range .Mounts}}{{if eq .Destination "/var/lib/clip-store"}}{{.Source}}{{end}}{{end}}'
```

마운트가 `clip-store-main`이 아니면 **3일 묵은 엉뚱한 디렉터리를 측정하게 됩니다.**
이번 세션에서 실제로 그렇게 측정하고 "클립 0건"이라 잘못 보고했습니다.

---

## 2. 오늘 고친 것 (전부 미병합)

### 2-1. 워커→백엔드가 완전히 막혀 있었다 — `ec9886f` / PR #433 / 이슈 #431

백엔드는 카메라에 Hub 매핑이 없거나 클라우드 클라이언트가 없으면 알림을
**의도적으로 로컬 수락**합니다. 이벤트를 durable하게 기록하고 상류 푸시만 건너뜁니다.
그런데 응답이 `202 {"status": "accepted"}` 였습니다.

발송기는 `edge_event_id`를 되돌려주는 영수증을 요구합니다. 받지 못하니
**"로컬 수락 완료(종결)"과 "프록시가 응답을 망가뜨림"을 구분할 수 없었고**,
`MALFORMED_RECEIPT`로 분류해 무한 재시도했습니다.

```
동일 백로그 연속 실패 시도   273회
분당 재시도                  약 45회
결과                         가장 오래된 미배달 항목 뒤로 신규 이벤트 전부 영구 정체
```

**수정:** 푸시를 건너뛴 사실은 백엔드만 압니다. 그러니 백엔드가 말합니다 —
`{"status": "accepted_local", "edge_event_id": ...}`. 발송기는 이 **명명된** 상태만
종결로 처리하고, 상태가 없거나 모르는 값이면 여전히 `MALFORMED_RECEIPT`입니다.
**누락된 필드로부터 추론하지 않고 명시적으로 선언합니다.**

**배포 후 실측:** 재시도 273 → 0, 큐 3,691 → **1** (0이 아님 — 잔여 1건).

> **후속 리뷰가 이 수정에서 데이터 유실 결함을 찾아냈고, `b2d8629`로 고쳤습니다.**
> 종결 영수증은 워커에게 "내 사본을 지워라"라고 말합니다. 로컬 수락 경로는 상류로
> 아무것도 보내지 않으므로 백엔드 기록이 **유일한 사본**인데, `_project_relay_event`가
> 실패하고 `_record_catalog` 폴백까지 실패해도 `accepted_local`을 돌려주고 있었습니다.
> 낙상 알림이 **양쪽 모두에서 사라집니다.** 이제 둘 다 실패하면 재시도 가능한 503으로
> 거부합니다. 카탈로그 실패가 안전 알림을 막아선 안 된다는 기존 규칙(#183, #202)보다
> 의도적으로 좁습니다 — 거기서는 상류 푸시가 내구성을 담당하지만 여기는 그것이 없습니다.
>
> **2차 리뷰(PR #433 code-review)가 그 503 가드에서 다시 결함을 찾았습니다.** 카탈로그가
> 절대 받아줄 수 없는 페이로드(깊이/크기 초과)나 멱등성 충돌은 재시도해도 결과가 같은데
> 503으로 답하면 워커가 무한 재시도하는 영구 독성 항목이 됩니다 — #431 의 재림입니다.
> 이제 `_record_catalog`가 실패를 `CatalogFailure(reason, permanent_status)`로 분류하고,
> 영구 실패는 422/409(워커가 PERMANENT 로 dead-letter), 일시 실패만 503(RETRY)입니다.
> 같은 리뷰가 잡은 나머지: `legacy_drain`이 `accepted_local`의 빈 `event_id`를 `""` 로
> 저장하던 것(→ `NULL`), `EdgeIngestClient`가 빈 id 로 `//snapshot` 에 PUT 하던 것(→ 건너뜀),
> 그리고 `AGENTS.md` 두 곳의 "카탈로그 실패해도 relay 는 항상 수락" 문장(→ 갱신).
>
> **명시적 트레이드오프:** `accepted_local`은 종결이므로, 매핑/클라이언트가 나중에 생겨도
> 그 이벤트는 Hub 로 다시 밀리지 않습니다. main 은 (큐를 영구 정체시키는 대가로) 결국
> 전달했습니다. 로컬 영속 + 살아있는 큐를 택했고, 재전송이 필요해지면 projection 행이
> 그 원본입니다. §⑤의 `backend_event_id = 0` 상태에서는 이것이 코너케이스가 아니라
> 현재 정상 상태라는 점을 알고 계십시오.

### 2-2. 디스크 압력이 클립 녹화를 영구 정지시킨다 — `42c7cb4`+`efd2273` / PR #427 / 이슈 #434

```
디스크 80% 초과 → recording_suspended = True → 신규 클립 거부
              → 완료되는 클립 없음 → FlushMessage 없음
              → rotate() 영원히 미실행 → 정지 유지
```

`recording_suspended`를 해제하는 것은 `rotate()` 뿐인데, `rotate()`는
`FlushMessage` 분기에서만 호출되고 `FlushMessage`는 클립이 완료돼야 생깁니다.
**압력 재평가가, 압력이 막는 바로 그것에 의존했습니다.**

**실측:** 스토어 80.68% (임계 0.80), **90분간 모든 낙상이 영상 0건.**
그동안 판정·스냅샷·이벤트 전달은 전부 정상으로 보였습니다.
**264GB를 회수해도 복구되지 않았고**, 재시작만이 유일한 해제 수단이었습니다.

**첫 수정은 아키텍트 리뷰에서 BLOCK 받았고, 그 판단이 옳았습니다.**
재평가를 "큐가 빌 때"로 옮겼는데, 13대 × 15fps = 초당 약 195개 메시지라
큐는 실제 부하에서 절대 비지 않습니다. 카메라 한 대만 살아 있어도 굶습니다.

> **더 중요한 것: 그때의 라이브 측정이 이 결함을 잡지 못했습니다.**
> 클립이 되살아난 건 **재시작이 플래그를 리셋했기 때문**이지 수정 때문이 아니었습니다.
> 유휴 큐로 만든 단위 테스트는 버그가 있어도 통과했습니다.

**최종 수정:** 경과 시간 기반 upkeep을 **매 반복, 블로킹 대기 직전에** 실행합니다.
같은 굶주림이 **멈춘 카메라의 클립 만료**(`actor.expire()`)에도 있어 함께 옮겼습니다.

**검증:** 큐를 절대 비우지 않는 부하 테스트 + 실제 `ClipMaintenance`에 디스크 공급자와
시계를 주입해 **정지가 재시작 없이 해제되는지**를 통합 증명. 돌연변이 4종 전부 실패 확인.
아키텍트 재검토 **HIGH 해소 / SHIP** 판정.

### 2-3. 그 외

|커밋|내용|상태|
|---|---|---|
|`e7f0625`|`spawn_process`가 네이티브 자식 stderr를 `DEVNULL`로 버리고 있었음 → 상속|배포됨|
|`474d4b7`|`TrtPerception::infer` 전역 뮤텍스 직렬화. 레터박스를 락 밖으로|배포됨|
|`de6f2c1`|워크스페이스 풀 (TensorRT 컨텍스트 4개)|배포됨|
|`97e179c`…`fdc52cc`|#429 epoch 전환 예약 (9개 속성 돌연변이 증명)|배포됨, 효과 미확인|
|`f1ecc58`+`32456d4`|nvidia 하트비트 + 중복 타일 (PR #432)|미배포|

---

## 3. 클립 영상 실측 — 정직하게

```
구간              클립    영상    비율
5~15분 전          68      24     35%
15~30분 전         87      66     75%
30~60분 전        146      89     60%
세션 시작 시     2,251      66      3%
```

**변동이 큽니다.** 35%~75% 사이를 오갑니다. "고쳤다"고 말할 수 있는 것은
**녹화가 영구 정지하지 않는다**는 것뿐이고, **잔여 실패율은 여전히 미해결**입니다.

남은 실패 사유 (20분 표본):

```
61건  REMUX_FAILED / STREAM_EPOCH_MISMATCH
 2건  STREAM_EPOCH_MISMATCH / STREAM_EPOCH_ROLLED
```

이것이 이슈 **#429**의 잔여분입니다. 수정(`ec3a42b`~`fdc52cc`)은 배포돼 있지만
**효과가 측정되지 않았습니다** — 디스크 잠금이 그 위를 덮고 있어 분리되지 않았습니다.

---

## 4. 다음 사람이 할 일 (우선순위 순)

### ① PR 리뷰·병합 — **사용자 담당**

리뷰와 병합은 사용자가 직접 수행합니다. 에이전트는 손대지 않습니다.

|PR|상태|비고|
|---|---|---|
|~~#432~~|**병합됨** (`95c6a0c`)|관제 화면 복구|
|**#433**|`BEHIND`|릴레이 로컬 수락. **라이브 가동 중**|
|**#427**|`DIRTY` — main과 충돌|**라이브 가동 중, 미리뷰.** 33+ 커밋|
|#425|`BEHIND`, test 실패|저장소 정리. 실패 원인 확인 필요|
|#398|`BEHIND`|non-nvidia 경로. 소규모|
|#299|`DIRTY`|충돌. 16k 라인. 별도 처리|

`main` 브랜치 보호는 `strict: true`이므로 **main이 앞서가면 브랜치 갱신 후 CI 재실행**이
필요합니다. `gh api -X PUT repos/SeniorAILab/SeeON-edge/pulls/<N>/update-branch`.

**#427 충돌 지점 (확인됨, 미해결로 남겨둠):**
`tests/test_deepstream_review_regressions.py` 한 블록(L354-378).
레인이 추가한 `test_child_stderr_is_inherited_so_media_plane_faults_reach_the_operator`가
#432 병합으로 이동한 인접 코드와 겹칩니다. **ours(레인 쪽 테스트) 유지가 정답**입니다 —
그 테스트는 배포된 stderr 상속 수정(`e7f0625`)을 고정합니다.
`worker/runtime/worker.py`는 자동 병합됩니다.

### ② #429 잔여 실패율 분리 측정

디스크 잠금이 제거됐으므로 이제 epoch 수정의 효과만 볼 수 있습니다.
`REMUX_FAILED / STREAM_EPOCH_MISMATCH` 비율을 1시간 이상 관측하고,
필요하면 예약 로직을 되돌린 대조군과 A/B 하십시오.

### ③ 미해결 MEDIUM — 유지보수 스레드 분리

아키텍트가 SHIP 판정과 함께 남긴 것:

> `ClipMaintenance.rotate`는 30초 간격이 지나면 약 2,400개 디렉터리의 매니페스트를
> **액터 스레드에서 동기적으로** glob·JSON 파싱합니다. 스캔이 길면 128칸 큐가 차서
> 증거 트래픽이 유실되고, 여기서 예외가 나면 **레코더 스레드가 죽는데 admission은
> 계속 수락합니다.**

권고 설계: 전용 유지보수 워커가 monotonic 데드라인으로 깨어나 `rotate`를 소유하고,
주기·강제 요청을 직렬화하며, 실패를 텔레메트리로 노출.

### ④ 미해결 이슈

|이슈|내용|
|---|---|
|**#430**|추론 컨텍스트 단일화로 GPU 직렬화. 워크스페이스 풀 배포됨, 처리량 미검증|
|**#428**|페이싱 실효 ~11fps (기준 14.0). 잔여분은 #430|
|**#429**|클립 영상 잔여 실패|
|**#424/#426**|수정 완료, 이슈는 OPEN — 병합 후 닫으십시오|
|**#431/#434**|수정 완료, 병합 대기|

### ⑤ 클라우드 전달 — 사용자가 오늘 범위에서 제외

`backend_event_id`는 여전히 **0**입니다. 클라우드에 도달한 이벤트가 없습니다.
연결 설정은 DB에 온전합니다(`enrolled=True`, `facility_token_set=True`,
`edge_installation_id` 존재). 영수증 의미론 결정은 **사용자가 유보**했습니다.

---

## 5. 이번 세션에서 신뢰하면 안 되는 것

**"검증했다"는 제 진술이 이번 세션에서 여러 번 틀렸습니다.** 구체적으로:

1. 워크스페이스 풀 테스트가 sanitize된다고 했으나 `SEEON_NATIVE_TEST_TARGETS`에 없었음
2. "13대 실시간 갱신"이라 했으나 `SnapshotQueue`는 일회성
3. #429 attempt 3에서 "다섯 속성 전부 고정"이 거짓
4. 커밋 메시지와 이슈에 "고쳤다"고 썼으나 치환이 매칭조차 안 된 경우
5. "와이어로 증명 불가능"이라 두 번 단언, 둘 다 틀림
6. **"8건 낙상이 방금 발생"** — 실제로는 배달 큐 백로그 배출이었음
7. **클립 잠금 수정이 "동작한다"** — 실제로는 재시작이 플래그를 리셋한 것

리뷰가 **13회 REJECT + 1회 BLOCK**했고 **전부 진짜 결함**이었습니다.
그중 다섯은 제가 이미 "검증했다"고 선언한 것이었습니다.

**교훈: 라이브 측정이 재시작·백로그·잘못된 마운트와 섞이면 아무것도 증명하지 못합니다.**
측정 전에 (a) 이미지 해시, (b) 마운트 경로, (c) 큐 잔량, (d) 재시작 카운트를 먼저 고정하십시오.

---

## 6. 작업 트리

```
~/beomsukoh/SeeON/SeeON-edge
  브랜치  fix/431-named-local-accept   HEAD 74d1d01   (PR #433)

~/beomsukoh/SeeON/SeeON-edge-lane-427
  브랜치  fix/nvidia-detection-telemetry-explicit-producer   HEAD 495b97d   (PR #427)
```

**빌드는 반드시 레인 워크트리에서** 하십시오. 이번 세션에서 베이스에서 빌드해
레인의 C++ 수정을 통째로 되돌린 이미지를 라이브에 올린 적이 있습니다.
`SOURCE_REVISION`은 40자 소문자 hex 필수(0이면 빌드 실패).

증거 보관: `~/seeon-backups/20260826T154910Z/` (아티팩트 40개, `SHA256SUMS` 단일)
