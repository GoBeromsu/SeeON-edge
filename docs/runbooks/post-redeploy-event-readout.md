# Runbook: 재배포 후 fall / bed_exit 판독 절차

13대 카메라 재배포 후 "이벤트가 왜 안 오는지"를 판단하기 위한 **읽기 전용** 절차다.
이 문서는 배포를 수행하지 않는다 — `lane-edge-redeploy`가 실행하는 쪽이고, 이 문서는
그 결과를 읽는 쪽이다. 아키텍처 설명은 [`docs/architecture.md`](../architecture.md)에
있으니 여기서는 반복하지 않는다.

> [!WARNING]
> **`worker-config` 응답(`GET /api/v1/cameras/worker-config`)은 13대 카메라의 RTSP
> URL을 자격증명 포함 평문으로 담고 있다.** 이 런북의 어떤 점검도 그 엔드포인트를
> 직접 호출하지 않는다 — 아래 점검들은 전부 카메라 ID, 불리언 상태, 타임스탬프만
> 필요하기 때문이다. 정말 그 엔드포인트를 확인해야 하는 상황이 오면, 응답 전체를
> 절대 로그/이슈/터미널 스크롤백에 남기지 말고 `jq`로 `camera_id`/`domains`만 즉시
> 추출해서 버릴 것. 원격 프로세스의 `ps` 출력도 마찬가지로 읽힌다 — 토큰을 커맨드라인
> 인자로 넘기지 말 것(예: `curl -H "Authorization: Bearer $TOKEN"`는 `$TOKEN`이
> 셸 확장된 뒤 `ps aux`에 평문으로 남는다). 이 저장소는 **공개** 저장소다. 노드
> 주소, SSH 계정, 키 경로, 카메라 IP, 비밀번호는 이 문서에도, 이 문서가 만들어내는
> 어떤 출력에도 남지 않아야 한다.

## 0. 사전 확인 — 지금 뭘 보고 있는지부터 확정한다

두 가지를 배포 직후 가장 먼저 확인한다. 아래에서 읽는 모든 신호는 **이 두 가지가
맞다는 전제** 위에 있다.

### 0-1. 실제로 뜬 이미지가 기대한 커밋을 담고 있는가

```sh
cd /opt/eldercare-fall-ml
# 오늘 밤(소스 빌드, dev 모드) 기준 $DC — `docker compose ... config`로 실제
# 렌더링해서 검증한 목록이다(추측 아님). 순서가 중요하다: dev가 cpu보다
# 먼저 와야 한다(compose.edge.dev.yaml 자신의 헤더 주석이 이 순서를 명시).
#   -f compose.edge.yaml   : base. image:는 ${ML_API_IMAGE:?...}/
#                            ${ML_WORKER_IMAGE:?...}로 필수 — .env.edge.prod에
#                            eldercare-fall-ml-api:dev / eldercare-fall-ml-worker:dev로
#                            채워져 있어야 한다(오늘 밤 전용 값, GHCR pull용 값 아님).
#   -f compose.edge.dev.yaml : 두 서비스에 build:+pull_policy:never를 얹어
#                            소스에서 직접 빌드하게 만든다. #237(GHCR 403)이
#                            풀릴 때까지의 임시 경로.
#   -f compose.edge.cpu.yaml : ml-worker의 무조건적 NVIDIA GPU deploy 예약을
#                            지운다 — 이 노드는 GPU 드라이버가 없다.
#                            compose.edge.igpu.yaml은 다른 호스트(Intel
#                            VAAPI)용이라 여기선 안 쓴다.
# compose.edge.local.yaml(커밋 안 되는 노드 로컬 파일)은 안 넣는다 — pull_policy:never
# 와 GPU deploy 제거를 각각 dev/cpu 오버레이가 이제 공식적으로 담당하므로,
# #239 이전에 임시 로컬 빌드를 굴리려고 만들어 둔 이 파일은 오늘 밤 경로에서
# 중복이다(내용 대조 완료). GHCR을 pull하는 평상시 배포로 돌아가면 이 목록에서
# `-f compose.edge.dev.yaml`만 빼면 된다 — 나머지 두 개(yaml/cpu)는 그대로다.
# 다음 배포 전엔 실행 중인 컨테이너의 실제 오버레이 목록을 아래로 재확인할 것:
#   docker inspect <container> --format \
#     '{{index .Config.Labels "com.docker.compose.project.config_files"}}'
DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml -f compose.edge.dev.yaml -f compose.edge.cpu.yaml'
docker inspect --format '{{index .Config.Image}} {{.Image}}' "$($DC ps -q ml-worker)"
```

이미지 태그/다이제스트만으로는 "그 안에 어떤 PR이 들어있는지" 알 수 없다. 배포
직전에 반드시 아래처럼 **빌드에 쓴 git ref**가 기대하는 fix 커밋의 후손인지
확인한다(스쿼시 머지는 브랜치 tip SHA를 직접 ancestor로 남기지 않으므로, PR
브랜치 tip이 아니라 **머지 커밋**으로 확인한다 — `gh pr view <n> --json
mergeCommit`으로 얻는다):

```sh
git fetch origin main
git merge-base --is-ancestor <fix-merge-commit-sha> origin/main && echo "포함됨" || echo "안 포함"
```

**이 문서를 쓰는 시점(2026-08-06) 기준 상태 — 다음 배포 때 반드시 재확인할 것,
시간이 지나면 이 목록은 낡는다:**

| 이슈 | 내용 | 상태 |
| --- | --- | --- |
| #217 | fall `operating_threshold`가 검출기까지 안 닿던 문제 | PR #222 머지됨(`ffb6bb9`), **main에 포함** |
| #224 | `bed_region_counters`/freshness 관측 노출 | PR #224 머지됨(`811ffa7`), **main에 포함** |
| #208 | RTSP 재연결마다 bed-region 캐시 초기화 | PR #232 머지됨(`a892550`), **main에 포함** — 단, 폴리곤 저장된 카메라엔 영향 없음(0-2 참고) |
| #219 (H2) | 48점 폴리곤이 AABB로 축소되어 판정 면적이 최대 2배 넓어짐 | PR #227 머지됨(`1b61304`), **main에 포함** |
| #218 (H1) | 트랙 소실 시 배정이 최대 ~6초 얼어붙었다가 이벤트 없이 조용히 삭제 | PR #234 머지됨(`ecfeda5`), **main에 포함** — 단 `grace_frames > 0`(트랙이 배정을 어느 정도 유지하다 끊긴 경우)만 수정됨. `grace_frames == 0`인 즉발 이탈은 PR 바디가 명시한 별도 후속 갭으로 **여전히 미수정**(아래 단락/B절 ③ 참고) |
| #220 (H3) | 다른 침대에 순간 접촉하면 원래 침대 grace_frames만 리셋되고 무기한 유지 | 이슈만 존재, **PR 없음** |
| #226 | `expired` 카운터가 flap 횟수가 아니라 cycle 수를 셈 | 이슈만 존재, 문서 각주만 반영(#224) |
| (번호 없음) | `worker/__main__.py`의 `logging.basicConfig`가 `extra=` 키를 포맷 문자열에서 참조하지 않아 `bed_region`/`stage_timings`/`bus`/`encode`가 `docker compose logs`에 안 보임 | PR #242 머지됨(`6f35357`), **main에 포함** — A-1/로그 확인 절 참고 |
| #238 | bed_region은 멀쩡한데 (b) 폴리곤 안에서 점수화가 안 됐는지 (c) 점수화는 됐지만 exit 카운터가 threshold를 못 넘었는지, 로그만으로 구분 안 됨 | PR #241 오픈(head `5e8fd29`, `6f35357` 위로 리베이스됨), **미머지** — B절 참고 |

**#219(H2)의 수정은 이제 main에 머지돼 있다(#227, `1b61304`)** — 오늘 밤 배포 이미지가
이 커밋의 후손이라면(0-1의 `merge-base --is-ancestor` 방법으로 직접 확인할 것) H2는
후보에서 빠진다. **#218(H1)도 이제 main에 머지돼 있다(#234, `ecfeda5`)** — 하지만
원래 버그 전체가 아니라 `grace_frames > 0`인 트랙 소실만 고쳤다: 배포 이미지가
`ecfeda5`의 후손이면 이 케이스는 이제 `BedExitEvent`를 정상 발화하고 `exit_beds`에도
잡힌다. `grace_frames == 0`인 **즉발 이탈**(트랙이 배정을 쌓을 시간도 없이 바로
소실된 경우)은 PR #234 바디가 스스로 명시한 별도 후속 갭으로 여전히 미수정이다 —
이건 **#220(H3, PR 없음)**과 겉모습이 정확히 같다. 아래 B절의 판단 트리에서 이
좁아진 잔여 갭(#218 즉발 케이스)과 #220을 갈래 ③의 1순위 용의자로 쓴다. 이미지가
`ecfeda5`의 후손이 아니라면(구형 이미지) #234 이전과 동일하게 `grace_frames > 0`
케이스까지 포함한 원래 범위 그대로 적용된다는 점도 배포 전 확인해둘 것.

### 0-2. 폴리곤 보유 여부로 어떤 카메라가 어떤 경로를 타는지 미리 나눠둔다

`SceneState.resolve_bed_regions()`가 `worker/pipeline/perception/scene_state.py:125`의
`if self.persisted_bed_regions:`에서 즉시 분기한다(#232 머지로 캐시 리셋 로직이
정리되면서 줄 번호가 바뀌었다 — 다음 배포 때도 재확인할 것) — 운영자가 저장한 폴리곤이
있으면 라이브 세그멘테이션이나 프레임 카운터 상태와 무관하게 매 프레임 `FRESH`를
반환하고 그 아래의 캐시/만료 로직 전체를 건너뛴다. 13대 중 10대(침실 카메라)는
폴리곤을 갖고 있고, 3대(11 프로그램실 / 12 중앙복도 / 13 우측복도)는 없다 — 이
3대는 침대가 없는 공간이라 bed_exit이 구조적으로 비활성이다.

**따라서 #208/#226은 오늘 밤 이 13대 중 어느 하나에도 영향을 주지 않는다.** 폴리곤
없는 3대는 애초에 bed_exit 대상이 아니고, 폴리곤 있는 10대는 프레임 카운터 로직
자체에 도달하지 않는다. 이 두 이슈는 이 판독 절차에서 빠진다 — 배포 대상이자 fix
대상이지만, 오늘 밤 관측할 신호에는 나타나지 않는다.

## A. 세 가지 결정적 판독값

### A-1. `bed_region.freshness` — 침실 10대는 전부 `fresh`여야 한다

**#242가 머지되면서(`6f35357`) 이 신호는 이제 `docker compose logs`로 보인다 —
아래는 배포 이미지가 이 커밋의 후손인지 확인하는 방법과, 안 보였던(그리고 지금도
`bed_exit_scoring` 하나는 여전히 안 보이는) 이유다.**

코드는 원래부터 존재했다 — `worker/runtime/telemetry/local_metrics.py`의
`log_snapshot()`이 카메라마다 `logger.info(...)`를 찍고, `bed_region.freshness`/
`counters`가 그 안에 들어있다(#224). 문제는 `worker/__main__.py`의
`logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s -
%(message)s")`가 `extra`의 키를 포맷 문자열이 참조하지 않으면 출력에 넣지
않는다는 데 있었다 — `#242` 이전엔 `bed_region`/`stage_timings`/`bus`/`encode`가
전부 `extra=`로만 전달돼서 어느 것도 로그 줄에 안 남았다(`#207`/`#224` 이전부터
있던 구조적 문제, 재현은 `git show 6f35357^:worker/runtime/telemetry/local_metrics.py`
참고). `#242`의 수정은 이 값들을 `extra=`뿐 아니라 로그 **메시지 문자열 자체에도**
`%s`로 렌더링하도록 `log_snapshot()`을 바꿨다 — 포맷터가 뭘 참조하든 살아남는다.

**배포 이미지가 `6f35357`의 후손인지 확인**(0-1과 같은 방법):

```sh
git fetch origin main
git merge-base --is-ancestor 6f35357828c41665646bd2394a17143e13ae9fb2 origin/main && echo "포함됨" || echo "안 포함"
```

포함돼 있다면 `$DC logs --tail 200 ml-worker`에서 `bed_region=`/`stage_timings=`/
`bus=`/`encoder=`/`encode=`가 이제 `worker.runtime.telemetry` 줄 자체에 값과 함께
찍힌다. **`bed_exit_scoring=`만은 예외다** — 이 필드 자체가 `#241`(현재 오픈,
head `5e8fd29`, `6f35357` 위로 리베이스돼 있어 머지되면 같은 방식으로 렌더링될
것으로 코드상 확인됨)이 붙기 전엔 아예 존재하지 않는다. 즉 지금 시점에서 로그로
안 보이는 건 렌더링 결함이 아니라 "그 필드가 아직 없다"는 것이다 — B절/아래
[!NOTE] 참고.

**오늘 밤 실무적 의미:** 침실 10대는 전부 폴리곤을 갖고 있으므로(0-2), 이 신호는
설령 늘 볼 수 있어도 오늘 밤은 구조적으로 늘 `fresh`다 — 그 자체로 새 정보를
주진 않는다. 이 판독값의 진짜 용도는 "**혹시 `fresh`가 아니면**"이다: 침실
카메라 중 하나가 `fresh` 이외의 값을 낸다면, 그건 0-2의 전제(폴리곤 short-circuit이
항상 이긴다는 모델) 자체가 틀렸다는 뜻이고, 다른 모든 판독보다 그게 먼저 조사
대상이 된다. 배포 이미지가 `6f35357`의 후손이라면 이 경보는 이제 로그만으로
켜진다 — **라이브 뷰 없이도 확인 가능하다.** 후손이 아니라면(구형 이미지로
롤백했거나 확인을 건너뛴 경우) 여전히 A-1은 관측 불가이고, 그 사실 자체가
"침실 카메라들이 조용히 `fresh`가 아니게 됐다"는 경보를 낼 수 없다는 뜻이라는
걸 알고 있는 채로 밤을 보내야 한다 — 그러니 위 `merge-base` 확인을 건너뛰지
말 것.

### A-2. fall 이벤트의 `audit` 블록 — #217 threshold가 적용됐는지 확인

이벤트는 로그가 아니라 **워커 상태 볼륨의 SQLite**에 남는다:
`ml-worker-state` 네임드 볼륨, `ML_WORKER_STATE_DIR` 아래
`worker-state.sqlite3`, `evidence_events` 테이블.

```sh
$DC exec ml-worker sqlite3 "$ML_WORKER_STATE_DIR/worker-state.sqlite3" \
  "SELECT edge_event_id, detected_at, delivery_state, \
          json_extract(payload_json,'\$.event_type') AS event_type, \
          json_extract(payload_json,'\$.audit') AS audit \
   FROM evidence_events \
   WHERE detected_at > '<재시작 UTC 타임스탬프>' \
   ORDER BY detected_at;"
```

**`edge_event_id`가 아니라 `detected_at`으로 "재시작 이후"를 판단한다.**
`edge_event_id`(fall 이벤트의 경우 `FallEventLatch.event_count`, 프로세스 재시작마다
0으로 리셋되는 인메모리 카운터)는 재시작 전후로 값이 겹칠 수 있어 "이 이벤트가
재시작 이후 발생했다"의 증거가 못 된다. `detected_at`은 실제 UTC 벽시계
타임스탬프다.

`audit.operating_threshold`가 기대한 값(env로 재정의한 값, `.env.edge.prod`의
`ML_FALL_OPERATING_THRESHOLD` 등)과 일치하면 #217 fix가 살아있는 것이다.
일치하지 않으면 — 0-1에서 확인했듯 #217은 이미 main에 있으므로, 이미지가
기대한 커밋을 안 담고 있거나(0-1로 돌아가서 재확인) 아니면 env 자체가 잘못
설정된 것이다.

### A-3. 실제 bed_exit 이벤트

같은 테이블, `event_type='bed-exit'` 필터:

```sh
$DC exec ml-worker sqlite3 "$ML_WORKER_STATE_DIR/worker-state.sqlite3" \
  "SELECT edge_event_id, detected_at, delivery_state, last_error_code, attempt_count \
   FROM evidence_events \
   WHERE json_extract(payload_json,'\$.event_type')='bed-exit' \
   ORDER BY detected_at DESC LIMIT 50;"
```

한 건이라도 있으면 B절은 필요 없다 — 파이프라인 전체(스코어링 → grace_frames →
night window → 스테이징 → 릴레이)가 최소 한 번은 끝까지 살아있었다는 뜻이다.
`delivery_state != 'ACKED'`인 행이 있으면 발화는 됐지만 backend까지 확인응답을
못 받은 것 — 이건 릴레이 전송 문제고, C절이 아니라 `last_error_code`로 바로
좁혀진다(재시도는 자동이고 최대 백오프 300초, 유실은 없다 — 아래 참고).

## B. bed_exit 0건일 때 — 판단 트리

`BedExitMonitor` 파이프라인을 인과 순서대로 나열한다. 각 갈래에서 **구분해주는
필드/로그가 있으면 그 이름을, 없으면 명시적으로 "없음"**을 적는다.

```
① bed region 사용 불가
       │  (없으면 다음 프레임에서 update()가 즉시 return () — _assignments 자체가
       │   갱신되지 않는다: worker/domains/bed_exit/detector.py:68)
       ▼
② 사람이 폴리곤 안에서 점수화된 적이 없음
       │  (containment_ratio가 min_containment=0.35를 넘겨 hold_frames=2 연속
       │   되지 않으면 assignment.bed_id가 계속 None)
       │  → PR #241(미머지)이 붙으면 카메라별 max_containment_observed로 이 갈래를
       │    사후에도 구분할 수 있게 된다. 아래 표 참고.
       ▼
③ 배정은 됐지만(bed_id 있음) exit 카운터가 threshold를 넘은 적이 없음
       │  (grace_frames > grace_frames=3 이 크로싱 조건. #220(H3)이 여기서
       │   무기한 리셋시킬 수 있다.)
       │  → PR #241(미머지)이 붙으면 assignments_made/grace_positive_transitions로
       │    이 갈래도 사후 구분이 가능해진다. 단 max_containment_observed가
       │    threshold를 넘었는데 assignments_made == 0이면 hold_frames를 연속으로
       │    못 채운 세 번째 경우다 — ②도 ③도 아니다.
       ▼
④ 카운터가 넘어서 이벤트는 만들어졌는데, night window 게이트가 버림
       │  (detector.py:90-91, self._night_window.contains(...)가 False —
       │   단 last_debug_snapshot은 이 게이트보다 먼저 채워져서, 오버레이엔
       │   bed:exit이 그대로 뜬다. 아래 표 ④ 참고)
       │  [상태 의존적: 이 문서 작성 시점엔 관찰 편의를 위해 창이 00:00-23:59로
       │   넓혀져 있어 사실상 죽어 있는 갈래다 — 표 ④ 각주 참고]
       ▼
⑤ 이벤트가 릴레이로 나갔는데 backend가 확인 안 함
```

| 갈래 | 구분 신호 | 명령/필드 | 비고 |
| --- | --- | --- | --- |
| ① | 대시보드 라이브 뷰의 침대 오버레이 라벨(`bed:empty`/`bed:occupied`/`bed:exit`)이 아예 안 뜸, 또는 폴리곤 자체가 비어 보임 | `GET /api/v1/streams/{camera_id}/snapshot`(대시보드 자체 인증 토큰 사용, 아래 참고) | A-1의 로그 갭 때문에 `bed_region.source`를 직접 볼 수는 없다 — 폴리곤이 있는 카메라라면(0-2) 이 갈래는 구조적으로 배제된다 |
| ② | 오버레이가 계속 `bed:empty` — 한 번도 `bed:occupied`로 안 바뀜 | 같은 스냅샷 엔드포인트, 사람이 침대 위에 있을 때 관찰 | #219(H2)의 수정(#227)은 이미 main에 머지돼 있어 오늘 밤 이 갈래를 밀어내는 방향으로는 더 이상 작용하지 않는다(0-1 확인 요). **PR #241(오픈, head `5e8fd29`)이 머지되면** `bed_exit_scoring.max_containment_observed`가 0에 가까운 채로 남는 것이 ②의 사후 확증 신호가 되고, `#242`(머지됨, `6f35357`) 위로 이미 리베이스돼 있으므로 머지되는 즉시 `docker compose logs`에도 값이 찍힌다 — 아래 [!NOTE] 참고 |
| ③ | 오버레이가 `bed:occupied` → `bed:empty`로 **`bed:exit`을 거치지 않고** 바로 바뀜 | 같은 엔드포인트, 실시간 관찰 필요(사후 조회 불가) | **`#234`(머지됨, `ecfeda5`) 이후 이 갈래의 범위가 좁아졌다** — 배포 이미지가 `ecfeda5`의 후손인지는 0-1 방법으로 확인할 것. 후손이면 `grace_frames > 0`이었던 트랙 소실은 이제 `BedExitEvent`를 발화하며 ③에서 빠지고 정상적으로 A-3 사후 조회가 된다. 남는 건 **`grace_frames == 0`인 즉발 이탈**(트랙이 배정을 쌓을 시간도 없이 바로 소실 — #234 PR 바디가 명시한 후속 갭)뿐이고, 이건 여전히 **#220(H3, PR 없음)과 정확히 같은 겉모습**(occupied→empty, exit 없음)을 만든다. 코드상 두 경로 다 카운터/이벤트를 안 남기므로 사후에는 구분 불가 — 어느 쪽인지 알려면 그 순간 라이브 뷰를 보면서 사람이 실제로 방을 나갔는지(즉발 이탈이면 트랙을 놓쳤을 뿐 사람은 안 나갔을 수도 있음) 육안 대조가 유일한 방법이다. **이게 team-lead가 미리 알아야 한다고 한 바로 그 종류의 갭이다.** `assignments_made`/`grace_positive_transitions`(PR #241)는 ②/③ 경계는 구분해줘도 ③ 내부에서 잔여 #218 즉발 케이스 vs #220까지는 못 가른다. 이미지가 `ecfeda5`의 후손이 **아니라면**(구형 이미지) `grace_frames > 0` 케이스까지 포함한 원래 범위 그대로 ③이 적용된다 |
| ④ | night window 밖 시간대에 ③까지는 확실히 넘었는데 이벤트가 안 옴 | 오버레이 라벨의 `bed:exit` 플래시 유무 — `detector.py`는 `last_debug_snapshot`을 night window 게이트(90-91행)보다 **먼저** 채우고, `overlay.py`의 `_draw_bedexit_beds`는 그 스냅샷의 `statuses[].occupancy`를 추가 게이팅 없이 그대로 그린다 | 그래서 게이트가 이벤트를 억제해도 오버레이엔 `bed:exit`이 그대로 짧게 뜬다 — **그 순간 `evidence_events`에 대응 행이 없으면 ④**다. ③(#220, 그리고 배포 이미지가 `ecfeda5`의 후손이 아니면 #218 전체, 후손이면 #218의 `grace_frames == 0` 잔여 케이스)은 애초에 `exit_beds`를 채우지 않으므로 `bed:exit` 자체가 절대 안 뜬다 — 이 플래시 유무가 ③/④를 가르는 신호다. window 경계(`night_window` 설정값)와 관찰 시각 대조도 병행할 것. **현재 상태 의존적으로 사실상 비활성** — 아래 [!NOTE] 참고, 관찰 시점에 `/api/v1/detection-settings`로 반드시 재확인할 것(이 노트를 그대로 믿지 말 것) |
| ⑤ | A-3의 `delivery_state`/`last_error_code` | SQLite 쿼리(A-3) — 재시도 대기 행은 `delivery_state='PENDING'`으로 조회할 것 | **유실은 구조적으로 없다** — `DurableEvidenceStager`가 네트워크 시도 전에 이미 SQLite에 영속 기록한다. 실패해도 `PENDING`(재시도 대기, 최대 백오프 300초)이나 `PERMANENT`(그래도 행은 남는다)로만 간다 — **`RETRY_SCHEDULED`는 `delivery_state` 값이 아니라 `evidence_sender.py`의 `SenderStep` enum 멤버명이다**; `evidence_outbox_schema.py`의 CHECK 제약이 허용하는 값은 `PENDING`/`ACKED`/`PERMANENT`/`COMPATIBILITY`뿐이라 `RETRY_SCHEDULED`로 쿼리하면 항상 0행이 나온다(재시도가 없다는 뜻이 아니라 쿼리가 틀렸다는 뜻). "이벤트가 발화됐는데 흔적이 아예 없다"는 이 파이프라인에서 일어날 수 없는 일이다 — 그런 게 보이면 이 문서가 기술한 경로 자체가 틀렸다는 뜻이니 바로 알려달라 |

> [!NOTE]
> **④는 이 문서 작성 시점 기준 상태 의존적으로 사실상 비활성이다.** `bed_exit`의
> 실제 운영 설정은 21:00-06:00 야간 창이지만, 오늘 밤 관찰 편의를 위해 의도적으로
> `{mode: 'window', start: '00:00', end: '23:59'}`로 넓혀둔 상태다 — 이 창에서
> `_WindowGatedDecider`가 억제하는 구간은 자정 전후 1분 안팎뿐이라, ④(night
> window 게이트 억제)가 오늘 밤 실제로 관측될 가능성은 낮다. **메커니즘 자체(위
> 표/다이어그램 설명)는 그대로 유효하다** — 창이 21:00-06:00으로 복원되면 이
> 갈래는 다시 살아난다. 이 값은 가변적이라 이 노트를 그대로 믿지 말고, 판독
> 시점에 직접 확인할 것:
> ```
> curl -s -b jar http://127.0.0.1:8000/api/v1/detection-settings
> ```
> (세션 쿠키가 없다면 `/api/v1/auth/session`으로 먼저 로그인) — 응답의
> `domains.bed_exit.start`/`.end`가 `00:00`/`23:59`가 아니면 창이 복원된 것이고,
> ④는 다시 활성 갈래다. 관찰이 끝나면 이 창을 21:00-06:00으로 되돌릴 것 —
> 아직 안 끝났다.

**실전 순서 제안**: A-3에서 0건을 확인했다면, night window 시간대에 실시간으로
대시보드 라이브 뷰를 침실 카메라 한 대에 띄워놓고 오버레이 라벨 전이를 직접
관찰하는 것이 **오늘 밤 시점 기준** ②/③/④를 구분하는 사실상 유일한 방법이다
(사후 로그로는 ②/③ 구분 불가). `bed:empty`가 지속되면 ②, `bed:occupied`가 뜨는데
`bed:exit`으로 안 넘어가면 ③, **`bed:exit`이 짧게라도 떴는데 그 순간
`evidence_events`에 대응 행이 없으면 ④**다(`last_debug_snapshot`이 게이트보다
먼저 찍히기 때문 — 위 표 ④ 참고). 이후 (배포 이미지가 `ecfeda5`의 후손이 아니면
#218 전체, 후손이면 #218의 `grace_frames == 0` 잔여 케이스만) vs #220은 위 표대로
구분 불가이니 어느 쪽 가설이 맞는지는 코드 수정 없이는 확정할 수 없다는 점을 그대로
보고할 것.

> [!NOTE]
> **`PR #242`는 머지됐다(`6f35357`, main에 포함 확인됨)** — `bed_region`/
> `stage_timings`/`bus`/`encode`는 이제 배포 이미지가 그 후손이기만 하면
> 라이브 뷰 없이 로그만으로 보인다(A-1/로그 확인 절 참고). **`PR #241`은 아직
> 열려 있다**(head `5e8fd29`, 작성 시점 CI 진행 중) — `bed_exit_scoring`
> (`max_containment_observed`/`grace_positive_transitions`/`assignments_made`,
> boot 이후 누적)은 이 PR로 처음 생기는 필드라서, 머지되기 전엔 코드 자체가
> 없어 로그에도 당연히 없다. **머지된 뒤엔 별도로 기다릴 게 없다** — 이 PR의
> 현재 diff가 이미 `#242`의 렌더링 수정 위에 얹혀 있어서, 머지되는 순간
> `bed_exit_scoring=`도 로그 줄에 값과 함께 찍힌다(코드 대조로 확인함, 추측
> 아님). 즉 ②/③을 라이브 뷰 없이 사후 로그만으로 구분하려면 **`#241`이
> 머지됐는지 하나만** 확인하면 된다 — 배포 전 0-1 방법으로
> `5e8fd29`(또는 그 머지 커밋)가 배포 이미지의 후손인지 직접 확인할 것.
> 안 들어가 있으면 위 라이브 뷰 절차가 ②/③ 구분의 유일한 방법으로 남는다.

## 라이브 뷰 접근

```sh
curl -sS -H "Authorization: Bearer ${OPERATOR_TOKEN}" \
  "https://<backend-host>/api/v1/streams/<camera_id>/snapshot" -o snapshot.jpg
```

`${OPERATOR_TOKEN}`은 대시보드 로그인 세션이 쓰는 일반 운영자 토큰이다 — 카메라
RTSP 자격증명이 아니다. `camera_stream`/`camera_snapshot`은
`backend/app/features/cameras/streams_router.py`가 워커의 `dev_mjpeg` 스트림을
그대로 프록시하는 것이라, 이 호출 경로 어디에도 RTSP URL이 등장하지 않는다.

## 로그 확인

```sh
cd /opt/eldercare-fall-ml
# $DC는 0-1에서 정의한 것과 동일 — 여기서 다시 정의하지 말고 그대로 재사용할 것
# (예전 판은 이 섹션에서 compose.edge.yaml 하나만 넣은 다른 정의를 또 두고 있었다)
$DC logs --tail 200 ml-worker
```

**`#242`(머지됨, `6f35357`) 후손 이미지라면** `bed_region`/`stage_timings`/`bus`/
`encode`는 `worker.runtime.telemetry` 줄에 값과 함께 보인다(A-1 참고, 확인 방법도
동일) — 더 이상 안 보인다고 가정하지 말 것. **`bed_exit_scoring`만은 `#241`이
머지되기 전까진 여전히 안 보인다**(필드 자체가 없어서 — 위 [!NOTE] 참고).
`#242` 이전 이미지로 롤백했거나 확인을 건너뛰었다면 다섯 필드 전부 예전처럼 안
보인다고 가정할 것. 부팅 시퀀스(프로파일 해석, 모델 워밍업, 카메라 활성화)는
이 갭과 무관하게 항상 정상적으로 보인다.

## Related

- [`docs/architecture.md`](../architecture.md) — 워커 레이어, 엔트리포인트, 장애
  매트릭스.
- [`docs/runbooks/worker-migration-rollback.md`](worker-migration-rollback.md) —
  이 배포가 잘못됐을 때 되돌리는 절차. 볼륨 보존 규칙은 여기와 동일하다.
- 이슈 #217, #218, #219, #220, #226, #238 — 이 문서의 표/판단 트리가 인용하는
  원본 근거.
- PR #234(#218 수정)는 **머지됨**(`ecfeda5`) — 단 `grace_frames > 0` 케이스만
  수정, `grace_frames == 0` 즉발 이탈은 PR 바디가 명시한 후속 갭으로 미수정
  남음(0-1/B절 ③ 참고). PR #241(#238 수정 — 이 런북의 B절이 다루는 (b)/(c)
  구분 신호, 오픈/미머지, head `5e8fd29`)은 아직 오픈 상태이고 이 문서의 판단
  트리 정확도에 직접 영향을 준다. PR #242(로그 렌더링 갭 수정)는 **머지됨**
  (`6f35357`) — A-1/로그 확인 절이 그 상태를 반영한다.
