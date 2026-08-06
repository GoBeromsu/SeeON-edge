---
name: edge-bringup
description: >
  Tailscale 뒤의 엣지 노드에 접속해 IDIS IP 카메라(DirectIP/WebGuard) RTSP 스트림을
  살리고 CPU-only 로 docker compose 스택을 띄우는 현장 브링업 절차. 엣지 서버,
  요양원 카메라, RTSP 454/401, "카메라가 안 붙는다", "서버에 스택 띄워줘",
  compose 가 GPU 때문에 실패, ml-worker 가 카메라를 0 대로 인식, WebGuard 설정
  화면이 멈춤 — 이런 이야기가 나오면 반드시 이 스킬을 먼저 읽어라. 사용자가
  "엣지", "카메라", "RTSP", "요양원", "happy-nursing-home" 을 언급하면 명시적으로
  브링업을 요청하지 않았더라도 적용된다. 이 저장소의 엣지 운영은 전부 여기서 시작한다.
---

# 엣지 브링업 (IDIS 카메라 + CPU-only compose)

현장 엣지 노드에서 카메라를 살리고 추론 스택을 띄우는 절차다. 실제로 13 대를
0 대에서 13 대로 만든 경로를 그대로 담았다.

**대상 하드웨어를 먼저 확인해라.** 카메라 쪽 절차는 **IDIS 계열 IP 카메라**
(OpenIP/DirectIP, WebGuard 웹 설정 UI) 전용이다. `Page305`, `#use-rtp-encryption`,
`REG.*` 레지스트리 트리, 8016 DirectIP 포트는 전부 IDIS 펌웨어의 것이라 Hanwha,
Hikvision, Axis 같은 다른 벤더에는 하나도 통하지 않는다. 다른 벤더라면 compose
브링업(4단계)만 가져다 쓰고 카메라 부분은 새로 조사해라.

## 이 절차가 지키는 것

세 가지 원칙이 있고, 어기면 실제로 사고가 났던 것들이다.

**증거는 엣지에서 만든다.** 카메라는 요양원 내부망에만 있다. 노트북에서 터널로
찔러 본 결과는 워커가 붙을 수 있다는 증거가 못 된다. 최종 검증은 항상 엣지에서
`ffprobe` 로 한다.

**비밀번호는 함부로 시도하지 않는다.** IDIS 카메라 계정은 오인증 누적으로 잠긴다.
401 을 보면 다른 값을 연달아 넣지 말고 **기록된 값이 맞는지부터 의심해라.** 실제로
자격증명 노트 쪽이 틀렸고, 그것도 노트 스스로 "미검증" 이라고 적어둔 값이었다.

**고친 것은 그때그때 이슈로 남긴다.** 브링업 도중 발견되는 결함은 대부분 "이 환경
에서만 드러나는" 것들이라 지나가면 다시 못 찾는다. 고친 직후에 등록해라.

## 0. 엣지에 붙는다

```bash
export EDGE_HOST=... EDGE_USER=... EDGE_KEY=~/.ssh/...
CAM_CHANNELS='1:10.0.0.11,2:10.0.0.12' scripts/edge.sh open
scripts/edge.sh run 'hostname && docker ps -a'
```

`scripts/edge.sh` 가 SSH ControlMaster 를 열고 채널 N 마다 로컬 `184NN` → 카메라
`:443` 포워드를 건다.

노드 주소·계정·키 경로와 카메라 IP 는 이 저장소에 적지 않는다. 저장소가 공개돼
있어서 그 값들을 합치면 어느 현장의 어느 노드에 누구로 붙는지가 그대로 드러난다.
자격증명 노트에서 읽어 환경변수로 넘겨라.

`tailscale switch` 는 쓰지 마라. 이 머신은 동시에 다른 tailnet 을 쓰고 있어서
전환하면 그쪽 작업이 끊긴다. 엣지 노드는 **공유 노드**로 들어와 있으므로 전환
없이 그대로 닿는다. `tailscale status | grep <node>` 로 보이면 된 것이다.

## 1. 카메라가 실제로 어떤 상태인지 먼저 잰다

고치기 전에 전수 측정부터 한다. 무엇이 깨졌는지 모르는 채로 설정을 만지면
멀쩡한 것까지 망가뜨린다.

```bash
CAM_PASSWORD='<자격증명 노트에서>' \
CAM_CHANNELS='1:10.0.0.11,2:10.0.0.12,...' \
scripts/rtsp_sweep.sh
```

출력을 이렇게 읽는다. **이 분기가 진단의 전부다.**

| 결과 | 뜻 | 다음 행동 |
|---|---|---|
| `hevc,640,360,30/1` | 정상 | 건드리지 마라 |
| `454 Session Not Found` | RTP 보안이 켜져 있다 | 3 단계 |
| `401 Unauthorized` | 자격증명이 틀렸다 | 2 단계 — **멈추고 확인** |
| `TIMEOUT` / 무응답 | 네트워크 도달 실패 | 카메라 설정 문제가 아니다. 경로부터 봐라 |

454 와 401 을 섞어 보면 진단이 통째로 어긋난다. 454 는 인증을 통과한 뒤 나오는
것이라 자격증명은 맞다는 뜻이고, 401 은 그 반대다.

## 2. 401 이 나왔을 때

**다른 비밀번호를 시도하기 전에 멈춰라.** 계정 잠금은 되돌리기 어렵고 현장 방문이
필요해질 수 있다.

순서는 이렇다.

1. 같은 자격증명으로 **웹 UI(443)** 가 되는지 본다. 웹이 200 이면 계정은 잠기지
   않았고, 문제는 RTSP 쪽 값이거나 URL 조립이다.
2. URL 인코딩을 의심해라. IDIS 카메라는 userinfo 를 퍼센트 디코딩하지 않는다.
   비밀번호에 `!` 가 있으면 `%21` 로 쓰면 안 되고 리터럴로 넣어야 한다. 이것만으로
   전 채널 401 이 난 적이 있다.
3. 기록이 여러 개면 **최근 실접속 기록이 있는 값**을 믿어라. "1 순위"라고 적혀
   있어도 검증 표시가 없으면 근거가 없는 값이다.
4. 그래도 안 되면 사람에게 넘겨라. NVR / IDIS Center 에서 계정을 확인하는 것이
   추측으로 시도하는 것보다 싸다.

값이 확정되면 **자격증명 노트를 그 자리에서 고쳐라.** 틀린 기록을 남겨두면 다음
사람이 같은 함정에 빠지고, 그때는 잠길 수도 있다.

## 3. RTP 보안 해제 (454 의 원인)

`NetworkConfig.Rtsp.RtpEncryptionType` 가 `1` 이면 RTSP DESCRIBE 가 454 로 죽는다.
`0` 이어야 스트림이 나온다.

절차와 함정은 `references/camera-webguard.md` 에 있다. 브라우저로 카메라를 만지기
전에 **반드시 읽어라** — 특히 왜 레지스트리를 스크립트로 직접 POST 하면 안 되는지
(admin 비밀번호가 깨질 수 있다), 그리고 어떤 다이얼로그를 절대 건드리면 안 되는지.

요약하면 이렇다.

```bash
# 1. 프록시를 띄운다 (Digest 주입 + 죽은 외부 스크립트 대체 + 캐시)
CAM_PASSWORD='...' CAM_CHANNELS='...' python3 scripts/camproxy.py &

# 2. 채널마다 캐시를 데운 뒤 브라우저로 연다
curl -s -o /dev/null http://127.0.0.1:194NN/setup/setup.html
curl -s -o /dev/null http://127.0.0.1:194NN/setup/page.js
#    -> http://localhost:194NN/setup/setup.html
```

브라우저에서는 다이얼로그를 닫고 `movePage('Page305')` 로 이동해
`#use-rtp-encryption` 를 **클릭**으로 해제한 뒤 `#setup-apply` 로 저장한다.
`.checked = false` 를 대입하면 안 된다 — 레지스트리를 쓰는 것은 change 핸들러다.

저장 후에는 브라우저 밖에서 두 번 검증한다. 레지스트리 readback 이 `0` 인지,
그리고 엣지에서 `rtsp_sweep.sh` 가 스트림을 잡는지. 페이지 안의 상태만 보고
"고쳤다"고 하면 안 된다.

## 4. CPU-only 로 스택을 띄운다

`compose.edge.yaml` 은 NVIDIA 예약을 무조건 건다. GPU 없는 노드에서는 이것만으로
`up` 이 하드 실패한다. `compose.edge.cpu.yaml` 오버레이가 그 예약을 지운다.

```bash
scripts/edge.sh run "cd '<repo>' && \
  docker compose --env-file .env.edge.prod \
    -f compose.edge.yaml -f compose.edge.cpu.yaml up -d"
```

**env 파일은 `.env.edge.prod` 하나면 된다** (#179, #182 이후). `compose.edge.yaml`
이 요구하는 모든 변수는 `.env.edge.prod.example` 안에 기본값이나
`<placeholder>` 로 이미 채워져 있다. `up` 하기 전에 항상 preflight 스크립트로
먼저 확인해라 — `config` 렌더 실패뿐 아니라 남아 있는 `<placeholder>` 도 잡아준다.
여기서 걸리는 것이 컨테이너를 띄운 뒤 디버깅하는 것보다 훨씬 싸다.

```bash
scripts/edge.sh run "cd '<repo>' && \
  scripts/edge-preflight/check-env.sh .env.edge.prod -f compose.edge.cpu.yaml"
```

GPU 예약이 실제로 지워졌는지도 같이 본다.

```bash
docker compose ... config | grep -i nvidia   # 아무것도 안 나와야 한다
```

포트도 미리 본다. 엣지에서 `tailscale serve` 가 `100.x.x.x:8000` → `127.0.0.1:8000`
을 이미 프록시하고 있을 수 있는데, 이건 충돌이 아니다. 진짜 충돌은 이전 브링업의
맨프로세스(uvicorn, `python -m worker`)가 남아 있는 경우다. 죽일 때는 **PID 로**
죽여라 — `pkill -f 'python -m worker'` 는 그 패턴이 자기 SSH 명령줄에도 들어 있어서
자기 세션을 끊는다.

기동 후 확인:

```bash
scripts/edge.sh run "cd '<repo>' && docker compose ... ps -a"
scripts/edge.sh run "docker logs <api-container> 2>&1 | tail -40"
```

`ml-api` 가 unhealthy 로 재시작 루프에 빠지면 `depends_on` 때문에 `ml-worker`
까지 못 뜬다. 로그의 마지막 예외를 봐라. 흔한 실패들은
`references/troubleshooting.md` 에 정리돼 있다.

## 5. 워커에 카메라를 물린다

RTSP 가 살아 있어도 워커 로스터가 비어 있으면 아무 일도 일어나지 않는다.
등록 경로와 그 함정은 `references/worker-roster.md` 를 봐라. 핵심만 말하면,
카메라를 등록한 뒤에는 **`ml-worker` 를 재시작해야 한다.** 주기 폴링은 외부 백엔드의
`config_version` 을 보기 때문에 로컬 등록만으로는 절대 트리거되지 않는다.

## 6. 고친 것을 이슈로 남긴다

브링업 도중 고친 것은 그때그때 등록해라. 좋은 이슈는 이렇게 생겼다.

- **증상**: 실제 출력을 붙인다. 요약하지 말고 그대로.
- **원인**: 파일:줄 로 짚는다. "설정이 잘못됨" 같은 말은 다음 사람에게 도움이 안 된다.
- **왜 여태 안 잡혔나**: 이게 가장 값어치 있는 부분이다. 로컬 개발에서는 왜
  안 보였는지 설명하면 재발 방지책이 저절로 나온다.
- **재발 방지**: 이 부류를 통째로 잡는 검사 하나를 제안한다.

실제 예로, `httpx` 가 런타임 의존성인데 test 그룹에 선언돼 있어서 슬림 백엔드
이미지가 부팅 즉시 죽은 적이 있다. 로컬 `uv run` 은 기본 그룹에 test 가 포함돼
있어서 이 결함을 완전히 가리고 있었다. 그래서 재발 방지책은 "httpx 를 옮긴다"가
아니라 "이미지와 같은 조건으로 import 스모크를 돌린다"가 된다.

## 참고 문서

- `references/camera-webguard.md` — IDIS WebGuard 조작. 브라우저로 카메라를 만지기 전에 필독.
- `references/troubleshooting.md` — 이 환경에서 실제로 겪은 실패와 원인.
- `references/worker-roster.md` — 카메라 등록 경로와 워커 설정 갱신 규칙.
