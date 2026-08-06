#!/usr/bin/env bash
# 엣지 노드로 가는 SSH 를 열고 카메라 웹 UI 로 가는 포트포워드를 건다.
#
# 이 스크립트가 존재하는 이유는 두 가지다.
#
#  1. 엣지는 Tailscale 뒤에 있고, 노드가 다른 tailnet 에 "공유"돼 있다.
#     `tailscale switch` 를 쓰면 안 된다 — 이 머신은 동시에 다른 tailnet 을
#     쓰고 있어서 전환하면 그쪽 작업이 끊긴다. 공유 노드는 전환 없이
#     ProxyCommand 로 바로 닿는다.
#
#  2. 카메라는 요양원 내부망에만 있다. 엣지를 점프호스트로 삼아 각 카메라의
#     443 을 로컬 184NN 로 끌어온다. 채널 N -> 로컬 184NN 이라는 규칙은
#     camproxy.py 와 공유한다.
#
# 사용:
#   EDGE_HOST=... EDGE_USER=... EDGE_KEY=~/.ssh/... \
#     CAM_CHANNELS='1:10.0.0.11,2:10.0.0.12' ./edge.sh open    # 접속 + 포워드
#   ./edge.sh run 'docker ps'                                   # 원격 명령
#   ./edge.sh close                                             # 정리
#
# 노드 주소, 계정, 키 경로는 여기에 적지 않는다. 이 저장소는 공개돼 있고,
# 그 셋을 합치면 어느 현장의 어느 노드에 누구로 붙는지가 그대로 드러난다.
# 값 자체가 비밀은 아니지만 공개할 이유도 없다. 자격증명 노트에서 읽어
# 환경변수로 넘기거나, 커밋되지 않는 로컬 파일에 두고 source 해라.
set -euo pipefail

: "${EDGE_HOST:?EDGE_HOST 필요 (엣지 노드의 Tailscale 주소)}"
: "${EDGE_USER:?EDGE_USER 필요 (엣지 SSH 계정)}"
: "${EDGE_KEY:?EDGE_KEY 필요 (SSH 개인키 경로)}"
TAILSCALE="${TAILSCALE:-$(command -v tailscale || echo /usr/bin/tailscale)}"
CP="${EDGE_CONTROL_PATH:-$HOME/.ssh/cm/hn.sock}"

mkdir -p "$(dirname "$CP")"

ssh_opts=(
  -o ControlMaster=auto -o ControlPath="$CP" -o ControlPersist=120m
  -o BatchMode=yes -o ConnectTimeout=25 -o IdentitiesOnly=yes
  -i "$EDGE_KEY"
  -o ProxyCommand="$TAILSCALE nc $EDGE_HOST %p"
)

case "${1:-}" in
  open)
    # 마스터 연결부터. 이게 되면 나머지는 이 소켓을 재사용한다.
    ssh "${ssh_opts[@]}" "$EDGE_USER@$EDGE_HOST" true
    echo "ssh master up ($CP)"

    # 카메라 포워드. 이미 열려 있으면 ssh 가 알아서 거절하므로 무시한다.
    if [[ -n "${CAM_CHANNELS:-}" ]]; then
      IFS=',' read -ra items <<< "$CAM_CHANNELS"
      for item in "${items[@]}"; do
        ch="${item%%:*}"; ip="${item##*:}"
        [[ -z "$ch" || -z "$ip" ]] && continue
        port=$((18400 + ch))
        ssh -S "$CP" -O forward -L "127.0.0.1:${port}:${ip}:443" \
            "$EDGE_USER@$EDGE_HOST" 2>/dev/null \
          && echo "  forward 127.0.0.1:${port} -> ${ip}:443 (ch${ch})" \
          || echo "  forward ${port} 이미 열려 있음 (ch${ch})"
      done
    else
      echo "CAM_CHANNELS 미설정 — 포워드는 건너뛴다"
    fi
    ;;

  run)
    shift
    ssh -S "$CP" "$EDGE_USER@$EDGE_HOST" "$@"
    ;;

  close)
    ssh -S "$CP" -O exit "$EDGE_USER@$EDGE_HOST" 2>/dev/null || true
    echo "closed"
    ;;

  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
