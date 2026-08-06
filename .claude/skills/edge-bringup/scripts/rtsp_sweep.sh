#!/usr/bin/env bash
# 엣지 서버에서 전 채널 RTSP 를 실측한다.
#
# 반드시 엣지에서 돌려야 한다. 카메라는 요양원 내부망에만 있고, 여기서 나오는
# 결과가 곧 ML 워커가 보게 될 현실이다. 노트북에서 터널로 찔러본 결과는
# 워커가 붙을 수 있다는 증거가 되지 못한다.
#
# 사용:
#   CAM_PASSWORD=... CAM_CHANNELS='1:10.0.0.11,2:10.0.0.12' ./rtsp_sweep.sh
#
# 출력 해석이 진단의 핵심이다:
#   hevc,640,360,30/1          정상
#   454 Session Not Found      RTP 보안이 켜져 있다 -> references/camera-webguard.md
#   401 Unauthorized           자격증명이 틀렸다. 비밀번호를 "고치기" 전에 멈춰라
#   TIMEOUT                    네트워크 도달 실패. 카메라 설정 문제가 아니다
#
# 401 이 뜨면 다른 비밀번호를 연달아 시도하지 마라. 카메라 계정은 오인증
# 누적으로 잠긴다. 기록된 값이 최신인지부터 확인해라 — 실제로 노트 쪽이
# 틀렸던 전례가 있다.
set -euo pipefail

: "${CAM_PASSWORD:?CAM_PASSWORD 필요}"
: "${CAM_CHANNELS:?CAM_CHANNELS 필요 (예: '1:10.0.0.11,2:10.0.0.12')}"
CAM_USER="${CAM_USER:-admin}"
TRACK="${CAM_TRACK:-trackID=2}"   # 2 = 서브스트림(ML 입력용), 1 = 메인
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 원격 프로그램은 argv 로, 비밀은 stdin 으로 나눠 보낸다.
# 비밀번호를 원격 명령줄에 얹으면 `ps` 에 그대로 보이고, 공유 서버에서는
# 그 자체가 유출이다. `bash -s` 는 stdin 에서 프로그램을 읽으므로 쓸 수 없다 —
# 프로그램을 -c 인자로 주고 stdin 은 데이터 전용으로 비워둔다.
REMOTE_PROG='
IFS= read -r CAM_USER
IFS= read -r CAM_PASSWORD
IFS= read -r CAM_CHANNELS
IFS= read -r TRACK
ok=0; bad=0
IFS="," read -ra items <<< "$CAM_CHANNELS"
for item in "${items[@]}"; do
  ch="${item%%:*}"; ip="${item##*:}"
  [ -z "$ch" ] && continue
  out=$(timeout 25 ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
        -show_entries stream=codec_name,width,height,avg_frame_rate -of csv=p=0 \
        "rtsp://${CAM_USER}:${CAM_PASSWORD}@${ip}:554/${TRACK}" 2>&1 | head -1)
  case "$out" in
    hevc*|h264*) ok=$((ok+1)) ;;
    *)           bad=$((bad+1)) ;;
  esac
  printf "ch%-3s %-16s %s\n" "$ch" "$ip" "${out:-TIMEOUT}"
done
echo "---- OK=$ok FAIL=$bad ----"
'

printf '%s\n%s\n%s\n%s\n' "$CAM_USER" "$CAM_PASSWORD" "$CAM_CHANNELS" "$TRACK" \
  | "$SKILL_DIR/edge.sh" run "bash -c $(printf '%q' "$REMOTE_PROG")"
