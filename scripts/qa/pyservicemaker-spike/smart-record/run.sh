#!/usr/bin/env bash
# Smart Record measurement against a LIVE RTSP source (the DS docs: only RTSP
# sources are enabled for smart record; recording cannot start until an
# I-frame is cached; overlapping records are unsupported).
#
# Credentials never enter the repository: pass the camera URI via SR_RTSP_URI.
set -euo pipefail
: "${SR_RTSP_URI:?set SR_RTSP_URI to the camera RTSP URI (credentials stay in your environment; never commit them)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-/tmp/pyservicemaker-spike-sr}"
mkdir -p "$WORK/records"
cp "$HERE/sr_stop.c" "$HERE/sr_live.py" "$WORK/"
docker build -q -f "$HERE/Dockerfile.sr" -t p1b-sr:local "$HERE" >/dev/null
docker run --rm --entrypoint bash -v "$WORK:/work" p1b-sr:local -lc \
  'cd /work && gcc -o sr_stop sr_stop.c -I/opt/nvidia/deepstream/deepstream/sources/includes $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0)'
for mode in duration stop overlap; do
  rm -f "$WORK/records/"*
  echo "== plugin-level mode=$mode"
  docker run --rm --gpus all --network host --entrypoint bash -v "$WORK:/work" \
    -e SR_RTSP_URI -e SR_OUT_DIR=/work/records -e SR_MODE="$mode" p1b-sr:local -lc "/work/sr_stop" 2>&1 | grep -E '^\[' | tee "$WORK/plugin-$mode.log"
  for f in "$WORK"/records/*.mp4; do
    [ -f "$f" ] && echo "  $(basename "$f") $(stat -c%s "$f")B $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")s"
  done
done
rm -f "$WORK/records/"*
echo "== pyservicemaker Pipeline.stop_recording (binding)"
docker run --rm --gpus all --network host --entrypoint bash -v "$WORK:/work" \
  -e SR_RTSP_URI -e SR_OUT_DIR=/work/records -e SR_REPORT=/work/binding-stop.json p1b-sr:local -lc "python3 /work/sr_live.py" >/dev/null 2>&1 || true
python3 -c "import json;d=json.load(open('$WORK/binding-stop.json'));[print(' ',e['event'],e['args']) for e in d['events']];print('  sizes',d.get('sizes'))"
