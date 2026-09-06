---
name: deepstream-serving-qa
description: >
  DeepStream Flow(nvinfer + NvDCF) 서빙 경로의 검출/추적 연속성을 실험으로 판정하는
  QA 절차. "ID가 끊긴다", "박스가 깜빡인다", "YOLO로는 잘 되는데 DeepStream은 왜",
  "사람이 앉아 있으면 안 잡힌다", "모델 탓인지 파이프라인 탓인지", 배치/엔진/ONNX
  교체, 임계값·해상도 변경 검증 — 이런 이야기가 나오면 추측하지 말고 이 스킬의
  사다리를 그대로 밟아라. 모든 도구는 `scripts/qa/`에 커밋되어 있고 운영 컨테이너는
  건드리지 않는다(격리 컨테이너 + 파일 소스). 결론은 숫자로만 낸다.
---

# DeepStream serving QA

## 언제 쓰나
- 운영 화면에서 사람 박스/ID가 간헐적으로 사라진다.
- 모델·임계값·해상도·배치·엔진을 바꾸기 전후를 비교해야 한다.
- "모델 recall 문제"와 "서빙 파이프라인 결함"을 분리해야 한다.

## 먼저 알아야 할 사실 (2026-09 #503에서 확정)
- 배치 결함의 서명: 운영 trace에서 트랙이 프레임의 11–17%에만 존재, 등장 간격이
  거의 전부 **짝수**(gap-2 ≈ 42%), 객체 행이 한쪽 `seq` 패리티에 90% 몰림, 같은 NvDCF
  id가 수십~수백 번 `new`로 재탄생. 원인은 batch=1 고정 ONNX를 nvinfer가
  `batch-size=13`(프로파일 min1/opt13/max13)으로 돌린 것. 배치당 한 슬롯만 유효 검출.
- 정상 서빙의 서명: gap-1 share ≥ 0.9, 패리티 편중 없음, 강한 검출(점수 ≥ 0.6)의
  presence ≥ 0.95, id 재탄생 한 자리 수.
- 증명하지 못하는 것들: HTTP 200/JPEG/fps/heartbeat 개수는 프레임 동일성·recall을
  증명하지 않는다. SDK 객체 수 ≠ 사람 수. `matched_tracks == objects`는 pose 매칭이
  됐다는 뜻이지 검출이 됐다는 뜻이 아니다. 격리 batch-1 결과 ≠ 운영 batch-13.
- 임계값 근처(0.2 pre-cluster) 박스는 FP16 노이즈로 켜졌다 꺼졌다 한다. 개수 불일치가
  전부 0.20–0.22 점수면 그건 결함이 아니다.
- 남는 끊김이 "야간 저신뢰 타깃"(점수 p50 < 0.5)에만 있으면 검출기/임계값/해상도
  영역이지 파이프라인이 아니다. 640×360 mux 입력에서 카운터 뒤 앉은 사람은 20–30px라
  nano 검출기가 놓친다.

## 사다리 (위에서 아래로, 각 단계는 숫자 통과선이 있다)

### 0. 운영 trace 연속성 (코드 변경 없음, 1분)
```bash
uv run python scripts/qa/trace_continuity.py /tmp/p1b-flow/traces/<camera>.jsonl --last-rows 27000
```
- 카메라 id ↔ trace 파일은 파일 마지막 행의 `camera_id`로 확인.
- 읽는 값: `presence_median`, `gap_share`, `object_row_seq_parity`, `max_rebirths_one_id`,
  `people_per_frame`, `score_p50`.
- gap-2 ≫ gap-1 이거나 패리티 편중이면 서빙 결함. gap-1 ≥ 0.9인데 presence가 낮으면
  검출 점수를 봐라(`score_p50`).

### 0b. 엔진 프로파일 확인
```bash
docker logs p1b-proof 2>&1 | grep -A2 'FullDims Engine Info'
uv run python -c "import onnxruntime as o;print(o.InferenceSession('<onnx>',providers=['CPUExecutionProvider']).get_inputs()[0].shape)"
```
- INPUT `min: 1x3x640x640 opt: 13x... Max: 13x...`인데 ONNX dim0가 정수(1)면 #503 재발.
  `edge_engine_build`/`cold_start`가 거부해야 정상이다.

### 1. ORT 배치 동등성 (CPU, 3분)
```bash
uv run python scripts/qa/ort_batch_parity.py --onnx models/pose/yolo26n-pose.onnx \
  --frames /tmp/pose-compare/parity-frames --batch 13 --out /tmp/parity.json
```
- 실제 프레임 ≥ 160장. 통과선: 좌표/키포인트 ≤ 1e-3, 점수 ≤ 1e-4, violations 0.
- 실패하면 export 자체가 배치 불안전. `python -m worker.tools.export_pose_onnx --force`로
  재export(dynamic, 재현 가능한 digest, `.sha256` 사이드카).

### 2. 격리 13소스 pre-tracker 동등성 (GPU, 10분)
소스는 **서로 다른 내용의 파일 13개**(같은 스트림 13개는 슬롯 누수를 못 잡는다).
샘플 길이(`ffprobe`)를 넘는 `-ss`를 주면 빈 파일이 나온다.
```bash
mkdir -p /tmp/p1b-flow/corpus/hetero
for i in $(seq 0 12); do docker run --rm -v /tmp/p1b-flow/corpus:/corpus --entrypoint ffmpeg \
  jrottenberg/ffmpeg:6-ubuntu -y -ss $((i*3)) -i /corpus/sample_1080p_h264.mp4 -t 8 -an \
  -c:v libx264 -preset veryfast -g 30 -pix_fmt yuv420p /corpus/hetero/clip$(printf %02d $i).mp4; done
URIS=$(python3 -c "print(','.join(f'file:///corpus/hetero/clip{i:02d}.mp4' for i in range(13)))")
# nvinfer 설정: 배포 served config를 복사하고 model-engine-file만 마운트 경로로 바꾼다
run() { docker run --rm --gpus all --name "$1" "${@:4}" -e NO_TRACKER=1 -e N_SOURCES=13 \
  -e DIAG_URIS="$URIS" -e INFER_CFG="$2" -e SECONDS=25 -e OUT="/out/$3" -e PYTHONPATH=/app \
  -v "$PWD/worker:/app/worker:ro" -v "$PWD/contracts:/app/contracts:ro" -v "$PWD/scripts/qa:/qa:ro" \
  -v /tmp/p1b-flow/corpus:/corpus:ro -v <models dir>:/app/models:ro -v <engine dir>:/trt:ro \
  -v <b1 assets>:/diagnostic:ro -v /tmp/pose-compare:/out --entrypoint python3 <edge image> /qa/batch_probe.py; }
run subject /out/nvinfer-b13.txt subject.json -e INFER_BATCH=13
run control /diagnostic/nvinfer-b1.txt control.json -e INFER_BATCH=1
run subject-perm /out/nvinfer-b13.txt subject-perm.json -e INFER_BATCH=13 -e PAD_PERMUTATION=12,3,7,0,9,1,11,5,2,10,4,8,6
uv run python scripts/qa/batch_probe_compare.py --subject /tmp/pose-compare/subject.json.rows.jsonl --control /tmp/pose-compare/control.json.rows.jsonl
uv run python scripts/qa/batch_probe_compare.py --subject /tmp/pose-compare/subject-perm.json.rows.jsonl --control /tmp/pose-compare/control.json.rows.jsonl --pad-permutation 12,3,7,0,9,1,11,5,2,10,4,8,6
```
- 통과선: `frame_keys_only_in_one_run` 0, IoU median ≥ 0.99, 개수 불일치는 전부 게이트
  경계 점수, |Δscore| > 0.05 극소수, 키포인트 p99 ≤ 2px, 순열 결과 = 비순열 결과.
- IoU 0.0 쌍이 점수 0.002 차이로 나오면 정렬 순서 뒤집힘이지 기하 오차가 아니다.
- 그 다음 `NO_TRACKER` 없이 한 번 더: presence per id, gap 히스토그램(보조 지표).
- **주의**: probe에서 `list(batch_meta.frame_items)`처럼 SDK 반복자를 materialize하면
  exit 139(segfault). 반드시 루프 안에서 읽는다. NvDCF accuracy 설정은 ReID TAO 모델이
  이미지에 없어서 초기화 실패한다(perf 설정만 쓴다).

### 3. 실카메라 13대 15분 게이트
- 배포는 `docker restart`가 아니라 **컨테이너 재생성**(마운트/환경은 restart로 못 바꾼다):
  이전 컨테이너를 `docker rename`으로 보존, 새 튜플(이미지 digest, worker 트리 revision,
  models dir, engine dir)을 `/tmp/p1b-flow/deploy-tuple-<sha>.json`에 기록. 롤백 = 이전 튜플.
- 부팅 후 heartbeat `frames=` 카운터를 t0/t1에 스냅샷해서 fps 계산:
  카메라당 ≥ 14.85, 합계 ≥ 193.05(현재는 30fps × 13 = 390), outage 0, segfault 0.
- 같은 창에서 0단계를 다시 돌려 gap-1 ≥ 0.9, 강한 타깃 presence ≥ 0.95 확인.

### 4. 눈으로 확인 (사람 수 논쟁이 있을 때)
```bash
curl -s -c /tmp/c -X POST http://127.0.0.1:8000/api/v1/auth/session -H 'content-type: application/json' -d '{"username":"admin","password":"<pw>"}'
curl -s -b /tmp/c -o /tmp/snap.jpg "http://127.0.0.1:8000/api/v1/streams/<camera>/snapshot?refresh=1"
```
- 스냅샷은 640×360 오버레이 프레임. 놓친 사람의 픽셀 높이를 재라. < 30px면 해상도
  문제이지 파이프라인이 아니다. 1080p 원본은 `trackID=1` RTSP에서 ffmpeg로 뽑아 CPU ORT
  640 vs 1280으로 비교한 뒤에만 해상도 변경을 제안한다.

## 하지 말 것
- 격리 결과 하나로 "모델 탓"/"파이프라인 탓" 단정. 사다리를 끝까지 밟는다.
- 임계값(#491)과 서빙 수정을 한 커밋에 섞기 — 회복량을 측정할 수 없게 된다.
- `worker/adapters/deepstream/configs/nvinfer-yolo26-pose.txt`의 dirty 변경을 쓸어 담기.
- 운영 컨테이너에 실험 설정 주입. 실험은 항상 `--rm` 격리 컨테이너.
- RTSP 자격증명/주민 이미지 로그·이슈 게시.

## 관련 파일
- `scripts/qa/trace_continuity.py`, `scripts/qa/ort_batch_parity.py`,
  `scripts/qa/batch_probe.py`, `scripts/qa/batch_probe_compare.py`
- `worker/tools/export_pose_onnx.py`, `worker/tools/edge_engine_build.py`,
  `worker/runtime/flow/onnx_shape.py`, `worker/runtime/flow/cold_start.py`
- 근거: GitHub #503, PR #504, `docs/runbooks/edge-image-publish.md`
