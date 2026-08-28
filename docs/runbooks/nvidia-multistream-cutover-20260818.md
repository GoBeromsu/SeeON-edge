# Cutover record — NVIDIA multistream serving (#312), 2026-08-18

Deployment-path exception, explicitly user-approved (plan todo 13(c)): this is the **local compose
project** `seeon-edge-wt-alert-api` on the workstation, running hand-built local image tags via
direct `docker compose up`. The sanctioned sealed-release flow in `edge-image-publish.md` governs
official facility deployments to the remote edge host and was **not** used here.

Full evidence (raw command output, 29-row soak table, before/after telemetry):
`.omo/evidence/task-13-nvidia-multistream-serving.md` in the canonical checkout.

## What shipped

| | Before | After |
|---|---|---|
| ml-worker | `local/fall-ml-worker:41d2a0d` | `local/fall-ml-worker:e2c1d37` |
| ml-api | `local/fall-ml-api:settingsfix` | `local/fall-ml-api:e2c1d37` |

Built from `feat/nvidia-multistream-serving` @ `e2c1d37698be3cd2fe7754720e893b67d6b0fe7f`
(Waves 1–4). Both Dockerfiles require a valid 40-hex `SOURCE_REVISION`:

```sh
cd <worktree>
SHA=$(git rev-parse HEAD)
docker build -f Dockerfile.edge --target runtime    --build-arg SOURCE_REVISION=$SHA -t local/fall-ml-worker:e2c1d37 .
docker build -f Dockerfile.backend --build-arg SOURCE_REVISION=$SHA -t local/fall-ml-api:e2c1d37 .
```

Cutover is image-only — compose carries `build: !reset null`, so it never rebuilds from source.
Edit the two image lines in the **untracked** `.env` of the canonical checkout, then:

```sh
cd /home/seniorsailab/beomsukoh/SeeON/SeeON-edge
docker compose -p seeon-edge-wt-alert-api -f compose.edge.yaml -f compose.edge.nvidia.yaml up -d --pull never
```

Project name and the `edge-state` volume **must** stay unchanged — they carry the 13-camera
registry (`registry_version=94`). Never `down -v`.

Models are a read-only bind mount from the canonical checkout `./models`, resolved against the
compose CWD — they are not baked into the image. Post-start assert:
`docker exec <ml-worker> ls /app/models/fall/lstm/model.pt`.

## Result

**#312 crash loop eliminated.** `RestartCount` 35 → **0**, sustained across a 33-minute
13-camera soak with zero watchdog trips and zero tracebacks.

Verified working in production:
- **Out-of-process NVDEC decode engaged** — 12 `ffmpeg -hwaccel cuda -c:v hevc_cuvid -i pipe:0
  -f rawvideo pipe:1` children of the worker process, stable for the whole soak.
- **Batched serving engaged** — `batch_sizes` bucket `12` is the only one still incrementing.
- **Inference drop 97.6% → 0.11%.**
- **Live preview lane decoupled** — was `taken: 0 / dropped: 24029` (starved), now
  `taken: 7851 / dropped: 0`.
- Evidence lane still `dropped: 0` (ADR-0001 preservation intact).
- Dashboard login works with the b14420a auth changes (`POST /api/v1/auth/session` → 204).
- No stall ever exceeded 4.27 s (the startup max); the 20 s+ stalls of #312 did not recur.

## Known gap — throughput target NOT met

Steady-state batched pose forward is **~2.3 s per 12-frame batch**, giving **~0.43 fps/camera
(~5.6 inf/s aggregate)** against the plan target of 5 fps/camera (≥65 inf/s). `pose last_sec`
sits at **2.2–3.0 s** against the todo-13 gate of **< 1 s**.

Bottleneck localisation (diagnosis only — no live tuning was attempted):
- GPU util **0–5%**, NVDEC util **0–1%**, worker CPU **115%** of 2400% — nothing is saturated.
- Batching is genuinely engaged, so this is not a batch-admission starvation problem.
- `pose last_sec` is the elapsed time of the whole `_forward()` call in
  `worker/pipeline/inference_coordinator.py:201`, recorded identically for every camera in the
  batch. The ~2.3 s is therefore spent **inside the batched forward path**
  (`BatchServingClient.infer_batch` — preprocessing / host↔device transfer), not in GPU compute
  and not in idle waiting (`idle_sleep_sec` is hard-capped to 5–10 ms).

Next investigation should profile `infer_batch` preprocessing rather than the model itself.

Caveat: **12** cuvid children for **13** cameras (`max_batch_size` is 12). All 13 cameras report
`failure_category=None` and accrue pose samples, so none is unserved, but the asymmetry is worth
a look.

## Reverting

Rollback ceremony was waived by the user; the previous tags are retained untouched. To revert,
restore `ML_WORKER_IMAGE=local/fall-ml-worker:41d2a0d` and
`ML_API_IMAGE=local/fall-ml-api:settingsfix` in `.env` and re-run the same `up -d --pull never`.
Note this returns the stack to the crash-looping build.
