# seeon-edge v0.1.1

Second sealed edge release. Built from the tagged commit and published to GHCR
as two digest-pinned images; the sealed compose stack now provisions every
runtime model artifact from the fetch manifest.

## Highlights

**Serving correctness (#503, #504)**

- The pose model was exported with a fixed batch dimension of 1 and served by
  nvinfer at `batch-size=13`; only one frame per 13-frame batch received valid
  detections, which showed up as people tracked in 11–17 % of frames with an
  even-gap signature. The pose ONNX is now exported with a dynamic batch axis
  by an owned tool, `edge-engine-build` and the boot gate refuse a fixed-batch
  graph at any batch above 1, and the build verifies nvinfer's real profile
  (`min 1 / opt 13 / max 13`). Live result on 13 cameras: 30 fps each,
  consecutive-frame share of track observations 0.93–0.99 (was ~0.10).

**Evidence clips (#500, #508)**

- Cameras deliver HEVC and Smart Record keeps the original bytes; browsers
  without hardware HEVC decode could not play them. A derived H.264 playback
  rendition is written beside each clip off the critical path and served by
  `/clips/{id}/video` after digest verification; the original stays immutable
  and is what `/artifacts` exports. `worker.tools.clip_playback_backfill`
  covers existing stores.

**Bed-zone recognition (#507)**

- One-off recognition runs on a native-resolution frame grabbed from the
  camera's registered RTSP URL with `yolo26l-seg` at 1280 instead of the
  640×360 DeepStream snapshot with the medium model; recognition deadlines are
  20 s (worker) / 25 s (backend). Facility default confidence is 0.15.
- Overlay labels read `사람 84% · 침대2 · 정상` (detection confidence, saved
  bed containing the foot point, fall state) instead of the tracker counter.

**Model provisioning and CI (#499, #509, #510)**

- The fall bundle `model.onnx` and the pose/bed ONNX artifacts are published
  (Hugging Face revision `2c46e52e`, GitHub release `models-onnx-2026-09-07`)
  and pinned in `worker/tools/fetch_models/manifest.json`; a fetch verifies
  15/15 artifacts.
- Pull-request CI fetches public artifacts only and skips the fifteen
  private-bundle test modules explicitly; a job that never runs on
  `pull_request` provisions the private bundle and runs the full suite.
- The fall-model selection mount is an opt-in overlay
  (`compose.edge.model-selection.yaml`); the unconditional bind made Docker
  create a directory on a fresh host and the worker refused to boot (#498).

## Operator notes

- Rebuild the engine on upgrade: the boot gate refuses the previous
  fixed-batch ONNX at batch 13 by design. `edge-engine-build` runs on every
  `up` and prints the profile receipt.
- Existing HEVC clips need `python -m worker.tools.clip_playback_backfill
  /var/lib/clip-store` once (run from the worker image).
- Cameras registered on the 1080p main stream (`trackID=1`) get
  native-resolution bed recognition; substream registrations recognize at
  640×360.

## Known limitations

- Lying residents under bedding in patient rooms score below every
  off-the-shelf pose model (#497); fine-tuning on facility frames is the
  planned path. NvDCF acceptance gates for borderline detections are under
  study (#505); a global sensitivity control is planned (#506).
- Seated people far from the camera flicker around the 0.2 pre-cluster gate
  at the 640×360 inference size.
