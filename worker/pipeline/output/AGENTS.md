# worker/pipeline/output: post-admission side effects

Event staging, alert evidence attachment, operator live view, and the worker's
HTTP surface. Earned its own file: 40 modules including `evidence/`, 51 external
importing files (score 12, distinct domain). `evidence/AGENTS.md` owns
durability; this file covers the direct modules.

## Where to look

| Task | File | Notes |
| --- | --- | --- |
| Stage an admitted event | `event_sink.py` | `EvidenceEventSink.emit_for_frame(event, trigger)`; `emit` raises `ValueError`; a camera mismatch raises |
| Audit plus snapshot JPEG | `evidence_attacher.py` | `AlertEvidenceAttacher`, `SnapshotRenderer` |
| Latest JPEG per camera | `live_view.py` | `LatestFrameStore`, `LiveViewRenderer`, `LiveViewSubscriber` |
| Overlay drawing | `preview_renderer.py` | CPU `PreviewRenderer`, `BedZoneGeometry`, `PreviewTrack` |
| HTTP wire contract | `live_view_api.py` | path constants, request parsers, response models |
| HTTP handlers | `_mjpeg_http.py` | 828 LOC hotspot; `build_http_server`, `MjpegProbeError`, `BedZoneNotFoundError` |
| Clip re-analysis controls | `_clip_analysis_http.py` | relay-protected `handle_get` / `handle_post` |
| Server lifecycle | `mjpeg_server.py` | `MjpegServer`, `start_optional_mjpeg_server`; default `127.0.0.1:8090` |

## HTTP surface

The worker is the provider; `ml-api` is the only consumer and reaches it
server-side. Routes live as constants in `live_view_api.py`: `/probe`,
`/replay`, `/stream/<camera>`, `/snapshot/<camera>`, `/overlay/<camera>`, plus
the `/pose` and `/bed-zone/recognize` suffixes. `/replay` runs
`worker.replay.replay_recovered` on inputs the backend supplies; it is not a
local trace reader.

## Conventions

- `_`-prefixed modules are handler internals. Compose through `mjpeg_server.py`
  and parse through `live_view_api.py`.
- No package `__all__`; import from the concrete module.
- Overlay subjects arrive as `worker.types.preview.OverlaySelection`. Rendering
  is CPU work on a JPEG tap and never feeds a decision.
- `worker.domains` and `worker.pipeline.perception` may not import this package
  (import-linter). Runtime wires sinks in.

## Anti-patterns

- Calling `EvidenceEventSink.emit`: clip identity would be lossy without the
  trigger frame.
- Adding a route or query field in a handler without its constant and parser in
  `live_view_api.py`.
- Reading frames from anywhere except `LatestFrameStore` for live view.
- Putting queue, manifest, or clip-store logic here instead of `evidence/`.

## Focused tests

`tests/test_worker_event_sink.py`, `tests/test_worker_mjpeg_server.py`,
`tests/test_worker_live_view_composition.py`, `tests/test_preview_renderer.py`,
`tests/test_worker_replay_http.py`, `tests/test_worker_clip_analysis_http.py`,
`tests/test_replay_transfer_bound.py`, `tests/test_runtime_latest_frame.py`,
`tests/test_relay_audit_envelope_contract.py`.
