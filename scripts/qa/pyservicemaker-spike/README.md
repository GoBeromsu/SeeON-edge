# pyservicemaker P1b spike

Measurement-only G8a harness. It is intentionally under `scripts/` and imports
`pyservicemaker` only inside the DeepStream container; production code must not
import it.

Run from the repository root:

```bash
scripts/qa/pyservicemaker-spike/run.sh
```

The wrapper pins DeepStream 9.1 by digest, bind-mounts the repository read-only,
installs the image's missing `pyyaml` dependency, and writes raw JSON to
`/tmp/pyservicemaker-p1b-spike.raw.json` (or a supplied output path). The configured input is
DeepStream's bundled looping-capable MP4 rather than a facility camera or RTSP
endpoint. It does not create an engine, run an inference pipeline, or write
production artifacts.

The current result is a gate failure: the harness proves local Smart Record
start/stop calls and the FP16 engine build, but the local recording creates no
completion callback/file and no YOLO-pose parser configuration maps the
`1×300×57` output into DeepStream metadata. See the paired research report for
raw measured output and the required owner decision.
