"""P1b gate items 2/3/4/6 on live facility cameras (generation 3).

Production-shaped probe over ``batch_capture().infer().track()``:
- tensor-meta rows are copied device->host through cudart (no Torch in the
  media-plane process);
- rows are mapped from network space (letterboxed 640x640) into frame space and
  matched to tracked objects one-to-one with an IoU gate; a row is consumed at
  most once; an object without a gated row is explicitly ``unmatched``;
- NvDCF lifecycle is recorded per id (first/last frame, frames present, gaps);
- an id switch is counted only when a new id is born within 5 s of an id that
  actually disappeared (absent >= 2 frames) and their boxes overlap.

Camera URIs arrive only via ``LIVE_URIS`` in the environment.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from pyservicemaker import BatchMetadataOperator, Flow, Pipeline, Probe, RenderMode

REASSOC_SEC = 5.0
IOU_GATE = 0.5
SCORE_MIN = 0.05
NET_W, NET_H = 640, 640  # nvinfer infer-dims, maintain-aspect-ratio=1, asymmetric padding

_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cudart.cudaMemcpy.restype = ctypes.c_int
_CUDA_D2H = 2
_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


def tensor_rows(tensor) -> np.ndarray:  # noqa: ANN001 - vendor type
    """Copy the output tensor to host with cudart. No Torch, no second context.

    cudart uses the process's primary CUDA context, the same one DeepStream's
    plugins use, so this adds no CUDA context. The tensor is released with the
    buffer; the rows are our own copy.
    """
    capsule = tensor.__dlpack__(None)
    managed = ctypes.cast(
        _PyCapsule_GetPointer(capsule, b"dltensor"), ctypes.POINTER(_DLManagedTensor)
    ).contents
    dl = managed.dl_tensor
    n = 1
    for i in range(dl.ndim):
        n *= int(dl.shape[i])
    host = np.empty(n, dtype=np.float32)
    if dl.device.device_type == 1:  # kDLCPU
        ctypes.memmove(host.ctypes.data, dl.data + dl.byte_offset, n * 4)
    else:
        rc = _cudart.cudaMemcpy(host.ctypes.data, dl.data + dl.byte_offset, n * 4, _CUDA_D2H)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed: {rc}")
    return host.reshape(-1, 57)


def net_to_frame(row: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """Inverse of nvinfer's letterbox: scale = min(NET/W, NET/H), pad at right/bottom."""
    scale = min(NET_W / frame_w, NET_H / frame_h)
    return (row[0] / scale, row[1] / scale, row[2] / scale, row[3] / scale)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class _Track:
    __slots__ = ("first_frame", "frames_present", "gaps", "last_box", "last_frame", "last_t")

    def __init__(self, frame: int, box: tuple, t: float) -> None:  # noqa: ANN001
        self.first_frame = frame
        self.last_frame = frame
        self.frames_present = 1
        self.gaps = 0
        self.last_box = box
        self.last_t = t


class ProductionShapedProbe(BatchMetadataOperator):
    def __init__(self, sources: int, seconds: float, frame_w: int, frame_h: int) -> None:
        super().__init__()
        self.seconds = seconds
        self.deadline: float | None = None
        self.frame_w, self.frame_h = frame_w, frame_h
        self.frames = 0
        self.objects = 0
        self.batches = 0
        self.latencies_ms: list[float] = []
        self.matched = 0
        self.unmatched = 0
        self.match_ious: list[float] = []
        self.rows_reused_blocked = 0
        self.queues = [collections.deque(maxlen=64) for _ in range(sources)]
        self.drops = 0
        self.tracks: dict[int, dict[int, _Track]] = collections.defaultdict(dict)
        self.frame_index: dict[int, int] = collections.defaultdict(int)
        self.births: list[dict] = []
        self.switches = 0
        self.first_pts: dict[int, int] = {}
        self.last_pts: dict[int, int] = {}
        self.done = threading.Event()

    def handle_metadata(self, batch_meta) -> None:  # noqa: ANN001 - vendor type
        started = time.perf_counter_ns()
        now = time.monotonic()
        if self.deadline is None:
            self.deadline = now + self.seconds
        for frame in batch_meta.frame_items:
            src = int(frame.pad_index)
            self.frames += 1
            self.frame_index[src] += 1
            fidx = self.frame_index[src]
            self.first_pts.setdefault(src, int(frame.buffer_pts))
            self.last_pts[src] = int(frame.buffer_pts)
            rows = None
            for tm in frame.tensor_items:
                layers = tm.as_tensor_output().get_layers()
                t = layers.get("output0")
                if t is not None:
                    rows = tensor_rows(t)
                break
            candidates: list[tuple[int, tuple[float, float, float, float]]] = []
            if rows is not None:
                for i, r in enumerate(rows):
                    if r[4] > SCORE_MIN:
                        candidates.append((i, net_to_frame(r, self.frame_w, self.frame_h)))
            objects = []
            for obj in frame.object_items:
                rp = obj.rect_params
                box = (
                    float(rp.left),
                    float(rp.top),
                    float(rp.left + rp.width),
                    float(rp.top + rp.height),
                )
                objects.append((int(obj.object_id), box, float(obj.confidence)))
            # deterministic one-to-one gated matching: best IoU first, each row consumed once
            pairs = sorted(
                (
                    (iou(box, cbox), oi, ri)
                    for oi, (_, box, _) in enumerate(objects)
                    for ri, (_, cbox) in enumerate(candidates)
                ),
                reverse=True,
            )
            used_rows: set[int] = set()
            assigned: dict[int, int] = {}
            for score, oi, ri in pairs:
                if score < IOU_GATE:
                    break
                if oi in assigned:
                    continue
                if ri in used_rows:
                    self.rows_reused_blocked += 1
                    continue
                assigned[oi] = ri
                used_rows.add(ri)
                self.match_ious.append(score)
            live_ids: set[int] = set()
            tracks = self.tracks[src]
            for oi, (oid, box, conf) in enumerate(objects):
                self.objects += 1
                live_ids.add(oid)
                ri = assigned.get(oi)
                kps = None
                if ri is not None:
                    self.matched += 1
                    kps = rows[candidates[ri][0]][6:57].reshape(17, 3).tolist()
                else:
                    self.unmatched += 1
                record = {
                    "source": src,
                    "id": oid,
                    "box": box,
                    "keypoints": kps,
                    "confidence": conf,
                    "pts": int(frame.buffer_pts),
                }
                q = self.queues[src]
                if len(q) == q.maxlen:
                    self.drops += 1
                q.append(record)
                tr = tracks.get(oid)
                if tr is None:
                    # birth: switch only if some id truly disappeared (>= 2 frames absent)
                    # within the window and its last box overlaps this one
                    lost = [
                        (lid, lt)
                        for lid, lt in tracks.items()
                        if fidx - lt.last_frame >= 2
                        and now - lt.last_t <= REASSOC_SEC
                        and iou(box, lt.last_box) > 0.3
                    ]
                    self.births.append(
                        {
                            "t": round(now, 2),
                            "source": src,
                            "id": oid,
                            "frame": fidx,
                            "switch_from": [lid for lid, _ in lost],
                        }
                    )
                    if lost:
                        self.switches += 1
                    tracks[oid] = _Track(fidx, box, now)
                else:
                    if fidx - tr.last_frame > 1:
                        tr.gaps += 1
                    tr.last_frame = fidx
                    tr.frames_present += 1
                    tr.last_box = box
                    tr.last_t = now
        self.batches += 1
        self.latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
        if self.deadline is not None and now >= self.deadline:
            self.done.set()


def pct(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(fraction * (len(s) - 1))))]


def _json_safe(value: object) -> object:
    """numpy scalars are not JSON serialisable; keep the receipt writable."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unserialisable {type(value).__name__}")


def write_report(
    probe: ProductionShapedProbe,
    sources: int,
    elapsed: float,
    error: str | None,
    out: str,
    extra: dict,
) -> None:
    lat = probe.latencies_ms
    per_source_fps = {
        str(s): round(n / max(1e-9, (probe.last_pts[s] - probe.first_pts[s]) / 1e9), 2)
        for s, n in probe.frame_index.items()
        if s in probe.last_pts and probe.last_pts[s] > probe.first_pts[s]
    }
    lifecycle = {
        str(s): {
            str(oid): {
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
                "frames_present": t.frames_present,
                "gaps": t.gaps,
            }
            for oid, t in tr.items()
        }
        for s, tr in probe.tracks.items()
    }
    report = {
        "sources": sources,
        "error": error,
        "elapsed_sec": round(elapsed, 1),
        "frames": probe.frames,
        "objects": probe.objects,
        "batch_callbacks": probe.batches,
        "batch_callbacks_per_sec": round(probe.batches / max(elapsed, 1e-9), 2),
        "frame_items_per_sec": round(probe.frames / max(elapsed, 1e-9), 2),
        "per_source_fps_from_pts": per_source_fps,
        "probe_latency_ms": {
            "p50": pct(lat, 0.5),
            "p95": pct(lat, 0.95),
            "p99": pct(lat, 0.99),
            "max": max(lat) if lat else None,
            "mean": round(statistics.fmean(lat), 4) if lat else None,
        },
        "batch_interval_budget_ms": round(1000 / 30, 2) if sources else None,
        "queue_drops_no_consumer": probe.drops,
        "association": {
            "iou_gate": IOU_GATE,
            "matched": probe.matched,
            "unmatched": probe.unmatched,
            "row_reuse_blocked": probe.rows_reused_blocked,
            "match_iou_p50": pct(probe.match_ious, 0.5),
            "match_iou_p05": pct(probe.match_ious, 0.05),
        },
        "distinct_ids_by_source": {str(s): len(v) for s, v in probe.tracks.items()},
        "id_births": len(probe.births),
        "id_switches_disappearance_gated": probe.switches,
        **extra,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {**report, "lifecycle": lifecycle, "births": probe.births[:200]},
            fh,
            indent=2,
            default=_json_safe,
        )
    print(json.dumps(report, indent=2), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-config", required=True)
    ap.add_argument("--tracker-config", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ort-fall-model", default=None)
    ap.add_argument("--cuda-apps-out", default=None)
    args = ap.parse_args()
    uris = [u.strip() for u in os.environ["LIVE_URIS"].splitlines() if u.strip()]
    frame_w, frame_h = 640, 360
    probe = ProductionShapedProbe(len(uris), args.seconds, frame_w, frame_h)
    extra: dict = {"torch_imported": "torch" in __import__("sys").modules}
    if args.ort_fall_model:
        import onnxruntime as ort

        session = ort.InferenceSession(args.ort_fall_model, providers=["CPUExecutionProvider"])
        extra["ort_providers"] = session.get_providers()

        def score_forever() -> None:
            window = np.zeros((1, 30, 56), dtype=np.float32)
            n = 0
            t0 = time.perf_counter()
            while True:
                session.run(None, {"window": window})
                n += 1
                extra["ort_inferences"] = n
                extra["ort_per_sec"] = round(n / max(1e-9, time.perf_counter() - t0), 1)
                time.sleep(1 / 15 / 13)

        threading.Thread(target=score_forever, daemon=True).start()

        def snapshot_cuda() -> None:
            time.sleep(45)
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
                capture_output=True,
                text=True,
            ).stdout
            rows = [line.strip() for line in out.splitlines() if line.strip()]
            extra["cuda_compute_apps"] = rows
            extra["cuda_process_count"] = max(0, len(rows) - 1)
            extra["torch_imported_at_snapshot"] = "torch" in __import__("sys").modules
            if args.cuda_apps_out:
                Path(args.cuda_apps_out).write_text(json.dumps(extra, indent=2))

        threading.Thread(target=snapshot_cuda, daemon=True).start()

    pipeline = Pipeline("p1b-live")
    flow = (
        Flow(pipeline)
        .batch_capture(uris, width=frame_w, height=frame_h)
        .infer(args.infer_config)
        .track(
            ll_config_file=args.tracker_config,
            ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        )
        .attach(what=Probe("prod-probe", probe))
        .render(mode=RenderMode.DISCARD, enable_osd=False, sync=False)
    )
    for i in range(len(uris)):
        pipeline[f"batch_capture-source-0_{i}"].set(
            {
                "select-rtp-protocol": 4,
                "latency": 200,
                "init-rtsp-reconnect-interval": 5,
                "rtsp-reconnect-interval": 5,
            }
        )

    started = time.monotonic()

    def stopper() -> None:
        probe.done.wait()
        write_report(probe, len(uris), time.monotonic() - started, None, args.out, extra)
        pipeline.stop()

    threading.Thread(target=stopper, daemon=True).start()
    error = None
    try:
        flow()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    write_report(probe, len(uris), time.monotonic() - started, error, args.out, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
