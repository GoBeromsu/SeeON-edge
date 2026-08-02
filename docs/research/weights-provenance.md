# Model-weight provenance and disclosure scope

**Scope.** This records what must be disclosed for model artifacts that the
edge worker can load. It does **not** make a licence decision: this project is
already settled as AGPL-3.0 with full public disclosure. In particular, removing
a runtime `ultralytics` import would not alter the scope described here.

## Method and artifact inventory

The inventory comes from loader call sites, not filenames. These citations
point at the current `worker/` package (the `edge/` tree they were ported
from is being retired; see `docs/architecture.md` "Source-to-target
ownership" for the historical mapping):

- `worker/runtime/worker.py:724-732` constructs the normal worker's `pose`,
  `person`, and `bed` runners and uses LSTM when `models.fall` is configured;
  `worker/runtime/worker.py:734-754`'s `_create_fall_model` fails closed
  (raises and refuses to boot) when `models.fall` is absent instead -- there
  is no registry fall fallback.
- `contracts/artifacts.py:7-24`, `worker/adapters/model/yolo_pose.py:33-69`,
  `worker/adapters/model/yolo_bed_seg.py:41-90`, and
  `worker/adapters/model/yolo_person.py:23-71` resolve and load the three
  YOLO checkpoints with `load_yolo_model(...)`.
- `worker/adapters/model/torch_lstm_fall.py:66-178,204-225` loads the LSTM
  manifest, architecture, and PyTorch state dict;
  `worker/ml-worker.example.yaml:9-22` pins the normal configuration to
  `models/fall/lstm`.
- `worker/adapters/model/sklearn_fall.py:86-150` loads the sklearn fall
  registry model's joblib model and metadata; retained as a loadable adapter
  (see `worker/adapters/model/__init__.py`) but no longer wired as an
  automatic runtime fallback.

`person` is a supported registry loader but is not included by the normal
`_build_runner_bundle`; it is retained below so the disclosure inventory covers
every worker loader call site. Likewise, each listed companion architecture or
metadata file is a worker-read artifact associated with the checkpoint/model.

## Verdicts

| Worker artifact(s) | Worker loading condition | ultralytics-trained | Was a pretrained Ultralytics weight the starting point? | Evidence in `eldercare-dataset-ops` |
| --- | --- | --- | --- | --- |
| `models/pose/yolo26n-pose.pt` | Normal `pose` runner | **unknown** | **unknown** | `ml/training/pose_extraction.py:32-35` and `ml/training/extract_poses.py:120-151` use `ultralytics.YOLO` only for pose **inference** and cache generation. `imported/eldercare-fall-ai-training/experiments/nh-pose-scale-measurement.md:36-37` says domain fine-tuning was deferred. `manifests/models/README.md` defines the needed provenance-manifest contract, but no run manifest exists. These do not establish the supplied checkpoint's origin. |
| `models/bed/yolo26m-seg.pt` | Normal `bed` runner (bed-exit) | **unknown** | **unknown** | `docs/adr/0006-model-handoff-contract.md` documents the bed artifact contract, but the sibling repository has no bed training script, dataset YAML, run log, or populated model provenance manifest. `manifests/models/README.md` is contract-only. |
| `models/person/yolo26n.pt` | Supported `person` registry runner; not created by normal bundle | **unknown** | **unknown** | No person-training pipeline or populated model-run manifest was found. The sibling repository's pose files only prove inference use, not this checkpoint's production. `manifests/models/README.md` remains contract-only. |
| `models/fall/lstm/model.pt`, `models/fall/lstm/arch.json`, `models/fall/lstm/metadata.yaml` | Normal fall configuration | **no** | **no** | `ml/training/models/lstm.py` defines the custom PyTorch `nn.LSTM`; `ml/training/models/base.py:182-195` trains with `module.train()`, Adam, and CrossEntropyLoss and saves a PyTorch state dict. No Ultralytics training/fine-tuning path or pretrained-Ultralytics initialization is present for this model family. |
| `models/fall/random-forest/model.pkl`, `models/fall/random-forest/metadata.json` | Sklearn fall registry model, loadable via `worker/adapters/model/sklearn_fall.py`; **not** an automatic fallback -- `WorkerRuntime._create_fall_model` (`worker/runtime/worker.py:734-754`) fails closed and refuses to boot when `models.fall` is absent, rather than switching to this model | **no** | **no** (not applicable to this sklearn model) | `ml/training/models/rf.py:16,36-54` defines and fits `RandomForestClassifier`; `ml/training/models/catalog.py` lists the supported non-YOLO fall families. The worker additionally requires `framework == "sklearn"` in `worker/adapters/model/sklearn_metadata.py:34-41`. |

The release trees cannot fill the YOLO gaps: `ml/data/releases/v1/DATA_CARD.md`
describes test-only diagnostic data with empty train/validation splits, and
`ml/data/releases/v2/DATA_CARD.md` records zero evaluable rows and prohibits
production/clinical claims. They are dataset releases, not model releases.

## Questions required to resolve the unknowns

1. **Pose checkpoint:** For the exact `yolo26n-pose.pt` SHA-256, provide its
   model-run manifest and Ultralytics training/fine-tuning run configuration/log,
   including the initial checkpoint identifier and dataset/release pin.
2. **Bed checkpoint:** For the exact `yolo26m-seg.pt` SHA-256, provide the same
   model-run manifest, training/fine-tuning config/log, initial checkpoint, and
   dataset/release pin.
3. **Person checkpoint:** For the exact `yolo26n.pt` SHA-256, provide the same
   provenance record; if it was obtained rather than trained here, identify the
   upstream release and its licence/source provenance.

Until those records exist, the three YOLO verdicts remain **unknown**. Their
names are not evidence of either Ultralytics training or an Ultralytics
pretrained starting point.

## Disclosure scope conclusion

Full public disclosure is already required under the settled AGPL-3.0 decision.
This provenance finding scopes the material that must accompany it:

- Disclose the custom LSTM and Random Forest training code, configurations,
  input/data-release pins, evaluation/provenance metadata, and the resulting
  model artifacts/checkpoints. The evidence supports that neither family was
  trained or initialized with Ultralytics.
- Do **not** exclude the pose, bed, or person checkpoints from disclosure scope.
  Their missing provenance prevents a narrower claim. Before distribution,
  publish or obtain the SHA-256-linked model-run manifest, training/fine-tuning
  code and configuration, dataset pin, logs, and initial-checkpoint provenance
  for each. If any was trained or fine-tuned with Ultralytics code, disclose the
  relevant training code and model produced by that code; a scratch training run
  with that code does not remove this requirement.
- The absence of a runtime import does not change this production-history-based
  scope.
