# LASER Paper-Compatible Point-Map Evaluation Design

**Date:** 2026-08-11

**Base branch:** `codex/pi3-ordinary-prediction-cache`

**Base commit:** `fcbe67981e905b13c4c1620b04a372760ecc87db`

**Implementation branch:** `codex/laser-paper-pointmap-eval`

## Summary

Implement a protocol-driven, paper-compatible point-map evaluation tool for
LASER while leaving reconstruction behavior unchanged. The first acceptance
target is not a local reproduction run: the implementation must let the user
run the full 7-Scenes and NeuralRGBD evaluation in a supported cloud
environment and compare its outputs directly with LASER Table 4.

The existing `mv_recon/eval.py` already contains the official center crop,
Umeyama alignment, point-to-point ICP, and bidirectional nearest-neighbour
metrics. Its reconstruction configuration is not paper-compatible, however.
It loads `configs/pipeline/default.yaml` without overrides and therefore
inherits Atomic segmentation and a 0.30 registration confidence keep ratio.
The official indoor point-map protocol uses Depth segmentation and behavior
equivalent to keeping the top 50 percent of confidence values for both depth
layer extraction and adjacent-window registration.

The new evaluator introduces an immutable `laser_paper` profile, validates the
fully resolved pipeline before model construction, preserves official metric
math, writes auditable result artifacts, and supports safe cloud smoke runs and
resume. Extended diagnostics remain separate from the six Table 4 metrics.

## Goals

1. Reproduce the LASER CVPR 2026 Table 4 protocol with the Pi3 backbone.
2. Make every algorithmic evaluation setting explicit and machine-verifiable.
3. Reject protocol drift before loading Pi3 or using GPU memory.
4. Preserve the official Acc, Comp, and NC mean/median definitions.
5. Aggregate by the fixed sequence map without stale files or wrong
   denominators.
6. Produce complete JSON, CSV, resolved configuration, and provenance
   artifacts suitable for cloud experiments and paper tables.
7. Reuse the existing ordinary prediction cache without redesigning it.
8. Add useful completeness and alignment diagnostics without mixing them into
   the paper-compatible primary result.
9. Support deterministic smoke runs and safe resume for long cloud jobs.

## Non-goals

- Changing Pi3 inference or model weights.
- Changing window registration, confidence selection, segmentation, anchor
  propagation, loop closure, or aggregation mathematics.
- Refactoring `pipeline/`, `inference_engine/`, `pi3/`, or `loop_closure/`.
- Replacing Open3D ICP or normal estimation with a custom implementation.
- Combining pose, video-depth, and point-map evaluation into one new framework.
- Running the complete 7-Scenes and NeuralRGBD benchmark locally.
- Treating partial-dataset or extended-metric results as LASER Table 4 results.

## Confirmed Base-Branch Problems

### The evaluator silently selects the wrong segmentation method

`mv_recon/eval.py::create_streaming_pi3()` loads
`configs/pipeline/default.yaml`, overrides the window to 20/5, enables anchor
propagation, and disables loop closure. It does not override
`segmentation.method`. The default is `atomic`, so a result produced by the
current script is not the LASER Depth baseline.

### Adjacent-window registration keeps the wrong confidence fraction

The modular engine uses `loop.registration.confidence_keep_ratio` for
sequential registration even when loop closure is disabled. The base default
is 0.30. Official `StreamingWindowEngine` behavior uses a 0.5 quantile and
therefore keeps the top 50 percent. A paper-compatible profile must lock this
field to 0.5 as well as locking
`segmentation.confidence_keep_ratio` to 0.5.

### Pipeline replacement hides the actual resolved protocol

The current evaluator first loads one resolved config and then changes it with
`dataclasses.replace()`. Those replacements are not represented by the
loader's `resolved_yaml` or SHA256. The new evaluator must pass overrides into
`load_pipeline_config()` so the resolved pipeline and its hash describe the
actual execution.

### Aggregation assumes that dataset length equals sequence-map length

The current script iterates the sequence map but divides totals by
`len(dataset)`. Paper-compatible aggregation must use the exact attempted
sequence-map entries, assert the expected coverage, and refuse to produce a
complete summary if any sequence is missing or failed.

### Existing outputs are not sufficient to audit a cloud run

The legacy CSVs omit the effective pipeline, checkpoint digest, sequence-map
digest, dependency versions, completion state, and cache identity. They also
do not safely distinguish current results from stale output in a reused
directory.

## Scope and Architecture

Implementation changes are restricted to:

```text
configs/evaluation/
mv_recon/
tests/
docs/
```

The implementation must not modify:

```text
pipeline/
inference_engine/
pi3/
loop_closure/
```

The focused module structure is:

```text
mv_recon/
├── eval.py                 # Hydra entry point and dataset orchestration
├── protocol.py             # typed protocol loading and strict validation
├── geometry_metrics.py     # crop, alignment, and metric computation
├── results.py              # typed aggregation and atomic result writers
└── eval_utils.py           # existing official Umeyama/NN helpers, unchanged
```

`mv_recon/eval.py` remains the user-facing command. It delegates protocol
resolution, metric computation, and output serialization to focused modules.
The existing dataset loaders and `StreamingPipelineModel` remain the
reconstruction boundary.

## Paper Protocol Configuration

Add `configs/evaluation/mv_recon_laser_paper.yaml` as a Hydra evaluation
profile. Its logical contents are:

```yaml
# @package _global_

defaults:
  - override /data: mv_recon_dense

name: mv_recon_laser_paper

protocol:
  name: laser_cvpr2026_table4_pi3
  version: 1
  strict: true
  preflight_only: false
  resume: false
  max_sequences: null

  pipeline_config: configs/pipeline/default.yaml
  pipeline_overrides:
    - window.size=20
    - window.overlap=5
    - segmentation.method=depth
    - segmentation.confidence_keep_ratio=0.5
    - segmentation.depth_merge_threshold=0.1
    - segmentation.temporal_iou_threshold=0.3
    - segmentation.felzenszwalb.scale=300
    - segmentation.felzenszwalb.sigma=1.1
    - segmentation.felzenszwalb.min_size=500
    - anchor_propagation.enabled=true
    - anchor_propagation.correspondence_iou_threshold=0.4
    - loop.enabled=false
    - loop.method=traditional
    - loop.registration.confidence_keep_ratio=0.5

  geometry:
    center_crop_size: 224
    alignment: umeyama_sim3_then_icp
    icp_type: point_to_point
    icp_threshold_m: 0.1
    normal_estimation: open3d_default
    fscore_thresholds_m: [0.01, 0.02, 0.05]

  paper_reference:
    7scenes-dense:
      accuracy: {mean: 0.013, median: 0.005}
      completion: {mean: 0.017, median: 0.006}
      normal_consistency: {mean: 0.607, median: 0.665}
    NRGBD-dense:
      accuracy: {mean: 0.020, median: 0.010}
      completion: {mean: 0.012, median: 0.004}
      normal_consistency: {mean: 0.713, median: 0.856}

eval_datasets:
  - 7scenes-dense
  - NRGBD-dense
```

The actual structured schema may use nested dataclasses or OmegaConf nodes,
but unknown keys and missing required keys must be rejected.

### Locked algorithmic fields

Strict mode validates the resolved `PipelineConfig`, not only the textual
override list. These fields must match exactly:

- window size 20 and overlap 5;
- Depth segmentation;
- segmentation confidence keep ratio 0.5;
- depth merge threshold 0.1;
- temporal IoU threshold 0.3;
- Felzenszwalb scale 300, sigma 1.1, and minimum size 500;
- anchor propagation enabled with correspondence IoU threshold 0.4;
- loop closure disabled and traditional aggregation path selected;
- adjacent-window registration confidence keep ratio 0.5;
- 224-pixel square center crop;
- Umeyama Sim(3) followed by point-to-point ICP at 0.1 metres;
- Open3D default normal estimation;
- the fixed kf10 sequence maps for 7-Scenes and NeuralRGBD.

Changing a locked field while retaining the `laser_paper` label is an error
before model construction.

### Operational fields

Cloud execution may change checkpoint path, CUDA device, output directory,
temporary cache directory, ordinary prediction cache root, and cache mode.
Those values are recorded in the run manifest. Model checkpoint content is
identified by SHA256, so path changes cannot hide a different model.

## Runtime Flow

1. Hydra composes the general, dataset, and paper evaluation configuration.
2. `protocol.py` validates the protocol schema and builds pipeline overrides,
   including operational model and cache paths from the Hydra config.
3. The overrides are passed directly to `load_pipeline_config()`.
4. Strict validation checks every locked field on the fully resolved
   `PipelineConfig`.
5. Preflight validates the checkpoint, dataset roots, sequence maps, frame
   selection, and output/resume state without constructing Pi3.
6. In normal mode, one `StreamingPipelineModel` is created from the validated
   pipeline config.
7. For every sequence-map entry, the existing dataset loader supplies images,
   point maps, validity mask, and selected image paths.
8. The existing streaming interface reconstructs predicted point maps, using a
   fresh window engine and fresh segmentation/anchor state for the sequence.
9. `geometry_metrics.py` performs paper-compatible alignment and metrics.
10. `results.py` atomically persists the current sequence results and run
    state.
11. Only a full successful sequence-map run receives `complete` status and a
    valid Table 4 summary.

The cloud command remains compatible with current Hydra usage:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  output_dir=outputs/mv_recon_laser_paper
```

## Dataset and Sampling Contract

The evaluator keeps `configs/data/mv_recon_dense.yaml` and the shipped maps:

```text
datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json
datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json
```

Preflight verifies that:

- both maps exist and their SHA256 values are recorded;
- 7-Scenes contains exactly 18 configured sequences and NeuralRGBD exactly 9;
- sequence names are unique and non-empty;
- frame ID lists are non-empty, strictly increasing integers;
- consecutive frame IDs use interval 10;
- the dataset returns the requested number of images, GT point maps, and masks;
- predicted and GT spatial dimensions can support a 224 by 224 center crop.

`max_sequences` is a per-dataset limit. It deterministically selects the first
entries in each selected dataset's sequence-map insertion order. The selected
sequence names are written to the manifest before model construction. Any
limited run receives `subset` status and cannot be reported as a Table 4
reproduction.

## Paper-Compatible Geometry Alignment

For each sequence:

1. Use the existing streaming interface to resize the predicted point maps to
   the GT spatial resolution.
2. Center-crop prediction, GT, images, and GT valid mask to 224 by 224.
3. Validate that coordinates selected by the GT mask are finite. Strict mode
   reports non-finite data as an error instead of silently changing the mask.
4. Estimate Umeyama scale, rotation, and translation from same-pixel predicted
   and GT correspondences selected by the GT valid mask.
5. Apply the Sim(3) transform to the full cropped predicted point maps.
6. Flatten prediction and GT using the same GT valid mask.
7. Run Open3D point-to-point ICP with identity initialization and maximum
   correspondence distance 0.1 metres.
8. Apply the returned ICP transformation to the predicted cloud.
9. Estimate prediction and GT normals after alignment by calling Open3D's
   default `estimate_normals()` behavior.
10. Compute both nearest-neighbour directions and their normal scores.

The final point-map evaluation does not apply a predicted-confidence filter.
The two locked 0.5 confidence ratios affect reconstruction: one affects Depth
layer extraction and one affects adjacent-window registration. They do not
remove points from the indoor Table 4 metric calculation.

Strict mode has no alternate ICP or normal-estimation fallback because such a
fallback would change the benchmark definition.

## Metric Semantics

Let `d_pred` be each predicted point's distance to its nearest GT point and
`d_gt` be each GT point's distance to its nearest predicted point.

The six primary per-sequence metrics are:

```text
accuracy_mean_m       = mean(d_pred)
accuracy_median_m     = median(d_pred)
completion_mean_m     = mean(d_gt)
completion_median_m   = median(d_gt)
normal_consistency_mean
normal_consistency_median
```

Let NC1 compare each predicted normal with the normal of its nearest GT point,
and NC2 compare each GT normal with the normal of its nearest predicted point.
The absolute dot product removes normal orientation sign ambiguity.

The official directional aggregation is retained:

```text
normal_consistency_mean   = (NC1 mean + NC2 mean) / 2
normal_consistency_median = (NC1 median + NC2 median) / 2
```

The NC median is not the median of concatenated NC1 and NC2 samples.

### Dataset aggregation

Each dataset metric is the arithmetic macro average of the corresponding
per-sequence metric. Points from different scenes are not pooled. The complete
paper denominator is the exact number of entries in the sequence map. If one
entry fails, the run remains incomplete instead of silently shrinking the
denominator.

### Extended diagnostics

The following fields are computed from the same aligned clouds and nearest
neighbour arrays but are stored separately from the primary result:

- NC1 and NC2 mean/median;
- Chamfer-L1 in metres, defined as
  `(accuracy_mean_m + completion_mean_m) / 2`;
- precision, recall, and F-score at 0.01, 0.02, and 0.05 metres;
- ICP fitness and inlier RMSE;
- Umeyama scale;
- predicted and GT point counts.

Extended fields never enter the Table 4 summary columns.

## Paper Reference Values

The profile contains the Pi3+LASER Table 4 values for comparison:

| Dataset | Acc Mean | Acc Median | Comp Mean | Comp Median | NC Mean | NC Median |
|---|---:|---:|---:|---:|---:|---:|
| 7-Scenes | 0.013 | 0.005 | 0.017 | 0.006 | 0.607 | 0.665 |
| NeuralRGBD | 0.020 | 0.010 | 0.012 | 0.004 | 0.713 | 0.856 |

The result writer emits `delta_to_paper` for every primary value. These deltas
are diagnostic only: the tool does not automatically declare mathematical
reproduction success or failure because the final tolerance and cloud runtime
variation remain under the user's control.

## Result Contract

Each run writes:

```text
outputs/mv_recon_laser_paper/
├── resolved_protocol.yaml
├── resolved_pipeline.yaml
├── protocol_manifest.json
├── results.json
├── summary.csv
├── sequences.csv
└── failures.jsonl
```

`results.json` is canonical. It includes schema version, run state, protocol,
provenance, per-sequence results, dataset macro averages, paper-reference
deltas, and completion counts. CSV files are derived views for inspection and
paper-table preparation.

`protocol_manifest.json` records:

- Git commit;
- the full resolved-protocol SHA256, resume-compatible protocol-identity
  SHA256, and resolved-pipeline SHA256;
- checkpoint SHA256;
- sequence-map SHA256 values;
- PyTorch, CUDA, Open3D, NumPy, and SciPy versions;
- GPU model and effective dtype;
- every locked and operational setting;
- per-sequence input-manifest and ordinary-prediction cache keys;
- attempted, successful, failed, and expected sequence counts.

Writers use temporary sibling files followed by atomic replacement. They never
aggregate by scanning prior output files.

## Resume and Output Safety

The default behavior refuses to overwrite a non-empty result directory.
`protocol.resume=true` explicitly enables resume. Existing sequence results
are reusable only when all of these identities match:

- resolved protocol hash;
- resolved pipeline hash;
- checkpoint hash;
- sequence-map hash;
- per-sequence input image manifest hash;
- metric implementation/schema version.

The full resolved-protocol hash covers every serialized field. Resume compares
a second canonical protocol-identity hash that excludes only the
`protocol.resume` and `protocol.preflight_only` control switches; otherwise
turning resume on would invalidate the run it is meant to resume. The identity
still includes `max_sequences`, all algorithmic settings, reference values,
dataset selection, and operational pipeline inputs.

Any mismatch rejects resume. A resumed sequence skips reconstruction and
metrics only when its stored result is complete under the exact identity. The
ordinary prediction cache remains independent: a metric implementation change
may invalidate sequence results while still allowing the validated Pi3
ordinary predictions to be reused.

The paper profile places Hydra's own job log directory outside the canonical
result directory, so Hydra bootstrap files cannot accidentally trip or bypass
the non-empty-directory guard. Preflight and numerical runs use distinct
output directories unless resume is explicitly requested.

Interrupted or failed jobs preserve atomically written sequence results and
set the run state to `incomplete` or `failed`. They do not emit a valid complete
Table 4 status.

## Error Handling

Preflight errors are fatal before model construction:

- protocol drift;
- missing or unreadable checkpoint;
- missing dataset root, map, or requested frame;
- malformed sequence map or non-kf10 interval;
- incompatible resume artifacts;
- result-directory collision without resume.

Sequence-time errors atomically record the dataset, sequence, error category,
and message, mark the run incomplete, and exit non-zero:

- CUDA out of memory;
- prediction/GT shape mismatch;
- crop too large;
- non-finite selected points;
- fewer than three valid Umeyama correspondences;
- degenerate Umeyama variance or non-finite transform;
- empty flattened cloud;
- ICP exception or non-finite transformation;
- invalid nearest-neighbour or normal arrays.

Strict paper mode never skips a failure and continues with a smaller
denominator. GPU cache cleanup occurs after every sequence and in failure
cleanup paths.

## Prediction Cache Contract

The existing ordinary prediction cache is reused unchanged. Its fingerprint
continues to identify the model, checkpoint, image manifest, image shape,
sample stride, window size, overlap, and window specs.

Each sequence uses a fresh window engine, segmentation strategy, anchor
propagator, and loop state. The lazy Pi3 model handle and validated ordinary
predictions may be reused. The cache does not contain segmentation labels,
anchor state, Umeyama/ICP outputs, or point-map metrics, so changing evaluation
code cannot silently return stale metrics.

## Cloud Workflow

### Preflight only

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  protocol.preflight_only=true \
  output_dir=outputs/mv_recon_laser_paper_preflight
```

This prints and writes the resolved protocol and dataset summary without
constructing Pi3.

### One-sequence-per-dataset smoke run

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  protocol.max_sequences=1 \
  output_dir=outputs/mv_recon_laser_paper_smoke
```

The result is marked `subset`.

### Full run

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  output_dir=outputs/mv_recon_laser_paper
```

### Resume an identical run

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  protocol.resume=true \
  output_dir=outputs/mv_recon_laser_paper
```

The cloud environment must use a Python version for which the pinned Open3D
dependency is available. The local development host currently uses Python
3.13, for which no Open3D wheel is available; unit tests must therefore isolate
Open3D behind injectable wrappers, while the real cloud smoke run validates the
installed Open3D path.

## Test Strategy

Implementation follows red-green-refactor. Tests cover:

1. The shipped paper profile resolves every locked value exactly.
2. Atomic segmentation, a changed window, registration confidence 0.30, or
   any other locked-field drift is rejected before model construction.
3. Operational checkpoint, device, output, and cache paths remain configurable
   and are recorded.
4. The new metric functions match the existing official `eval_utils.py`
   results on deterministic fixtures.
5. Accuracy and completeness query the correct nearest-neighbour direction.
6. NC mean and median average the two directional statistics separately.
7. The final evaluator uses the GT valid mask and does not filter by prediction
   confidence.
8. Center crop, same-pixel Umeyama correspondences, and ICP invocation use the
   exact paper settings.
9. Dataset macro aggregation uses successful results only internally but a
   strict complete result requires the full sequence-map count.
10. Failed or subset runs cannot serialize as a complete paper result.
11. Results, CSV projections, failures, and manifests are atomic and reject
    non-finite JSON values.
12. Resume accepts exact identities and rejects any protocol, checkpoint,
    map, manifest, or metric-version mismatch.
13. A fake model and fake dataset exercise the evaluator end-to-end without a
    GPU or Open3D installation by injecting deterministic alignment results.
14. CLI preflight does not construct Pi3.
15. Git diff scope contains no reconstruction-core changes.

Before writing this design, the isolated base worktree compiled its required
Cython extensions and passed the existing baseline suite: 274 tests passed.
The implementation acceptance gate is the baseline suite plus all new tests.
The user's cloud run is the final numeric Table 4 validation.

## Acceptance Criteria

The implementation is complete when:

1. The paper profile resolves and passes strict preflight with exact LASER
   settings.
2. Attempts to label a drifted pipeline as `laser_paper` fail before loading
   the model.
3. The evaluator produces all six Table 4 primary fields per sequence and per
   dataset with exact official definitions.
4. Partial, failed, and subset runs are unambiguously non-complete.
5. Provenance is sufficient to identify the code, model, data selection,
   runtime, and cache inputs of every reported number.
6. Resume never mixes incompatible runs.
7. No files outside the approved evaluation, test, and documentation scope are
   changed.
8. All available local tests pass; the only local environment limitation is
   the unavailable Python 3.13 Open3D wheel.
9. The user can transfer the branch to the cloud, run preflight, run a
   one-sequence smoke test, and launch the full Table 4 benchmark without
   editing source code.
