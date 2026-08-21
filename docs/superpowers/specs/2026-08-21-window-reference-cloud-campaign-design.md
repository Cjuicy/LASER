# Window-Reference Cloud Campaign Design

**Date:** 2026-08-21  
**Status:** Proposed for implementation  
**Branch:** `codex/laser-paper-pointmap-eval`  
**Design baseline:** `977b261a8cd4809a69ef19b8d9738401b52c5a62`

## 1. Goal

Add a cloud-ready experiment sidecar that can be run from a fresh clone to
compare the existing `depth`, `geometry`, and `atomic` initial segmentation
methods with window-reference refinement disabled and enabled.

The primary campaign is a strict six-run ablation per scene:

| Initial segmentation | Window refinement | Stable identity |
|---|---:|---|
| `depth` | off | `depth__wr-off` |
| `depth` | on | `depth__wr-on` |
| `geometry` | off | `geometry__wr-off` |
| `geometry` | on | `geometry__wr-on` |
| `atomic` | off | `atomic__wr-off` |
| `atomic` | on | `atomic__wr-on` |

All six runs use the same images, frame selection, PI3 checkpoint, prediction
window schedule, reconstruction mode, and non-refinement configuration. The
only changing fields are `segmentation.method` and
`segmentation.window_reference.enabled`.

The campaign must:

- run from a fresh clone of the named branch;
- reuse one ordinary PI3 prediction cache across all six runs for a scene;
- support synthetic checks, point-cloud-GT scenes, and KITTI Odometry scenes;
- aggregate reconstruction, segmentation, refinement, cache, timing, and GT
  evaluation results into machine-readable JSON and CSV;
- resume safely after interruption;
- retain compact results while bounding disk use one scene at a time;
- never download protected datasets or store cloud credentials.

## 2. Approved Experimental Decisions

### 2.1 Reconstruction and window schedule

The primary matrix is fixed to:

```text
reconstruction.mode=no_loop
window.size=75
window.overlap=30
input.sample_stride=1 after staging
```

`no_loop` is deliberate. It keeps loop detection, loop-constraint generation,
and global loop optimization out of the causal comparison, and it is the only
mode accepted by the current point-cloud evaluator.

The `75/30` schedule is deliberately retained for comparison with the existing
KITTI cloud campaign. Historical KITTI compact results use `75/30`, but most of
those results use `corrected` reconstruction. They may be displayed as a
separately labelled historical reference; they must not be pooled with the new
`no_loop` on/off pairs as if reconstruction mode were identical.

### 2.2 Initial segmentation configuration

The three methods use the existing production implementations and defaults.
Atomic segmentation is fixed to:

```text
segmentation.method=atomic
segmentation.atomic.split_mode=conservative
```

The split mode is part of every run identity and is identical in atomic off/on
runs. It is not an additional experiment axis.

### 2.3 Window-reference configuration

Enabled runs use the shipped values:

```text
segmentation.window_reference.enabled=true
segmentation.window_reference.sampling_stride=4
segmentation.window_reference.max_keyframes=4
segmentation.window_reference.relative_depth_tolerance=0.05
segmentation.window_reference.min_reference_score=0.30
segmentation.window_reference.stop_coverage_ratio=0.90
segmentation.window_reference.min_coverage_gain=0.03
segmentation.window_reference.min_region_correspondences=8
segmentation.window_reference.min_region_coverage=0.10
segmentation.window_reference.min_region_purity=0.80
segmentation.window_reference.merge_vote_threshold=0.80
```

Disabled runs set only `enabled=false`; all remaining values are still resolved,
snapshotted, and included in the run configuration for auditability.

## 3. Scope Boundaries

### 3.1 In scope

- A typed campaign package under `experiments/window_reference_campaign/`.
- A root CLI, `run_window_reference_campaign.py`.
- A checked-in YAML manifest with presets and scene definitions.
- Cloud bootstrap planning, preflight, plan, run, resume, and summarize commands.
- Campaign-owned staging for frame-limited image/pose/point-map-GT alignment.
- Existing reconstruction and evaluator reuse.
- A compact diagnostics/results schema and CSV/JSON summaries.
- Synthetic CPU tests and no-GPU cloud plan/preflight validation.
- Documentation with exact clone and run commands.

### 3.2 Out of scope

- Changes to PI3, the three segmentation algorithms, refinement geometry, the
  reconstruction modes, artifact schema, prediction-cache schema, or metrics.
- Direct `.ply`, `.pcd`, or KITTI Velodyne `.bin` input. The production pipeline
  accepts ordered RGB images; point-cloud GT is evaluation-only.
- Automatic KITTI download, credential/cookie storage, or use of unofficial
  mirrors to bypass KITTI registration.
- KITTI leaderboard submission.
- Claiming the internal ATE/RPE evaluator implements the official KITTI
  100--800 m devkit metric.
- Multi-GPU scheduling in the first version. The schema may reserve `jobs`, but
  one scene and one run execute at a time on one GPU.
- Reusing old compact metrics as resume state. Historical results are read-only
  comparison inputs because they have different run identities.

## 4. Existing Source Contracts

The implementation must orchestrate existing boundaries rather than duplicate
them:

- `pipeline.config.InputConfig` and `pipeline.manifest.ImageManifest` accept an
  ordered image directory, not an external point cloud.
- `PipelineRunner` owns preprocessing, prediction fingerprinting, cache/provider
  construction, reconstruction, artifact writing, and provenance.
- `segment_and_refine_window()` already enforces
  `initial segmentation -> optional refinement`.
- `NoLoopReconstructionMode` carries per-frame segmentation/refinement summaries
  into `ReconstructionDiagnostics.segmentation_summaries`.
- `build_prediction_fingerprint()` intentionally excludes segmentation method,
  refinement, and reconstruction mode. It includes input content, preprocessing,
  checkpoint, PI3 runtime sources, dtype, sample stride, and window specs.
- `evaluate_pointcloud.py` and `evaluation.pointcloud` own point-cloud metrics.
- `evaluate_ate.py` and `evaluation.trajectory` own internal ATE/RPE metrics.
- Existing 7-Scenes/NeuralRGBD preparation logic in
  `tools/disposable_experiments/preflight_experiments.py` is the behavioural
  reference, but its historical source-SHA gate and method tables are not reused.

## 5. Cloud Environment Findings

The inspected no-GPU AutoDL instance contains an extracted historical checkout,
not a Git repository. It must remain a data/result source; the new branch must be
cloned separately.

Observed external source root:

```text
/root/autodl-tmp/LASER-Paper-Pointmap-Eval/
```

Observed inputs:

```text
data/KITTI/                  # normalized 00..10 layout
data/7-Scenes/
data/NeuralRGBD/
data/pointcloud_benchmark/
weights/PI3/model.safetensors
weights/dino_salad.ckpt
weights/dinov2_vitb14_pretrain.pth
outputs/disposable_experiments/metrics/
```

The primary `no_loop` matrix requires only the PI3 checkpoint. SALAD and DINO
weights are not required.

The existing `vggt` conda environment uses Python 3.11.15 and has a CUDA-enabled
PyTorch build. CUDA is unavailable in no-card mode, so this instance can run
`plan`, manifest-only checks, and CPU unit tests but not PI3 GPU inference.

Storage observed during design:

```text
system disk:       about 15 GiB free
/root/autodl-tmp:  about 99 GiB free
/root/autodl-fs:   about 12 GiB free (95% used)
```

Consequently, the default clone, output, staging, cache, and artifacts must be
placed under `/root/autodl-tmp`. The campaign must reject
`/root/autodl-fs` as its documented default and warn when available space is
below the configured threshold.

The old ordinary prediction caches have been cleaned. Each scene's first new
run must regenerate PI3 predictions once; the other five runs reuse them.

No hostname, username, password, token, SSH cookie, or KITTI login information
may be written to the repository, campaign manifest, result metadata, or logs.

## 6. Scene Presets

### 6.1 Synthetic contract preset

`synthetic-smoke` reuses small deterministic point-map fixtures to verify:

- exact six-run matrix expansion;
- refinement disabled identity;
- a positive merge;
- conservative fallback;
- occlusion/depth-rejection diagnostics;
- result-schema and summary behaviour.

This preset does not load PI3, require weights, or claim end-to-end quality.

### 6.2 Point-cloud-GT preset

The default `pointcloud-small` preset uses three already available scenes:

| Dataset | Scene | Selected frames | Purpose |
|---|---|---:|---|
| 7-Scenes | `chess/seq-03` | 100 | indoor handheld RGB-D and cross-window behaviour |
| NeuralRGBD | `thin_geometry` | 40 | short smoke and thin-structure stress case |
| NeuralRGBD | `complete_kitchen` | 122 | longer indoor cross-window quality case |

The observed prepared GT shapes are:

```text
chess/seq-03:       (100, 392, 518, 3)
thin_geometry:       (40, 392, 518, 3)
complete_kitchen:   (122, 392, 518, 3)
```

`thin_geometry` is shorter than the configured window and therefore exercises a
single partial window. The other two scenes exercise overlapping 75/30 windows.

The campaign may consume an externally prepared `ground_truth.npz` only after
validating its frame IDs and shape against the staged images. Otherwise, it
prepares GT from the raw dataset using campaign-owned output paths.

### 6.3 KITTI presets

The inspected KITTI root contains normalized directories
`<root>/<sequence>/image_2` and `<root>/<sequence>/poses.txt` for 00--10. The
resolver must also accept the official extracted layout
`dataset/sequences/<sequence>/image_2` plus `dataset/poses/<sequence>.txt`.

Presets:

| Preset | Sequences/slice | Purpose |
|---|---|---|
| `kitti-smoke` | sequence 04, frames 0:80 | fastest real RGB/pose/cache validation |
| `kitti-small` | complete 04, 03, 07 | economical multi-scene campaign |
| `kitti-formal-subset` | complete 00, 01, 04, 07 | longer formal subset without claiming 00--10 coverage |

Observed image/pose counts match for the selected complete sequences:

```text
00: 4541
01: 1101
03:  801
04:  271
07: 1101
```

KITTI uses the left colour camera `image_2`. The script must never download
KITTI automatically. Preflight explains that official registration and purpose
declaration are required and reports the expected source layout.

## 7. Architecture

### 7.1 Files

```text
run_window_reference_campaign.py
configs/experiments/window_reference_campaign.yaml
experiments/window_reference_campaign/
  __init__.py
  config.py
  matrix.py
  scenes.py
  staging.py
  preflight.py
  runner.py
  evaluation.py
  diagnostics.py
  results.py
  cli.py
docs/window-reference-cloud-campaign.md
tests/experiments/
  test_window_reference_campaign_config.py
  test_window_reference_campaign_scenes.py
  test_window_reference_campaign_runner.py
  test_window_reference_campaign_evaluation.py
  test_window_reference_campaign_results.py
  test_window_reference_campaign_preflight.py
  test_window_reference_campaign_cli.py
```

Files may be combined when a module would otherwise contain only trivial
delegation, but configuration, staging, execution, and result aggregation must
remain separately testable.

### 7.2 CLI

The root CLI exposes:

```text
bootstrap   Print or execute clone-environment preparation steps; never download datasets.
plan        Resolve a preset and matrix without importing torch or reading model weights.
preflight   Validate environment, paths, space, images, GT, weights, and CUDA.
run         Stage scenes, execute/resume the six-run matrix, and evaluate.
summarize   Validate completed compact records and atomically write summaries.
```

Common options:

```text
--config
--preset
--scene
--methods depth,geometry,atomic
--refinement off,on
--output-root
--data-root
--checkpoint
--dry-run
--resume / --no-resume
--fail-fast / --keep-going
--start-frame
--max-frames
--frame-stride
--gpu
--cache-policy auto|refresh|readonly|off
--keep-artifacts
```

`plan --dry-run` must not require torch, CUDA, weights, or datasets. `preflight`
may run without CUDA only when explicitly passed `--allow-no-gpu`; real `run`
must reject unavailable CUDA when `model.inference_device=cuda`.

### 7.3 Fresh-clone cloud workflow

The documented AutoDL flow is:

```bash
cd /root/autodl-tmp
git clone --recursive --branch codex/laser-paper-pointmap-eval \
  https://github.com/Cjuicy/LASER.git LASER-Window-Reference
cd LASER-Window-Reference

conda run -n vggt python run_window_reference_campaign.py plan \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign

conda run -n vggt python run_window_reference_campaign.py preflight \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n vggt \
  python run_window_reference_campaign.py run \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign \
  --resume --keep-going
```

The final documentation also gives a `screen` wrapper for SSH disconnect
survival. The campaign itself remains foreground and returns a meaningful exit
status.

## 8. Typed Configuration and Identities

The checked-in YAML defines campaign, matrix, window-reference preset, scenes,
runtime, storage, and evaluation sections. Unknown keys are rejected.

Booleans must remain booleans. In particular, YAML-like strings such as `off`,
`on`, `yes`, and `no` must not silently become the refinement axis; the typed
schema resolves explicit booleans and emits stable `wr-off`/`wr-on` labels.

Every run identity contains at least:

```text
schema_version
campaign_config_sha256
source_commit
source_dirty
dataset
scene
frame_start
frame_stop
frame_stride
staged_manifest_sha256
segmentation_method
atomic_split_mode
window_reference_enabled
window_reference_config
reconstruction_mode
window_size
overlap
model_name
model_dtype
checkpoint_sha256
prediction_key
```

Result validation and resume require exact identity equality plus finite required
metrics. Path existence alone is never sufficient.

## 9. Staging and Alignment

All limited-frame or mapped scenes are staged below the campaign output root.
Raw datasets are read-only.

Staging rules:

1. Resolve and naturally sort source images.
2. Resolve one explicit source-frame index vector.
3. Apply `start`, `stop/max_frames`, and `frame_stride` once.
4. Create a campaign-owned image directory of relative symlinks or validated
   absolute symlinks named `000000.png`, `000001.png`, and so on.
5. Slice KITTI poses or point-map GT with the exact same source-frame vector.
6. Write `source_frame_ids.npy` and a JSON staging manifest atomically.
7. Set pipeline `input.sample_stride=1`; sampling must not happen again.

Symlink targets must resolve inside an approved data root. Existing staging is
reused only when its manifest and file hashes match.

For KITTI, each pose row must contain 12 finite values. Image and pose counts
must match before slicing. For point-cloud GT, `point_maps`, `valid_mask`, and
frame IDs must have equal leading dimensions and match staged frames exactly.

## 10. Cache and Execution Semantics

Each scene slice owns one shared ordinary prediction-cache root:

```text
<campaign>/work/cache/<dataset>/<scene-id>/<scene-slice>/v2/<prediction-key>/
```

Run order is deterministic:

```text
depth__wr-off
depth__wr-on
geometry__wr-off
geometry__wr-on
atomic__wr-off
atomic__wr-on
```

Cache policy per scene:

1. Find the first incomplete run.
2. Use `auto` for that run, or `refresh` only when explicitly requested.
3. After a complete cache exists, use `readonly` for all remaining runs.
4. If resume skipped earlier results but the cache was cleaned, the first
   remaining run returns to `auto` rather than failing with blind `readonly`.
5. Assert all six completed records contain the same prediction key.

Runs are serial. A failed run leaves its attempt directory and log intact. A
retry creates or cleans only campaign-owned stale run paths after safety checks.
It never overwrites a validated immutable artifact.

Default storage policy is one scene at a time:

- retain compact run records, logs, evaluations, diagnostics summaries, and
  campaign summaries;
- retain the shared cache until all requested runs for the scene validate;
- remove validated large artifacts and the scene cache unless
  `--keep-artifacts` is set;
- never remove raw data, external prepared GT, weights, historical outputs, or
  anything outside the campaign root.

## 11. Evaluation

### 11.1 Point-cloud metrics

The adapter calls the existing evaluator and retains:

- accuracy mean/median;
- completion mean/median;
- normal consistency mean/median;
- Chamfer-L1;
- precision, recall, and F-score at 1, 2, and 5 cm.

The evaluator's no-loop and shape contracts remain authoritative. Non-finite
metrics invalidate the run.

### 11.2 KITTI internal trajectory metrics

The adapter calls the existing trajectory evaluator using its 12-value pose-row
loader and retains:

- Sim(3)-aligned ATE RMSE in metres;
- RPE translation RMSE in metres;
- RPE rotation RMSE in degrees;
- matched frame count.

Output columns and documentation use `internal_ate_*`/`internal_rpe_*` names for
KITTI to prevent confusion with the official KITTI devkit.

Official KITTI percentage/degree-per-metre metrics are not implemented in this
campaign. They may be added later as a separate evaluator and separate table.

### 11.3 No-GT scenes

Synthetic or repository image smoke without GT still produces execution,
cache, segmentation, refinement, timing, and fallback records. It does not emit
fabricated trajectory or point-cloud quality values.

## 12. Diagnostics Aggregation

The compact diagnostics record aggregates the artifact's per-frame
`segmentation_summaries` while preserving window observation semantics.

Required fields include:

- run status, attempt, failure stage, frame count, window count;
- elapsed reconstruction/evaluation time and per-frame/per-window rates;
- prediction key, cache policy, cache hits/misses/corruptions when available;
- segmentation region count distribution;
- refinement-enabled flag;
- keyframe count distribution and keyframe-index histogram/string records;
- coverage-ratio distribution;
- regions before/after and absolute/relative reduction;
- candidate, accepted, and conflict edge totals;
- projected, occluded, and depth-rejected sample totals;
- applied-frame count/rate;
- fallback-reason histogram.

Overlapping windows produce repeated frame observations. Summaries must expose
both `unique_frame_count` and `window_frame_observation_count`; they must not
silently treat observations as unique frames.

Disabled runs mark refinement-only aggregates as `null` or an explicit
`not_applicable` state. They must not be reported as refinement fallback or as
zero keyframe performance.

## 13. Result Layout

```text
outputs/window_reference_campaign/<campaign-id>/
  campaign.json
  plan.json
  preflight.json
  prepared/<dataset>/<scene-id>/<scene-slice>/
  work/cache/<dataset>/<scene-id>/<scene-slice>/
  runs/<dataset>/<scene-id>/<scene-slice>/<method>__wr-{off,on}/
    attempts/<attempt-id>/stdout.log
    run.json
    diagnostics_summary.json
    evaluation/
    artifact/                       # optional/temporary
  summary/
    runs.csv
    diagnostics.csv
    trajectory.csv
    pointcloud.csv
    summary.json
    failures.json
```

All JSON and CSV outputs use temporary files plus atomic rename. JSON forbids
NaN and infinity. `campaign.json` records argv with secrets redacted, Git
commit/dirty state, resolved configuration hash, Python/Torch/CUDA/GPU details,
checkpoint digest, timestamps, and final campaign status.

## 14. Bootstrap and Preflight

`bootstrap` must be usable before torch is installed. It imports only the Python
standard library at command startup and can either print or execute:

- submodule initialization;
- Python 3.11 environment checks;
- requirements installation;
- Cython extension build;
- public weight download through the existing repository script when weights
  are not supplied externally.

It never downloads KITTI, 7-Scenes, NeuralRGBD, or stores authenticated URLs.

`preflight` validates:

- supported Python version;
- importability of required packages and compiled extensions;
- source commit/dirty state;
- checkpoint existence, size, and SHA-256;
- selected image/pose/GT layout and counts;
- writable campaign root;
- minimum free space;
- CUDA/device/dtype compatibility unless `--allow-no-gpu` is set;
- deterministic matrix size and distinct output identities.

Preflight is read-only except for atomically writing its report below the
campaign root.

## 15. Error Handling and Resume

- Configuration, path, alignment, and identity errors fail before GPU work.
- `--fail-fast` stops at the first failed run and returns non-zero.
- `--keep-going` records the failure, continues independent runs/scenes, and
  returns non-zero if any requested run remains invalid.
- SIGINT/SIGTERM leave the active attempt and cache available for inspection.
- A run is resumable only from a valid compact record with an exact identity.
- Evaluation may resume from a valid artifact without rerunning reconstruction.
- A corrupted or incomplete cache is handled by the existing prediction-store
  validation and recorded; it is never silently treated as a hit.
- Cleanup resolves targets and proves they are descendants of the campaign-owned
  cleanup root before deletion.

## 16. Test Strategy

Implementation follows strict RED-GREEN-REFACTOR.

### 16.1 Unit and contract tests

- Typed schema, unknown keys, booleans, exact six-run order, and identity.
- Official and normalized KITTI layouts.
- Natural image sorting, exact frame slicing, pose alignment, GT alignment, and
  prevention of double sampling.
- First-pending cache policy, readonly reuse, cache-missing resume recovery,
  fail-fast, keep-going, and artifact-only evaluation resume.
- On/off diagnostics, fallback histograms, overlap observations, empty values,
  non-finite rejection, and atomic writes.
- Point-cloud and trajectory evaluator adapters using literal small fixtures.
- Bootstrap import isolation and no-dataset-download behaviour.
- Preflight behaviour with no GPU and missing/available paths.

### 16.2 Integration tests

- Synthetic six-run campaign without PI3/GPU.
- Fake prediction provider proving one prediction identity serves all six runs.
- Existing window-reference, cache-parity, artifact, point-cloud, and trajectory
  regression suites.
- CLI `plan` and `bootstrap --dry-run` from a minimal environment.

### 16.3 Verification sequence

```bash
python -m pytest -p no:cacheprovider -q tests/experiments/test_window_reference_campaign_*.py
python run_window_reference_campaign.py plan --preset synthetic-smoke
python run_window_reference_campaign.py run --preset synthetic-smoke
python run_window_reference_campaign.py plan --preset kitti-small
python -m compileall -q experiments pipeline reconstruction inference_engine
bash -n <any newly added shell wrapper>
python -m pytest -p no:cacheprovider -q
```

On the inspected no-GPU cloud instance:

```bash
python run_window_reference_campaign.py plan --preset pointcloud-small ...
python run_window_reference_campaign.py plan --preset kitti-small ...
python run_window_reference_campaign.py preflight --allow-no-gpu ...
```

After switching to GPU mode, the first real run is `kitti-smoke`, followed by
`pointcloud-small` and then `kitti-small`. The implementation task does not claim
these GPU campaigns passed unless they were actually executed with weights and
data.

## 17. Implementation Tasks

The detailed plan will expand these sequential TDD units:

1. Typed campaign configuration, six-run matrix, identities, and `plan`.
2. Scene resolution and exact campaign-owned staging.
3. Runner, cache reuse, attempts, resume, and failure policy.
4. Diagnostics and compact result aggregation.
5. Point-cloud and KITTI internal trajectory evaluation adapters.
6. Bootstrap and preflight.
7. Synthetic integration smoke, cloud documentation, and no-GPU cloud checks.
8. Full review, focused/full verification, final push, and GPU handoff command.

## 18. Acceptance Criteria

1. A fresh branch clone can run `plan` without torch, weights, CUDA, or datasets.
2. Every selected scene expands to exactly six unique identities in the approved
   order and uses `no_loop`, `75/30`, and atomic conservative split.
3. A staged scene applies the frame index vector exactly once to images and GT.
4. The first pending run produces or finds one PI3 cache; remaining runs consume
   the same prediction key without ordinary PI3 inference.
5. Resume skips only exact, finite, schema-valid compact results.
6. Point-cloud-GT runs use the existing evaluator and KITTI runs emit clearly
   labelled internal ATE/RPE metrics.
7. Refinement diagnostics distinguish disabled, applied, and fallback states and
   expose overlap observation counts.
8. Failures preserve actionable logs and return non-zero; successful compact
   records and summaries are atomic.
9. Default cleanup cannot touch external data, weights, historical outputs, or
   paths outside the campaign root.
10. The no-GPU cloud instance passes `plan` and preflight with `--allow-no-gpu`.
11. The complete local test suite passes with no new warnings before push.
12. Documentation contains exact clone, no-GPU inspection, GPU smoke, resume,
    summary, and safe cleanup commands without any credential.

## 19. Residual Risks

- A 75-frame window increases refinement candidate projections and CPU work;
  `sampling_stride=4` and `max_keyframes=4` bound typical cost, but real runtime
  must be measured.
- `thin_geometry` is shorter than one configured window and cannot validate
  cross-window registration by itself.
- Existing cloud compact baselines do not retain ordinary prediction caches, so
  the first new run for every scene pays PI3 inference cost.
- Historical KITTI metrics use different reconstruction methods and are context,
  not a controlled off/on baseline.
- The current no-GPU instance cannot validate PI3 execution, CUDA memory, or
  empirical quality. The feature remains unvalidated on real data until GPU
  smoke and selected campaigns are run.
