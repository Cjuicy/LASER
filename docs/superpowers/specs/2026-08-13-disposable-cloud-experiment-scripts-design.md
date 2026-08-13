# Disposable Cloud Experiment Scripts Design

**Date:** 2026-08-13

## Goal

Add a small, disposable script bundle for running the requested point-cloud and
KITTI trajectory experiments on the existing cloud checkout. The scripts are
an orchestration layer only: they select existing reconstruction and evaluation
implementations, retain compact results and logs, and delete large per-scene
intermediate files after successful evaluation.

The source baseline is branch `codex/laser-paper-pointmap-eval`, commit
`cfe26f8b57341a9320eae2ef71532a353654d1a6`.

## Script Bundle

Create only these disposable entry points under
`tools/disposable_experiments/`:

1. `preflight_experiments.py` validates the environment, weights, exact dataset
   sequence maps, raw inputs, poses, and writable/free disk space.
2. `run_pointcloud_all.sh` prepares and evaluates one point-cloud scene at a
   time, then removes that scene's large intermediates.
3. `run_kitti_ate_all.sh` evaluates one KITTI sequence at a time, then removes
   that sequence's large intermediates.
4. `summarize_results.py` validates retained metric JSON files and produces
   long-form CSV plus screenshot-style XLSX summaries.
5. `run_all.sh` is the single foreground entry point and runs preflight,
   point-cloud experiments, KITTI ATE experiments, and final summarization in
   order.

The directory has no long-term runtime dependency: after the campaign and
result download it may be deleted.

## Experiment Matrix

### Point-cloud evaluation

- Datasets: all exact kf10 sequences in the checked-in maps:
  - 7-Scenes: 18 sequences, 1,700 selected frames;
  - NeuralRGBD: 9 scenes, 1,101 selected frames.
- Schedule: selected kf10 images, window size 20, overlap 5.
- Reconstruction: `no_loop`; anchor propagation enabled.
- Five method identities:
  - `depth`: `segmentation.method=depth`;
  - `geometry`: `segmentation.method=geometry`;
  - `atomic-original`: `segmentation.method=atomic`,
    `segmentation.atomic.split_mode=none`;
  - `atomic-split-assisted`: `segmentation.method=atomic`,
    `segmentation.atomic.split_mode=conservative`;
  - `atomic-split-no-assisted`: `segmentation.method=atomic`,
    `segmentation.atomic.split_mode=normal_only`.
- Metrics retained per method and scene: accuracy mean/median, completion
  mean/median, normal consistency mean/median, Chamfer-L1, and precision,
  recall, and F-score at 1/2/5 cm.

### KITTI ATE evaluation

- Dataset: KITTI odometry sequences `00` through `10` inclusive.
- Schedule: sample stride 1, window size 75, overlap 30.
- Six method identities:
  - baseline: `depth + traditional`;
  - new loop: `corrected` with `depth`, `geometry`, `atomic-original`,
    `atomic-split-assisted`, or `atomic-split-no-assisted`.
- Registration confidence keep ratio remains `0.5` for all six methods.
- Metrics retained per method and sequence: ATE RMSE in metres, RPE translation
  RMSE in metres, RPE rotation RMSE in degrees, and matched frame count.

## Execution and Resume Behavior

Each shell runner uses a deterministic output path for every
dataset/scene/method. A method is skipped only when its compact metric JSON is
present, parses successfully, contains the expected identity fields, and all
required metrics are finite. There is no database or campaign state machine.

Methods within one scene run serially and share the existing content-addressed
ordinary PI3 prediction cache. A failed command stops the script immediately,
leaves intermediates intact, and returns a non-zero status. Running `run_all.sh`
again skips validated results, removes only the failed method's stale
campaign-owned directory, and retries the first missing method.

Logs are written per dataset, scene, and method. The scripts run in the
foreground; the delivered command wraps `run_all.sh` in `screen` so it survives
SSH disconnection without agent monitoring.

## Safe Cleanup

Each method artifact is removed immediately after its compact metric JSON
validates. Prepared benchmark data and prediction-cache entries are removed
only after every requested method for the current scene or sequence validates.
Cleanup targets are resolved below campaign-owned roots and checked before
removal.

The scripts may delete only:

- per-scene reconstruction artifacts;
- prepared point-cloud benchmark data for the completed scene, including
  generated 7-Scenes projected depth files;
- ordinary prediction-cache entries proven to belong to the completed scene;
- temporary configuration and staging files.

The scripts never delete:

- `data/7-Scenes`, `data/NeuralRGBD`, or `data/KITTI` raw source files;
- model, SALAD, or DINO weights;
- retained metric JSON/CSV/XLSX files;
- logs or the preflight report.

Cleanup is disabled by `KEEP_INTERMEDIATES=1` for debugging.

## Result Layout

Compact outputs live below a dedicated root, defaulting to
`outputs/disposable_experiments/`:

```text
preflight.json
metrics/pointcloud/<dataset>/<scene>/<method>.json
metrics/ate/kitti/<sequence>/<method>.json
logs/...
summary/pointcloud_long.csv
summary/pointcloud_summary.xlsx
summary/kitti_ate_long.csv
summary/kitti_ate_summary.xlsx
```

The point-cloud workbook uses one sheet for 7-Scenes and one for NeuralRGBD.
Rows are methods, columns are scenes, and each cell contains the requested
point-cloud metrics. The ATE workbook uses method rows and KITTI `00`-`10`
columns with ATE/RPE values in each cell. Both include a machine-friendly long
table sheet. Formatting follows the supplied screenshots without encoding
metric values as styled text in the long tables.

## Verification

- Unit tests cover method-to-override mapping, successful-result detection,
  incomplete/non-finite result rejection, safe cleanup path validation, exact
  sequence coverage, and summary-table contents.
- Shell syntax is checked with `bash -n`.
- Preflight is exercised against fixtures without loading GPU models.
- Workbook generation is rendered and visually inspected, and key ranges and
  formula-error scans are checked before delivery.
- Cloud deployment runs preflight and dry-run planning only; the long campaign
  is left for the user to start with the single delivered command.

## Acceptance Criteria

1. One command starts the full campaign without interactive monitoring.
2. The requested 5-method point-cloud and 6-method KITTI matrices are exact.
3. Re-running resumes from valid compact results.
4. A failed or interrupted scene retains its intermediates.
5. Successful scenes retain only compact results/logs unless
   `KEEP_INTERMEDIATES=1`.
6. Raw datasets and weights cannot be selected by cleanup code.
7. Final CSV and XLSX files cover every requested scene and method or fail
   loudly instead of silently producing a partial final table.
