# Pi3 Ordinary Prediction Cache: Validation Guide

## Scope

This branch keeps the modular segmentation/loop pipeline as the baseline and
adds one shared, persistent cache around ordinary Pi3 sliding-window
predictions.

The cache boundary is intentionally narrow:

```text
preprocessed RGB window
        |
        v
ordinary Pi3 forward       (only on a cache miss)
        |
        v
depth + confidence + camera poses
        |
        +--> persistent v2 prediction cache
        |
        v
local-point reconstruction + current RGB attachment
        |
        +--> depth / geometry / atomic segmentation
        +--> traditional / corrected loop method
```

SALAD detection and candidate joint A/B inference are outside this cache.
Joint inference uses the same lazy Pi3 handle and is counted separately.

## Configuration

```yaml
model:
  name: pi3
  checkpoint: weights/model.safetensors
  inference_device: cuda
  process_device: cpu
  dtype: bfloat16

prediction_cache:
  root: inference_cache/predictions
  mode: auto
```

The checkpoint must be a local file. State-dict loading is strict.

### Cache modes

| Mode | Read | Write | Missing/corrupt behavior |
|---|---:|---:|---|
| `auto` | yes | yes | compute missing windows; quarantine corrupt artifacts and rebuild |
| `refresh` | no old reuse | yes | snapshot the old entry under `invalid/` and rebuild |
| `readonly` | yes | no | fail immediately on a missing or corrupt artifact; do not quarantine |
| `off` | no | no | run ordinary Pi3 for every requested window |

For a ten-entry matrix, requested `refresh` applies only to entry zero.
Entries one through nine are forced to `readonly`, so the refresh cannot
erase the result it is meant to share.

## What is cached

Cached:

- one sequence reference intrinsic, shape `(3,3)`;
- per-window depth, shape `(N,H,W)`;
- per-window confidence, shape `(N,H,W)`;
- per-window camera poses, shape `(N,4,4)`;
- the canonical `WindowSpec(index, frame_start, frame_end)`;
- artifact digests and prediction provenance.

Not cached:

- RGB tensors;
- full local-point maps;
- segmentation labels or split results;
- anchor-propagation state;
- traditional/corrected method caches;
- SALAD descriptors or loop candidates;
- joint A/B predictions;
- loop constraints, optimizer state, or final reconstruction.

On both cold and warm paths, local points are reconstructed as
`unproject_depth_to_local_points(depth, reference_intrinsic)`. The current
call's RGB tensors are then attached, so downstream code receives the same
four-key contract: `local_points`, `camera_poses`, `conf`, and `images`.

## Fingerprint

The prediction key is SHA-256 over canonical JSON. It includes:

- prediction-cache schema version and adapter contract version;
- model name (`pi3`);
- checkpoint file bytes;
- the ordered image-content manifest;
- runtime source bytes for the Pi3 adapter, loader, preprocessing, Pi3 model,
  geometry, and DINOv2 implementation;
- model dtype;
- preprocessing contract and resulting `(N,3,H,W)` shape;
- input sample stride;
- window size and overlap;
- the complete ordered window schedule.

It excludes experiment state that cannot change ordinary Pi3 output:

- inference/process device identity;
- segmentation and atomic split settings;
- anchor-propagation settings;
- traditional/corrected loop selection;
- SALAD, constraint, and optimizer settings;
- output directories, Git metadata, and hardware identity.

Changing image bytes/order, checkpoint bytes, dtype, preprocessing/runtime
source, or the window schedule therefore creates a different prediction key.
Changing only a downstream experiment method reuses the same key.

## v2 disk layout

```text
<prediction_cache.root>/
  v2/
    <64-character prediction key>/
      manifest.json
      sequence.json
      complete.json
      windows/
        000000.pt
        000001.pt
        ...
      locks/
        entry.lock
      invalid/
        <timestamp>-<reason>/
          ...
```

Writes use a same-directory temporary file, sync that temporary file, and
atomically replace the destination. Tensor payloads are read back and validated;
JSON validates its canonical serialized representation before publication. This
provides atomic visibility, but does not claim power-loss durability because the
parent directory is not synced. A process-level file lock serializes one
prediction entry.
Partially written valid entries are resumable: existing windows hit, and only
missing windows forward through Pi3.

Each sequence/window artifact has its own value digest bound to the prediction
key. Truncation, same-shape value mutation, a copied artifact from another
key, an invalid window range, or a false completion marker is rejected.

## Diagnostics

Inspect `<result_dir>/<scene_name>/run_summary.json`:

```text
model_name
checkpoint_digest
ordinary_prediction_key
prediction_cache_mode
model_constructed
ordinary_hits
ordinary_misses
ordinary_forward_count
joint_forward_count
corrupt_count
prediction_cache_read_ms
prediction_cache_write_ms
saved_window_count
stored_bytes
prediction_cache_events
```

`prediction_cache_events` records readonly failures with the exact window and
absolute frame range, and records automatic quarantine with its reason plus the
absolute original and quarantine paths.

Expected cold/warm behavior for a two-window sequence:

| Run | ordinary hits | ordinary forwards | model constructed |
|---|---:|---:|---:|
| first `auto` run | 0 | 2 | true |
| second `auto` or `readonly` run | 2 | 0 | false, unless joint inference is needed |

If valid loop candidates exist on a warm run, ordinary forward count remains
zero while joint forward count may be positive and model construction becomes
necessary.

## CPU test suite and dry run

```bash
python setup.py build_ext --inplace
python -m pytest -q

python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/test.yaml \
  --set model.name=pi3 \
  --dry-run
```

The dry run must print ten unique names beginning with `pi3_`. Method-cache
and result paths differ per entry; `prediction_cache.root` remains shared.

## 15-frame two-window smoke test

With 15 frames, `window.size=10`, and `window.overlap=5`, the canonical
schedule is exactly:

```text
WindowSpec(0, 0, 10)
WindowSpec(1, 5, 15)
```

Run the first configuration with `auto`:

```bash
python run_laser.py \
  --config configs/pipeline/default.yaml \
  --set input.image_dir=/data/smoke-15 \
  --set output.scene_name=smoke_depth_traditional_cold \
  --set output.cache_dir=inference_cache/method/smoke_depth_traditional_cold \
  --set output.result_dir=viser_results \
  --set prediction_cache.root=inference_cache/predictions \
  --set prediction_cache.mode=auto \
  --set window.size=10 \
  --set window.overlap=5 \
  --set segmentation.method=depth \
  --set loop.enabled=false \
  --set loop.method=traditional
```

Then replay with a different downstream method and `readonly`:

```bash
python run_laser.py \
  --config configs/pipeline/default.yaml \
  --set input.image_dir=/data/smoke-15 \
  --set output.scene_name=smoke_atomic_corrected_warm \
  --set output.cache_dir=inference_cache/method/smoke_atomic_corrected_warm \
  --set output.result_dir=viser_results \
  --set prediction_cache.root=inference_cache/predictions \
  --set prediction_cache.mode=readonly \
  --set window.size=10 \
  --set window.overlap=5 \
  --set segmentation.method=atomic \
  --set segmentation.atomic.split_mode=conservative \
  --set loop.enabled=false \
  --set loop.method=corrected
```

Acceptance:

- cold summary: `ordinary_forward_count == 2`;
- warm summary: `ordinary_forward_count == 0`;
- warm summary: `ordinary_hits == 2`;
- both summaries have the same `ordinary_prediction_key`;
- the warm run can complete without constructing Pi3 when loop inference is
  disabled or yields no candidates.

## Ten-entry matrix acceptance

```bash
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/default.yaml \
  --set input.image_dir=/data/smoke-15 \
  --set output.scene_name=smoke_matrix \
  --set prediction_cache.root=inference_cache/predictions \
  --set prediction_cache.mode=refresh
```

For the ten summaries, accept the ordinary-forward vector:

```text
[W, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

For the 15-frame smoke, `W == 2`. A second matrix run with `readonly` must
produce ten zeros. Do not add `joint_forward_count` to this total.

## Full KITTI/GPU validation

When CUDA, weights, and the dataset are available:

1. Run one full sequence with `refresh`; record wall time, peak GPU memory,
   prediction-cache bytes, and ordinary/joint counts.
2. Run depth, geometry, and all atomic split modes against the same prediction
   root.
3. Run traditional and corrected loop methods with separate method-cache and
   result directories.
4. Repeat in `readonly` mode.
5. Confirm all warm runs have zero ordinary forwards and the same prediction
   key. Joint counts may differ with loop candidates.
6. Compare reconstruction metrics against the baseline using the existing
   evaluation formulas; cache reuse must not change image order or formulas.

## Cloud clone

```bash
git clone --recursive \
  --branch codex/pi3-ordinary-prediction-cache \
  --single-branch \
  https://github.com/Cjuicy/LASER.git
cd LASER

python setup.py build_ext --inplace
python -m pytest -q
```

After cloning, `git rev-parse --abbrev-ref HEAD` should print
`codex/pi3-ordinary-prediction-cache`.
