# LASER Modular Pipeline

This branch has one streaming reconstruction entry point, one strict
configuration schema, three segmentation strategies, one anchor-propagation
implementation, and two loop-closure strategies.

## Quick start

```bash
python run_laser.py --config configs/pipeline/default.yaml
```

Every override uses a canonical dotted configuration path:

```bash
python run_laser.py \
  --config configs/pipeline/default.yaml \
  --set input.image_dir=data/00/image_2 \
  --set output.scene_name=kitti_00_atomic_normal_traditional \
  --set segmentation.method=atomic \
  --set segmentation.atomic.split_mode=normal_only \
  --set loop.method=traditional
```

The CLI deliberately accepts only `--config` and repeated
`--set KEY=VALUE`. Unknown, missing, or retired fields fail during strict
configuration loading.

## Default configuration

```yaml
version: 1

input:
  image_dir: data/00/image_2
  sample_stride: 1

output:
  scene_name: kitti_00
  cache_dir: inference_cache/kitti_00
  result_dir: viser_results
  save_diagnostics: true

model:
  checkpoint: weights/model.safetensors
  inference_device: cuda
  process_device: cpu
  dtype: bfloat16

window:
  size: 10
  overlap: 5

segmentation:
  method: atomic
  confidence_keep_ratio: 0.5
  depth_merge_threshold: 0.1
  temporal_iou_threshold: 0.3

  felzenszwalb:
    scale: 300
    sigma: 1.1
    min_size: 500

  geometry:
    normal_method: cross
    normal_threshold_degrees: 20.0

  atomic:
    split_mode: conservative
    split_score_threshold: 0.10

anchor_propagation:
  enabled: true
  correspondence_iou_threshold: 0.4

loop:
  enabled: true
  method: corrected

  registration:
    confidence_keep_ratio: 0.30

  detection:
    method: salad
    salad_checkpoint: weights/dino_salad.ckpt
    dino_checkpoint: weights/dinov2_vitb14_pretrain.pth
    image_size: [336, 336]
    batch_size: 32
    similarity_threshold: 0.7
    top_k: 5
    nms_enabled: true
    nms_frame_radius: 25

  constraint:
    chunk_size: 20

  optimizer:
    implementation: cpp
    use_sim3: true
    max_iterations: 30
    initial_damping: 1.0e-6
```

## Exact method choices

The public enums are intentionally closed:

- `segmentation.method`: `depth`, `geometry`, `atomic`
- `segmentation.atomic.split_mode`: `none`, `conservative`, `normal_only`
- `loop.method`: `traditional`, `corrected`
- `segmentation.geometry.normal_method`: `cross`, `sobel`
- `model.dtype`: `float16`, `bfloat16`, `float32`

`segmentation.confidence_keep_ratio` and
`loop.registration.confidence_keep_ratio` are positive keep ratios in
`(0, 1]`. For example, `0.30` keeps the highest-confidence 30 percent of
finite pixels. They are not rejection quantiles.

All three segmentation methods share the configured Felzenszwalb parameters.
`depth` segments the depth map. `geometry` segments surface-normal geometry.
`atomic` first merges layer atoms and then applies exactly one of these split
modes:

- `none`: returns the compact merge-only labels and performs no split work.
- `conservative`: uses
  `score = normal_gain × max(rgb_contrast, normalized_gap_contrast)`.
- `normal_only`: uses `score = normal_gain` and does not compute RGB or gap
  confirmation.

Both active modes use the same split threshold and guards: two to four
children, minimum child area, full parent coverage, at most four leaves,
finite values, compact labels, and no recursive splitting.

## Anchor propagation

There is one anchor-propagation implementation:
`AnchorPropagator.propagate`. The original IoU-weighted scale propagation
math is unchanged. `anchor_propagation.enabled` controls whether it runs;
selecting a segmentation or loop method never selects a second anchor
algorithm.

## Loop methods

Both loop strategies receive the same immutable image manifest, SALAD
candidate tuple, registration keep ratio, constraint type, and optimizer
configuration.

- `traditional` preserves the baseline delayed semantics. Window caches store
  relative Sim(3) transforms and unapplied anchor scale masks. Aggregation
  accumulates transforms and applies anchor scaling once.
- `corrected` registers against the previous corrected overlap, applies the
  absolute Sim(3) and anchor mask immediately, stores absolute and edge
  transforms, optimizes edges, and applies only the optimized-vs-original
  delta during aggregation.

When no valid loop candidate remains, both methods use a no-loop path and do
not invoke constraint estimation or the optimizer solve.

## Ten-configuration experiment matrix

The matrix is an explicit list, not an automatically expanding Cartesian
product:

| Segmentation | Atomic split | Loop |
|---|---|---|
| depth | n/a | traditional |
| depth | n/a | corrected |
| geometry | n/a | traditional |
| geometry | n/a | corrected |
| atomic | none | traditional |
| atomic | none | corrected |
| atomic | conservative | traditional |
| atomic | conservative | corrected |
| atomic | normal_only | traditional |
| atomic | normal_only | corrected |

Validate all ten resolved configurations without loading Pi3 or SALAD:

```bash
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/test.yaml \
  --dry-run
```

Run the complete GPU matrix:

```bash
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/default.yaml \
  --set input.image_dir=data/00/image_2 \
  --set output.scene_name=kitti_00
```

Each entry gets unique scene, cache, and result directory suffixes.

## Outputs and diagnostics

The resolved configuration is written before model construction. A completed
run writes the reconstruction artifacts plus:

- `resolved_config.yaml`
- `run_summary.json`
- `loop_candidates.json`
- `loop_constraints.json`
- `segmentation_diagnostics.json`

`run_summary.json` records the SHA-256 hash of the resolved configuration, Git
commit, selected methods, atomic split mode, manifest endpoints, image/window/
candidate/constraint counts, no-loop status, split totals, and stage timings.
Writes use a temporary file followed by an atomic replacement.

## AutoDL / cloud example

```bash
git clone --recursive \
  --branch codex/modular-segmentation-loop-integration \
  https://github.com/Cjuicy/LASER.git
cd LASER

conda create -n laser -y python=3.11
conda activate laser
pip install -r requirements.txt
pip install faiss-gpu-cu12
python setup.py build_ext --inplace
bash scripts/download_weights.sh

python -m pytest -q
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/test.yaml \
  --dry-run

python run_laser.py \
  --config configs/pipeline/default.yaml \
  --set input.image_dir=data/00/image_2 \
  --set output.scene_name=kitti_00 \
  --set output.cache_dir=inference_cache/kitti_00 \
  --set output.result_dir=viser_results
```

Preflight rejects missing images/checkpoints, invalid window sizes, unsupported
dtypes/devices, and unavailable CUDA before constructing Pi3 or SALAD.
