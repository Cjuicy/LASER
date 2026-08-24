# Cloud Validation: Decoupled LASER Reconstruction and Evaluation

This workflow targets branch `codex/keyframe-ate-residual-sim3`, based on
`codex/laser-paper-pointmap-eval`.

## Clone and environment

```bash
git clone --recursive --branch codex/keyframe-ate-residual-sim3 https://github.com/Cjuicy/LASER.git
cd LASER
git submodule update --init --recursive
conda create -n laser-decoupled python=3.11 -y
conda activate laser-decoupled
pip install -r requirements.txt
python setup.py build_ext --inplace
bash scripts/download_weights.sh
```

Expected files:

```text
weights/model.safetensors
weights/dino_salad.ckpt
weights/dinov2_vitb14_pretrain.pth
```

Prepare an ordered image directory for each sequence. ATE ground truth can be
Sintel, Replica, TUM, or TartanAir trajectory format. Point-cloud ground truth
must be an NPZ containing `point_maps` `(N,H,W,3)` and `valid_mask` `(N,H,W)`
for the exact reconstructed frames.

## Locally executable smoke commands

These commands require no model, GPU, weights, or dataset. They are executed by
`tests/test_cloud_validation_commands.py`.

```bash
# smoke-test
python run_reconstruction.py --help
python evaluate_ate.py --help
python evaluate_pointcloud.py --help
python run_experiment_matrix.py --help
```

## Tests and matrix planning

```bash
pytest -q
python run_experiment_matrix.py --config configs/experiments/ate_matrix.yaml --dry-run
python run_experiment_matrix.py --config configs/experiments/pointcloud_matrix.yaml --dry-run
```

The dry runs report exactly 9 ATE entries and 3 point-cloud entries. A dry run
still hashes the configured image files and model checkpoint to show the real
artifact identity; set those paths in the reconstruction YAML or with `--set`.

## Keyframe-aware ATE and dual evaluation

The version-2 capability matrix is
`configs/experiments/keyframe_metrics_matrix.yaml`. It reconstructs each
variant once, then runs every evaluator supported by both the scene and the
reconstruction mode:

- `no_loop` runs ATE and point-cloud evaluation when both ground truths are
  declared. Its post-anchor residual Sim(3) updates local points and camera
  poses, so the same artifact is used by both evaluators.
- `corrected` runs ATE. Its post-anchor residual Sim(3) refines absolute
  transforms and sequential edges before the new loop optimization.
- `traditional` runs only ATE, exactly once per segmentation method, with
  window-reference keyframe refinement forced off. Its reconstruction and
  loop-closure implementations are unchanged.

For a dual-capability scene, edit these declarations in the checked-in config:

```yaml
dataset:
  name: fixture
  sequence: sequence-0
  trajectory_ground_truth:
    path: data/groundtruth.txt
    format: tum
  pointcloud_ground_truth:
    path: data/groundtruth_pointmaps.npz
evaluation:
  trajectory_config: configs/evaluation/ate.yaml
  pointcloud_config: configs/evaluation/pointcloud.yaml
```

Then run all 15 variants (five per segmentation method):

```bash
python run_experiment_matrix.py \
  --config configs/experiments/keyframe_metrics_matrix.yaml \
  --set input.image_dir=/data/sequence/images \
  --set model.checkpoint=weights/model.safetensors \
  --set window.size=20 \
  --set window.overlap=5
```

For an ATE-only scene, remove both `pointcloud_ground_truth` and
`pointcloud_config`. The 15 reconstruction variants remain, and unsupported
point-cloud evaluator records are `skipped`.

For a pointcloud-only scene, remove both `trajectory_ground_truth` and
`trajectory_config`. Only the six `no_loop` variants are scheduled: window
reference off/on for each of depth, geometry, and atomic segmentation. At
least one ground-truth declaration is required. A declaration with a missing
or invalid file is an error, not an omitted capability.

The version-2 output layout is:

```text
outputs/experiments/keyframe-metrics/
  artifacts/<reconstruction-identity>/
  evaluation/<entry-name>/
    ate/trajectory_metrics.json
    ate/evaluator_record.json
    pointcloud/pointcloud_metrics.json
    pointcloud/pointcloud_sequences.csv
    pointcloud/evaluator_record.json
  evaluation_summary.json
```

Every entry records both canonical evaluator slots with one of four statuses:

- `passed`: evaluation completed and its output exists;
- `skipped`: the scene did not declare that capability, or the reconstruction
  mode does not support it;
- `failed`: a declared evaluator input was invalid or that evaluator raised an
  error; a passed sibling output is preserved;
- `blocked`: reconstruction failed, so a scheduled evaluator could not run.

The command exits with status 1 when any evaluator is `failed` or `blocked`.
It otherwise exits 0 and prints counts for all four statuses. Reconstruction
continues with later variants after an entry failure.

Rerunning the same command resumes each evaluator independently. A `passed`
record is reused only when the artifact manifest, that evaluator's config,
its ground truth, and the source revision all match and the recorded output
still exists. Changing only point-cloud ground truth therefore reruns only the
point-cloud evaluator; it does not reconstruct the artifact or rerun ATE.

## Single reconstruction examples

```bash
# no_loop + depth, 10/5
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser_no_loop.yaml \
  --set input.image_dir=/data/sequence/images \
  --set model.checkpoint=weights/model.safetensors \
  --set output.scene_name=sequence \
  --set segmentation.method=depth \
  --set window.size=10 \
  --set window.overlap=5

# Traditional + geometry, 20/5
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set input.image_dir=/data/sequence/images \
  --set output.scene_name=sequence \
  --set segmentation.method=geometry \
  --set reconstruction.mode=traditional \
  --set window.size=20 \
  --set window.overlap=5

# Corrected + atomic, 20/10
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set input.image_dir=/data/sequence/images \
  --set output.scene_name=sequence \
  --set segmentation.method=atomic \
  --set reconstruction.mode=corrected \
  --set window.size=20 \
  --set window.overlap=10
```

Artifacts are written under:

```text
outputs/reconstruction/<scene>/<segmentation>-<mode>/
```

## ATE: 3 × 3

Edit the dataset and output fields in `configs/experiments/ate_matrix.yaml`, or
provide reconstruction overrides:

```bash
python run_experiment_matrix.py \
  --config configs/experiments/ate_matrix.yaml \
  --set input.image_dir=/data/sequence/images \
  --set model.checkpoint=weights/model.safetensors \
  --set window.size=20 \
  --set window.overlap=5
```

Results are written below:

```text
outputs/experiments/laser/evaluation/ate/<segmentation>-<mode>/trajectory_metrics.json
```

## Point-cloud quality: 3 × no_loop

Edit `configs/experiments/pointcloud_matrix.yaml`, then run:

```bash
python run_experiment_matrix.py \
  --config configs/experiments/pointcloud_matrix.yaml \
  --set input.image_dir=/data/sequence/images \
  --set model.checkpoint=weights/model.safetensors \
  --set window.size=20 \
  --set window.overlap=5
```

The runner forces `reconstruction.mode=no_loop` and the Table 4 `nearest`
confidence quantile. Results are written below:

```text
outputs/experiments/laser/evaluation/pointcloud/<segmentation>-no_loop/
  pointcloud_metrics.json
  pointcloud_sequences.csv
```

Complete reconstruction artifacts are content-addressed below
`outputs/experiments/laser/artifacts/`. Evaluation configuration is excluded
from artifact identity, so a matching reconstruction is reused instead of
being recomputed.
