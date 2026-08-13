# Cloud Validation: Decoupled LASER Reconstruction and Evaluation

This workflow targets branch `codex/laser-paper-pointmap-eval`.

## Clone and environment

```bash
git clone --recursive --branch codex/laser-paper-pointmap-eval https://github.com/Cjuicy/LASER.git
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
