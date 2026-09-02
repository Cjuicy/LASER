# Traditional Second-Global Cloud Validation

This guide validates the new `traditional_second_global` reconstruction mode
on a GPU machine. It produces three trajectories from the same images and
configuration:

1. a standalone Traditional baseline;
2. the new run's nested Stage 1 Traditional artifact; and
3. the new run's Stage 2 residual-global-optimization artifact.

Do not interpret the Stage 2 ATE until the standalone baseline and nested
Stage 1 trajectory tensors are exactly equal.

## 1. Clone the implementation branch

```bash
git clone --recursive --branch codex/keyframe-selection \
  https://github.com/Cjuicy/LASER.git LASER-Improved
cd LASER-Improved
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

The first command printed by Git must identify `codex/keyframe-selection`.
Record the commit printed by the second command with the experiment results.

## 2. Create the environment and download weights

```bash
conda create -n laser-second-global python=3.11 -y
conda activate laser-second-global
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e viser
bash scripts/download_weights.sh
```

The reconstruction config expects these local files:

```text
weights/model.safetensors
weights/dino_salad.ckpt
weights/dinov2_vitb14_pretrain.pth
```

Check them before starting a long run:

```bash
test -f weights/model.safetensors
test -f weights/dino_salad.ckpt
test -f weights/dinov2_vitb14_pretrain.pth
```

## 3. Select one scene and fixed experiment controls

Replace the first four paths/names for the target dataset. Keep every other
setting identical between the baseline and second-global runs.

```bash
export LASER_IMAGE_DIR=/data/sequence/images
export LASER_GT_PATH=/data/sequence/groundtruth.txt
export LASER_GT_FORMAT=tum
export LASER_SCENE=scene-name
export LASER_SEGMENTATION=geometry
export LASER_WINDOW_SIZE=20
export LASER_WINDOW_OVERLAP=5
export LASER_RESULT_ROOT=outputs/reconstruction
export LASER_ATE_ROOT=outputs/evaluation/ate-second-global
```

For Replica, Sintel, or another supported trajectory source, set
`LASER_GT_FORMAT` to the matching value accepted by `evaluate_ate.py`.

## 4. Run the standalone Traditional baseline

```bash
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set input.image_dir="$LASER_IMAGE_DIR" \
  --set output.scene_name="$LASER_SCENE" \
  --set output.result_dir="$LASER_RESULT_ROOT" \
  --set segmentation.method="$LASER_SEGMENTATION" \
  --set reconstruction.mode=traditional \
  --set window.size="$LASER_WINDOW_SIZE" \
  --set window.overlap="$LASER_WINDOW_OVERLAP"
```

## 5. Run Traditional plus second-global optimization

```bash
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set input.image_dir="$LASER_IMAGE_DIR" \
  --set output.scene_name="$LASER_SCENE" \
  --set output.result_dir="$LASER_RESULT_ROOT" \
  --set segmentation.method="$LASER_SEGMENTATION" \
  --set reconstruction.mode=traditional_second_global \
  --set window.size="$LASER_WINDOW_SIZE" \
  --set window.overlap="$LASER_WINDOW_OVERLAP"
```

The second command reuses the ordinary PI3 prediction cache. SALAD candidate
detection executes once inside the new run; joint PI3 loop evidence is
estimated once for Stage 1 and again against the corrected Stage 2 window
states.

Define the three artifact paths:

```bash
export LASER_BASELINE_RESULT="$LASER_RESULT_ROOT/$LASER_SCENE/$LASER_SEGMENTATION-traditional"
export LASER_SECOND_RESULT="$LASER_RESULT_ROOT/$LASER_SCENE/$LASER_SEGMENTATION-traditional_second_global"
export LASER_STAGE1_RESULT="$LASER_SECOND_RESULT/stage1"
```

## 6. Enforce exact Stage 1 baseline equality

```bash
python - <<'PY'
import os
from pathlib import Path
import torch

baseline_path = Path(os.environ["LASER_BASELINE_RESULT"]) / "trajectory.pt"
stage1_path = Path(os.environ["LASER_STAGE1_RESULT"]) / "trajectory.pt"
baseline = torch.load(
    baseline_path,
    map_location="cpu",
    weights_only=True,
)["camera_poses"]
stage1 = torch.load(
    stage1_path,
    map_location="cpu",
    weights_only=True,
)["camera_poses"]
if not torch.equal(baseline, stage1):
    difference = torch.max(torch.abs(baseline - stage1)).item()
    raise SystemExit(
        "Stage 1 differs from Traditional baseline; "
        f"max_abs_difference={difference}"
    )
print(f"Stage 1 exact baseline equality: PASS; shape={tuple(stage1.shape)}")
PY
```

Stop and report a regression if this check fails.

## 7. Evaluate Stage 1 and Stage 2 with identical ATE settings

```bash
python evaluate_ate.py \
  --artifact "$LASER_STAGE1_RESULT" \
  --config configs/evaluation/ate.yaml \
  --ground-truth "$LASER_GT_PATH" \
  --ground-truth-format "$LASER_GT_FORMAT" \
  --output "$LASER_ATE_ROOT/$LASER_SCENE/stage1"

python evaluate_ate.py \
  --artifact "$LASER_SECOND_RESULT" \
  --config configs/evaluation/ate.yaml \
  --ground-truth "$LASER_GT_PATH" \
  --ground-truth-format "$LASER_GT_FORMAT" \
  --output "$LASER_ATE_ROOT/$LASER_SCENE/stage2"
```

The desired experimental outcome is:

```text
ATE(stage2) < ATE(stage1)
```

Record ATE, translation/rotation RPE, candidate counts, accepted constraint
counts, the reconstruction commit, dataset sequence, window settings,
segmentation method, and all resolved YAML files. A Stage 2 regression is an
experimental result to diagnose; a Stage 1 mismatch is an implementation
failure.

## 8. Inspect diagnostics

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LASER_SECOND_RESULT"])
for label, path in (("stage1", root / "stage1"), ("stage2", root)):
    diagnostics = json.loads((path / "diagnostics.json").read_text())
    print(label)
    print("  candidates:", diagnostics["candidate_count"])
    print("  constraints:", diagnostics["constraint_count"])
    print("  mode_scalars:", diagnostics["mode_scalars"])
PY
```

For a multi-window Stage 2 run,
`stage2_residual_applied_window_count` must equal `window_count - 1`.
