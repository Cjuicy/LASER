# LASER Reconstruction Configuration

LASER uses a strict version-2 reconstruction schema. The public command is:

```bash
python run_reconstruction.py --config configs/reconstruction/pi3_laser.yaml
```

Every experiment change is a repeated `--set KEY=VALUE`. Unknown and retired
fields fail during configuration loading.

## Public choices

- `model.name`: `pi3`
- `segmentation.method`: `depth`, `geometry`, `atomic`
- `segmentation.atomic.split_mode`: `none`, `conservative`, `normal_only`
- `segmentation.confidence_quantile_method`: `higher`, `nearest`
- `reconstruction.mode`: `no_loop`, `traditional`, `corrected`
- `prediction_cache.mode`: `auto`, `refresh`, `readonly`, `off`
- `model.dtype`: `float16`, `bfloat16`, `float32`

The retired loop enable/method flags are not accepted. `reconstruction.mode`
is the sole top-level choice:

- `no_loop` uses the LASER Table 4 incremental assembly path and may omit the
  entire `loop` section.
- `traditional` and `corrected` require loop detection, constraint, and
  optimizer configuration.

## Windows and segmentation

The only window invariant is:

```text
window.size > window.overlap >= 1
```

Examples include `10/5`, `20/5`, and `20/10`. Evaluators do not inspect or
override these values.

All segmentation methods use the same `SegmentationStrategy` interface and the
same LASER `AnchorPropagator`. Depth uses Felzenszwalb depth regions; Geometry
adds surface-normal criteria; Atomic performs layer atom merge/split with the
configured exclusive split mode.

`segmentation.confidence_keep_ratio` and
`registration.confidence_keep_ratio` are positive keep ratios in `(0,1]`.
ATE defaults to the `higher` NumPy quantile rule. The point-cloud experiment
matrix explicitly selects `nearest` to retain the LASER Table 4 protocol.

## Reconstruction modes

Traditional retains deferred processing: ordinary prediction and per-window
state are completed first; loop detection/evidence/optimization run afterward;
the final aggregation applies the accumulated Sim(3) and anchor scale.

Corrected retains online processing: adjacent Sim(3) and anchor scale are
applied as each window arrives; loop detection runs after the last window;
sequential edges are optimized and the optimized-vs-original delta is applied
once.

No-loop is a complete independent mode. It never constructs loop detection,
joint evidence, or an optimizer.

## Examples

```bash
# no loop + depth, 10/5
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser_no_loop.yaml \
  --set segmentation.method=depth \
  --set window.size=10 \
  --set window.overlap=5

# Traditional + geometry, 20/5
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set reconstruction.mode=traditional \
  --set segmentation.method=geometry \
  --set window.size=20 \
  --set window.overlap=5

# Corrected + atomic, 20/10
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set reconstruction.mode=corrected \
  --set segmentation.method=atomic \
  --set window.size=20 \
  --set window.overlap=10
```

## Outputs

The runner writes one immutable directory with:

- `manifest.json`: schema, modes, tensor digests, resolved config hash,
  checkpoint hash, Git commit, and diagnostics.
- `trajectory.pt`: frame IDs and camera poses.
- `pointmap.pt`: local/global point maps and reconstruction mode.
- `confidence.pt`: dense confidence maps.

Writers reject an existing artifact directory instead of overwriting it.
Trajectory and point-cloud evaluators use narrow validated loaders.

## Valid experiment matrices

```bash
python run_experiment_matrix.py \
  --config configs/experiments/ate_matrix.yaml \
  --dry-run

python run_experiment_matrix.py \
  --config configs/experiments/pointcloud_matrix.yaml \
  --dry-run
```

ATE expands to 9 entries (`3 segmentation × 3 reconstruction`). Point-cloud
evaluation expands to 3 entries (`3 segmentation × no_loop`). A point-cloud
loop entry fails before any reconstruction or metric backend call.

See [reconstruction-evaluation-cloud-validation.md](reconstruction-evaluation-cloud-validation.md)
for cloud setup and full evaluation commands.
