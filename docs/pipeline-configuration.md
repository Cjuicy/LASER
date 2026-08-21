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

### Window reference segmentation refinement

`segmentation.window_reference.enabled` defaults to `false`. When enabled,
LASER selects reference frames adaptively from the current PI3 window, performs
sparse geometry/visibility checks, and may merge complete adjacent regions from
the selected initial segmentation method. It never splits a region. Missing or
unreliable geometry preserves the initial labels.

The geometry contract uses the authoritative provider intrinsic: traditional and
corrected modes use `WindowPrediction.reference_intrinsic` propagated from the
provider, while no-loop uses the current intrinsic actually used to
re-unproject its point maps. PI3 camera poses are OpenCV camera-to-world poses;
projection transforms source local points with
`inv(T_target_c2w) @ T_source_c2w` before applying `K`. The refiner never
estimates or replaces `K`. Missing, invalid, or incompatible `K` falls back to
the initial labels unchanged.

The complete configuration is:

| Field | Default | Accepted values | Purpose |
| --- | ---: | --- | --- |
| `segmentation.window_reference.enabled` | `false` | boolean | Enable the optional refinement stage. |
| `segmentation.window_reference.sampling_stride` | `4` | positive integer (`>= 1`) | Pixel stride for sparse projection and visibility checks; larger values reduce CPU work. |
| `segmentation.window_reference.max_keyframes` | `4` | positive integer (`>= 1`) | Maximum number of adaptively selected reference frames; smaller values reduce CPU work. |
| `segmentation.window_reference.relative_depth_tolerance` | `0.05` | finite float in `(0, inf)` | Relative depth agreement required for a projected sample. |
| `segmentation.window_reference.min_reference_score` | `0.30` | finite float in `(0, 1]` | Minimum directed pair score `S(k→t)` for a reference-to-target projection to count as reliable evidence; it is not a frame-quality threshold. |
| `segmentation.window_reference.stop_coverage_ratio` | `0.90` | finite float in `(0, 1]` | Stop selecting references once this coverage is reached. |
| `segmentation.window_reference.min_coverage_gain` | `0.03` | finite float in `[0, 1]` | Minimum additional coverage contributed by a reference. |
| `segmentation.window_reference.min_region_correspondences` | `8` | positive integer (`>= 1`) | Minimum sparse correspondences supporting a region mapping. |
| `segmentation.window_reference.min_region_coverage` | `0.10` | finite float in `(0, 1]` | Minimum target-region coverage for a mapping. |
| `segmentation.window_reference.min_region_purity` | `0.80` | finite float in `(0, 1]` | Minimum source-label purity for a mapping. |
| `segmentation.window_reference.merge_vote_threshold` | `0.80` | finite float in `(0, 1]` | Vote threshold for accepting an adjacent-region merge. |

The refiner runs on the CPU process device and is deliberately bounded by
`sampling_stride` and the final selected-reference ceiling
`min(max_keyframes, window frame count)`; `max_keyframes` limits the number of
references retained for the final refinement, not the number of candidates
considered. If candidates are repeatedly rejected for insufficient coverage
gain, the selection projection work can approach the worst-case `N²` directed
pair evaluations (without allocating an all-pairs tensor). Use a larger stride
or fewer keyframes when CPU budget matters. It is a merge-only stage: labels are compacted after
accepted merges, and no region is ever split. A single-frame window, missing or
invalid intrinsics, invalid/non-finite geometry, incompatible geometry, no
usable reference, or insufficient/conflicting support falls back to the
initial labels. The per-frame diagnostics record the fallback reason and the
before/after region counts.

The defaults and thresholds are implementation defaults, not a quality claim.
Their empirical effect must be calibrated with the real PI3 weights, dataset,
and prediction cache; this repository does not claim reconstruction-metric
improvement without that real-cache evaluation gate.

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
