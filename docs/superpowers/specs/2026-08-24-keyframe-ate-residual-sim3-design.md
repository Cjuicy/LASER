# Keyframe-Aware ATE Residual Sim(3) Design

## Status

Approved in design review on 2026-08-24.

## Problem

The current window-reference/keyframe refinement can improve segmentation and
the anchor-derived point scaling, but the improvement is not guaranteed to
reach trajectory evaluation.

Both `no_loop` and `corrected` estimate their coarse inter-window Sim(3) before
segmentation, window-reference refinement, temporal graph construction, and
anchor propagation. Anchor propagation then changes `local_points`, while the
already-computed camera poses remain unchanged. In `corrected`, `sim3_abs` and
`sim3_edge` also remain unchanged. As a result:

- point-cloud output can reflect anchor changes;
- ATE still consumes the pre-anchor camera poses;
- corrected loop optimization consumes pre-anchor sequential edges; and
- a keyframe refinement can be visible in diagnostics without affecting ATE.

The evaluation workflow is also single-metric per run. It cannot express a
scene that should evaluate one reconstruction artifact with both trajectory
and point-cloud evaluators, while tolerating scenes that only provide one kind
of ground truth.

## Goals

1. Add an explicit post-anchor residual Sim(3) stage shared by `no_loop` and
   `corrected`.
2. Make the residual transform affect both points and poses so the resulting
   artifact is internally coordinate-consistent.
3. Make corrected loop constraints and optimization consume sequential edges
   derived from the post-anchor refined absolute transforms.
4. Keep the traditional implementation and behavior unchanged.
5. Evaluate a no-loop reconstruction artifact with every evaluator supported
   by the scene, including ATE and point-cloud evaluation in the same run.
6. Evaluate traditional and corrected reconstruction only with ATE.
7. Represent missing evaluator capability as an explicit skip, while treating
   declared-but-invalid inputs and evaluator errors as failures.
8. Preserve successful evaluator output when another evaluator for the same
   artifact fails.
9. Reuse the ordinary PI3 prediction cache across reconstruction modes,
   keyframe variants, segmentation methods, and evaluators.

## Non-goals

- Changing the traditional reconstruction or loop-closure implementation.
- Enabling point-cloud benchmark scores for traditional or corrected modes.
- Changing PI3 inference, prediction cache payloads, or cache identity.
- Changing the window-reference keyframe selection algorithm.
- Changing segmentation or anchor propagation mathematics.
- Claiming empirical ATE or point-cloud improvements without real dataset and
  GPU campaign results.
- Silently recovering from an invalid residual transform.

## Current pipeline and root cause

The relevant current ordering is:

```text
PI3 cache
  -> WindowPrediction
  -> coarse registration
  -> segmentation
  -> window-reference/keyframe refinement
  -> temporal graph
  -> anchor point scaling
  -> aggregation or loop optimization
```

In `no_loop`, aggregation computes global points from anchor-adjusted local
points and coarse camera poses. In `corrected`, the processor optimizes
`sim3_edge` values created before anchor scaling. Neither path performs a
post-anchor alignment that can propagate anchor-derived geometry back into
camera poses.

## Proposed reconstruction architecture

### Shared post-anchor residual alignment

Introduce a focused reconstruction component for residual inter-window
alignment. It owns estimation, validation, application, composition, and
diagnostic measurement for one residual Sim(3). It does not own segmentation,
anchor propagation, aggregation, loop detection, or evaluation.

For every non-first window in `no_loop` and `corrected`, the mode executes:

```text
raw WindowPrediction
  -> coarse registration
  -> segmentation
  -> window-reference/keyframe refinement
  -> temporal graph
  -> anchor point scaling
  -> residual registration against the finalized previous overlap
  -> finalized window state
```

The residual estimator receives:

- the finalized previous window's overlap points and poses;
- the current anchor-adjusted overlap points;
- the current coarse-aligned overlap poses;
- both overlap confidence tensors;
- overlap size; and
- the configured registration confidence keep ratio.

It uses the existing mutual-confidence mask semantics and adjacent-window
registration primitive. This deliberately keeps coarse and residual
registration on the same Sim(3) convention.

### Sim(3) convention and composition

For `S = (s, R, t)`, the action on a world-space point is:

```text
S(x) = s R x + t
```

`accumulate_sim3(A, B)` means `A ∘ B`: apply `B`, then apply `A`.

For window `i`:

- `C_i` is the coarse absolute Sim(3);
- `D_i` is the post-anchor residual Sim(3); and
- `A_i = D_i ∘ C_i` is the refined absolute Sim(3).

The residual is applied consistently as:

```text
local_points_i <- d_scale * local_points_i
camera_poses_i <- D_i ∘ camera_poses_i
```

Only the scale is applied to camera-local points. Rotation and translation
belong to the camera-to-world pose. Applying the residual scale to both local
point coordinates and camera-center translations makes the derived global
points equal to applying `D_i` to the pre-residual global points.

The first window uses the identity absolute transform and has no residual
registration.

### Gating

The residual stage runs only when all of the following are true:

- the mode is `no_loop` or `corrected`;
- the window is not the first window; and
- `anchor_propagation.enabled` is true.

When anchor propagation is disabled, no geometry changes after coarse
registration. The residual stage is therefore recorded as `skipped` and is
not called. `segmentation.window_reference.enabled` does not independently
gate residual registration: keyframe refinement changes graph structure, and
anchor propagation is the stage that converts that graph decision into point
geometry.

### No-loop path

`no_loop` retains a finalized state for the previous window. Each current
window is coarse-aligned, segmented, keyframe-refined, graph-constructed,
anchor-scaled, and residual-aligned before it becomes the next registration
source.

Aggregation trims overlap exactly once and emits:

- residual-refined local points;
- residual-refined camera poses;
- global points derived from those two tensors; and
- the unchanged confidence tensors.

The same reconstruction artifact is the source for both point-cloud and ATE
evaluation. No evaluator is allowed to reconstruct independently.

### Corrected path

`CorrectedWindowState` stores the refined absolute transform `A_i`, not the
pre-anchor coarse transform. For every window after the first, the sequential
edge is rebuilt as:

```text
E_i = inverse(A_(i-1)) ∘ A_i
```

Loop evidence converts global alignments into local loop constraints using the
refined `A_i` transforms. The corrected optimizer consumes only the refined
sequential edges and refined loop constraints.

If the optimizer returns optimized absolute transforms `O_i`, aggregation
applies exactly one final delta:

```text
Delta_i = O_i ∘ inverse(A_i)
```

`Delta_i` scales the already residual-refined local points and transforms the
already residual-refined camera poses. This preserves the existing one-delta
invariant and prevents applying either the residual or loop correction twice.

Even though corrected artifacts are not point-cloud benchmark inputs, their
local points, global points, camera poses, absolute transforms, and edges must
remain mutually consistent.

### Traditional path

The following implementation files are not modified:

- `reconstruction/modes/traditional.py`
- `loop_closure/methods/traditional.py`

Traditional remains the original baseline. Campaign planning forces
`segmentation.window_reference.enabled=false` for traditional and emits one
traditional variant per segmentation method and ATE-capable scene. It does not
produce a keyframe on/off pair.

## Residual data contract

The shared residual component returns an immutable result with:

- the validated residual Sim(3);
- whether the residual was applied or skipped;
- the number of selected overlap correspondences;
- absolute log scale magnitude;
- rotation angle in radians; and
- translation norm.

Validation requires:

- a scalar, finite, strictly positive scale;
- finite rotation and translation tensors;
- rotation shape `(3, 3)` and translation shape `(3,)`;
- an orthonormal, right-handed rotation within the repository's numerical
  tolerance; and
- output points and poses with unchanged shapes and finite values.

An enabled residual stage never silently returns identity after an estimation
failure. A degenerate confidence mask, invalid transform, invalid shape, or
non-finite output fails that reconstruction variant.

## Diagnostics

Reconstruction artifacts retain their current tensor layout and schema-facing
load contract. Diagnostics are extended with numeric mode scalars sufficient
to audit whether keyframe-derived geometry reached the pose path:

- `residual_applied_window_count`;
- `residual_skipped_window_count`;
- `max_abs_log_residual_scale`;
- `mean_abs_log_residual_scale`;
- `max_residual_rotation_rad`;
- `mean_residual_rotation_rad`;
- `max_residual_translation_norm`;
- `mean_residual_translation_norm`; and
- for corrected, `refined_edge_count`.

The first window is counted as skipped. Anchor-disabled runs count all windows
as skipped. A successful anchor-enabled multi-window run must report at least
one applied residual. Existing segmentation, candidate, constraint, cache, and
timing diagnostics remain intact.

## Capability-aware evaluation campaign

### Scene capability declaration

Scene configuration explicitly declares optional evaluator inputs:

```yaml
trajectory_ground_truth:
  path: data/groundtruth.txt
  format: tum
pointcloud_ground_truth:
  path: data/groundtruth_pointmaps.npz
```

A scene must declare at least one evaluator input. The scheduler never infers
capability from a file that happens to exist.

- An omitted declaration means the evaluator is unsupported and is skipped.
- A declared path that is missing, unreadable, or invalid is a failure.

Evaluation config paths remain evaluator-specific and are not part of the
reconstruction artifact identity.

### Variant scheduling

For every configured segmentation method, scene variants are capability
driven:

| Scene capability | no_loop | traditional | corrected |
| --- | --- | --- | --- |
| ATE + point cloud | keyframe off/on; run both evaluators on each artifact | baseline ATE | keyframe off/on; ATE |
| ATE only | keyframe off/on; ATE | baseline ATE | keyframe off/on; ATE |
| Point cloud only | keyframe off/on; point cloud | not scheduled | not scheduled |

The prediction key remains shared because segmentation method,
window-reference state, reconstruction mode, and evaluation choice do not
change ordinary PI3 predictions.

### Evaluation bundle execution

Each reconstruction variant writes one complete artifact. An evaluation bundle
then invokes every scheduled evaluator independently against that same
artifact.

Each evaluator record has one terminal status:

- `passed`: metrics were written successfully;
- `skipped`: the scene does not declare that capability or the mode does not
  support that evaluator;
- `failed`: the capability was declared, but its input or computation failed;
  or
- `blocked`: reconstruction failed, so the evaluator could not run.

Evaluator output is isolated and atomic. If one evaluator fails after another
passes, the successful metrics remain valid and resumable. The variant's final
status is failed when any scheduled evaluator fails. An unsupported evaluator
skip does not fail the variant.

Errors are recorded in compact structured form with evaluator kind, exception
type, and a sanitized message. Tracebacks stay in the attempt log and are not
embedded in portable summary records.

### Resume and identity

Resume accepts an evaluator result only when all of the following match:

- reconstruction artifact identity and manifest digest;
- scene capability declaration digest;
- evaluator configuration digest;
- ground-truth input digest; and
- evaluator implementation/source revision.

A previously passed evaluator may be reused independently of another failed or
missing evaluator. A changed declaration, config, truth file, or source
revision invalidates only the affected evaluation record; it does not silently
reuse stale metrics.

The new campaign uses a distinct schema/campaign identity so existing
single-evaluator window-reference campaign records cannot be mistaken for
multi-evaluator results.

## Failure handling

### Reconstruction failures

Residual estimation and validation failures terminate that reconstruction
variant. All scheduled evaluator records for the variant become `blocked`.
Other variants and scenes continue under keep-going execution.

### Evaluation failures

Evaluator failures are isolated from each other. A passed result is never
deleted because a sibling evaluator failed. The campaign exits non-zero if any
scheduled evaluator failed or any reconstruction was blocked, while still
writing the complete summary of passed, skipped, failed, and blocked records.

### Configuration failures

Configuration and planning fail before reconstruction when:

- a scene declares neither ground-truth type;
- a traditional or corrected point-cloud-only variant is requested directly;
- traditional is configured with window-reference refinement enabled;
- evaluator configuration for a declared capability is missing; or
- duplicate variant or output identities are generated.

## Testing strategy

Implementation follows test-driven development.

### Residual unit tests

- Verify the Sim(3) convention and `D_i ∘ C_i` composition order.
- Verify local-point scale and full pose application produce consistent global
  points.
- Reject non-scalar, non-finite, non-positive, reflected, or non-orthonormal
  transforms.
- Reject degenerate overlap confidence selection.
- Verify skip results for first-window and anchor-disabled cases.

### No-loop mode tests

- Inject distinct coarse and residual transforms and prove residual runs after
  anchor propagation.
- Prove the finalized window becomes the next window's registration source.
- Prove residual scale changes local points and residual rotation/translation
  change camera poses.
- Prove `trajectory.pt` contains residual-refined poses and `pointmap.pt`
  contains points derived from the same finalized state.
- Prove anchor-disabled output matches the pre-change behavior and does not
  call the residual estimator.

### Corrected mode tests

- Prove state absolute transforms equal `residual ∘ coarse`.
- Prove sequential edges are recomputed from adjacent refined absolutes.
- Prove loop constraint construction receives refined absolute transforms.
- Prove the optimizer receives refined sequential edges.
- Prove the final optimized delta is applied exactly once to points and poses.
- Prove the no-constraint corrected path still emits the residual-refined
  trajectory.
- Prove anchor-disabled behavior matches the pre-change behavior.

### Traditional regression tests

- Run the existing traditional characterization suite unchanged.
- Assert traditional campaign variants always have keyframe refinement off.
- Verify no diff is introduced in the two traditional implementation files.

### Evaluation campaign tests

- A dual-capability no-loop artifact invokes ATE and point-cloud evaluators
  exactly once each.
- An ATE-only scene skips point-cloud evaluation and schedules all three modes
  according to the table.
- A point-cloud-only scene schedules only no-loop variants.
- Traditional has one baseline variant while no-loop and corrected have
  keyframe off/on variants.
- A failed evaluator preserves a sibling passed result and yields a failed
  variant.
- A reconstruction failure yields blocked evaluator records.
- A declared missing or malformed ground truth fails; an omitted declaration
  skips.
- Resume reuses each passed evaluator independently and invalidates changed
  inputs precisely.
- CPU synthetic integration covers artifact creation, both evaluator calls,
  status serialization, summary generation, and resume without model or GPU.

### Verification commands

At minimum, completion requires:

```bash
pytest -q
python run_reconstruction.py --help
python evaluate_ate.py --help
python evaluate_pointcloud.py --help
python run_experiment_matrix.py --help
```

Campaign planning/dry-run verification must additionally prove the exact
capability-derived variant and evaluator counts for the checked-in synthetic,
ATE-only, and point-cloud-only fixtures.

## Compatibility and migration

- Existing reconstruction tensor filenames and loader contracts remain
  unchanged.
- Existing single-evaluator experiment configs remain readable and retain
  their current behavior.
- The new capability-aware campaign uses a new config/schema version and a new
  output identity.
- Existing prediction cache entries remain reusable because prediction schema
  and identity are unchanged.
- Existing window-reference campaign records are read-only historical records;
  they are not upgraded in place or pooled with the new campaign.

## Acceptance criteria

The work is complete only when all of the following are true:

1. A non-identity post-anchor residual changes no-loop camera poses and is
   observable in the trajectory artifact used for ATE.
2. The same no-loop artifact can be evaluated by both ATE and point-cloud
   evaluators without reconstruction duplication.
3. Corrected sequential edges and loop optimization derive from post-anchor
   refined absolute transforms.
4. A corrected run with no accepted loop constraints still emits the refined
   no-loop trajectory rather than the pre-anchor coarse trajectory.
5. Points and poses remain coordinate-consistent after residual and corrected
   optimization deltas.
6. Anchor-disabled no-loop and corrected runs preserve their pre-change
   outputs.
7. Traditional implementation files are unchanged and traditional campaign
   variants force keyframe refinement off.
8. Scene capability declarations schedule the exact evaluator set defined in
   the table.
9. Missing capability is skipped, declared invalid input fails, sibling
   evaluator success is preserved, and reconstruction failure blocks dependent
   evaluators.
10. The full local test suite passes after the required Cython extensions are
    built.
