# Traditional Second-Global Loop Optimization Design

## Status

Approved in design review on 2026-09-02.

## Decision summary

Add an opt-in reconstruction mode named `traditional_second_global`. It runs
the existing Traditional reconstruction as an unchanged Stage 1, materializes
the full overlap-preserving Stage 1 window geometry, rebuilds both adjacent
and loop Sim(3) constraints from that corrected geometry, performs a second
global optimization, and applies the final correction to points and camera
poses exactly once.

The run writes two independently evaluable reconstruction artifacts. The
output root is the Stage 2 artifact, and `stage1/` below it is the Traditional
baseline artifact. The existing `traditional` mode remains unchanged.

## Motivation

The current Traditional pipeline estimates adjacent-window Sim(3) transforms
before segmentation and anchor-scale propagation. It detects loops, estimates
loop constraints, performs one global optimization, and only then applies the
optimized transforms and anchor scale masks while aggregating the final point
map.

Consequently, the first global optimizer does not consume the complete
geometry that exists after its own transforms and the anchor corrections have
been applied. A second pass can use this corrected geometry to estimate a new
residual pose graph and test whether it improves trajectory accuracy.

The experimental comparison must preserve a strict baseline:

```text
ATE(stage1) == ATE(standalone traditional)
ATE(stage2) is the proposed result
```

The first equality is a regression requirement, not an empirical expectation.
It is enforced first on artifact tensors and then checked in real-data ATE
runs.

## Goals

1. Preserve the existing Traditional Stage 1 computation and output tensors.
2. Build a second adjacent-window Sim(3) chain from Stage 1 corrected point
   geometry and poses.
3. Re-estimate loop constraints from Stage 1 corrected windows while reusing
   the image-only loop candidates detected during Stage 1.
4. Optimize the second residual pose graph with the existing Sim(3) optimizer.
5. Apply the Stage 2 optimization delta consistently to local points and
   camera poses and recompute global points from them.
6. Write Stage 1 and Stage 2 as standard artifacts accepted by the existing
   ATE evaluator.
7. Keep existing reconstruction modes, default experiment matrices,
   prediction cache identity, segmentation, anchor propagation, loop
   detection, and optimizer mathematics unchanged.

## Non-goals

- Replacing joint PI3 loop-constraint estimation with ICP, GICP, or direct
  same-pixel matching between distant frames.
- Changing SALAD candidate ranking, thresholds, NMS, or checkpoints.
- Changing segmentation, window-reference refinement, temporal graphs, or
  anchor-scale propagation.
- Changing `Sim3LoopOptimizer` residuals, damping, solver, or convergence.
- Automatically adding the new mode to the existing canonical nine-entry ATE
  matrix.
- Enabling point-cloud benchmark evaluation for loop reconstruction modes.
- Claiming an ATE improvement before a controlled real-data GPU campaign.
- Persisting private overlap-window caches as a new public artifact format.

## Existing behavior that must remain stable

The following files define the Traditional baseline and are not modified:

- `reconstruction/modes/traditional.py`
- `loop_closure/methods/traditional.py`

The new mode invokes `TraditionalReconstructionMode.run()` with a context whose
mode is `traditional`. It injects recording wrappers around the detector and
processor to retain the detector candidates and Stage 1 `LoopSolution` needed
by Stage 2. The wrappers delegate every Stage 1 call and return value without
alteration.

Stage 1 therefore continues to execute:

```text
ordinary PI3 predictions
  -> early adjacent registration
  -> segmentation and optional window-reference refinement
  -> temporal graphs
  -> deferred anchor scale masks
  -> one SALAD detection
  -> Stage 1 joint-PI3 loop constraints
  -> Stage 1 global Sim(3) optimization
  -> existing Traditional aggregation
  -> Stage 1 ReconstructionArtifact
```

The test contract compares the four dense Stage 1 tensors against an
independent `traditional` run with the same injected deterministic services:

- `local_points`;
- `global_points`;
- `camera_poses`; and
- `confidence`.

The comparison uses `torch.equal`, not an approximate tolerance.

## Selected architecture

### Top-level flow

```text
same WindowPrediction stream
  -> unchanged Traditional Stage 1
       -> Stage 1 standard artifact
       -> recorded raw window trace, candidates, and Stage 1 solution
  -> materialize full Stage 1 windows without overlap trimming
  -> sequential residual registration on corrected geometry
  -> Stage 2 joint-PI3 loop-constraint estimation with the same candidates
  -> Stage 2 residual Sim(3) graph optimization
  -> one final optimization delta per window
  -> Stage 2 standard artifact
```

Loop detection is not repeated because its input is the image sequence, which
does not change between stages. Joint PI3 evidence estimation is repeated
because its cached alignment targets are the corrected Stage 1/Stage 2 window
states and therefore do change.

### Internal staged result

The new mode returns an internal immutable staged result containing:

- the Stage 1 `ReconstructionArtifact`;
- the Stage 2 `ReconstructionArtifact`; and
- numeric stage diagnostics needed by the runner.

`PipelineRunner` recognizes this result, atomically writes both artifacts, and
continues to return the Stage 2 `ReconstructionArtifact` from `run()`. Existing
callers that only consume the primary artifact therefore retain their current
return type and behavior.

## Sim(3) convention

For `S = (s, R, t)`, the action on a point in the relevant world coordinate
system is:

```text
S(x) = s R x + t
```

`accumulate_sim3(A, B)` is `A compose B`: apply `B`, then apply `A`.

The first window anchors every graph and uses identity. All scales must be
finite, scalar, and strictly positive. Rotations and translations must have
shapes `(3, 3)` and `(3,)`, be finite, and satisfy the existing validation
contracts. Residual alignment additionally retains the repository's
orthonormal and right-handed rotation checks.

## Stage 1 window materialization

The existing Traditional artifact removes duplicate overlap frames. Stage 2
needs every full window so it can re-register adjacent overlaps and select
candidate-centered loop ranges. It therefore reconstructs full Stage 1 window
states from the recorded raw trace and Stage 1 optimized transforms.

Let `T_i^(1)` be the Stage 1 optimized relative transforms and let:

```text
G_0^(1) = identity
G_i^(1) = G_(i-1)^(1) compose T_i^(1)
```

The materializer reproduces the existing Traditional aggregation semantics
exactly for each untrimmed window:

- without an anchor mask, local points receive the current absolute scale;
- with an anchor mask, local points receive the previous absolute scale times
  the stored anchor mask, matching the existing baseline implementation;
- camera poses receive the full current absolute Sim(3);
- confidence is cloned unchanged; and
- frame ranges and segmentation metadata are retained.

The materializer never mutates the Stage 1 trace. A characterization test
trims and concatenates its full windows using the existing overlap rule and
proves exact equality with the Stage 1 artifact. This prevents the second pass
from quietly operating on a different interpretation of Stage 1.

## Stage 2 adjacent residual chain

Let `W_i^(1)` be a full Stage 1 materialized window. Stage 2 builds corrected
states `W_i^(2-init)` sequentially.

For the first window:

```text
D_0 = identity
W_0^(2-init) = W_0^(1)
```

For every later window, the existing post-anchor residual registration
primitive receives:

- the previous Stage 2 initialized overlap points and poses;
- the current Stage 1 materialized overlap points and poses;
- both confidence tensors;
- the configured overlap; and
- `registration.confidence_keep_ratio`.

It estimates `D_i`, applies the residual scale to the current local points,
and applies the full residual Sim(3) to the current camera poses. The resulting
state becomes the reference for the next window.

The residual sequential edge is:

```text
E_i = inverse(D_(i-1)) compose D_i
```

Every non-first window must produce a valid residual. This stage is intrinsic
to `traditional_second_global` and is not gated by anchor propagation: even
when anchor propagation is disabled, Stage 1 global optimization has produced
the geometry that the explicitly selected second pass must consume.

## Stage 2 loop constraints

Stage 2 iterates over the exact candidate tuple returned by the single Stage 1
SALAD call. This includes candidates that failed to produce a Stage 1
constraint; corrected geometry may make such a candidate valid in Stage 2.

For each valid cross-window candidate, the existing joint PI3 evidence
provider aligns both Stage 2 initialized cached windows to their respective
sides of one joint prediction, producing `L_A` and `L_B`.

The global correction and residual-node measurement are:

```text
C_global = L_B compose inverse(L_A)
C_AB = inverse(D_B) compose C_global compose D_A
```

The measurement conversion is the same coordinate-safe conversion used by
the corrected loop method. It is required because the evidence targets are
already in the Stage 2 initialized common frame, while the optimizer consumes
constraints in the residual node coordinates.

Candidate-to-window mapping uses the existing latest-containing-window rule.
Same-window candidates are ignored, duplicate window pairs are deduplicated
in candidate order, and candidate-local `ValueError` failures are logged and
skipped. Runtime/model/resource failures propagate.

## Stage 2 optimization and one-delta aggregation

The optimizer receives:

- all `N - 1` Stage 2 residual sequential edges; and
- every accepted Stage 2 loop constraint.

It returns optimized residual edges, which are accumulated from identity:

```text
O_0 = identity
O_i = O_(i-1) compose optimized_E_i
```

For each Stage 2 initialized state, aggregation computes exactly one delta:

```text
Delta_i = O_i compose inverse(D_i)
```

The delta is applied once:

- `local_points_i <- delta_scale * local_points_i`;
- `camera_poses_i <- Delta_i compose camera_poses_i`;
- confidence remains unchanged;
- duplicate overlap frames are trimmed once; and
- final global points are recomputed from the resulting local points and
  camera poses.

Stage 1 transforms, anchor masks, residual transforms, and optimized deltas
must never be applied twice.

If no Stage 2 loop constraints survive, the graph optimizer is not called.
The Stage 2 output is still the sequentially residual-aligned geometry, and
diagnostics record that the optimizer used its no-loop path. With one input
window, Stage 2 is identity and equals Stage 1.

## Configuration and registry

Add:

```text
ReconstructionMode.TRADITIONAL_SECOND_GLOBAL = "traditional_second_global"
```

The mode requires the same complete loop services as `traditional` and
`corrected`. It reuses the existing loop detector, evidence provider, and
optimizer configuration. No new numerical hyperparameters are introduced.

The canonical default ATE matrix remains explicitly limited to:

```text
no_loop, traditional, corrected
```

Adding the enum must not silently turn the existing 3 x 3 matrix into a 3 x 4
matrix. The new mode is selected explicitly through reconstruction overrides
or a dedicated future experiment configuration.

Prediction cache fingerprints remain unchanged because reconstruction mode,
segmentation, anchor propagation, loop settings, and evaluation settings do
not participate in ordinary PI3 prediction identity.

## Artifact layout and atomicity

The result directory is both a valid Stage 2 artifact and the staged bundle
root:

```text
outputs/reconstruction/<scene>/<segmentation>-traditional_second_global/
  manifest.json
  trajectory.pt
  pointmap.pt
  confidence.pt
  diagnostics.json
  resolved_reconstruction.yaml
  stage1/
    manifest.json
    trajectory.pt
    pointmap.pt
    confidence.pt
    diagnostics.json
    resolved_reconstruction.yaml
```

Existing loaders and evaluators read explicit filenames and ignore the extra
`stage1/` directory. Therefore:

```bash
python evaluate_ate.py --artifact <result>/stage1 ...
python evaluate_ate.py --artifact <result> ...
```

evaluate Stage 1 and Stage 2 independently.

The staged writer builds the complete tree in a temporary sibling directory
and renames the finished Stage 2 root into place. It refuses to overwrite an
existing result. A Stage 2 computation or write failure leaves no completed
bundle at the requested destination.

Both artifact manifests retain the same ordinary prediction key, checkpoint
digest, source commit, and run configuration provenance. Their own
`reconstruction_mode` fields distinguish `traditional` Stage 1 from
`traditional_second_global` Stage 2.

`PipelineRunner.artifact_dir` remains the Stage 2 root. A new immutable stage
directory mapping exposes `stage1` and `stage2` paths for CLI reporting and
programmatic use.

## Diagnostics

The Stage 1 artifact retains its existing Traditional diagnostics unchanged.

The Stage 2 artifact uses its top-level `candidate_count` and
`constraint_count` for the shared detector candidates and accepted Stage 2
constraints. Numeric mode scalars include:

- `window_count`;
- `stage1_candidate_count`;
- `stage1_constraint_count`;
- `stage1_used_no_loop_path`;
- `stage2_candidate_count`;
- `stage2_constraint_count`;
- `stage2_used_no_loop_path`;
- `stage2_residual_applied_window_count`;
- `stage2_residual_skipped_window_count`;
- maximum and mean absolute log residual scale;
- maximum and mean residual rotation in radians;
- maximum and mean residual translation norm; and
- maximum and mean absolute log final optimization scale delta.

The first window is counted as a skipped residual. A multi-window run must
report exactly `window_count - 1` applied adjacent residuals.

Stage timings distinguish at least Stage 1 reconstruction, Stage 1 window
materialization, Stage 2 adjacent residual alignment, Stage 2 constraint
estimation/optimization, and Stage 2 aggregation when the existing timing
boundary can do so without changing Stage 1 diagnostics.

## Failure handling

The new mode fails the run when:

- Stage 1 does not produce a trace, candidate tuple, solution, or artifact;
- Stage 1 solution length differs from the raw window trace;
- Stage 1 materialized windows do not cover the Stage 1 artifact frame set;
- a Stage 2 adjacent confidence mask has no shared correspondences;
- a Stage 2 residual or optimization transform is malformed or non-finite;
- an applied transform changes tensor shapes or produces non-finite values;
- the optimizer returns the wrong number of edges/transforms; or
- Stage 2 aggregation does not cover every frame exactly once.

There is no silent fallback from a requested second pass to the Stage 1
artifact. Candidate-local validation failures remain isolated in the same way
as existing loop methods. CUDA OOM, model failures, unexpected runtime errors,
and filesystem failures propagate.

## Testing strategy

Implementation follows test-driven development.

### Stage 1 preservation

- Run standalone `traditional` and the new mode with identical deterministic
  predictions and injected services.
- Assert exact equality of all four Stage 1 tensors.
- Assert equal Stage 1 segmentation summaries, candidate count, constraint
  count, and mode scalars.
- Prove SALAD is invoked exactly once in the new mode.
- Prove the two existing Traditional implementation files are not modified by
  this feature.

### Materialization

- Exercise identity, scale, rotation, translation, anchor-mask, and overlap
  cases.
- Assert the input trace is not mutated.
- Trim the full materialized windows and prove exact tensor equality with the
  existing Traditional aggregation.
- Reject state/solution count mismatch and non-finite output.

### Stage 2 adjacent residuals

- Inject distinct Stage 1 and residual transforms and verify call order.
- Prove each adjacent estimator receives Stage 1 materialized current geometry
  and the already residual-refined previous geometry.
- Prove local points receive residual scale and poses receive full residual
  Sim(3).
- Prove `E_i = inverse(D_(i-1)) compose D_i` with non-commuting transforms.
- Reject degenerate confidence masks and invalid transforms.
- Verify one-window identity behavior.

### Stage 2 loop constraints and optimization

- Verify detector candidates are reused without a second detector call.
- Verify joint evidence is called independently for Stage 1 and Stage 2.
- Verify Stage 2 iterates over all detector candidates, including candidates
  rejected by Stage 1.
- Verify constraints use corrected Stage 2 window tensors and residual-node
  coordinate conversion.
- Verify the optimizer receives exactly `N - 1` rebuilt sequential edges and
  all accepted Stage 2 loop constraints.
- Verify no accepted loop constraints skip the optimizer while retaining the
  sequential residual result.

### Final aggregation and artifacts

- Prove `Delta_i = O_i compose inverse(D_i)` is applied exactly once.
- Prove local points, poses, and recomputed global points remain coordinate
  consistent.
- Verify overlap trimming and frame order.
- Round-trip both Stage 1 and Stage 2 through existing artifact loaders.
- Verify the Stage 2 root and `stage1/` paths are directly ATE-loadable.
- Verify staged writes refuse overwrite and leave no destination on injected
  failure.
- Verify `PipelineRunner.run()` returns Stage 2 and reports both stage paths.

### Regression and commands

At minimum, completion requires fresh successful runs of:

```bash
pytest -q
python run_reconstruction.py --help
python evaluate_ate.py --help
python run_experiment_matrix.py --help
```

A CPU synthetic integration run must exercise both stages without loading
production checkpoints. Real-data acceptance uses the same images, weights,
sample stride, window size, overlap, segmentation method, anchor settings,
loop configuration, and ATE configuration for both artifact evaluations.

## Real-data experiment protocol

For each selected scene:

1. Run `traditional` once and record its artifact digest and ATE.
2. Run `traditional_second_global` with otherwise identical configuration.
3. Assert its `stage1/trajectory.pt` tensor equals the standalone Traditional
   trajectory tensor.
4. Evaluate `<result>/stage1` and `<result>` with the same ground truth and ATE
   configuration.
5. Record Stage 1 and Stage 2 ATE/RPE plus both stages' candidate and constraint
   counts.
6. Treat a Stage 1 mismatch as an implementation regression; do not interpret
   the Stage 2 number until it is fixed.

The desired outcome is `ATE(stage2) < ATE(stage1)`, but the implementation is
accepted on data-flow correctness and regression safety before empirical
improvement is claimed.

## Acceptance criteria

The change is complete only when all of the following are true:

1. `traditional` retains its existing behavior and implementation files.
2. The new mode's Stage 1 tensors exactly equal a standalone Traditional run.
3. Full Stage 1 windows preserve overlap and reproduce the baseline artifact
   after the baseline trim rule.
4. Every non-first Stage 2 window is re-registered from Stage 1 corrected
   geometry against the already residual-refined predecessor.
5. Stage 2 loop constraints are re-estimated from corrected geometry using the
   same detector candidates and joint PI3 evidence path.
6. Rebuilt adjacent edges and accepted loop constraints enter one second
   Sim(3) graph optimization.
7. The final optimization delta is applied once to both points and poses.
8. Final global points are recomputed from the final local points and poses.
9. Stage 1 and Stage 2 are independently loadable by the existing ATE path.
10. The default nine-entry experiment matrix is unchanged.
11. Failure behavior is explicit and staged output is atomic.
12. Focused tests and the full baseline suite pass.
