# Joint Loop-Constraint Estimation Repair

## Status

Approved repair direction for `codex/modular-segmentation-loop-integration`.
This design fixes the loop-constraint regression diagnosed on the cloud
without changing segmentation, anchor propagation, loop detection, or graph
optimization.

## Problem

The pre-integration loop pipeline estimated every loop constraint by:

1. selecting a bounded image neighborhood around both loop-candidate frames;
2. concatenating the two neighborhoods;
3. running Pi3 jointly on the concatenated images;
4. aligning each cached window side to the corresponding side of the joint
   Pi3 prediction; and
5. converting the two alignments into either a traditional or corrected loop
   constraint.

The modular pipeline replaced this production path with a direct single-frame
registration between two independently predicted cached point maps. Pixels in
two distant loop frames are not correspondences, so these measurements can
lower the graph optimizer's internal residual while damaging the physical
trajectory.

The regression is visible on KITTI:

- sequences without detected loops retain plausible trajectories;
- sequences with several loop constraints show large ATE regressions;
- changing the shared confidence keep ratio from `0.5` to `0.3` does not
  remove the failure; and
- `loop.constraint.chunk_size` is parsed and validated but is not consumed by
  the runtime.

## Scope

The repair will:

- restore joint Pi3 inference for production loop-constraint estimation;
- share that inference and alignment implementation between `traditional` and
  `corrected`;
- consume the canonical `loop.constraint.chunk_size` and
  `loop.registration.confidence_keep_ratio`;
- retain the immutable image manifest and the already preprocessed input
  tensor used by the main run;
- skip only candidate-local validation failures while allowing model,
  device, and resource failures to terminate the run; and
- preserve the existing loop-specific constraint conversion and optimizer
  implementations.

The repair will not:

- change depth, geometry, or atomic segmentation;
- change atomic split modes;
- change anchor propagation;
- change SALAD candidate detection or NMS;
- change the Sim(3) optimizer;
- reintroduce retired CLI parameters; or
- make traditional pose trajectories depend on the selected segmentation
  method.

## Architecture

### Shared joint-alignment estimator

Create `loop_closure/constraint_estimation.py` with a focused callable class:

```python
class JointPi3AlignmentEstimator:
    def __init__(
        self,
        *,
        model,
        images: torch.Tensor,
        manifest: ImageManifest,
        chunk_size: int,
        confidence_keep_ratio: float,
        inference_device: str,
        dtype: torch.dtype,
    ) -> None: ...

    def __call__(
        self,
        cache_a: WindowCache,
        cache_b: WindowCache,
        candidate: LoopCandidate,
        keep_ratio: float,
    ) -> tuple[Sim3, Sim3]: ...
```

The callable signature deliberately matches the existing strategy injection
point. The estimator owns production dependencies; strategies remain
responsible only for converting two alignments into their method-specific
measurement.

The constructor validates that the image tensor length equals the manifest
length and that `chunk_size` and the configured keep ratio are in their
canonical ranges. Each call rejects a strategy-supplied keep ratio that does
not match the configured value. This prevents labels or test names from
silently diverging from the resolved configuration.

### Candidate neighborhood mapping

For each candidate side:

1. use the selected `WindowCache.frame_start` and `frame_end` as hard bounds;
2. select at most exactly `chunk_size` consecutive frames centered on the
   candidate, shifting the range at a cache boundary instead of shortening it
   when enough cached frames remain;
3. slice the already preprocessed image tensor with the resulting absolute
   frame range;
4. concatenate side A followed by side B; and
5. run one Pi3 forward pass for that candidate.

The side lengths are recorded before concatenation. Joint prediction side A is
the leading slice and side B is the trailing slice. Cached points, poses, and
confidence use the same absolute-to-cache-relative slices.

The default `chunk_size=20` therefore restores the reference neighborhood size
of up to 20 frames per side while respecting sequence and selected-window
boundaries. Odd values and `chunk_size=1` remain valid and have unambiguous
exact-size behavior.

### Dual alignment

For each side, the estimator:

1. selects the highest-confidence fraction from the cached prediction;
2. selects the same fraction from the joint prediction;
3. intersects the two masks;
4. calls `register_adjacent_windows` with the cached side as the source and the
   joint side as the target; and
5. validates the returned Sim(3).

It returns `(alignment_a, alignment_b)`. It never directly treats side A and
side B pixels as correspondences.

### Strategy responsibilities

`TraditionalLoopClosureStrategy` will:

- require a joint estimator whenever non-empty cross-window candidates exist;
- compute its measurement with
  `compute_sim3_ab(alignment_a, alignment_b)`; and
- retain its existing sequential-edge optimizer and delayed aggregation.

`CorrectedLoopClosureStrategy` will:

- require the same joint estimator;
- convert the two global alignments with `build_local_loop_constraint` using
  the cached absolute transforms; and
- retain its existing edge optimization and optimization-delta aggregation.

The direct single-frame production fallback will be removed. Tests may inject
a deterministic estimator explicitly.

### Runner wiring

`PipelineRunner` already owns all required inputs: model, preprocessed images,
manifest, resolved config, caches, and candidates. Before constraint
construction it will build one `JointPi3AlignmentEstimator` and pass it
explicitly to `build_constraints`.

The legacy evaluation entry point will use the same factory with its model,
image tensor, manifest, and resolved config. No-loop paths will not construct
or invoke the estimator.

`PipelineDependencies` will expose the estimator factory so runner tests can
verify dependency propagation without loading Pi3.

## Candidate Failure Handling

The two strategies will isolate `ValueError` raised while mapping, masking,
aligning, or validating one candidate:

- log the candidate frames, selected window indices, and reason;
- omit that constraint; and
- continue with later candidates.

Exceptions representing system or model failures, including CUDA OOM and
unexpected `RuntimeError`, will propagate. This avoids hiding infrastructure
failures as rejected loop candidates.

Candidates mapping to the same window are ignored. Duplicate window pairs are
deduplicated in candidate order so the graph receives at most one measurement
for a window pair, matching the reference behavior.

If no valid constraints remain, the existing no-loop optimization path is
used.

## Diagnostics

Existing diagnostics remain the source of truth:

- `candidate_count` records detector output;
- `constraint_count` records accepted, deduplicated measurements;
- `used_no_loop_path` records whether optimization was skipped; and
- stage timings include the restored joint inference in `loop_constraints`.

Candidate-local rejection messages will include enough context to identify the
failed frames and windows in the captured run log. This repair does not add a
second diagnostics schema.

## Testing

Tests will prove the missing behavior rather than only the final type contract:

1. a joint-estimator test verifies that Pi3 receives side A followed by side B
   neighborhoods whose sizes come from `chunk_size`;
2. a registration test verifies two cached-to-joint alignment calls and
   rejects any direct cached-A-to-cached-B call;
3. traditional and corrected strategy tests verify that non-empty candidates
   require an explicitly supplied estimator;
4. strategy tests verify their distinct conversion formulas using the same
   injected alignments;
5. a failure-isolation test verifies that one invalid candidate is skipped and
   the next valid candidate remains;
6. a deduplication test verifies one constraint per window pair;
7. runner tests verify that the resolved model, images, manifest, chunk size,
   keep ratio, device, and dtype reach the estimator factory;
8. no-loop tests verify the estimator factory and optimizer are not invoked;
9. the full unit and matrix suites must remain green; and
10. the cloud acceptance run will evaluate KITTI 00 first before rerunning the
    full matrix.

Cloud acceptance requires:

- completed diagnostics and trajectory output;
- three traditional segmentation variants producing the same trajectory;
- traditional KITTI 00 no longer showing the observed approximately
  `61 m` ATE regression;
- corrected depth and atomic results returning to the range established by the
  pre-integration corrected pipeline; and
- logged candidate and accepted-constraint counts matching the reference run
  after candidate-local rejection and deduplication.

## Compatibility

The strict public YAML schema is unchanged. The repair activates the already
canonical fields:

```yaml
loop:
  registration:
    confidence_keep_ratio: 0.30
  constraint:
    chunk_size: 20
```

Window-cache schema version 1 remains unchanged because joint loop inference
consumes existing cache fields without altering serialization.
