# Window Reference Segmentation Refinement Design

Date: 2026-08-21

Status: design direction approved in chat; this written specification awaits review
before implementation.

Target branch: `codex/laser-paper-pointmap-eval`

## 1. Goal

Add an optional window-level segmentation refinement stage that uses PI3 geometry,
camera poses, confidence, and adaptively selected reference frames to consolidate
fragmented per-frame regions.

The stage runs after the existing `depth`, `geometry`, or `atomic` segmentation
strategy and before temporal graph construction. It may merge complete existing
regions, but it may never split a region or modify PI3 prediction tensors.

The feature is disabled by default. When disabled, segmentation results,
diagnostics, temporal graphs, anchor inputs, and reconstruction tensors remain
identical to the current branch.

## 2. Scope

### 2.1 In scope

- Reuse the current window's canonical local point maps, camera-to-world poses,
  PI3 confidence logits, and reference intrinsic.
- Select one or more reference frames based on frame quality and sparse geometric
  coverage. The selected count is data dependent.
- Project reference-frame points and initial labels into non-reference target
  frames with a deterministic CPU implementation.
- Reject occluded, low-confidence, and geometrically inconsistent projections.
- Convert reliable projected correspondences into evidence over the target
  frame's existing four-connected region adjacency graph.
- Merge adjacent regions conservatively with deterministic union-find.
- Record scalar diagnostics suitable for the current artifact JSON contract.
- Preserve prediction-cache and reconstruction-artifact schemas.

### 2.2 Out of scope

- A fourth initial segmentation method.
- Pixel-level label replacement, region splitting, or rerunning segmentation.
- Semantic or object identity prediction.
- Optical flow, learned affinities, new neural networks, feature matching, point
  cloud registration, bundle adjustment, or intrinsic optimization.
- Cross-window or global label persistence.
- Refinement after anchor propagation or loop optimization.
- GPU/CUDA kernels or new third-party dependencies.
- Repairing existing no-loop intrinsic re-estimation, view-direction conventions,
  diagnostic-output switches, or segmentation smoke warnings.
- Persisting segmentation labels as a new reconstruction artifact.

## 3. Verified existing contracts

### 3.1 Segmentation

`SegmentationStrategy.segment()` consumes point maps shaped `(N,H,W,3)`, optional
confidence `(N,H,W)`, and optional images. It returns one `SegmentationResult` per
frame. Each result contains a two-dimensional label array whose unique values are
exactly `0..R-1` and a mapping of scalar diagnostics.

All three current strategies segment each frame independently. The new feature is
a postprocessor and is not registered as a new `SegmentationMethod`.

`build_temporal_graphs()` consumes segmentation results immediately after the
strategy call. The required order is therefore:

```text
initial strategy.segment()
        -> window reference refinement
        -> build_temporal_graphs()
        -> anchor propagation
```

### 3.2 PI3 prediction geometry

The model constructs per-view local points in each camera coordinate system and
left-multiplies them by `camera_poses` to construct global points. Consequently,
`camera_poses` are OpenCV camera-to-world transforms.

The ordinary prediction provider stores depth, confidence, camera poses, and one
sequence-level `reference_intrinsic`. Every returned runtime local point map is
reconstructed from depth with that intrinsic, including cache-off inference.
The provider already owns the authoritative intrinsic used for reconstruction.

Confidence values are raw PI3 logits. Ranking may use the logits directly because
sigmoid is monotonic, but confidence weights must use a numerically stable sigmoid.

### 3.3 Reconstruction modes

The three modes call segmentation at different numerical states:

- `no_loop` has re-unprojected the point maps with its current intrinsic and has
  applied sequential window scale/pose changes.
- `traditional` records, but has not yet applied, its deferred adjacent Sim(3).
- `corrected` has applied the current absolute Sim(3) scale and pose transform.

Within each window, the point maps, poses, and intrinsic remain mutually
consistent. The refiner always consumes the tensors at the exact point where the
mode currently invokes segmentation. It does not reload cached predictions.

For no-loop, the intrinsic passed to the refiner is the current intrinsic that
no-loop used to re-unproject the point maps. Traditional and corrected pass the
provider intrinsic carried by `WindowPrediction`.

## 4. Considered approaches

### 4.1 Single reference frame

Select only the highest-quality frame and project it into all targets. This is
small and useful as an ablation, but it is fragile under occlusion and large view
changes and cannot detect conflicts between independent references.

### 4.2 Adaptive sparse references and region merging

Select references incrementally, stop when the window is sufficiently covered,
project sparse points with visibility checks, and merge only adjacent target
regions. This is the selected V1 because it is conservative, explainable,
bounded, independently testable, and requires no new dependency.

### 4.3 Dense all-pairs or global 3D region graph

Project every frame pair or maintain a cross-window 3D region graph. This offers a
higher theoretical coverage ceiling but introduces quadratic work, persistent
state, harder visibility conflicts, and coupling to cache, loop, and artifact
boundaries. It is excluded from V1.

## 5. Configuration

Add a required nested block to every complete pipeline YAML:

```yaml
segmentation:
  window_reference:
    enabled: false
    sampling_stride: 4
    max_keyframes: 4
    relative_depth_tolerance: 0.05
    min_reference_score: 0.30
    stop_coverage_ratio: 0.90
    min_coverage_gain: 0.03
    min_region_correspondences: 8
    min_region_coverage: 0.10
    min_region_purity: 0.80
    merge_vote_threshold: 0.80
```

The refiner reuses `segmentation.confidence_keep_ratio` and
`segmentation.confidence_quantile_method`; it does not introduce a second
confidence selection policy.

Validation rules are:

- `enabled` is a boolean.
- `sampling_stride`, `max_keyframes`, and `min_region_correspondences` are positive
  integers. Booleans are not accepted as integers.
- `relative_depth_tolerance` is finite and strictly positive.
- `min_reference_score`, `stop_coverage_ratio`, `min_region_coverage`,
  `min_region_purity`, and `merge_vote_threshold` are finite and in `(0,1]`.
- `min_coverage_gain` is finite and in `[0,1]`.

`max_keyframes` is a safety ceiling, not a requested selected count. A window may
stop with one selected frame when that frame already provides sufficient coverage.
The runtime ceiling is `min(max_keyframes, N)`.

The defaults are conservative experiment seeds. The 5% relative-depth threshold
matches the upstream PI3 geometry-warp default; the remaining values must be
reported as empirical controls and evaluated on real prediction caches before the
feature can be enabled by default.

## 6. Public interfaces and component boundaries

### 6.1 Configuration type

Add `WindowReferenceConfig` to `pipeline/config.py` and add it as a nested field of
`SegmentationConfig`.

### 6.2 Intrinsic propagation

`OrdinaryPredictionProvider` adds a cloned `reference_intrinsic` to its returned
mapping after the sequence intrinsic is established. This does not alter the
stored window artifact or cache schema.

`WindowPrediction` gains:

```python
reference_intrinsic: torch.Tensor | None = None
```

When present, `from_mapping()` validates shape `(3,3)`, floating dtype, finite
values, positive focal lengths, and a homogeneous last row close to `[0,0,1]`,
then moves it to `process_device`. The optional default preserves compatibility
with test providers. An enabled refiner treats a missing intrinsic as a fallback,
not as permission to estimate another intrinsic.

### 6.3 Refiner

Add `inference_engine/segmentation/window_reference.py` with a protocol-compatible
disabled implementation and the enabled `WindowReferenceRefiner`.

The enabled interface is:

```python
def refine(
    self,
    results: list[SegmentationResult],
    *,
    point_maps: torch.Tensor,
    camera_poses: torch.Tensor,
    confidence: torch.Tensor,
    reference_intrinsic: torch.Tensor | None,
) -> list[SegmentationResult]:
    ...
```

The method never mutates input tensors, labels, or diagnostic mappings. It returns
the same number and resolution of results. The disabled implementation returns the
original result list without copying it or adding diagnostics.

### 6.4 Construction and shared call site

`PipelineRunner` builds the refiner from `SegmentationConfig` through an injectable
dependency and stores it in `ReconstructionContext`.

`reconstruction/shared.py` adds one `segment_and_refine_window()` wrapper. It
performs the existing NumPy strategy call exactly once and invokes the refiner only
when enabled. The three modes replace their direct strategy calls with this wrapper
but retain their current registration, graph, anchor, state, and aggregation order.

The mode constructors accept the shared wrapper as an injected callable, matching
their existing testable injection style for registration and graph construction.

## 7. Algorithm

### 7.1 Input validation and preparation

Mismatched frame counts, ranks, or spatial shapes are programming errors and raise
`ValueError`. Runtime geometric unreliability uses fallback instead of raising.

For enabled execution:

1. Detach the necessary window tensors and copy them to CPU.
2. Use float64 for pose inversion and relative transforms; use float32 or the input
   floating precision for point/confidence arithmetic; use int64/`np.intp` indices.
3. Require finite K, positive `fx/fy`, and a last row close to `[0,0,1]`.
4. Define a valid point as finite XYZ with `Z > 1e-6`.
5. Require every used pose to be finite and invertible.
6. On the centered sparse grid, project local XYZ through K back to its source
   pixel. Require at most 0.5 pixel error. This compatibility guard detects a
   future point-map/intrinsic contract change. Insufficient or incompatible data
   triggers fallback rather than intrinsic re-estimation.

### 7.2 Confidence and frame quality

For each frame, reuse the current quantile rule to retain valid high-confidence
pixels. Equal values at the threshold remain selected.

Convert a logit to a weight with:

\[
p(c)=\sigma(\operatorname{clip}(c,-20,20)).
\]

Frame quality is:

\[
Q_i=\frac{|V_i|}{HW}\operatorname{mean}_{p\in V_i}p(c_{ip}),
\]

where `V_i` is the valid high-confidence set. A frame with no valid selected point
has quality zero and cannot become a reference.

### 7.3 Pairwise sparse reprojection

Use a deterministic centered grid with spacing `sampling_stride`. For source
reference `k` and target `t`, transform a source local point with:

\[
X_t=(T_t^{c2w})^{-1}T_k^{c2w}[X_k,1]^T.
\]

Project positive-depth target-camera points with:

\[
u=f_xX_t/Z_t+c_x,\qquad v=f_yY_t/Z_t+c_y.
\]

Round with `floor(value + 0.5)` and reject out-of-bounds coordinates. Build a
deterministic z-buffer over `(target_pixel_id, projected_z, source_flat_index)`:
the smallest positive depth wins and a depth tie selects the smaller source index.

The target point and target confidence must also be valid. A winner is geometrically
consistent only when:

\[
e_z=\frac{|Z_{projected}-Z_{target}|}{\max(|Z_{target}|,10^{-6})}
    < \texttt{relative_depth_tolerance}.
\]

The correspondence weight is:

\[
w=\sqrt{p(c_{source})p(c_{target})}
  \max\left(0,1-\frac{e_z}{\texttt{relative_depth_tolerance}}\right).
\]

For coverage, a coarse cell is `(u // stride, v // stride)`. Let `C` be the number
of coarse target cells receiving at least one consistent correspondence divided by
the number of coarse cells containing at least one valid high-confidence target
pixel. Let `G` be the consistent-winner count divided by the z-buffer winner count
that also has valid high-confidence target geometry. Let `F` be the mean confidence
weight before the depth residual factor over consistent correspondences.

The directed pair score is:

\[
S_{k\rightarrow t}=C\sqrt{GF}.
\]

Empty denominators produce zero. Clamp reported components and scores to `[0,1]`.
A reference is reliable for a target when
`S >= min_reference_score`.

### 7.4 Adaptive reference selection

Select the first reference by descending frame quality. A quality tie chooses the
frame closest to the temporal center and then the lower frame index.

After evaluating each selected reference against every other frame, maintain:

\[
B_t=\max_{k\in K}S_{k\rightarrow t}.
\]

Selected reference frames are considered covered for the window-level stop ratio,
but self-projections are never used as merge evidence.

For each unselected and non-rejected frame, candidate priority is:

\[
P_i=Q_i(1-B_i).
\]

Choose the highest priority, breaking ties by lower frame index, and evaluate that
candidate against all targets. Its marginal gain is the mean
`max(B_t, S_i_to_t) - B_t` over frames that are neither already selected nor the
candidate itself.

- If gain is at least `min_coverage_gain`, select the candidate and update `B`.
- Otherwise mark only that candidate rejected and continue with the next candidate.

Stop when any of the following holds:

- the fraction of frames with `B >= min_reference_score`, including selected
  references, reaches `stop_coverage_ratio`;
- the safety ceiling `max_keyframes` is reached;
- every remaining candidate has zero quality or has been rejected.

This performs selected/rejected-candidate-to-window projection lazily. Typical work
is `K*N` sparse projections; the explicitly bounded worst case includes rejected
candidates and may approach `N*N`, without allocating an all-pairs tensor.

### 7.5 Region correspondence

All selected reference and target labels used here are the original initial
segmentation labels. Refinement is not iterative, so frame or edge processing order
cannot change the evidence.

For each reliable non-self pair, aggregate valid correspondences by
`(target_region, source_region)` using sparse encoded pairs. For each target region,
the source region with the largest total weight is its dominant mapping. A weight
tie chooses the smaller source label.

The mapping is reliable only when all conditions hold:

- unique target hits are at least `min_region_correspondences`;
- approximate coverage
  `min(1, unique_hits * stride**2 / target_region_area)` is at least
  `min_region_coverage`;
- dominant weight divided by total weight is at least `min_region_purity`.

Define region support as `min(coverage, purity)`.

### 7.6 Merge voting and deterministic union

Build target adjacency only from different labels touching horizontally or
vertically. Store each undirected edge as sorted `(lower_label, higher_label)` and
deduplicate it. Non-adjacent regions are never direct merge candidates.

For an adjacent target edge `(a,b)` and each reliable non-self reference:

- if both mappings are reliable and have the same dominant source label, add
  `pair_score * min(support_a, support_b)` to merge evidence;
- if both are reliable and have different dominant labels, add the same form of
  weight to separate evidence;
- otherwise abstain.

With positive merge evidence, define:

\[
r=\frac{E_{merge}}{E_{merge}+E_{separate}}.
\]

The edge is eligible only when `r >= merge_vote_threshold`. Sort eligible edges by
descending ratio, descending merge evidence, and ascending label pair.

The union-find component stores at most one dominant source label per reference.
Before joining two components, reject the union if any reference has known but
different labels on the two sides. Otherwise merge the larger root into the smaller
root and combine compatible signatures. This prevents an accepted chain from
transitively joining regions that a reference explicitly separates.

Reference-frame labels are not refined. For every non-reference target, map every
pixel only through its original region's final union-find root, then call
`compact_labels()`. The result dtype is `np.intp` and the labels remain exactly
`0..R_new-1`.

This construction establishes both invariants:

\[
R_{new}\le R_{initial}
\]

and every initial region maps to exactly one refined region.

## 8. Fallback and error behavior

The enabled refiner keeps labels unchanged for:

- a single-frame window;
- missing or invalid intrinsic;
- intrinsic/point-map incompatibility;
- invalid or non-invertible geometry;
- no usable reference frame;
- no reliable target support;
- merge candidates rejected by conflicting evidence.

Individual out-of-bounds, occluded, low-confidence, or depth-inconsistent samples
abstain locally. They do not fail the whole window.

Disabled execution adds no diagnostics. Enabled execution records why unchanged
labels were retained.

## 9. Diagnostics

Enabled execution adds only finite scalar values to copied per-frame diagnostics:

```text
window_reference_applied: bool
window_reference_keyframes: str
window_reference_keyframe_count: int
window_reference_is_keyframe: bool
window_reference_coverage_ratio: float
window_reference_regions_before: int
window_reference_regions_after: int
window_reference_candidate_edges: int
window_reference_accepted_edges: int
window_reference_conflict_edges: int
window_reference_projected_samples: int
window_reference_occluded_samples: int
window_reference_depth_rejected_samples: int
window_reference_fallback: str
```

`window_reference_keyframes` is the selected ascending indices joined by commas.
Fallback values are one of:

```text
none
disabled
single_frame
missing_intrinsic
invalid_intrinsic
intrinsic_incompatible
invalid_geometry
no_reference
insufficient_support
conflict_only
```

The disabled path does not emit the `disabled` value because it emits no new keys;
the value remains part of the closed diagnostic vocabulary for explicit internal
outcomes and tests.

When enabled, `region_count` is replaced with the final compact region count while
all existing strategy-specific diagnostics are preserved.

`pipeline/diagnostics.py` may summarize applied frames, accepted/conflict edge
totals, and mean window coverage. It does not place strings, booleans, or keyframe
lists in `mode_scalars` and does not change artifact schema version 1.

## 10. Determinism, performance, and dependencies

The V1 implementation runs on CPU even when the reconstruction process device is
CUDA. It copies only the current window data after initial segmentation and only
when enabled.

Determinism is defined by:

- a fixed centered sampling grid;
- stable quantile and tie rules;
- lower-index frame, pixel, label, and union roots for all ties;
- deterministic z-buffer ordering;
- sorted adjacency and merge edges;
- no GPU scatter or atomic operations.

For `P=H*W`, sparse samples `Q approximately P/stride**2`, selected references `K`,
and window frames `N`, projection work is approximately
`O(K*N*Q*log(Q))`, plus rejected candidates. Each reference-target pair is
processed and released independently. The implementation must not allocate a
`[K,N,H,W,...]` tensor or dense source-region-by-target-region table.

No PyTorch3D, networkx, learned model, native extension, or additional package is
introduced.

## 11. File-level change map

Create:

- `inference_engine/segmentation/window_reference.py`
- `tests/test_window_reference_refinement.py`

Modify:

- `pipeline/config.py`
- `configs/pipeline/default.yaml`
- `configs/pipeline/test.yaml`
- `configs/reconstruction/pi3_laser.yaml`
- `configs/reconstruction/pi3_laser_no_loop.yaml`
- `inference_engine/prediction_cache/provider.py`
- `reconstruction/prediction_stream.py`
- `inference_engine/segmentation/__init__.py`
- `pipeline/runner.py`
- `reconstruction/modes/base.py`
- `reconstruction/shared.py`
- `reconstruction/modes/no_loop.py`
- `reconstruction/modes/traditional.py`
- `reconstruction/modes/corrected.py`
- focused configuration, provider, prediction-stream, mode, pipeline, diagnostic,
  and artifact tests
- `docs/pipeline-configuration.md`
- `README.md` only if its architecture or command examples otherwise become false

Do not modify initial strategy implementations, PI3 model code, prediction cache
artifacts/fingerprint, temporal matching, anchor propagation, loop mathematics,
artifact writer/schema, or evaluation modules.

## 12. Verification requirements

Implementation follows strict RED-GREEN-REFACTOR cycles. Required behavior tests
cover:

1. Configuration defaults, typed overrides, and every validation boundary.
2. Provider intrinsic cloning across cache miss, hit, and off modes.
3. `WindowPrediction` optional intrinsic and invalid intrinsic rejection.
4. Identity reprojection and analytically known c2w translation direction.
5. Positive-depth, image-bounds, deterministic z-buffer, and target-depth checks.
6. No mutation of point, pose, confidence, labels, or diagnostics inputs.
7. Adaptive one-reference stop, additional-reference selection, rejected-candidate
   continuation, coverage stop, safety ceiling, and deterministic ties.
8. Reliable region mapping boundaries for hit count, coverage, and purity.
9. Adjacent merge, non-adjacent preservation, low-support abstention, weighted
   multi-reference conflict, and transitive component conflict.
10. Compact `np.intp` output, region-count monotonicity, and no-split mapping.
11. Disabled result-list identity, diagnostics identity, and reconstruction tensor
    equality.
12. Exact `segment < refine < graph < anchor` order in no-loop, traditional, and
    corrected modes, with each mode's current tensor state and intrinsic.
13. Scalar finite diagnostics, canonical JSON round-trip, unchanged artifact schema
    and file set, and no new forbidden import edge.
14. Repeated CPU execution produces identical labels, selected references, and
    diagnostics.

The final local verification commands are:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_window_reference_refinement.py

PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_pipeline_config.py \
  tests/test_prediction_provider.py \
  tests/reconstruction/test_no_loop_mode.py \
  tests/reconstruction/test_traditional_mode.py \
  tests/reconstruction/test_corrected_mode.py \
  tests/test_reconstruction_artifacts.py

PYTHONDONTWRITEBYTECODE=1 python scripts/verify_segmentation_modes.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
git status --short
```

The existing segmentation smoke currently passes while emitting numerical warnings
from `inference_engine/utils/depth.py:171`. The implementation must not conceal or
attribute those pre-existing warnings to this feature.

## 13. Real-cache experiment gate

The current local environment has no model weights, dataset, or prediction cache,
so implementation correctness is separated from empirical threshold promotion.

When assets are available, first evaluate one cached 20-frame window for each
initial strategy. Record selected-reference count, pair score components,
projected/occluded/depth-rejected samples, accepted/conflict edges, before/after
region count, wall time, and peak memory. Visually audit the highest-evidence merge
edges before running full sequences.

Required ablations are:

- disabled, one reference, adaptive references, and all-frame offline upper bound;
- stride 2, 4, and 8;
- depth tolerance 0.02, 0.05, and 0.10;
- region purity 0.70, 0.80, and 0.90;
- region coverage 0.05, 0.10, and 0.20;
- correspondence minimum 4, 8, and 16;
- weighted conflict voting against an any-separate-veto ablation;
- z-buffer and target-depth checks enabled versus disabled negative controls.

Promotion beyond default-disabled requires all three strategies and all three modes
to remain finite and crash-free, disabled outputs to remain identical, visual merge
audits to meet the agreed false-merge limit, at least one reconstruction metric to
improve without a worst-sequence regression, and CPU overhead to remain within the
experiment budget.

## 14. Research references

The design was checked against these primary or upstream references:

- [PI3 README](https://github.com/yyfz/Pi3/blob/main/README.md) for the model's
  point-map and camera prediction contract.
- [PI3 model implementation](https://github.com/yyfz/Pi3/blob/main/pi3/models/pi3.py)
  for the construction of global points by applying predicted camera poses to
  homogenized local points.
- [OpenCV camera calibration documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
  for the pinhole projection convention used by the sparse reprojector.
- [scikit-image RAG documentation](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_merge.html)
  as background for region-adjacency merging; V1 implements only the smaller
  deterministic four-connected graph specified above and does not depend on its
  merge implementation.

The branch-local source remains authoritative for integration boundaries and data
states. In particular, `Pi3/models/pi3.py`,
`inference_engine/prediction_cache/provider.py`, and the three reconstruction mode
implementations were inspected directly.

## 15. Known independent risks

- Non-depth thresholds are empirical seeds until real-cache evaluation is run.
- Existing no-loop code repeats intrinsic estimation/unprojection despite the
  canonical provider path. This design passes the intrinsic actually used by the
  mode and does not repair the existing path.
- An existing geometry helper uses a `-Z` viewing direction while PI3 publishes
  OpenCV `+Z` camera coordinates. This refiner does not reuse that convention.
- The current segmentation smoke emits divide-by-zero, overflow, and invalid-value
  warnings in temporal IoU matrix multiplication while still passing. That issue is
  tracked as existing behavior and is not bundled into this implementation.
