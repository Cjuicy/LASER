# LASER Loop-Closure Corrected Pipeline Design

## 1. Purpose

This change makes loop-closure reconstruction build on the same forward
correction pipeline as non-loop reconstruction:

1. process every window sequentially;
2. immediately apply whole-window Sim(3) alignment;
3. immediately apply the existing segmentation scale mask;
4. use the corrected window as the next registration reference;
5. detect loops only after the initial trajectory is complete;
6. optimize sequential and loop Sim(3) constraints together; and
7. apply only the optimized correction delta during final aggregation.

The loop-closure path will also use one canonical image manifest and one
unambiguous registration-confidence ratio. The approved default is `0.3`,
meaning that each registration input keeps its highest-confidence
approximately 30 percent of finite pixels before the two masks are
intersected.

## 2. Baseline and Scope

The implementation branch is `codex/loop-closure-corrected-pipeline`, based
on commit `98cce5f9f470599aca0cf5a6614f39409d929d58` from
`codex/unified-segmentation-methods`.

### In scope

- canonical natural image ordering across streaming inference, SALAD, and
  joint Pi3 loop prediction;
- a shared `0.3` registration-confidence ratio for adjacent-window and
  loop-side registration;
- a backward-compatible registration-confidence hook in the base streaming
  engine so non-loop and loop first stages can be compared with identical
  effective settings;
- immediate forward correction in `StreamingWindowEngineLC`;
- explicit absolute and sequential Sim(3) cache fields;
- loop constraints built from already corrected window caches;
- optimized transform accumulation and one-time final correction;
- no-loop equivalence; and
- focused CPU unit tests plus a documented real-sequence validation command.

### Out of scope

- changing the three segmentation methods;
- changing segmentation parameters or confidence selection;
- changing segment graph construction;
- changing region matching;
- changing scale-anchor estimation or propagation;
- changing `refine_depth_segments()`;
- changing SALAD descriptor extraction;
- changing the mathematical form of `Sim3LoopOptimizer`; and
- changing the default behavior of the non-loop entry point; and
- adding a new loop-candidate ranking or anchor-propagation algorithm.

## 3. Findings in the Current Code

The current implementation has four independent problems.

### 3.1 Image indices are not stable across the pipeline

`demo_lc.py` discovers a naturally sorted, sampled sequence, but
`LoopClosureEngine` and `LoopDetector` scan the directory again and use
lexicographic sorting. Numbered names can therefore become
`frame1, frame10, frame2`, and the three scanners do not accept exactly the
same extension set. SALAD indices can consequently refer to different frames
than the window and joint-Pi3 paths.

### 3.2 Forward correction is deferred in loop mode

`StreamingWindowEngineLC` saves the estimated Sim(3) and scale mask but does
not immediately apply them. The following window registers against an
uncorrected cache, so loop mode does not reproduce the non-loop forward
pipeline.

### 3.3 Final aggregation applies ambiguous transforms

The existing loop aggregation accumulates cached transforms and applies
segmentation masks late. Its mask branch uses the previous cumulative scale
instead of the current cumulative scale. More importantly, late application
cannot make the earlier corrected geometry participate in later registration.

### 3.4 The `top_conf_percentile` name has opposite meanings

The streaming engine converts a keep ratio to a quantile with
`1 - top_conf_percentile`. `LoopClosureEngine` uses the supplied value
directly as a quantile. Setting both call sites to `0.3` would therefore keep
about 30 percent in one path and about 70 percent in the other. The streaming
value is also forwarded to segmentation, so changing that existing field
would unintentionally change segmentation behavior.

## 4. Selected Approach

The selected approach separates registration confidence from segmentation
confidence, makes the canonical image manifest an explicit dependency, and
stores both absolute and relative transforms.

A minimal literal replacement of `0.5` with `0.3` is rejected because it
preserves the opposite quantile interpretations and changes segmentation
inputs. An exact combined-confidence top-k selection is also rejected for this
iteration because it would replace the current per-side thresholding and
intersection algorithm.

## 5. Canonical Image Manifest

Add `utils/image_paths.py` with:

```python
natural_sort_key(path) -> list[tuple[int, str | int]]
discover_images(data_path, sample_interval=1) -> list[str]
```

`discover_images()` will:

- require `sample_interval >= 1`;
- require an existing directory;
- accept case-insensitive `.png`, `.jpg`, and `.jpeg` files;
- sort by the numeric portions of the basename; and
- apply `sample_interval` only after sorting.

`demo_lc.py` creates this manifest once. The exact list is then passed to:

- `StreamingWindowEngineLC` window inference;
- `LoopClosureEngine`; and
- `LoopDetector`.

An explicitly supplied manifest is already sorted and sampled. Downstream
components must not rescan, resort, or resample it. Standalone loop APIs may
use `discover_images()` only when no manifest was supplied.

Before loop detection, the engine verifies that the canonical manifest
produces the same number of windows as the cache list.

## 6. Registration Confidence

Add a registration-only helper in
`inference_engine/utils/registration_confidence.py`:

```python
select_top_confidence_mask(
    confidence: torch.Tensor,
    keep_ratio: float = 0.3,
) -> torch.Tensor
```

The contract is:

- `keep_ratio` must satisfy `0 < keep_ratio <= 1`;
- non-finite confidence values are always excluded;
- the quantile is `1 - keep_ratio`;
- quantile interpolation remains `nearest`;
- the mask uses `confidence >= threshold`; and
- equal values at the threshold may retain slightly more than the requested
  ratio.

Registration keeps the existing mutual-mask rule:

```python
mask = (
    select_top_confidence_mask(source_conf, 0.3)
    & select_top_confidence_mask(target_conf, 0.3)
)
```

This rule is used for:

- the previous and current overlap during forward registration;
- the joint-Pi3 A slice and corrected A cache slice; and
- the joint-Pi3 B slice and corrected B cache slice.

`StreamingWindowEngine` gains an optional
`registration_top_confidence_ratio=None` compatibility hook. When omitted,
it derives the existing registration behavior from the legacy
`top_conf_percentile` argument, so the non-loop default remains unchanged.
The new field controls registration masks only; the existing
`top_conf_percentile` field continues to control segmentation exactly as it
does at the baseline.

`StreamingWindowEngineLC` sets
`registration_top_confidence_ratio=0.3` by default and forwards it through
the base hook. `LoopClosureEngine` uses the same registration argument.
`demo_lc.py` exposes one argument with a default of `0.3` and forwards the
same value to both engines. A parity test can instantiate both streaming
engines with an explicit registration ratio of `0.3` while leaving their
segmentation setting identical.

No segmentation function receives the new registration ratio.

## 7. Sim(3) Convention

For a Sim(3) `S = (s, R, t)`, application is:

```text
S(x) = s R x + t
```

Composition follows the existing `accumulate_sim3(S1, S2)` convention:

```text
accumulate_sim3(S1, S2) = S1 compose S2
```

### 7.1 Absolute transform

Because adjacent registration uses the previous corrected point cloud and
propagated camera poses as its target, the returned transform maps the current
raw window into the first-stage global coordinate system. It is stored as:

```text
G_i = sim3_abs[i]
```

The first window uses `G_0 = identity`.

### 7.2 Sequential edge

The optimizer requires a relative edge, not the absolute transform:

```text
E_(i-1,i) = inverse(G_(i-1)) compose G_i
```

The implementation must preserve the invariant:

```text
G_(i-1) compose E_(i-1,i) == G_i
```

The first cache has no sequential edge. For `N` windows, the optimizer
receives exactly `N - 1` sequential edges.

## 8. First-Stage Forward Pipeline

`StreamingWindowEngineLC` retains its loop-specific cache metadata but changes
its per-window processing order to match `StreamingWindowEngine`.

### 8.1 First window

1. run Pi3 inference;
2. estimate and fix the reference intrinsic;
3. unproject the depth with that intrinsic;
4. build the existing segment graph when depth refinement is enabled;
5. set `sim3_abs` to identity;
6. omit `sim3_edge`; and
7. cache the window as the next registration reference.

### 8.2 Later windows

1. unproject the raw Pi3 depth with the fixed intrinsic;
2. build the mutual registration mask with the shared `0.3` keep ratio;
3. register the current raw overlap against the previous corrected overlap;
4. treat the returned `(s, R, t)` as the current `sim3_abs`;
5. derive `sim3_edge` from the previous and current absolute transforms;
6. immediately multiply the current local points by the absolute scale;
7. immediately apply the absolute Sim(3) to current camera poses;
8. build the existing segment graph on the globally scaled current points;
9. call the unchanged `refine_depth_segments()` implementation;
10. immediately multiply local points by the returned scale mask;
11. retain the mask only as diagnostic metadata; and
12. cache the corrected points, propagated poses, and current graph as the
    next registration reference.

This ordering guarantees that correction from `W_i` participates in
registration of `W_(i+1)`.

## 9. Window Cache Contract

Each cache contains:

| Field | Contract |
|---|---|
| `local_points` | Whole-window scale and segmentation mask already applied |
| `camera_poses` | First-stage propagated camera poses |
| `sim3_abs` | First-stage absolute window transform `G_i` |
| `sim3_edge` | Relative sequential edge for non-first windows |
| `scale_mask` | Already-applied diagnostic mask; never applied again |
| `conf` | Original Pi3 confidence |

Existing image and model-output fields remain unchanged. The ambiguous
loop-only `sim3` field is replaced by the explicit absolute and edge fields,
and every consumer is updated in the same change.

## 10. Loop Detection and Constraint Construction

SALAD runs only after all first-stage caches are complete.

For each accepted loop pair:

1. map the canonical frame indices to the two window indices and bounded
   frame ranges;
2. jointly run Pi3 on the A and B ranges;
3. select the corresponding slices from corrected cache A and cache B;
4. build A-side and B-side mutual masks with the shared `0.3` ratio;
5. register the joint A prediction to corrected cache A, producing `L_A`;
6. register the joint B prediction to corrected cache B, producing `L_B`;
   and
7. construct the existing-direction loop measurement:

```text
C_AB = L_B compose inverse(L_A)
```

The loop constraint is stored as:

```python
(window_index_a, window_index_b, C_AB)
```

The existing `Sim3LoopOptimizer` receives all `sim3_edge` values and all
valid loop constraints. Its residuals, solver, damping, and convergence logic
are unchanged.

## 11. Optimization Output and Final Aggregation

The optimizer returns `N - 1` optimized sequential edges:

```text
E_hat_(0,1), ..., E_hat_(N-2,N-1)
```

They are accumulated from identity to obtain one optimized absolute transform
per window:

```text
G_hat_0 = identity
G_hat_i = G_hat_(i-1) compose E_hat_(i-1,i)
```

For each already corrected cache, compute only the optimization delta:

```text
D_i = G_hat_i compose inverse(G_i)
```

Apply `D_i` exactly once:

- multiply `local_points` by the uniform scale of `D_i`;
- apply the complete `D_i` to `camera_poses`;
- do not apply `scale_mask`;
- do not apply `G_i` again; and
- do not rerun segmentation or anchor propagation.

After delta application, final aggregation only:

- removes duplicate overlap frames;
- concatenates local points and camera poses;
- computes world points from the corrected poses and local points; and
- saves the existing output format.

The aggregation helper must not mutate the input cache dictionaries in place.

## 12. No-Loop Behavior

No loop is a valid result, not an error.

If SALAD produces no valid loop constraints:

- do not run joint Pi3;
- do not invoke the optimizer;
- use the original `sim3_abs` list as the optimized absolute list; and
- apply identity deltas during aggregation.

The output is therefore the first-stage corrected trajectory. Under the same
window, overlap, confidence, and segmentation configuration, it must match the
non-loop forward result within floating-point tolerance.

## 13. Validation and Error Handling

The implementation fails early for:

- `registration_top_confidence_ratio` outside `(0, 1]`;
- a missing or empty image directory when fallback discovery is used;
- a non-positive sample interval;
- a canonical manifest/cache window-count mismatch;
- an empty finite mutual registration mask;
- a malformed cached transform; and
- an optimizer result whose edge count is not `N - 1`.

An invalid loop candidate whose frame indices cannot map to valid cache ranges
is skipped with the pair and reason recorded. If all candidates are skipped,
the pipeline follows the valid no-loop path.

## 14. File Boundaries

### Create

- `utils/image_paths.py`: canonical discovery and natural sorting only.
- `inference_engine/utils/registration_confidence.py`: registration-only
  ratio validation and mask selection.
- `tests/test_loop_image_manifest.py`: manifest ownership and index stability.
- `tests/test_registration_confidence.py`: `0.3` semantics and validation.
- `tests/test_streaming_window_engine_lc_pipeline.py`: first-stage propagation,
  cache invariants, and one-time mask application.
- `tests/test_loop_closure_pipeline.py`: corrected-cache loop constraints,
  no-loop behavior, optimizer edge flow, and final deltas.

### Modify

- `demo_lc.py`: create and forward the canonical manifest and shared
  registration ratio; remove redundant loop-cache rewriting.
- `loop_closure/loop_model.py`: consume an explicit manifest and avoid
  rescanning.
- `loop_closure/loop_closure.py`: consume corrected caches and explicit edges,
  build loop constraints, and return optimized absolute transforms.
- `inference_engine/streaming_window_engine.py`: add the
  backward-compatible registration-only confidence hook and shared mask
  call; preserve its default effective ratio and all non-loop processing.
- `inference_engine/streaming_window_engine_lc.py`: immediate first-stage
  correction, explicit transform cache fields, and delta-only aggregation.
- existing focused tests where public constructor forwarding must be asserted.

### Deliberately unchanged

- segmentation, matching, and propagation modules;
- `refine_depth_segments()`;
- SALAD model and descriptor extraction internals; and
- `loop_closure/utils/sim3loop.py` optimizer mathematics.

## 15. Test Matrix

### 15.1 Image ordering

- natural ordering of `frame1`, `frame2`, and `frame10`;
- case-insensitive `.jpg`, `.jpeg`, and `.png`;
- filtering before sampling;
- sampling exactly once;
- explicit manifest identity preserved by `LoopClosureEngine` and
  `LoopDetector`; and
- window, SALAD, and joint-Pi3 indices refer to the same paths.

### 15.2 Registration confidence

- `keep_ratio=0.3` uses the `0.7` quantile;
- non-finite values are excluded;
- source and target masks are separately selected and intersected;
- adjacent, loop-A, and loop-B paths call the same helper;
- the non-loop default keeps its baseline effective ratio when the new
  argument is omitted;
- invalid ratios fail early; and
- segmentation receives its original confidence parameter.

### 15.3 First-stage propagation

- the whole-window scale is applied before refinement;
- camera poses are propagated immediately;
- the segmentation mask is applied immediately and once;
- corrected `W_1` points are the registration source for `W_2`;
- `sim3_abs` and `sim3_edge` satisfy their composition invariant; and
- first-stage LC output matches the non-loop engine with the same effective
  configuration.

### 15.4 Loop constraints

- A and B registrations receive corrected cache point maps;
- `C_AB` has the approved composition direction;
- exactly `N - 1` sequential edges reach the optimizer;
- all valid loop constraints reach the same optimizer call; and
- invalid loop ranges degrade to the no-loop path when none remain.

### 15.5 Final aggregation

- optimized edges are accumulated from identity;
- delta scale and pose are each applied once;
- `scale_mask` is not reapplied;
- identity deltas preserve cache values;
- aggregation does not mutate its inputs; and
- overlap removal and output tensor shapes remain unchanged.

### 15.6 Regression and real-sequence validation

- build the two repository Cython extensions with
  `python setup.py build_ext --inplace`;
- run the complete `pytest -q` suite;
- run one identical real sequence with loop detection disabled or with no
  accepted loop and compare against the non-loop result;
- run one real sequence with an accepted loop and verify that the optimized
  trajectory uses both sequential and loop constraints; and
- keep model weights, images, sampling, window size, overlap, segmentation
  mode, and random environment identical across comparisons.

## 16. Acceptance Criteria

The change is accepted when:

1. one naturally sorted and sampled image manifest drives the complete loop
   pipeline;
2. adjacent and loop-side registration use one `0.3` keep-ratio contract;
3. segmentation methods and parameters are unchanged;
4. loop-mode first-stage processing follows the non-loop correction order;
5. each corrected window participates in registration of the next window;
6. every non-first window caches an explicit relative Sim(3) edge;
7. every window caches an explicit first-stage absolute Sim(3);
8. SALAD runs only after the initial trajectory is complete;
9. joint Pi3 is aligned to already corrected A and B caches;
10. sequential and loop constraints enter the existing optimizer together;
11. optimized edges are re-accumulated into absolute transforms;
12. final aggregation applies only the optimized delta;
13. the segmentation mask is never applied twice;
14. the absence of any valid loop produces the unchanged first-stage result;
15. all new CPU unit tests and the existing test suite pass; and
16. the three segmentation methods, anchor propagation, SALAD extraction, and
    optimizer mathematics remain untouched.
