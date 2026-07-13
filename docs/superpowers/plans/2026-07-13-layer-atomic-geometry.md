# Layer-Atomic Geometry Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LASER's final coarse-depth segmentation with a full-image, atom-level 3D continuity merge that prevents turn/far-field fragmentation while preserving baseline runtime characteristics and every non-segmentation setting.

**Architecture:** Keep LASER's existing depth Felzenszwalb output as immutable atoms and its existing mean-depth merge as a weak coarse-layer prior. Build a region adjacency graph only between those atoms, measure each atom's normal 3D sampling scale and each shared boundary's 3D gap, then run one DSU merge. Geometry may separate atoms inside one coarse layer or reconnect atoms across different coarse layers; it never splits an initial atom and never masks low-confidence pixels.

**Tech Stack:** Python 3, NumPy, scikit-image, existing Cython `merge_regions`, pytest.

## Global Constraints

- Start from `origin/main@3a67cdcf52c55e543b5f6ae014748754ea095ad8` on `feature/layer-atomic-geometry` in `/tmp/LASER-LayerAtomic`.
- Do not copy or patch code from `feature/dphgs-segmentation`; implement against clean `origin/main` interfaces.
- Change segmentation and the minimal point-map input wiring only; do not change scale propagation, registration, model inference, windowing, or confidence selection.
- Keep source defaults unchanged: `seg_scale=300`, `seg_sigma=1.1`, `seg_min_size=500`, `depth_merge_thresh=0.1`, `corr_iou_thresh=0.3`, and the existing `top_conf_percentile` supplied by the engine.
- Add no CLI option, configuration field, or new tunable threshold.
- Confidence may determine the existing baseline depth-range merge threshold, but it must never remove pixels from the segmentation graph.
- Output must cover the complete image, contain compact non-negative integer labels, be deterministic, and consist only of unions of initial Felzenszwalb atoms.
- The geometry rule is `G_AB = d_AB / sqrt(s_A * s_B)`. Merge when `G_AB <= 1 + depth_merge_thresh` for atoms in the same coarse layer and when `G_AB <= 1` for atoms in different coarse layers.
- Use horizontal and vertical pixel edges once each. Compute atom internal scales and atom-pair boundary gaps with vectorized reductions; do not compute normals, planes, per-pixel region growing, or sorted floating edge queues.

---

### Task 1: Expose the unchanged LASER segmentation stages

**Files:**
- Modify: `inference_engine/utils/depth.py`
- Create: `tests/test_depth_segmentation_stages.py`

**Interfaces:**
- Produces: `segment_depth_felzenszwalb_rag_stages(...) -> tuple[np.ndarray, np.ndarray, float]` returning `(initial_labels, coarse_labels, merge_threshold)`.
- Preserves: `segment_depth_felzenszwalb_rag(...) -> np.ndarray` with byte-for-byte-equivalent label values for identical inputs.

- [ ] **Step 1: Write a failing regression test for the stages API and legacy output**

```python
import numpy as np

from inference_engine.utils.depth import (
    segment_depth_felzenszwalb_rag,
    segment_depth_felzenszwalb_rag_stages,
)


def test_stages_preserve_legacy_coarse_labels_and_threshold():
    yy, xx = np.mgrid[:40, :60]
    depth = (1.0 + 0.002 * xx + 0.004 * yy).astype(np.float64)
    depth[:, 30:] += 0.8

    initial, coarse, threshold = segment_depth_felzenszwalb_rag_stages(
        depth,
        depth_merge_thresh=0.1,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=20,
    )
    legacy = segment_depth_felzenszwalb_rag(
        depth,
        depth_merge_thresh=0.1,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=20,
    )

    np.testing.assert_array_equal(coarse, legacy)
    assert initial.shape == depth.shape
    assert threshold == np.float64(0.1 * (depth.max() - depth.min()))
```

The test must also build an independent oracle by calling the existing
`felzenszwalb(...)` and Cython `merge_regions(...)` directly, then compare
`initial`, `coarse`, `merge_threshold`, and the legacy wrapper with that
oracle. Parameterize it for both the no-confidence path and the
`conf_map + batch_idx` path; comparing two wrappers that share the same new
implementation is not sufficient regression coverage.

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_depth_segmentation_stages.py`

Expected: collection fails because `segment_depth_felzenszwalb_rag_stages` does not exist.

- [ ] **Step 3: Extract the existing stages without changing their operations**

Implement `segment_depth_felzenszwalb_rag_stages` by moving the current `felzenszwalb`, confident-depth range calculation, and `merge_regions` calls into the new function. Keep argument order and defaults identical. Make `segment_depth_felzenszwalb_rag` call it and return only `coarse_labels`.

```python
def segment_depth_felzenszwalb_rag_stages(
        depth_map,
        depth_merge_thresh,
        conf_map=None,
        top_conf_percentile=None,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=500,
        batch_idx=None
):
    initial_labels = felzenszwalb(
        depth_map,
        scale=seg_scale,
        sigma=seg_sigma,
        min_size=seg_min_size,
    )
    if conf_map is not None and top_conf_percentile is not None:
        frame_conf = conf_map[batch_idx]
        conf_thresh = np.quantile(
            frame_conf.reshape(-1),
            top_conf_percentile,
            method='nearest',
        )
        conf_depth = depth_map[frame_conf >= conf_thresh]
    else:
        conf_depth = depth_map
    merge_threshold = depth_merge_thresh * (
        np.max(conf_depth) - np.min(conf_depth)
    )
    coarse_labels = merge_regions(initial_labels, depth_map, merge_threshold)
    return initial_labels, coarse_labels, merge_threshold
```

- [ ] **Step 4: Run the focused test and full baseline suite**

Run: `pytest -q tests/test_depth_segmentation_stages.py && pytest -q`

Expected: the focused test and all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add inference_engine/utils/depth.py tests/test_depth_segmentation_stages.py
git commit -m "refactor: expose LASER segmentation stages"
```

---

### Task 2: Implement atom-level 3D continuity merging

**Files:**
- Create: `inference_engine/utils/layer_atomic_geometry.py`
- Create: `tests/test_layer_atomic_geometry.py`

**Interfaces:**
- Consumes: `segment_depth_felzenszwalb_rag_stages` from Task 1.
- Produces: `merge_layer_atoms(point_map, initial_labels, coarse_labels, depth_merge_thresh) -> np.ndarray`.
- Produces: `segment_point_map_layer_atomic(point_map, depth_merge_thresh, conf_map=None, top_conf_percentile=None, seg_scale=300, seg_sigma=1.1, seg_min_size=500, batch_idx=None) -> np.ndarray`.

- [ ] **Step 1: Write failing behavior tests for the core geometry rule**

Use synthetic point maps with two immutable atoms. Tests must assert these independent behaviors:

```python
def test_continuous_turn_merges_atoms_despite_surface_direction_change(): ...
def test_real_3d_gap_splits_atoms_inside_the_same_coarse_layer(): ...
def test_continuous_atoms_can_remerge_across_coarse_layers(): ...
def test_weak_layer_prior_accepts_only_same_layer_small_excess_gap(): ...
def test_result_is_invariant_to_global_point_scale(): ...
def test_degenerate_atoms_do_not_merge_or_break_scale_invariance(): ...
def test_result_is_compact_full_coverage_and_only_unions_atoms(): ...
def test_result_is_deterministic(): ...
```

For all fixtures, construct each atom with multiple valid horizontal and vertical internal edges. Set normal internal spacing to `1.0`; use boundary gap `1.0` for continuous contact, `1.05` for the weak-prior test, and `3.0` for a real separation. Use `depth_merge_thresh=0.1`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_layer_atomic_geometry.py`

Expected: collection fails because `inference_engine.utils.layer_atomic_geometry` does not exist.

- [ ] **Step 3: Implement compact labels and DSU**

Implement a private compact-label conversion with `np.unique(..., return_inverse=True)` and a deterministic DSU whose smaller root always becomes the parent. Never expose or emit `-1`.

- [ ] **Step 4: Implement vectorized edge statistics**

For right and bottom neighbors, compute Euclidean point differences. Reject only non-finite edge distances from statistics; never remove their pixels or atom labels.

For internal edges (`label_a == label_b`), compute `s_A` with `np.bincount(weights=distance)` divided by counts. Internal scale samples must be finite and strictly positive. Mark atoms without any valid positive internal edge as geometrically invalid; do not merge a boundary involving an invalid-scale atom. Never use an absolute fallback such as `1.0`, because that breaks global point-scale invariance. Clamp only the final positive denominator with machine epsilon to prevent numerical warnings.

For boundary edges (`label_a != label_b`), encode the ordered pair as `min_label * n_atoms + max_label`, call `np.unique(..., return_inverse=True)`, and compute each `d_AB` using weighted/count `np.bincount`. This keeps memory proportional to boundary edges rather than constructing per-region pixel masks.

- [ ] **Step 5: Apply the one weak-prior rule and relabel once**

Derive each atom's coarse-layer id from the flattened first occurrence of that atom. Compute:

```python
normalized_gap = boundary_gap / np.sqrt(scale_a * scale_b)
limit = np.where(same_coarse_layer, 1.0 + depth_merge_thresh, 1.0)
```

Union pairs satisfying `normalized_gap <= limit`. Map atom roots back to the image and compact with `np.unique(..., return_inverse=True)`.

- [ ] **Step 6: Implement the full segmentation wrapper**

`segment_point_map_layer_atomic` validates `(H, W, 3)`, extracts `depth_map = point_map[..., -1]`, obtains baseline stages with all existing parameters unchanged, and calls `merge_layer_atoms`.

- [ ] **Step 7: Run focused tests, refactor, and rerun**

Run: `pytest -q tests/test_layer_atomic_geometry.py tests/test_depth_segmentation_stages.py`

Expected: all focused tests pass with no warnings.

- [ ] **Step 8: Commit**

```bash
git add inference_engine/utils/layer_atomic_geometry.py tests/test_layer_atomic_geometry.py
git commit -m "feat: add layer-atomic geometry segmentation"
```

---

### Task 3: Route existing point maps through the new segmentation only

**Files:**
- Modify: `inference_engine/utils/lsa.py`
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `inference_engine/streaming_window_engine_lc.py`
- Modify: `inference_engine/inference_utils.py`
- Create: `tests/test_layer_atomic_integration.py`

**Interfaces:**
- Consumes: `segment_point_map_layer_atomic` from Task 2.
- Changes internal segmentation input: `make_sp_graph(point_maps, ...)`, where `point_maps.shape == (N, H, W, 3)`.
- Preserves all existing optional parameter names and defaults after the first argument.

- [ ] **Step 1: Write a failing dispatch test**

Monkeypatch `lsa.batched_image_op_wrapper` and `lsa.match_segmentation_seq`. Assert that `make_sp_graph` passes the complete point-map batch to `segment_point_map_layer_atomic`, forwards `depth_merge_thresh`, `conf_map`, and `top_conf_percentile` unchanged, and forwards resulting labels to `match_segmentation_seq` with the unchanged `corr_iou_thresh`.

- [ ] **Step 2: Run the dispatch test and verify RED**

Run: `pytest -q tests/test_layer_atomic_integration.py`

Expected: the captured operation is still `segment_depth_felzenszwalb_rag`, so the assertion fails.

- [ ] **Step 3: Replace only the segmentation dispatch in `lsa.py`**

Import `segment_point_map_layer_atomic`, rename the first local parameter from `depth` to `point_maps`, and select the new operation in `batched_image_op_wrapper`. Do not alter graph matching or scale refinement.

- [ ] **Step 4: Pass complete existing point maps at every production call site**

Replace each `[..., -1]` argument used only to call `make_sp_graph` with the already available full local point map. Keep confidence and percentile keywords unchanged. In `inference_utils.py`, make the existing mask positional argument explicit as `conf_map=...`; this is input wiring for segmentation, not a change to registration or propagation.

- [ ] **Step 5: Run focused and full tests**

Run: `pytest -q tests/test_layer_atomic_integration.py && pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Verify the runtime diff is segmentation-scoped**

Run:

```bash
git diff origin/main...HEAD -- inference_engine
git diff --check origin/main...HEAD
```

Expected: only the new segmentation implementation, baseline-stage extraction, and complete point-map arguments to `make_sp_graph` are changed.

- [ ] **Step 7: Commit**

```bash
git add inference_engine/utils/lsa.py inference_engine/streaming_window_engine.py inference_engine/streaming_window_engine_lc.py inference_engine/inference_utils.py tests/test_layer_atomic_integration.py
git commit -m "feat: use layer-atomic segmentation in streaming inference"
```

---

### Task 4: Verify performance and prepare the KITTI 00 experiment

**Files:**
- No production source changes.

**Interfaces:**
- Verifies the complete segmentation call on KITTI-sized synthetic point maps.
- Produces exact commands and output locations for baseline/new KITTI 00 ATE comparison.

- [ ] **Step 1: Run fresh build and full regression**

Run:

```bash
python setup.py build_ext --inplace
pytest -q
python -m compileall -q inference_engine tests
git diff --check origin/main...HEAD
```

Expected: build succeeds, all tests pass, compileall exits zero, and diff-check prints nothing.

- [ ] **Step 2: Measure segmentation runtime**

Generate deterministic smooth `(376, 1241, 3)` point maps with several depth discontinuities. Warm up both functions, then record the median and p95 of at least five calls for:

- baseline `segment_depth_felzenszwalb_rag(point_map[..., -1], ...)`;
- new `segment_point_map_layer_atomic(point_map, ...)`;
- the geometry-only `merge_layer_atoms(...)` stage.
- complete baseline and new `make_sp_graph(...)` paths, including mask creation and temporal graph matching.

Report absolute milliseconds and `new / baseline`. The acceptance target is `<= 1.20`; `> 1.30` blocks the ATE experiment and requires profiling before any threshold tuning.

Also report peak resident memory for the complete `make_sp_graph` call. The
downstream code materializes one full-image boolean mask per final region and
dense IoU arrays, so a large increase in final region count or peak RSS is a
performance failure even when segmentation-only timing passes.

- [ ] **Step 3: Verify segmentation invariants on the performance frame**

Report baseline initial-atom count, coarse-layer count, final-region count, pixel coverage, minimum label, maximum label, and whether every final region is a union of complete initial atoms. Include a fragmented/checkerboard stress frame to expose boundary-pair sorting and downstream region-mask growth, but do not turn timing into a flaky pytest assertion.

- [ ] **Step 4: Review the complete branch diff**

Run a whole-branch spec and code-quality review against `origin/main`, fix every Critical or Important issue, and rerun Steps 1-3 after any fix.

- [ ] **Step 5: Prepare controlled KITTI 00 commands**

Use the repository's existing KITTI invocation and evaluator. Keep checkpoint, image sampling, window size, overlap, confidence percentile, seed/environment, and Sim(3) evaluation exactly identical between:

- clean `origin/main` baseline;
- `feature/layer-atomic-geometry`.

Run all 4541 frames of sequence 00. Save trajectory, APE/RPE text, wall-clock time, and segmentation diagnostics in separate baseline/new output directories. Do not report ATE improvement until both complete evaluations exist.

- [ ] **Step 6: Commit any test-only fixes, then report branch state**

Run:

```bash
git status --short --branch
git log --oneline origin/main..HEAD
```

Expected: only intentionally ignored build artifacts remain untracked; the branch contains the plan and verified segmentation commits and is ready for the external GPU/KITTI run.
