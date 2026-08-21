# Window Reference Segmentation Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-disabled, deterministic window-level postprocessor that uses PI3 geometry and adaptive reference frames to merge fragmented initial segmentation regions without ever splitting them.

**Architecture:** The ordinary prediction provider propagates the authoritative intrinsic into each `WindowPrediction`. A CPU/NumPy refiner performs sparse c2w reprojection, adaptive reference selection, reliable region correspondence, conflict-aware merge voting, and compact relabeling; a shared wrapper inserts it between initial segmentation and temporal graph construction in all three reconstruction modes.

**Tech Stack:** Python 3.11, PyTorch tensors at pipeline boundaries, NumPy CPU geometry, OmegaConf structured configuration, pytest, existing SciPy/scikit-image dependencies only.

**Spec:** `docs/superpowers/specs/2026-08-21-window-reference-segmentation-refinement-design.md`

## Global Constraints

- The feature is a postprocessor after `depth`, `geometry`, or `atomic` segmentation and before temporal graph construction; it is not a fourth `SegmentationMethod`.
- The refiner may merge complete existing regions but may never split an initial region or mutate PI3 tensors, input labels, or input diagnostic mappings.
- The feature is disabled by default. Disabled execution returns the exact original result list and adds no diagnostics.
- Use the provider's authoritative reference intrinsic. Missing or incompatible intrinsics trigger unchanged-label fallback; never estimate an intrinsic inside the refiner.
- Poses are OpenCV camera-to-world. Source-to-target projection is `(T_target_c2w)^-1 @ T_source_c2w @ X_source` with positive target `Z`.
- V1 runs deterministically on CPU using a centered sparse grid, deterministic z-buffer, high-confidence filtering, relative-depth consistency, four-connected adjacency, weighted merge/separate votes, and conflict-aware union-find.
- Reference selection is adaptive, permits one selected reference, and is bounded by `min(max_keyframes, N)`.
- Output labels use `np.intp`, are compact `0..R-1`, and satisfy `R_new <= R_initial`; every initial region maps to exactly one output region.
- Runtime unreliability falls back to initial labels. Programming-contract shape/count errors raise `ValueError`.
- Do not change prediction cache artifacts/fingerprint, reconstruction artifact schema/version/file set, PI3 model code, initial strategy implementations, temporal matching, anchor propagation, loop mathematics, or evaluation modules.
- Do not add a third-party dependency or allocate a dense `[K,N,H,W,...]` tensor or dense source-region-by-target-region table.
- Follow strict RED-GREEN-REFACTOR: add one behavior test, run it and observe the expected failure, add minimal production code, rerun the focused test, then run the task regression set before committing.
- Test expectations must be literal or hand-derived and must exercise production behavior rather than asserting on mocks.

---

## File and interface map

- `pipeline/config.py` owns `WindowReferenceConfig`, structured parsing, and validation.
- `configs/pipeline/*.yaml` and `configs/reconstruction/*.yaml` provide the complete required nested block.
- `inference_engine/prediction_cache/provider.py` exposes a cloned `reference_intrinsic` in each runtime prediction mapping without changing stored artifacts.
- `reconstruction/prediction_stream.py` validates and carries `WindowPrediction.reference_intrinsic`.
- `inference_engine/segmentation/window_reference.py` owns the protocol, disabled implementation, enabled implementation, projection evidence, reference selection, region evidence, union-find, diagnostics, and factory.
- `reconstruction/shared.py` owns the single `segment_and_refine_window()` integration wrapper.
- `reconstruction/modes/base.py` carries the refiner in `ReconstructionContext`.
- `pipeline/runner.py` builds the refiner through `PipelineDependencies`.
- `reconstruction/modes/{no_loop,traditional,corrected}.py` call the shared wrapper at their existing segmentation sites.
- `tests/test_window_reference_refinement.py` contains hand-constructed geometric and region-level behavior tests.
- Existing config, provider, prediction-stream, runner, mode, diagnostics, and artifact tests cover boundary integration.

### Stable cross-task interfaces

```python
# pipeline/config.py
@dataclass(frozen=True)
class WindowReferenceConfig:
    enabled: bool = MISSING
    sampling_stride: int = MISSING
    max_keyframes: int = MISSING
    relative_depth_tolerance: float = MISSING
    min_reference_score: float = MISSING
    stop_coverage_ratio: float = MISSING
    min_coverage_gain: float = MISSING
    min_region_correspondences: int = MISSING
    min_region_coverage: float = MISSING
    min_region_purity: float = MISSING
    merge_vote_threshold: float = MISSING

# inference_engine/segmentation/window_reference.py
class WindowReferenceRefinement(Protocol):
    enabled: bool
    def refine(
        self,
        results: list[SegmentationResult],
        *,
        point_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        confidence: torch.Tensor,
        reference_intrinsic: torch.Tensor | None,
    ) -> list[SegmentationResult]: ...

class DisabledWindowReferenceRefiner:
    enabled = False

class WindowReferenceRefiner:
    enabled = True

def build_window_reference_refiner(
    config: SegmentationConfig,
) -> WindowReferenceRefinement: ...

# reconstruction/shared.py
def segment_and_refine_window(
    *,
    strategy: SegmentationStrategy,
    refiner: WindowReferenceRefinement,
    point_maps: torch.Tensor,
    camera_poses: torch.Tensor,
    confidence: torch.Tensor,
    images: torch.Tensor,
    reference_intrinsic: torch.Tensor | None,
) -> list[SegmentationResult]: ...
```

---

### Task 1: Structured configuration and complete YAML defaults

**Files:**
- Modify: `pipeline/config.py`
- Modify: `configs/pipeline/default.yaml`
- Modify: `configs/pipeline/test.yaml`
- Modify: `configs/reconstruction/pi3_laser.yaml`
- Modify: `configs/reconstruction/pi3_laser_no_loop.yaml`
- Test: `tests/test_pipeline_config.py`

**Interfaces:**
- Consumes: Existing `SegmentationConfig`, strict OmegaConf schema, and `_validate_config()`.
- Produces: `WindowReferenceConfig` and `SegmentationConfig.window_reference` with the exact fields and defaults in the approved specification.

- [ ] **Step 1: Add a failing test for parsing the complete default block**

Add assertions to the existing default-config test using literal values:

```python
window_reference = loaded.config.segmentation.window_reference
assert window_reference.enabled is False
assert window_reference.sampling_stride == 4
assert window_reference.max_keyframes == 4
assert window_reference.relative_depth_tolerance == 0.05
assert window_reference.min_reference_score == 0.30
assert window_reference.stop_coverage_ratio == 0.90
assert window_reference.min_coverage_gain == 0.03
assert window_reference.min_region_correspondences == 8
assert window_reference.min_region_coverage == 0.10
assert window_reference.min_region_purity == 0.80
assert window_reference.merge_vote_threshold == 0.80
```

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_pipeline_config.py
```

Expected: failure because structured `SegmentationConfig` has no `window_reference` field or the YAML block is unknown/missing.

- [ ] **Step 3: Add the dataclass and exact YAML block**

Add the stable interface shown above and nest it in `SegmentationConfig`:

```python
@dataclass(frozen=True)
class SegmentationConfig:
    method: SegmentationMethod = MISSING
    confidence_keep_ratio: float = MISSING
    confidence_quantile_method: ConfidenceQuantileMethod = MISSING
    depth_merge_threshold: float = MISSING
    temporal_iou_threshold: float = MISSING
    felzenszwalb: FelzenszwalbConfig = field(default_factory=FelzenszwalbConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    atomic: AtomicConfig = field(default_factory=AtomicConfig)
    window_reference: WindowReferenceConfig = field(
        default_factory=WindowReferenceConfig
    )
```

Put the exact approved block under `segmentation:` in every complete YAML.

- [ ] **Step 4: Add failing table-driven boundary tests**

Use existing override helpers and literal invalid values. Cover boolean-as-integer rejection for all integer fields, zero for positive integers, `nan`/`inf`, open/closed ratio endpoints, and a non-boolean `enabled` value:

```python
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("segmentation.window_reference.sampling_stride=0", "sampling_stride"),
        ("segmentation.window_reference.max_keyframes=true", "max_keyframes"),
        ("segmentation.window_reference.relative_depth_tolerance=0", "relative_depth_tolerance"),
        ("segmentation.window_reference.min_reference_score=0", "min_reference_score"),
        ("segmentation.window_reference.stop_coverage_ratio=1.1", "stop_coverage_ratio"),
        ("segmentation.window_reference.min_coverage_gain=-0.01", "min_coverage_gain"),
        ("segmentation.window_reference.min_region_correspondences=false", "min_region_correspondences"),
        ("segmentation.window_reference.min_region_coverage=nan", "min_region_coverage"),
        ("segmentation.window_reference.min_region_purity=0", "min_region_purity"),
        ("segmentation.window_reference.merge_vote_threshold=inf", "merge_vote_threshold"),
    ],
)
def test_window_reference_config_rejects_invalid_boundaries(config_path, override, message):
    with pytest.raises(ValueError, match=message):
        load_pipeline_config(config_path, [override])
```

Expected RED: at least one invalid value loads successfully because validation does not exist.

- [ ] **Step 5: Implement explicit finite/type/range validation**

Use a helper that rejects `bool` before `int` checks and emits field-qualified messages:

```python
def _positive_int(path: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")

window_reference = config.segmentation.window_reference
for name in ("sampling_stride", "max_keyframes", "min_region_correspondences"):
    _positive_int(
        f"segmentation.window_reference.{name}",
        getattr(window_reference, name),
    )
```

Implement the exact open/closed intervals from the spec with `math.isfinite()`.

- [ ] **Step 6: Run focused and YAML-load regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_pipeline_config.py
```

Expected: all configuration tests pass with no new warnings.

- [ ] **Step 7: Commit Task 1**

```bash
git add pipeline/config.py configs/pipeline/default.yaml configs/pipeline/test.yaml configs/reconstruction/pi3_laser.yaml configs/reconstruction/pi3_laser_no_loop.yaml tests/test_pipeline_config.py
git commit -m "feat: configure window reference refinement"
```

---

### Task 2: Propagate and validate the authoritative intrinsic

**Files:**
- Modify: `inference_engine/prediction_cache/provider.py`
- Modify: `reconstruction/prediction_stream.py`
- Test: `tests/test_prediction_provider.py`
- Test: `tests/reconstruction/test_prediction_stream.py`
- Test: `tests/test_prediction_cache_method_parity.py`

**Interfaces:**
- Consumes: Provider `_reference_intrinsic`, all cache modes, and `WindowPrediction.from_mapping(..., process_device)`.
- Produces: Mapping key `reference_intrinsic` as a clone and optional `WindowPrediction.reference_intrinsic: torch.Tensor | None` on `process_device`.

- [ ] **Step 1: Add failing provider tests for miss, hit, and off paths**

Extend the provider fixture matrix. For each path, assert value equality and clone isolation rather than implementation calls:

```python
prediction = provider.get(spec, images)
intrinsic = prediction["reference_intrinsic"]
assert intrinsic.shape == (3, 3)
assert torch.equal(intrinsic, provider.reference_intrinsic)
intrinsic[0, 0] = -123.0
assert provider.reference_intrinsic[0, 0].item() > 0.0
```

Expected RED: `KeyError: 'reference_intrinsic'`.

- [ ] **Step 2: Return a clone without changing cache artifacts**

Centralize runtime mapping completion so both model and replay paths use it:

```python
def _with_reference_intrinsic(
    self,
    prediction: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if self._reference_intrinsic is None:
        raise RuntimeError("reference intrinsic is unavailable")
    return {
        **prediction,
        "reference_intrinsic": self._reference_intrinsic.clone(),
    }
```

Do not add the tensor to `OrdinaryWindowArtifact`, `SequenceArtifact`, the fingerprint, or store filenames.

- [ ] **Step 3: Run provider tests GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_prediction_provider.py tests/test_prediction_cache_method_parity.py
```

Expected: all provider/cache-parity tests pass.

- [ ] **Step 4: Add failing `WindowPrediction` behavior tests**

Cover an absent key, a valid `(3,3)` tensor moved to the requested device, and invalid shape/dtype/finiteness/focal/last-row cases:

```python
prediction = WindowPrediction.from_mapping(
    {**mapping, "reference_intrinsic": torch.tensor([
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 1.0],
    ])},
    spec,
    "key",
    "cpu",
)
assert torch.equal(
    prediction.reference_intrinsic,
    torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]]),
)
```

Expected RED: constructor rejects the new keyword or result has no field.

- [ ] **Step 5: Add optional field and exact validation**

Append the dataclass field after required fields:

```python
reference_intrinsic: torch.Tensor | None = None
```

When the mapping key is absent, store `None`. When present, require a floating finite `(3,3)` tensor, `fx > 0`, `fy > 0`, and `torch.allclose(K[2], [0,0,1], atol=1e-6, rtol=0)`, then clone and move it to `process_device`.

- [ ] **Step 6: Run prediction-stream and provider regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/reconstruction/test_prediction_stream.py tests/test_prediction_provider.py tests/test_prediction_cache_method_parity.py
```

Expected: all tests pass; old mapping fixtures without the optional key remain valid.

- [ ] **Step 7: Commit Task 2**

```bash
git add inference_engine/prediction_cache/provider.py reconstruction/prediction_stream.py tests/test_prediction_provider.py tests/reconstruction/test_prediction_stream.py tests/test_prediction_cache_method_parity.py
git commit -m "feat: propagate prediction reference intrinsic"
```

---

### Task 3: Refiner boundary, disabled identity, confidence, and fallback

**Files:**
- Create: `inference_engine/segmentation/window_reference.py`
- Modify: `inference_engine/segmentation/__init__.py`
- Create: `tests/test_window_reference_refinement.py`

**Interfaces:**
- Consumes: `SegmentationConfig`, `WindowReferenceConfig`, `ConfidenceQuantileMethod`, `SegmentationResult`, and `compact_labels()`.
- Produces: `WindowReferenceRefinement`, `DisabledWindowReferenceRefiner`, `WindowReferenceRefiner`, and `build_window_reference_refiner()` with the stable signatures above.

- [ ] **Step 1: Add failing factory and disabled-identity tests**

Use real `SegmentationResult` values and input diagnostics mappings:

```python
def test_disabled_refiner_returns_exact_result_list(segmentation_config):
    results = [SegmentationResult(np.array([[0, 1]], dtype=np.intp), {"region_count": 2})]
    refiner = build_window_reference_refiner(segmentation_config)
    refined = refiner.refine(
        results,
        point_maps=torch.zeros((1, 1, 2, 3)),
        camera_poses=torch.eye(4).repeat(1, 1, 1),
        confidence=torch.zeros((1, 1, 2)),
        reference_intrinsic=None,
    )
    assert refined is results
    assert refined[0].diagnostics is results[0].diagnostics
```

Expected RED: import failure because the module/factory does not exist.

- [ ] **Step 2: Implement the protocol, factory, and disabled implementation**

Use the stable cross-task interface. The factory selects solely on `config.window_reference.enabled`. Export the builder and types from `inference_engine/segmentation/__init__.py`.

- [ ] **Step 3: Run the disabled test GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_window_reference_refinement.py -k disabled
```

Expected: disabled identity and no-diagnostic tests pass.

- [ ] **Step 4: Add failing enabled boundary/fallback tests**

Cover frame-count/rank/spatial mismatch as `ValueError`, plus unchanged copied results with fallback strings for `single_frame`, `missing_intrinsic`, `invalid_intrinsic`, `intrinsic_incompatible`, `invalid_geometry`, and `no_reference`. Also assert no input mutation by cloning tensors/labels/diagnostics before the call.

Use a hand-built identity fixture with this literal K and local point map:

```python
K = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
local = torch.tensor([
    [[-0.5, -0.5, 1.0], [0.0, -0.5, 1.0], [0.5, -0.5, 1.0]],
    [[-0.5,  0.0, 1.0], [0.0,  0.0, 1.0], [0.5,  0.0, 1.0]],
    [[-0.5,  0.5, 1.0], [0.0,  0.5, 1.0], [0.5,  0.5, 1.0]],
])
```

Expected RED: enabled class lacks validation/fallback behavior.

- [ ] **Step 5: Implement preparation, stable sigmoid, quantile masks, and fallback result copying**

Add private helpers with exact behavior:

```python
def _confidence_probability(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))

def _fallback_results(
    results: list[SegmentationResult],
    reason: str,
) -> list[SegmentationResult]:
    return [
        SegmentationResult(
            labels=result.labels.copy(),
            diagnostics={
                **dict(result.diagnostics),
                "window_reference_applied": False,
                "window_reference_fallback": reason,
                "window_reference_regions_before": int(np.unique(result.labels).size),
                "window_reference_regions_after": int(np.unique(result.labels).size),
            },
        )
        for result in results
    ]
```

Complete the enabled diagnostic key set from the spec with finite scalar values. Reuse NumPy quantile `method=config.confidence_quantile_method.value`, retaining values equal to the threshold. Calculate `Q_i = selected_valid_fraction * mean(sigmoid(logit))`. Validate source pixels project through K to within 0.5 pixel on the centered grid.

Define the centered coordinates without an image-size special case:

```python
def _centered_axis(length: int, stride: int) -> np.ndarray:
    start = ((length - 1) % stride) // 2
    return np.arange(start, length, stride, dtype=np.int64)
```

Build the confidence quantile only from pixels whose confidence and XYZ are finite
and whose local `Z > 1e-6`; the returned mask is that valid mask intersected with
`confidence >= threshold`.

- [ ] **Step 6: Run Task 3 tests and refactor without behavior changes**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_window_reference_refinement.py -k "disabled or fallback or validation or confidence"
```

Expected: focused tests pass and input equality assertions remain true.

- [ ] **Step 7: Commit Task 3**

```bash
git add inference_engine/segmentation/window_reference.py inference_engine/segmentation/__init__.py tests/test_window_reference_refinement.py
git commit -m "feat: add window reference refiner boundary"
```

---

### Task 4: Deterministic sparse c2w reprojection and pair scoring

**Files:**
- Modify: `inference_engine/segmentation/window_reference.py`
- Modify: `tests/test_window_reference_refinement.py`

**Interfaces:**
- Consumes: Prepared CPU arrays, centered sample indices, high-confidence masks, K, and camera-to-world poses from Task 3.
- Produces: Private immutable `_PairProjection` containing source/target flat indices, correspondence weights, coverage, geometry ratio, confidence score, pair score, and rejection counters.

- [ ] **Step 1: Add failing identity and translated-camera projection tests**

For identity poses and the 3x3 fixture, expect sampled source pixel 4 to map to target pixel 4. For translation direction, use source c2w translation `+0.5` on X, target identity, `fx=2`, `Z=1`; the center source point must project one pixel to the right. This hand-derived result detects accidentally using `T_target @ inv(T_source)`.

```python
source_pose = torch.eye(4)
source_pose[0, 3] = 0.5
target_pose = torch.eye(4)
pair = _project_pair(...)
assert pair.source_flat_indices.tolist() == [4]
assert pair.target_flat_indices.tolist() == [5]
```

Expected RED: `_project_pair` is absent.

- [ ] **Step 2: Implement transform, projection, rounding, bounds, and positive-depth filtering**

Use float64 for pose inversion/relative transform and release each pair result before evaluating the next pair:

```python
relative = np.linalg.inv(target_pose.astype(np.float64)) @ source_pose.astype(np.float64)
target_xyz = (relative @ source_homogeneous.T).T[:, :3]
positive = target_xyz[:, 2] > 1e-6
u = K[0, 0] * target_xyz[:, 0] / target_xyz[:, 2] + K[0, 2]
v = K[1, 1] * target_xyz[:, 1] / target_xyz[:, 2] + K[1, 2]
pixel_u = np.floor(u + 0.5).astype(np.int64)
pixel_v = np.floor(v + 0.5).astype(np.int64)
```

- [ ] **Step 3: Run translation tests GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_window_reference_refinement.py -k "identity_projection or c2w_translation"
```

Expected: both tests pass.

- [ ] **Step 4: Add failing z-buffer, confidence, occlusion, depth, and score tests**

Use literal fixtures that create two source samples at the same target pixel with depths `1.0` and `2.0`, then a tie at `1.0`. Assert the nearest depth wins and a tie selects the lower source flat index. Cover out-of-bounds/negative-Z removal, target low-confidence abstention, strict `relative_error < tolerance`, zero denominators, and this hand-derived perfect-pair score:

```python
assert pair.coverage == pytest.approx(1.0)
assert pair.geometry_ratio == pytest.approx(1.0)
assert pair.mean_confidence == pytest.approx(0.5)  # zero logits on both frames
assert pair.score == pytest.approx(np.sqrt(0.5))
```

Expected RED: winner ordering, rejection counters, or score fields are missing.

- [ ] **Step 5: Implement deterministic z-buffer and exact evidence formulas**

Sort by target pixel, projected depth, then source flat index; select the first record for each target. Apply target-valid/high-confidence filtering and the strict relative-depth predicate. Set correspondence weight to:

```python
confidence_weight = np.sqrt(source_probability * target_probability)
depth_weight = np.maximum(0.0, 1.0 - relative_error / tolerance)
weight = confidence_weight * depth_weight
pair_score = coverage * np.sqrt(geometry_ratio * mean_confidence)
```

Count coarse target cells with `u // stride, v // stride`. Clamp reported components to `[0,1]` and return zero for empty denominators.

- [ ] **Step 6: Run all projection tests and the entire refiner file**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_window_reference_refinement.py -k projection
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_window_reference_refinement.py
```

Expected: all current refiner tests pass with deterministic arrays and finite scores.

- [ ] **Step 7: Commit Task 4**

```bash
git add inference_engine/segmentation/window_reference.py tests/test_window_reference_refinement.py
git commit -m "feat: project sparse reference evidence"
```

---

### Task 5: Adaptive reference selection

**Files:**
- Modify: `inference_engine/segmentation/window_reference.py`
- Modify: `tests/test_window_reference_refinement.py`

**Interfaces:**
- Consumes: Per-frame `Q_i`, lazy `_project_pair(source, target)` callback, `min_reference_score`, `stop_coverage_ratio`, `min_coverage_gain`, and `max_keyframes`.
- Produces: Immutable `_ReferenceSelection` with ascending selected indices, per-target best scores, accepted reliable pair projections, and selection diagnostics.

- [ ] **Step 1: Add failing first-reference and one-reference-stop tests**

Pass a deterministic fake pair evaluator backed by literal score matrices; assert results rather than call counts. Cover quality ordering, temporal-center tie, lower-index tie, and a first frame whose directed scores already cover at least 90% of frames:

```python
selection = _select_references(
    qualities=np.array([0.8, 0.8, 0.2]),
    frame_count=3,
    evaluate=lambda source, target: pair_scores[source][target],
    config=config,
)
assert selection.indices == (1,)
```

Expected RED: selection helper is absent.

- [ ] **Step 2: Implement first selection, `B_t`, and coverage stop**

Choose descending quality, then distance to `(N-1)/2`, then lower index. Treat selected references as covered only for the stop ratio; never create self merge evidence. Store accepted output indices in ascending order while retaining selection order internally for computation.

- [ ] **Step 3: Add failing marginal-gain and candidate-rejection tests**

Use a four-frame literal matrix where the highest-priority second candidate has gain below `0.03`, but the next candidate has gain above it. Assert only the weak candidate is rejected and selection continues. Add safety-ceiling, zero-quality, and deterministic-priority ties.

```python
assert selection.indices == (0, 2)
assert selection.rejected_indices == (1,)
```

Expected RED: implementation stops after the first rejected candidate or selects a fixed count.

- [ ] **Step 4: Implement exact priority, marginal gain, and stop conditions**

Use:

```python
priority = qualities[candidate] * (1.0 - best_scores[candidate])
gain = np.mean([
    max(best_scores[target], candidate_scores[target]) - best_scores[target]
    for target in eligible_targets
]) if eligible_targets else 0.0
```

Reject only the tested candidate when gain is too small. Stop on coverage ratio, ceiling, or exhaustion/rejection of all positive-quality candidates. Evaluate candidates lazily and never allocate an all-pairs tensor.

- [ ] **Step 5: Run adaptive-selection tests and repeat for determinism**

Run twice:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_window_reference_refinement.py -k selection
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_window_reference_refinement.py -k selection
```

Expected: identical passing results in both runs.

- [ ] **Step 6: Commit Task 5**

```bash
git add inference_engine/segmentation/window_reference.py tests/test_window_reference_refinement.py
git commit -m "feat: select adaptive segmentation references"
```

---

### Task 6: Region correspondence, conflict-aware merge voting, and final diagnostics

**Files:**
- Modify: `inference_engine/segmentation/window_reference.py`
- Modify: `tests/test_window_reference_refinement.py`

**Interfaces:**
- Consumes: Original initial labels, reliable non-self `_PairProjection` values from selected references, and region thresholds.
- Produces: Reliable dominant mappings, four-connected adjacency votes, deterministic conflict-aware union results, compact refined `SegmentationResult` values, and the complete scalar diagnostic vocabulary.

- [ ] **Step 1: Add failing dominant-mapping boundary tests**

Create target regions and sparse literal `(target_pixel, source_label, weight)` evidence. Cover unique-hit minimum, approximate coverage `min(1, hits * stride**2 / area)`, purity `dominant_weight / total_weight`, and smaller-source-label tie breaking. Assert exact dominant label and `support=min(coverage,purity)` only when all thresholds pass.

Expected RED: region aggregation helper is absent.

- [ ] **Step 2: Implement sparse encoded-pair aggregation**

Encode only observed pairs, use weighted `np.unique`/sorting or a dictionary bounded by correspondence count, and never allocate a dense region-pair matrix. Count unique target pixels separately from total weight.

- [ ] **Step 3: Add failing adjacency and weighted vote tests**

Use these literal target labels to distinguish four-neighbor adjacency from diagonal contact:

```python
labels = np.array([
    [0, 0, 1],
    [0, 2, 1],
    [3, 3, 2],
], dtype=np.intp)
assert _adjacent_region_edges(labels) == ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))
```

Build two references with weighted agreement/disagreement and assert an edge is eligible only when merge evidence is positive and `merge / (merge + separate) >= 0.80`. Cover abstention when either region mapping is unreliable.

Expected RED: adjacency/vote helpers are absent.

- [ ] **Step 4: Implement deterministic edge building and sorting**

Deduplicate sorted horizontal/vertical label pairs. Accumulate `pair_score * min(support_a, support_b)` into merge or separate evidence. Sort eligible edges by `(-ratio, -merge_evidence, label_a, label_b)`.

- [ ] **Step 5: Add failing direct and transitive-conflict union tests**

Cover an accepted adjacent merge, a non-adjacent same-reference-label pair that remains separate, and three regions where `(0,1)` can merge but merging the result with `2` would combine different known labels for one reference. Assert the second union is rejected and counted as a conflict.

Expected RED: plain DSU permits the transitive conflict.

- [ ] **Step 6: Implement component signatures and compact merge-only labels**

Each root stores `{reference_index: dominant_source_label}`. Reject a union if an overlapping reference key has unequal labels. Otherwise attach the larger numeric root to the smaller numeric root and merge compatible signatures. Relabel each original region through its root, then call `compact_labels()` and cast to `np.intp`.

- [ ] **Step 7: Add failing end-to-end refiner invariants and diagnostics tests**

Use a two-frame identity fixture where the reference has one region and the target has two adjacent regions. Assert target merges to one region, keyframe labels remain byte-for-byte unchanged, each initial target region maps to one output label, repeated execution is identical, and every diagnostic value is `float|int|bool|str` and finite where numeric.

Assert all approved diagnostic keys and fallback vocabulary literally. Also cover `insufficient_support` and `conflict_only`, and verify `region_count` equals the final compact region count.

Expected RED: `refine()` has not connected selection/projection/merge or diagnostics are incomplete.

- [ ] **Step 8: Wire the enabled `refine()` orchestration and run the full unit file**

Use only original initial labels for all evidence and do not refine selected reference frames. Copy diagnostics before amendment. Process pair evidence one target/reference at a time. Preserve labels unchanged on every runtime fallback.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_window_reference_refinement.py
```

Expected: all projection, selection, mapping, conflict, fallback, mutation, invariant, and determinism tests pass.

- [ ] **Step 9: Commit Task 6**

```bash
git add inference_engine/segmentation/window_reference.py tests/test_window_reference_refinement.py
git commit -m "feat: merge window regions with reference evidence"
```

---

### Task 7: Insert the refiner in all reconstruction modes

**Files:**
- Modify: `reconstruction/shared.py`
- Modify: `reconstruction/modes/base.py`
- Modify: `pipeline/runner.py`
- Modify: `reconstruction/modes/no_loop.py`
- Modify: `reconstruction/modes/traditional.py`
- Modify: `reconstruction/modes/corrected.py`
- Create: `tests/reconstruction/test_shared.py`
- Test: `tests/test_pipeline_runner.py`
- Test: `tests/reconstruction/test_no_loop_mode.py`
- Test: `tests/reconstruction/test_traditional_mode.py`
- Test: `tests/reconstruction/test_corrected_mode.py`

**Interfaces:**
- Consumes: Task 2 `WindowPrediction.reference_intrinsic`, Task 3 factory/protocol, existing strategy and graph APIs, and each mode's current window tensor state.
- Produces: One shared `segment_and_refine_window()` call site and exact `segment -> refine -> graph -> anchor` order in all modes.

- [ ] **Step 1: Add failing shared-wrapper ordering and disabled-bypass tests**

Use small real result objects and recording fakes whose return values flow to the next real boundary. Assert event order and result identity:

```python
assert events == ["segment", "refine"]
assert returned is refined_results
```

For `refiner.enabled is False`, assert the strategy output is returned and the disabled refiner's `refine()` method is not invoked. This realizes the spec's integration bypass while retaining the disabled implementation's direct identity contract.

Expected RED: wrapper does not exist.

- [ ] **Step 2: Implement `segment_and_refine_window()`**

```python
results = strategy.segment(
    as_numpy(point_maps),
    as_numpy(confidence),
    as_numpy(images),
)
if not refiner.enabled:
    return results
return refiner.refine(
    results,
    point_maps=point_maps,
    camera_poses=camera_poses,
    confidence=confidence,
    reference_intrinsic=reference_intrinsic,
)
```

- [ ] **Step 3: Add failing runner construction/injection test**

Extend `PipelineDependencies` with `build_window_reference_refiner`. In the existing dependency-driven runner test, return a sentinel refiner and capture the `ReconstructionContext`; assert the builder receives `config.segmentation` and the context carries the same sentinel.

Expected RED: dependency keyword/context field is unknown.

- [ ] **Step 4: Build and carry the refiner**

Add:

```python
build_window_reference_refiner: Callable = build_window_reference_refiner
```

to `PipelineDependencies`, construct it beside the initial segmenter, and add:

```python
window_reference_refiner: WindowReferenceRefinement
```

to `ReconstructionContext`.

- [ ] **Step 5: Add failing mode-order and tensor-state tests**

For each mode, inject a shared wrapper callable into the constructor and record `segment`, `refine`, `graph`, and `anchor`. Assert:

```python
assert events.index("segment") < events.index("refine") < events.index("graph")
if anchor_enabled:
    assert events.index("graph") < events.index("anchor")
```

Also assert the wrapper receives:

- no-loop: re-unprojected/scaled `local_points`, adjusted `camera_poses`, and the no-loop current intrinsic;
- traditional: current untransformed window tensors and `prediction.reference_intrinsic`;
- corrected: current scale/pose-adjusted tensors and `prediction.reference_intrinsic`.

Expected RED: modes call `segmentation_strategy.segment()` directly.

- [ ] **Step 6: Replace the three direct calls with the injected shared wrapper**

Add a `segment_window: Callable = segment_and_refine_window` constructor dependency to each mode, store it, and call with the exact tensor state named above. Retain graph construction, trace contents, anchor ordering, state aggregation, loop services, and segmentation-summary collection.

- [ ] **Step 7: Add and pass disabled tensor/artifact parity tests**

Run each mode with a disabled refiner and compare local points, global points, poses, confidence, graph inputs, segmentation summaries, schema version, and artifact file set against the existing behavior fixture. Do not merely assert the disabled refiner was skipped.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_pipeline_runner.py \
  tests/reconstruction/test_shared.py \
  tests/reconstruction/test_no_loop_mode.py \
  tests/reconstruction/test_traditional_mode.py \
  tests/reconstruction/test_corrected_mode.py
```

Expected: all integration/order/parity tests pass.

- [ ] **Step 8: Commit Task 7**

```bash
git add reconstruction/shared.py reconstruction/modes/base.py pipeline/runner.py reconstruction/modes/no_loop.py reconstruction/modes/traditional.py reconstruction/modes/corrected.py tests/reconstruction/test_shared.py tests/test_pipeline_runner.py tests/reconstruction/test_no_loop_mode.py tests/reconstruction/test_traditional_mode.py tests/reconstruction/test_corrected_mode.py
git commit -m "feat: refine segmentation before temporal graphs"
```

---

### Task 8: Artifact contract, documentation, and complete verification

**Files:**
- Modify: `tests/test_reconstruction_artifacts.py`
- Modify: `docs/pipeline-configuration.md`
- Modify: `README.md` only if its architecture/command descriptions are false after Task 7

**Interfaces:**
- Consumes: Complete enabled diagnostics and unchanged `ReconstructionArtifact` schema version 1.
- Produces: User-facing configuration documentation, canonical JSON evidence, and final verification evidence.

- [ ] **Step 1: Add a failing canonical artifact round-trip test**

Build enabled segmentation summaries containing booleans, strings, integers, and finite floats from the approved diagnostic vocabulary; round-trip through the real artifact serializer and loader. Assert schema version remains `1`, `mode_scalars` contains only numeric scalars, no segmentation-label file is created, and all existing artifact filenames are unchanged.

Expected RED: only if current serialization or newly wired diagnostics violate the scalar/JSON contract. If the test passes immediately because existing serialization already satisfies the contract, record that it is a characterization test and do not change production artifact code.

- [ ] **Step 2: Run artifact and diagnostics tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_reconstruction_artifacts.py
```

Expected: pass without artifact schema/file changes.

- [ ] **Step 3: Document exact configuration and failure semantics**

Add this concrete structure to `docs/pipeline-configuration.md`:

```markdown
### Window reference segmentation refinement

`segmentation.window_reference.enabled` defaults to `false`. When enabled,
LASER selects reference frames adaptively from the current PI3 window, performs
sparse geometry/visibility checks, and may merge complete adjacent regions from
the selected initial segmentation method. It never splits a region. Missing or
unreliable geometry preserves the initial labels.
```

Document every field, accepted interval, default value, CPU cost control, and the fact that empirical thresholds require real-cache evaluation. Update `README.md` only where its described pipeline order would otherwise omit the new optional stage.

- [ ] **Step 4: Run formatting/static checks available in the repository**

Run:

```bash
git diff --check
python -m compileall -q inference_engine pipeline reconstruction
```

Expected: no whitespace errors and no Python syntax/import compilation errors.

- [ ] **Step 5: Run the focused feature suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_window_reference_refinement.py \
  tests/test_pipeline_config.py \
  tests/test_prediction_provider.py \
  tests/reconstruction/test_prediction_stream.py \
  tests/test_pipeline_runner.py \
  tests/reconstruction/test_shared.py \
  tests/reconstruction/test_no_loop_mode.py \
  tests/reconstruction/test_traditional_mode.py \
  tests/reconstruction/test_corrected_mode.py \
  tests/test_reconstruction_artifacts.py
```

Expected: all focused tests pass with no new warnings.

- [ ] **Step 6: Run segmentation smoke and record pre-existing warnings separately**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_segmentation_modes.py
```

Expected: depth, geometry, and atomic all report PASS. Existing numerical warnings at `inference_engine/utils/depth.py:171` may remain; no new warning may originate from the refiner.

- [ ] **Step 7: Run the complete suite twice for deterministic coverage**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
```

Expected: both runs pass with identical counts. Compare the two captured summaries and investigate any differing labels/diagnostics or intermittent failure before committing.

- [ ] **Step 8: Verify scope and commit Task 8**

Run:

```bash
git status --short
git diff --check
git diff --stat 8f94083
```

Confirm no cache/artifact/evaluation/PI3 model file is modified and no dependency was added. Then commit only Task 8 files:

```bash
git add tests/test_reconstruction_artifacts.py docs/pipeline-configuration.md
git commit -m "docs: document window reference refinement"
```

If `README.md` changed because its pipeline description was false, add that exact file
before committing. Do not modify `pipeline/diagnostics.py`: per-frame refinement
diagnostics already travel through `segmentation_summaries`, while `mode_scalars`
must remain numeric and schema-compatible.

---

## Final branch acceptance

After all task commits and task-scoped reviews:

1. Generate one whole-branch review package from `8f94083` to `HEAD`.
2. Review against the approved specification, not only this plan.
3. Confirm every new production branch is covered by a test observed RED before implementation.
4. Confirm disabled outputs are identical, output labels are merge-only and compact, all modes use the correct tensor state/intrinsic, and no schema/dependency/cache change slipped in.
5. Run the fresh final commands from Task 8 after any review fix.
6. Report the absence of real weights/data/cache as an empirical-calibration limitation; do not claim reconstruction-quality improvement until the real-cache experiment gate in the specification is executed.
