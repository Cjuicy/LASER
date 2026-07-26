# Joint Loop-Constraint Estimation Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore candidate-neighborhood joint Pi3 inference so both loop methods build geometrically valid constraints without changing segmentation, split, anchor-propagation, or optimizer behavior.

**Architecture:** A new `JointPi3AlignmentEstimator` owns centered frame selection, one joint Pi3 forward pass, and two cached-to-joint registrations. Traditional and corrected strategies require this shared estimator for cross-window candidates and retain only their distinct constraint conversion formulas. The modular runner and legacy evaluation entry point construct the same estimator from the resolved configuration.

**Tech Stack:** Python 3.11, PyTorch, OmegaConf-backed typed pipeline configuration, pytest.

## Global Constraints

- Keep depth, geometry, and atomic segmentation behavior unchanged.
- Keep atomic split modes `none`, `conservative`, and `normal_only` unchanged.
- Keep anchor propagation and both Sim(3) optimizers unchanged.
- Use only `loop.constraint.chunk_size` and `loop.registration.confidence_keep_ratio`; do not restore retired arguments.
- Use the immutable `ImageManifest` and the exact preprocessed image tensor from the active run.
- Catch candidate-local `ValueError`; propagate `RuntimeError`, CUDA OOM, and unexpected failures.
- Preserve window-cache schema version 1 and the existing diagnostics schema.

---

## File Structure

- Create `loop_closure/constraint_estimation.py`: centered candidate neighborhoods, joint Pi3 inference, cached-to-joint confidence masking, and dual Sim(3) alignment.
- Modify `loop_closure/methods/base.py`: extend the strategy protocol with an explicit optional constraint-estimator keyword.
- Modify `loop_closure/methods/traditional.py`: remove direct frame-to-frame registration and convert shared alignments with `compute_sim3_ab`.
- Modify `loop_closure/methods/corrected.py`: remove direct frame-to-frame registration and convert shared alignments with `build_local_loop_constraint`.
- Modify `pipeline/runner.py`: expose the estimator factory as a dependency and wire model/images/manifest/resolved constraint parameters only when candidates exist.
- Modify `eval_launch.py`: wire the same estimator into legacy `streaming_pi3_lc` evaluation.
- Create `tests/test_loop_constraint_estimation.py`: focused estimator behavior and boundary tests.
- Modify `tests/test_traditional_loop_method.py`: traditional estimator requirement, ratio propagation, and conversion tests.
- Modify `tests/test_corrected_loop_method.py`: corrected estimator requirement and conversion tests.
- Modify `tests/test_loop_method_contracts.py`: shared candidate rejection, failure isolation, deduplication, and system-error propagation.
- Modify `tests/test_pipeline_runner.py`: runner factory wiring and no-candidate bypass.
- Create `tests/test_eval_launch_lc.py`: legacy evaluation wiring without loading real Pi3 weights.
- Modify `docs/pipeline-configuration.md`: state that `chunk_size` is the per-side joint inference neighborhood size.

---

### Task 1: Shared Joint Pi3 Alignment Estimator

**Files:**
- Create: `tests/test_loop_constraint_estimation.py`
- Create: `loop_closure/constraint_estimation.py`

**Interfaces:**
- Consumes: `WindowCache`, `LoopCandidate`, `Sim3`, `ImageManifest`, `register_adjacent_windows`, `select_top_confidence_mask`, and `intersect_confidence_masks`.
- Produces: `centered_frame_range(frame_start: int, frame_end: int, center: int, chunk_size: int) -> tuple[int, int]`.
- Produces: `JointPi3AlignmentEstimator.__call__(cache_a: WindowCache, cache_b: WindowCache, candidate: LoopCandidate, keep_ratio: float) -> tuple[Sim3, Sim3]`.

- [ ] **Step 1: Write failing centered-range tests**

Add literal boundary cases:

```python
@pytest.mark.parametrize(
    ("bounds", "center", "chunk_size", "expected"),
    (
        ((4, 10), 8, 4, (6, 10)),
        ((4, 10), 4, 4, (4, 8)),
        ((4, 10), 6, 5, (4, 9)),
        ((4, 10), 7, 1, (7, 8)),
    ),
)
def test_centered_frame_range_returns_exact_bounded_neighborhood(
    bounds,
    center,
    chunk_size,
    expected,
):
    assert centered_frame_range(*bounds, center, chunk_size) == expected
```

These tests catch off-by-one, even/odd-size, and boundary-shortening regressions.

- [ ] **Step 2: Run the centered-range tests and verify RED**

Run:

```bash
pytest -q tests/test_loop_constraint_estimation.py
```

Expected: collection fails because `loop_closure.constraint_estimation` does not exist.

- [ ] **Step 3: Implement centered frame selection**

Implement:

```python
def centered_frame_range(
    frame_start: int,
    frame_end: int,
    center: int,
    chunk_size: int,
) -> tuple[int, int]:
    if frame_start < 0 or frame_end <= frame_start:
        raise ValueError("candidate cache frame range is invalid")
    if not frame_start <= center < frame_end:
        raise ValueError("candidate frame is outside its selected cache")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("loop constraint chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("loop constraint chunk_size must be at least 1")
    size = min(chunk_size, frame_end - frame_start)
    start = center - size // 2
    start = max(frame_start, min(start, frame_end - size))
    return start, start + size
```

- [ ] **Step 4: Run the centered-range tests and verify GREEN**

Run:

```bash
pytest -q tests/test_loop_constraint_estimation.py
```

Expected: all centered-range cases pass.

- [ ] **Step 5: Write failing joint inference and dual-registration tests**

Use a recording `torch.nn.Module` that returns the complete Pi3 prediction structure:

```python
class RecordingPi3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, images):
        self.inputs.append(images.detach().cpu().clone())
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch, frames, _, height, width = images.shape
        frame_ids = images[:, :, :1].permute(0, 1, 3, 4, 2)
        local_points = frame_ids.repeat(1, 1, 1, 1, 3)
        camera_poses = torch.eye(4).repeat(batch, frames, 1, 1)
        confidence = torch.ones((batch, frames, height, width))
        return {
            "points": local_points,
            "local_points": local_points,
            "camera_poses": camera_poses,
            "conf": confidence,
        }
```

Construct cache A with absolute range `[4, 10)`, cache B with `[0, 6)`,
candidate `(8, 1)`, frame-valued images, and `chunk_size=4`. Patch only
`register_adjacent_windows`, the slow geometric boundary below the estimator.
Assert:

```python
assert model.inputs[0][:, 0, 0, 0].tolist() == [
    6.0, 7.0, 8.0, 9.0, 0.0, 1.0, 2.0, 3.0
]
assert len(registration_calls) == 2
assert registration_calls[0].source_frame_count == 4
assert registration_calls[0].target_frame_ids == [6.0, 7.0, 8.0, 9.0]
assert registration_calls[1].source_frame_count == 4
assert registration_calls[1].target_frame_ids == [0.0, 1.0, 2.0, 3.0]
```

Also assert that the estimator rejects an image/manifest length mismatch and
a call-time keep ratio different from its configured ratio.

- [ ] **Step 6: Run joint-estimator tests and verify RED**

Run:

```bash
pytest -q tests/test_loop_constraint_estimation.py
```

Expected: failures show that `JointPi3AlignmentEstimator` is missing.

- [ ] **Step 7: Implement joint inference and two cached-to-joint registrations**

Implement the estimator with validated constructor state and this call flow:

```python
def __call__(self, cache_a, cache_b, candidate, keep_ratio):
    ratio = validate_confidence_keep_ratio(keep_ratio)
    if not math.isclose(
        ratio,
        self.confidence_keep_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "strategy keep ratio does not match joint estimator configuration"
        )
    range_a = centered_frame_range(
        cache_a.frame_start,
        cache_a.frame_end,
        candidate.frame_a,
        self.chunk_size,
    )
    range_b = centered_frame_range(
        cache_b.frame_start,
        cache_b.frame_end,
        candidate.frame_b,
        self.chunk_size,
    )
    images_a = self.images[slice(*range_a)]
    images_b = self.images[slice(*range_b)]
    joint_images = torch.cat((images_a, images_b), dim=0).to(
        self.inference_device
    )
    prediction = self._predict(joint_images)
    side_a_count = range_a[1] - range_a[0]
    joint_a = self._prediction_side(prediction, 0, side_a_count)
    joint_b = self._prediction_side(
        prediction,
        side_a_count,
        side_a_count + range_b[1] - range_b[0],
    )
    alignment_a = self._align_side(cache_a, range_a, joint_a, "side A")
    alignment_b = self._align_side(cache_b, range_b, joint_b, "side B")
    return alignment_a, alignment_b
```

`_predict` must use `torch.no_grad()`, use autocast only for float16/bfloat16,
require `local_points`, `camera_poses`, and `conf`, squeeze exactly one model
batch dimension, and move those tensors to CPU. `_align_side` must select
confidence independently on cached and joint predictions, intersect masks,
call `register_adjacent_windows(cached, joint, cached_pose, joint_pose, mask)`,
and validate the returned Sim(3).

- [ ] **Step 8: Run estimator tests and verify GREEN**

Run:

```bash
pytest -q tests/test_loop_constraint_estimation.py
```

Expected: all estimator tests pass with exactly one model call and two registrations.

- [ ] **Step 9: Commit the shared estimator**

```bash
git add loop_closure/constraint_estimation.py tests/test_loop_constraint_estimation.py
git commit -m "fix: restore joint Pi3 loop alignment"
```

---

### Task 2: Require Shared Estimation in Both Loop Strategies

**Files:**
- Modify: `loop_closure/methods/base.py:237-248`
- Modify: `loop_closure/methods/traditional.py:205-323`
- Modify: `loop_closure/methods/corrected.py:264-380`
- Modify: `tests/test_traditional_loop_method.py:165-247`
- Modify: `tests/test_corrected_loop_method.py:231-315`
- Modify: `tests/test_loop_method_contracts.py:120-155`

**Interfaces:**
- Consumes: estimator callable produced by Task 1.
- Produces: `build_constraints(caches, candidates, *, constraint_estimator=None) -> list[LoopConstraint]` on both strategies.
- Guarantees: one accepted constraint per ordered window pair; `ValueError` rejects only its candidate; all other exception classes propagate.

- [ ] **Step 1: Write failing strategy requirement and conversion tests**

For each concrete strategy, construct two caches and one cross-window
candidate. Assert that no estimator is rejected:

```python
with pytest.raises(ValueError, match="joint constraint estimator"):
    strategy.build_constraints(caches, (candidate,))
```

Inject literal alignments and verify method-specific conversion:

```python
alignment_a = identity_sim3(scale=2.0)
alignment_b = identity_sim3(scale=6.0)
strategy = strategy_fixture(
    constraint_estimator=lambda *arguments: (alignment_a, alignment_b)
)
constraint = strategy.build_constraints(caches, (candidate,))[0]
assert torch.as_tensor(constraint.measurement[0]).item() == pytest.approx(3.0)
```

For corrected, compare the returned measurement against a hand-derived
identity-rotation scale case using cache absolute scales and literal expected
scale, not by calling `build_local_loop_constraint` in the assertion.

- [ ] **Step 2: Write failing shared failure-isolation and deduplication tests**

Parameterize over traditional and corrected. Supply three candidates:
the first estimator call raises `ValueError("no mutual confidence")`, the
second succeeds, and the third maps to the same ordered window pair.
Assert one constraint remains and it belongs to the second candidate.

Add a separate estimator that raises `RuntimeError("CUDA out of memory")` and
assert that exact runtime error escapes `build_constraints`.

- [ ] **Step 3: Run strategy tests and verify RED**

Run:

```bash
pytest -q \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py \
  tests/test_loop_method_contracts.py
```

Expected: direct fallback tests fail because the current strategies still
register cached A directly to cached B, do not require the shared estimator,
do not isolate `ValueError`, and do not deduplicate window pairs.

- [ ] **Step 4: Remove both direct-pair implementations**

Delete `_estimate_direct_pair` from traditional and corrected. Remove their
loop-constraint imports of `register_adjacent_windows` and confidence mask
helpers while preserving sequential window-registration imports.

Resolve the estimator as:

```python
estimator = constraint_estimator or self.constraint_estimator
```

For a cross-window candidate with no estimator, raise:

```python
raise ValueError(
    f"{self.name.value} loop constraints require a joint constraint estimator"
)
```

- [ ] **Step 5: Add candidate isolation and pair deduplication**

Use candidate order as the stable precedence:

```python
constraints = []
seen_pairs = set()
for candidate in candidates:
    try:
        cache_a = self._cache_for_frame(caches, candidate.frame_a)
        cache_b = self._cache_for_frame(caches, candidate.frame_b)
    except ValueError as error:
        logger.warning(
            "Skipping %s loop candidate frames=%s->%s: %s",
            self.name.value,
            candidate.frame_a,
            candidate.frame_b,
            error,
        )
        continue
    pair = (cache_a.window_index, cache_b.window_index)
    if pair[0] == pair[1] or pair in seen_pairs:
        continue
    seen_pairs.add(pair)
    if estimator is None:
        raise ValueError(
            f"{self.name.value} loop constraints require "
            "a joint constraint estimator"
        )
    try:
        alignment_a, alignment_b = estimator(
            cache_a,
            cache_b,
            candidate,
            self.registration_confidence_keep_ratio,
        )
        measurement = self._convert_constraint(
            cache_a,
            cache_b,
            alignment_a,
            alignment_b,
        )
        validate_sim3(measurement, context=f"{self.name.value} loop measurement")
    except ValueError as error:
        logger.warning(
            "Skipping %s loop candidate frames=%s->%s: %s",
            self.name.value,
            candidate.frame_a,
            candidate.frame_b,
            error,
        )
        continue
    constraints.append(
        LoopConstraint(
            window_a=pair[0],
            window_b=pair[1],
            measurement=measurement,
            candidate=candidate,
        )
    )
return constraints
```

The missing-estimator case must be checked before entering the candidate-local
catch so configuration errors fail the run rather than silently producing a
no-loop result. Candidate mapping and candidate measurement each have their
own `ValueError` isolation boundary. A window pair is marked as seen before
estimation so a later duplicate cannot replace the first detected candidate.

- [ ] **Step 6: Preserve the two conversion formulas**

Traditional:

```python
measurement = compute_sim3_ab(alignment_a, alignment_b)
```

Corrected:

```python
measurement = build_local_loop_constraint(
    cache_a.loop_state["sim3_abs"],
    cache_b.loop_state["sim3_abs"],
    alignment_a,
    alignment_b,
)
```

Extend `LoopClosureStrategy.build_constraints` in `base.py` with the explicit
keyword argument so runner and evaluation code share one contract.

- [ ] **Step 7: Run strategy tests and verify GREEN**

Run:

```bash
pytest -q \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py \
  tests/test_loop_method_contracts.py
```

Expected: both strategies pass conversion, isolation, deduplication, and
system-error propagation tests.

- [ ] **Step 8: Commit strategy integration**

```bash
git add \
  loop_closure/methods/base.py \
  loop_closure/methods/traditional.py \
  loop_closure/methods/corrected.py \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py \
  tests/test_loop_method_contracts.py
git commit -m "fix: share loop alignment across strategies"
```

---

### Task 3: Wire the Modular Runner and Legacy Evaluation Entry Point

**Files:**
- Modify: `pipeline/runner.py:1-180,260-375`
- Modify: `eval_launch.py:1-184`
- Modify: `tests/test_pipeline_runner.py:30-280`
- Create: `tests/test_eval_launch_lc.py`

**Interfaces:**
- Consumes: `JointPi3AlignmentEstimator` from Task 1.
- Produces: `PipelineDependencies.build_constraint_estimator`.
- Guarantees: factory receives the active model, exact image tensor and manifest, `loop.constraint.chunk_size`, `loop.registration.confidence_keep_ratio`, inference device, and resolved dtype.

- [ ] **Step 1: Write failing runner dependency-wiring tests**

Extend `RecordingState` with estimator records and make the recording loop
strategy accept the explicit estimator:

```python
constraint_estimators: list[object] = field(default_factory=list)
estimator_factory_kwargs: list[dict[str, object]] = field(default_factory=list)

def build_constraints(
    self,
    caches,
    candidates,
    *,
    constraint_estimator=None,
):
    self.state.constraint_estimators.append(constraint_estimator)
    return []
```

Make a dependency fixture return one candidate and record the estimator
factory kwargs. Assert identity for model/images/manifest and literal resolved
values:

```python
assert kwargs["model"] is loaded_model
assert kwargs["images"] is loaded_images
assert kwargs["manifest"] is state.inference_manifests[0]
assert kwargs["chunk_size"] == 20
assert kwargs["confidence_keep_ratio"] == pytest.approx(0.30)
assert kwargs["inference_device"] == "cpu"
assert kwargs["dtype"] is torch.float32
assert state.constraint_estimators == [sentinel_estimator]
```

Strengthen the existing no-candidate test to assert the estimator factory has
zero calls.

- [ ] **Step 2: Run runner wiring tests and verify RED**

Run:

```bash
pytest -q tests/test_pipeline_runner.py
```

Expected: constructing `PipelineDependencies` with
`build_constraint_estimator` fails because that dependency does not exist.

- [ ] **Step 3: Implement conditional runner construction**

Add:

```python
build_constraint_estimator: Callable = JointPi3AlignmentEstimator
```

After detection:

```python
constraint_estimator = (
    dependencies.build_constraint_estimator(
        model=model,
        images=images,
        manifest=manifest,
        chunk_size=config.loop.constraint.chunk_size,
        confidence_keep_ratio=(
            config.loop.registration.confidence_keep_ratio
        ),
        inference_device=config.model.inference_device,
        dtype=resolve_model_dtype(config.model.dtype),
    )
    if candidates
    else None
)
constraints = loop_strategy.build_constraints(
    caches,
    candidates,
    constraint_estimator=constraint_estimator,
)
```

- [ ] **Step 4: Run runner wiring tests and verify GREEN**

Run:

```bash
pytest -q tests/test_pipeline_runner.py
```

Expected: factory wiring passes and no-candidate execution constructs no estimator.

- [ ] **Step 5: Write failing legacy evaluation wiring test**

Use a real `ImageManifest`, image tensor, fake engine with `delegate`,
`pipeline_config`, and a recording strategy. Patch `run_windows` and
`JointPi3AlignmentEstimator`, call `_run_modular_evaluation`, and assert the
recording strategy receives the estimator instance when one candidate exists.
The test must also assert no estimator is constructed for
`detect_loops=False`.

- [ ] **Step 6: Run legacy evaluation test and verify RED**

Run:

```bash
pytest -q tests/test_eval_launch_lc.py
```

Expected: the recording strategy receives no estimator because
`eval_launch.py` does not construct one.

- [ ] **Step 7: Wire the same estimator in legacy evaluation**

Construct only when `candidates` is non-empty:

```python
constraint_estimator = (
    JointPi3AlignmentEstimator(
        model=model.delegate,
        images=imgs,
        manifest=manifest,
        chunk_size=config.loop.constraint.chunk_size,
        confidence_keep_ratio=(
            config.loop.registration.confidence_keep_ratio
        ),
        inference_device=config.model.inference_device,
        dtype=resolve_model_dtype(config.model.dtype),
    )
    if candidates
    else None
)
constraints = model.loop_strategy.build_constraints(
    caches,
    candidates,
    constraint_estimator=constraint_estimator,
)
```

Expose the dtype conversion as a public `resolve_model_dtype` helper in
`pipeline.runner` and use it in both runner paths to avoid duplicated mappings.

- [ ] **Step 8: Run runner and evaluation tests together**

Run:

```bash
pytest -q tests/test_pipeline_runner.py tests/test_eval_launch_lc.py
```

Expected: both entry points pass and share the same estimator class and resolved parameters.

- [ ] **Step 9: Commit entry-point wiring**

```bash
git add pipeline/runner.py eval_launch.py tests/test_pipeline_runner.py tests/test_eval_launch_lc.py
git commit -m "fix: wire joint loop estimator into runners"
```

---

### Task 4: Documentation and Regression Verification

**Files:**
- Modify: `docs/pipeline-configuration.md:90-125`

**Interfaces:**
- Consumes: the completed estimator, strategy, and runner behavior.
- Produces: verified unit suite, six-method matrix smoke test, and cloud acceptance commands.

- [ ] **Step 1: Document active constraint parameters**

Add:

```markdown
`loop.constraint.chunk_size` is the maximum number of consecutive frames
selected around each candidate on each side of joint Pi3 inference. The
selection is centered, shifts at cache boundaries, and never crosses the
selected cache range. `loop.registration.confidence_keep_ratio` is applied
independently to cached and joint predictions before their confidence masks
are intersected.
```

- [ ] **Step 2: Run focused loop and runner tests**

Run:

```bash
pytest -q \
  tests/test_loop_constraint_estimation.py \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py \
  tests/test_loop_method_contracts.py \
  tests/test_pipeline_runner.py \
  tests/test_eval_launch_lc.py
```

Expected: all focused tests pass with no warnings introduced by the repair.

- [ ] **Step 3: Run the full unit suite**

Run:

```bash
pytest -q
```

Expected: the complete repository test suite passes.

- [ ] **Step 4: Run the six-combination pipeline matrix smoke test**

Run:

```bash
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/test.yaml \
  --dry-run
```

Expected: depth, geometry, and atomic each resolve with traditional and corrected loop methods.

- [ ] **Step 5: Inspect scope and generated files**

Run:

```bash
git diff --check
git status --short
git diff --stat origin/codex/modular-segmentation-loop-integration...HEAD
```

Expected: only planned source, test, and documentation files are staged or
committed; pre-existing generated `_segmentation_cy.cpp` and `fast_seg.cpp`
remain untracked and untouched.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/pipeline-configuration.md
git commit -m "docs: explain joint loop constraint parameters"
```

- [ ] **Step 7: Push the repaired branch**

```bash
git push origin codex/modular-segmentation-loop-integration
```

- [ ] **Step 8: Run cloud acceptance in increasing cost order**

On the cloud, first clone or update the branch, then run unit tests. Run KITTI
00 with depth/traditional and depth/corrected before the full segmentation ×
loop matrix. Preserve each result directory and verify `run_summary.json`,
`resolved_config.yaml`, `loop_candidates.json`, `loop_constraints.json`, and
`pred_traj.txt` before trajectory evaluation.

Acceptance requires the three traditional segmentation variants to produce
identical trajectories for the same loop parameters, both loop methods to
write complete outputs, and the prior approximately `61 m` KITTI 00
traditional ATE regression to be absent.
