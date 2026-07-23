# Loop Closure Constraint Coordinate Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the Sim(3) coordinate semantics of loop constraints built from globally corrected caches, while preserving the first-stage pipeline, the 30% confidence rule, and the existing optimizer.

**Architecture:** Keep `sim3_abs` as the transform from each raw window coordinate system into the first-stage global coordinate system. Convert the two joint-Pi3 global alignments back into the optimizer's required window-local relative measurement before optimization. Add mathematical validity checks and diagnostics without introducing empirical loop-rejection thresholds.

**Tech Stack:** Python 3.11, PyTorch, PyPose, NumPy, SciPy, pytest.

## Global Constraints

- Keep `registration_top_confidence_ratio` at `0.3`.
- Do not change image discovery, natural sorting, segmentation, depth refinement, or anchor propagation.
- Do not change SALAD similarity thresholds, `top_k`, or NMS.
- Do not change the Sim(3) optimizer residual, solver, damping, convergence logic, or edge weights.
- Reject only mathematically invalid Sim(3) values: non-finite components or scale less than or equal to zero.
- Preserve the no-loop fallback and the existing output trajectory format.
- Do not modify the unrelated untracked files `inference_engine/utils/_segmentation_cy.cpp` and `inference_engine/utils/fast_seg.cpp`.

---

## File Structure

- Modify `loop_closure/loop_closure.py`
  - Own the conversion from global loop alignments to a window-local loop constraint.
  - Validate Sim(3) values.
  - Preserve scored loop-candidate metadata and print per-constraint diagnostics.
- Reuse `loop_closure/loop_model.py`
  - Call its existing `save_results()` implementation so accepted SALAD frame pairs and similarities are persisted through `loop_closures.txt`.
- Modify `tests/test_loop_closure_pipeline.py`
  - Prove the old global-coordinate formula is wrong with non-unit transforms.
  - Cover invalid Sim(3), fallback, candidate persistence, and diagnostic output.
- Modify `docs/superpowers/specs/2026-07-23-loop-closure-corrected-pipeline-design.md`
  - Replace the obsolete loop-constraint formula with the corrected coordinate conversion.
- Use `docs/superpowers/specs/2026-07-23-loop-closure-constraint-coordinate-fix-design.md`
  - Treat this approved Chinese design as the source of truth.

---

### Task 1: Reproduce the coordinate-semantic bug with non-unit Sim(3)

**Files:**
- Modify: `tests/test_loop_closure_pipeline.py`

**Interfaces:**
- Consumes: `LoopClosureEngine.process_loops(raw_predictions)`.
- Produces: a failing regression test proving that global alignment correction must be conjugated by `sim3_abs_a` and `sim3_abs_b`.

- [ ] **Step 1: Add inverse Sim(3) support to the isolated test module**

Add this helper next to `compose_sim3()`:

```python
def inverse_sim3(transform):
    scale, rotation, translation = transform
    inverse_scale = 1.0 / scale
    inverse_rotation = rotation.T
    inverse_translation = (
        -inverse_scale * inverse_rotation @ translation
    )
    return inverse_scale, inverse_rotation, inverse_translation
```

Expose it from `fake_geometry` in `load_loop_engine_module()`:

```python
fake_geometry.closed_form_inverse_sim3 = inverse_sim3
```

- [ ] **Step 2: Change the existing scale-only expectation to the local-coordinate result**

Rename:

```python
test_loop_constraints_use_corrected_cache_points_and_ab_direction
```

to:

```python
test_loop_constraints_convert_global_alignments_to_local_measurement
```

The fixture has:

```text
G_A = scale 1
G_B = scale 2
L_A = scale 2
L_B = scale 6
```

The old implementation returns `L_B / L_A = 3`. The optimizer-compatible result is:

```text
inverse(G_B) * (L_B / L_A) * G_A = 1.5
```

Change the final assertion to:

```python
assert_sim3_close(constraint_ab, make_sim3(1.5))
```

- [ ] **Step 3: Add a nontrivial pure-transform regression test**

Add:

```python
def rotation_z(degrees):
    radians = torch.deg2rad(torch.tensor(float(degrees)))
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack((
        torch.stack((cosine, -sine, zero)),
        torch.stack((sine, cosine, zero)),
        torch.stack((zero, zero, one)),
    ))


def test_build_local_loop_constraint_recovers_joint_local_measurement(
    monkeypatch,
    tmp_path,
):
    module, _, _, _ = make_engine(monkeypatch, tmp_path)
    sim3_abs_a = (
        1.4,
        rotation_z(25.0),
        torch.tensor([3.0, -2.0, 1.0]),
    )
    sim3_abs_b = (
        0.8,
        rotation_z(-35.0),
        torch.tensor([-4.0, 1.5, 0.5]),
    )
    local_alignment_a = (
        1.1,
        rotation_z(12.0),
        torch.tensor([0.4, -0.2, 0.7]),
    )
    local_alignment_b = (
        0.9,
        rotation_z(-18.0),
        torch.tensor([-0.3, 0.8, -0.1]),
    )
    global_alignment_a = compose_sim3(
        sim3_abs_a,
        local_alignment_a,
    )
    global_alignment_b = compose_sim3(
        sim3_abs_b,
        local_alignment_b,
    )
    expected = compose_sim3(
        local_alignment_b,
        inverse_sim3(local_alignment_a),
    )

    actual = module.build_local_loop_constraint(
        sim3_abs_a,
        sim3_abs_b,
        global_alignment_a,
        global_alignment_b,
    )

    assert_sim3_close(actual, expected)
```

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
pytest -q \
  tests/test_loop_closure_pipeline.py::test_loop_constraints_convert_global_alignments_to_local_measurement \
  tests/test_loop_closure_pipeline.py::test_build_local_loop_constraint_recovers_joint_local_measurement
```

Expected:

- The integration assertion reports scale `3.0` instead of `1.5`.
- The pure-transform test fails because `build_local_loop_constraint` does not exist.

- [ ] **Step 5: Commit only after Task 2 turns these tests green**

Do not commit the RED state separately.

---

### Task 2: Convert global loop alignments into local optimizer measurements

**Files:**
- Modify: `loop_closure/loop_closure.py`
- Test: `tests/test_loop_closure_pipeline.py`

**Interfaces:**
- Consumes:
  - `sim3_abs_a: tuple[scale, rotation, translation]`
  - `sim3_abs_b: tuple[scale, rotation, translation]`
  - `global_alignment_a: tuple[scale, rotation, translation]`
  - `global_alignment_b: tuple[scale, rotation, translation]`
- Produces:
  - `build_local_loop_constraint(...) -> tuple[scale, rotation, translation]`
  - `C_AB = inverse(G_B) compose L_B compose inverse(L_A) compose G_A`

- [ ] **Step 1: Import the existing inverse operation**

Replace the geometry import with:

```python
from inference_engine.utils.geometry import (
    accumulate_sim3,
    closed_form_inverse_sim3,
)
```

- [ ] **Step 2: Implement the conversion helper**

Add above `LoopClosureEngine`:

```python
def build_local_loop_constraint(
        sim3_abs_a,
        sim3_abs_b,
        global_alignment_a,
        global_alignment_b,
):
    global_correction = accumulate_sim3(
        global_alignment_b,
        closed_form_inverse_sim3(*global_alignment_a),
    )
    return accumulate_sim3(
        closed_form_inverse_sim3(*sim3_abs_b),
        accumulate_sim3(
            global_correction,
            sim3_abs_a,
        ),
    )
```

- [ ] **Step 3: Use the helper in `process_loops()`**

Replace:

```python
s_ab, R_ab, t_ab = compute_sim3_ab(
    (s_a, R_a, t_a),
    (s_b, R_b, t_b),
)
```

with:

```python
s_ab, R_ab, t_ab = build_local_loop_constraint(
    raw_predictions[chunk_idx_a]["sim3_abs"],
    raw_predictions[chunk_idx_b]["sim3_abs"],
    (s_a, R_a, t_a),
    (s_b, R_b, t_b),
)
```

- [ ] **Step 4: Run the two coordinate tests and verify GREEN**

Run the same command from Task 1.

Expected: `2 passed`.

- [ ] **Step 5: Run the loop pipeline test file**

Run:

```bash
pytest -q tests/test_loop_closure_pipeline.py
```

Expected: all tests in the file pass.

- [ ] **Step 6: Commit the coordinate fix**

```bash
git add loop_closure/loop_closure.py tests/test_loop_closure_pipeline.py
git commit -m "fix: convert loop constraints to local coordinates"
```

---

### Task 3: Reject only mathematically invalid Sim(3) constraints

**Files:**
- Modify: `tests/test_loop_closure_pipeline.py`
- Modify: `loop_closure/loop_closure.py`

**Interfaces:**
- Produces:
  - `validate_sim3(transform, context) -> None`
  - Raises `ValueError` for invalid shape, non-finite values, or non-positive scale.
- Does not add empirical scale, rotation, translation, or residual thresholds.

- [ ] **Step 1: Write parameterized failing tests**

Add:

```python
@pytest.mark.parametrize(
    "invalid_alignment",
    [
        (
            float("nan"),
            torch.eye(3),
            torch.zeros(3),
        ),
        (
            0.0,
            torch.eye(3),
            torch.zeros(3),
        ),
        (
            -1.0,
            torch.eye(3),
            torch.zeros(3),
        ),
        (
            1.0,
            torch.full((3, 3), float("inf")),
            torch.zeros(3),
        ),
    ],
)
def test_process_loops_skips_mathematically_invalid_sim3(
    monkeypatch,
    tmp_path,
    invalid_alignment,
):
    module, engine, _, _ = make_engine(monkeypatch, tmp_path)
    caches = make_two_caches()
    engine.loop_list = [(0, 2)]
    monkeypatch.setattr(
        module,
        "process_loop_list",
        lambda *args, **kwargs: [
            (0, (0, 1), 1, (1, 2)),
        ],
    )
    engine.process_single_chunk = lambda *args, **kwargs: {
        "local_points": torch.ones((2, 1, 1, 3)),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
        "conf": torch.ones((2, 1, 1)),
    }
    alignments = iter((invalid_alignment, make_sim3(1.0)))
    monkeypatch.setattr(
        module,
        "register_adjacent_windows",
        lambda *args: next(alignments),
    )

    constraints = engine.process_loops(caches)

    assert constraints == []
```

- [ ] **Step 2: Run the parameterized test and verify RED**

Run:

```bash
pytest -q \
  tests/test_loop_closure_pipeline.py::test_process_loops_skips_mathematically_invalid_sim3
```

Expected: invalid constraints are currently appended or fail inside composition instead of being skipped cleanly.

- [ ] **Step 3: Implement exact mathematical validation**

Add above `build_local_loop_constraint()`:

```python
def validate_sim3(transform, context):
    if not isinstance(transform, (tuple, list)) or len(transform) != 3:
        raise ValueError(f"{context} must contain scale, rotation, translation")

    scale, rotation, translation = transform
    scale_tensor = torch.as_tensor(scale)
    rotation_tensor = torch.as_tensor(rotation)
    translation_tensor = torch.as_tensor(translation)

    if scale_tensor.numel() != 1:
        raise ValueError(f"{context} scale must be scalar")
    if tuple(rotation_tensor.shape) != (3, 3):
        raise ValueError(f"{context} rotation must have shape (3, 3)")
    if translation_tensor.numel() != 3:
        raise ValueError(f"{context} translation must contain 3 values")
    if not torch.isfinite(scale_tensor).all():
        raise ValueError(f"{context} scale is not finite")
    if not torch.isfinite(rotation_tensor).all():
        raise ValueError(f"{context} rotation is not finite")
    if not torch.isfinite(translation_tensor).all():
        raise ValueError(f"{context} translation is not finite")
    if float(scale_tensor.detach().cpu().item()) <= 0.0:
        raise ValueError(f"{context} scale must be positive")
```

Validate all four inputs and the output inside `build_local_loop_constraint()`:

```python
for context, transform in (
    ("sim3_abs_a", sim3_abs_a),
    ("sim3_abs_b", sim3_abs_b),
    ("global_alignment_a", global_alignment_a),
    ("global_alignment_b", global_alignment_b),
):
    validate_sim3(transform, context)

constraint = ...
validate_sim3(constraint, "loop_constraint_ab")
return constraint
```

- [ ] **Step 4: Skip invalid candidates with context**

Wrap the conversion in `process_loops()`:

```python
try:
    loop_constraint = build_local_loop_constraint(...)
except ValueError as error:
    print(
        "Skipping loop candidate "
        f"{chunk_idx_a}->{chunk_idx_b}: {error}"
    )
    continue
```

Append `loop_constraint` only after validation succeeds.

- [ ] **Step 5: Run invalid, fallback, and no-loop tests**

Run:

```bash
pytest -q \
  tests/test_loop_closure_pipeline.py::test_process_loops_skips_mathematically_invalid_sim3 \
  tests/test_loop_closure_pipeline.py::test_invalid_loop_candidates_do_not_invoke_optimizer \
  tests/test_loop_closure_pipeline.py::test_no_loop_returns_original_absolute_transforms
```

Expected: all tests pass.

- [ ] **Step 6: Commit validation**

```bash
git add loop_closure/loop_closure.py tests/test_loop_closure_pipeline.py
git commit -m "fix: reject invalid loop sim3 constraints"
```

---

### Task 4: Persist loop scores and print constraint diagnostics

**Files:**
- Modify: `tests/test_loop_closure_pipeline.py`
- Modify: `loop_closure/loop_closure.py`
- Reuse without modification: `loop_closure/loop_model.py`

**Interfaces:**
- Consumes: `LoopDetector.loop_closures` entries `(frame_a, frame_b, similarity)`.
- Produces:
  - Existing `loop_closures.txt` containing accepted pairs and scores.
  - Per-constraint logs containing frame pair, window pair, measurement metrics, and initial graph residual metrics.
  - Graph count log before optimization.
- Does not change candidate acceptance.

- [ ] **Step 1: Extend the recording detector and write a failing persistence test**

Give `RecordingDetector`:

```python
self.loop_closures = []
self.save_calls = 0
```

Make `run()` populate scored candidates from configured pairs:

```python
self.loop_closures = [
    (frame_a, frame_b, 0.9)
    for frame_a, frame_b in self.loop_list
]
```

Add:

```python
def save_results(self):
    self.save_calls += 1
```

Add:

```python
def test_get_loop_pairs_persists_scored_candidates(
    monkeypatch,
    tmp_path,
):
    _, engine, detector, _ = make_engine(monkeypatch, tmp_path)
    detector.loop_list = [(2, 0)]

    engine.get_loop_pairs()

    assert detector.save_calls == 1
```

- [ ] **Step 2: Write a failing diagnostic test**

Add:

```python
def test_process_loops_prints_constraint_diagnostics(
    monkeypatch,
    tmp_path,
    capsys,
):
    module, engine, _, _ = make_engine(monkeypatch, tmp_path)
    caches = make_two_caches()
    engine.loop_list = [(2, 0)]
    engine.loop_candidates = [(2, 0, 0.91)]
    monkeypatch.setattr(
        module,
        "process_loop_list",
        lambda *args, **kwargs: [
            (1, (1, 2), 0, (0, 1)),
        ],
    )
    engine.process_single_chunk = lambda *args, **kwargs: {
        "local_points": torch.ones((2, 1, 1, 3)),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
        "conf": torch.ones((2, 1, 1)),
    }
    alignments = iter((make_sim3(1.0), make_sim3(1.0)))
    monkeypatch.setattr(
        module,
        "register_adjacent_windows",
        lambda *args: next(alignments),
    )

    engine.process_loops(caches)
    output = capsys.readouterr().out

    assert "frames=2->0" in output
    assert "windows=1->0" in output
    assert "similarity=0.910000" in output
    assert "measurement_scale=" in output
    assert "measurement_rotation_deg=" in output
    assert "measurement_translation_norm=" in output
    assert "initial_residual_scale_log_abs=" in output
    assert "initial_residual_rotation_deg=" in output
    assert "initial_residual_translation_norm=" in output
```

Add:

```python
def test_run_prints_graph_constraint_counts(
    monkeypatch,
    tmp_path,
    capsys,
):
    _, engine, _, _ = make_engine(monkeypatch, tmp_path)
    caches = make_two_caches()

    def set_loop_pairs():
        engine.loop_list = [(2, 0)]

    engine.get_loop_pairs = set_loop_pairs
    engine.process_loops = lambda predictions: [
        (1, 0, make_sim3(2.0)),
    ]

    engine.run(caches)
    output = capsys.readouterr().out

    assert "sequential_constraints=1" in output
    assert "loop_constraints=1" in output
```

- [ ] **Step 3: Run diagnostic tests and verify RED**

Run:

```bash
pytest -q \
  tests/test_loop_closure_pipeline.py::test_get_loop_pairs_persists_scored_candidates \
  tests/test_loop_closure_pipeline.py::test_process_loops_prints_constraint_diagnostics \
  tests/test_loop_closure_pipeline.py::test_run_prints_graph_constraint_counts
```

Expected: persistence and diagnostic strings are absent.

- [ ] **Step 4: Persist the detector output**

In `LoopClosureEngine.get_loop_pairs()`:

```python
self.loop_detector.run()
self.loop_detector.save_results()
self.loop_candidates = list(
    self.loop_detector.loop_closures or []
)
self.loop_list = self.loop_detector.get_loop_list()
```

Initialize `self.loop_candidates = []` in `__init__`.

In `LoopDetector.run()`, call `self.save_results()` only through the engine; do not duplicate the write inside `run()`.

- [ ] **Step 5: Preserve frame pairs and scores while mapping candidates**

In `process_loops()`, process each scored candidate separately:

```python
candidates = (
    self.loop_candidates
    if self.loop_candidates
    else [
        (frame_a, frame_b, float("nan"))
        for frame_a, frame_b in self.loop_list
    ]
)
self.loop_results = []
for frame_a, frame_b, similarity in candidates:
    mapped = process_loop_list(
        self.chunk_indices,
        [(frame_a, frame_b)],
        half_window=(
            self.config["Model"]["loop_chunk_size"] // 2
        ),
    )
    if not mapped:
        continue
    self.loop_results.append(
        (*mapped[0], frame_a, frame_b, float(similarity))
    )
self.loop_results = remove_duplicates(self.loop_results)
```

The first four tuple fields remain unchanged, so existing range and duplicate logic remains valid. Fields 4–6 carry the frame pair and score.

At the beginning of the constraint-building loop over
`self.loop_predict_list`, unpack the metadata:

```python
frame_a = item[0][4]
frame_b = item[0][5]
similarity = item[0][6]
```

- [ ] **Step 6: Add deterministic Sim(3) metrics**

Add:

```python
def sim3_metrics(transform):
    scale, rotation, translation = transform
    scale_value = float(torch.as_tensor(scale).detach().cpu().item())
    rotation_tensor = torch.as_tensor(
        rotation,
        dtype=torch.float64,
    )
    cosine = torch.clamp(
        (torch.trace(rotation_tensor) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    rotation_degrees = float(
        torch.rad2deg(torch.acos(cosine)).item()
    )
    translation_norm = float(
        torch.linalg.vector_norm(
            torch.as_tensor(
                translation,
                dtype=torch.float64,
            ).reshape(-1)
        ).item()
    )
    return {
        "scale": scale_value,
        "scale_log_abs": abs(math.log(scale_value)),
        "rotation_deg": rotation_degrees,
        "translation_norm": translation_norm,
    }
```

Import `math`.

Compute the initial residual:

```python
sim3_abs_a = raw_predictions[chunk_idx_a]["sim3_abs"]
sim3_abs_b = raw_predictions[chunk_idx_b]["sim3_abs"]
initial_residual = accumulate_sim3(
    loop_constraint,
    accumulate_sim3(
        closed_form_inverse_sim3(*sim3_abs_a),
        sim3_abs_b,
    ),
)
```

Print one line per accepted constraint with the exact field names asserted in Step 2.

Use:

```python
measurement_metrics = sim3_metrics(loop_constraint)
residual_metrics = sim3_metrics(initial_residual)
print(
    "Loop constraint: "
    f"frames={frame_a}->{frame_b}, "
    f"windows={chunk_idx_a}->{chunk_idx_b}, "
    f"similarity={similarity:.6f}, "
    f"measurement_scale={measurement_metrics['scale']:.6f}, "
    "measurement_rotation_deg="
    f"{measurement_metrics['rotation_deg']:.6f}, "
    "measurement_translation_norm="
    f"{measurement_metrics['translation_norm']:.6f}, "
    "initial_residual_scale_log_abs="
    f"{residual_metrics['scale_log_abs']:.6f}, "
    "initial_residual_rotation_deg="
    f"{residual_metrics['rotation_deg']:.6f}, "
    "initial_residual_translation_norm="
    f"{residual_metrics['translation_norm']:.6f}"
)
```

- [ ] **Step 7: Print graph counts before optimization**

Immediately before `self.loop_optimizer.optimize(...)`:

```python
print(
    "Loop graph: "
    f"sequential_constraints={len(sequential_edges)}, "
    f"loop_constraints={len(loop_constraints)}"
)
```

- [ ] **Step 8: Run diagnostic and loop tests**

Run:

```bash
pytest -q tests/test_loop_closure_pipeline.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit diagnostics**

```bash
git add \
  loop_closure/loop_closure.py \
  tests/test_loop_closure_pipeline.py
git commit -m "feat: add loop constraint diagnostics"
```

---

### Task 5: Correct the original Chinese pipeline design

**Files:**
- Modify: `docs/superpowers/specs/2026-07-23-loop-closure-corrected-pipeline-design.md`

**Interfaces:**
- Consumes: approved coordinate-fix design.
- Produces: one consistent description across both Chinese specifications.

- [ ] **Step 1: Replace the obsolete formula**

Replace:

```text
C_AB = L_B compose inverse(L_A)
```

with:

```text
C_global = L_B compose inverse(L_A)

C_AB =
    inverse(G_B)
    compose C_global
    compose G_A
```

Explicitly state:

```text
L_A 和 L_B 的目标缓存已经处于第一阶段全局坐标系，
因此 C_global 不能直接作为优化器的窗口局部相对测量。
```

- [ ] **Step 2: Update the invariant and test sections**

Add the optimizer residual invariant:

```text
C_AB compose inverse(G_A) compose G_B == identity
```

Require a non-unit Sim(3) test rather than identity-only fixtures.

- [ ] **Step 3: Check both designs for contradictory formulas**

Run:

```bash
rg -n \
  "C_AB = L_B compose inverse\\(L_A\\)|C_global|inverse\\(G_B\\)" \
  docs/superpowers/specs
```

Expected:

- No document describes `L_B compose inverse(L_A)` as the final optimizer measurement.
- Both design documents contain the `inverse(G_B)` conversion.

- [ ] **Step 4: Check Markdown whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Commit documentation**

```bash
git add \
  docs/superpowers/specs/2026-07-23-loop-closure-corrected-pipeline-design.md
git commit -m "docs: correct loop constraint coordinate formula"
```

---

### Task 6: Full verification and publication

**Files:**
- Verify all modified files.
- Do not modify unrelated untracked files.

**Interfaces:**
- Produces: a verified branch ready for the user's KITTI 00 cloud experiment.

- [ ] **Step 1: Run the coordinate regression tests**

```bash
pytest -q \
  tests/test_loop_closure_pipeline.py::test_loop_constraints_convert_global_alignments_to_local_measurement \
  tests/test_loop_closure_pipeline.py::test_build_local_loop_constraint_recovers_joint_local_measurement
```

Expected: `2 passed`.

- [ ] **Step 2: Run all loop-related tests**

```bash
pytest -q \
  tests/test_loop_closure_pipeline.py \
  tests/test_streaming_window_engine_lc_pipeline.py \
  tests/test_loop_image_manifest.py \
  tests/test_registration_confidence.py \
  tests/test_demo_lc.py \
  tests/test_eval_launch_lc.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the full test suite**

```bash
pytest -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 4: Inspect the final diff**

```bash
git diff --check
git status --short
git diff origin/codex/loop-closure-corrected-pipeline...HEAD -- \
  loop_closure/loop_closure.py \
  tests/test_loop_closure_pipeline.py \
  docs/superpowers/specs/2026-07-23-loop-closure-corrected-pipeline-design.md \
  docs/superpowers/specs/2026-07-23-loop-closure-constraint-coordinate-fix-design.md \
  docs/superpowers/plans/2026-07-23-loop-closure-constraint-coordinate-fix.md
```

Expected:

- No whitespace errors.
- Only the planned files and the two pre-existing untracked C++ files appear.
- No segmentation, anchor propagation, confidence ratio, SALAD threshold, or optimizer math changes.

- [ ] **Step 5: Push the current branch**

```bash
git push origin codex/loop-closure-corrected-pipeline
```

Expected: remote branch advances to the verified local HEAD.

- [ ] **Step 6: Provide cloud verification instructions**

Tell the user to reuse the existing KITTI 00 first-stage caches where possible, rerun loop-constraint construction and optimization, and collect:

- `loop_closures.txt`;
- per-constraint diagnostic lines;
- optimized ATE/RPE;
- the pre-optimization baseline ATE `18.323618`;
- maximum RPE frame indices to check whether residual spikes still align with window boundaries.
