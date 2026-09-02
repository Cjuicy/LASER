# Traditional Second-Global Loop Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `traditional_second_global` reconstruction mode that preserves an exact Traditional Stage 1 artifact, rebuilds adjacent and loop Sim(3) constraints from Stage 1 corrected geometry, performs a second global optimization, and writes independently evaluable Stage 1 and Stage 2 artifacts.

**Architecture:** The new mode delegates Stage 1 to the unchanged `TraditionalReconstructionMode`, records its candidates and solution, materializes overlap-preserving Stage 1 windows, and feeds a residual window graph through the corrected coordinate-safe constraint and one-delta optimization math. `PipelineRunner` unwraps a typed two-artifact result, atomically writes Stage 2 at the requested root and Stage 1 under `stage1/`, and continues returning the Stage 2 `ReconstructionArtifact`.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, OmegaConf, pytest, existing LASER joint-PI3 evidence and `Sim3LoopOptimizer`.

**Spec:** `docs/superpowers/specs/2026-09-02-traditional-second-global-loop-design.md`

## Global Constraints

- Do not modify `reconstruction/modes/traditional.py` or `loop_closure/methods/traditional.py`.
- Preserve exact Stage 1 tensor equality with standalone `traditional` by testing with `torch.equal`.
- Reuse the single SALAD candidate tuple; rerun joint PI3 evidence estimation against Stage 2 corrected window states.
- Rebuild every non-first adjacent residual edge and apply the final optimization delta to local points and camera poses exactly once.
- Introduce no new numerical hyperparameters and do not change `Sim3LoopOptimizer` mathematics.
- Keep the canonical ATE matrix at `depth|geometry|atomic x no_loop|traditional|corrected = 9` entries.
- Keep ordinary PI3 prediction cache identity unchanged.
- Preserve existing artifact schema version 1 and make both stages readable by existing artifact and ATE loaders.
- Follow test-driven development: write each behavior test, observe the expected failure, then add minimal production code.
- Do not claim empirical ATE improvement without a controlled real-data GPU run.

## File structure

- Create `loop_closure/methods/second_global.py`: expose the Stage 2 processor name while reusing corrected constraint, optimization, and one-delta aggregation semantics.
- Create `reconstruction/modes/traditional_second_global.py`: own Stage 1 recording, full-window materialization, Stage 2 residual-state construction, diagnostics, and two-stage orchestration.
- Create `tests/reconstruction/test_traditional_second_global_mode.py`: characterize Stage 1 equality, detector/evidence call ownership, residual call order, and final artifact invariants.
- Create `tests/test_second_global_loop_method.py`: verify the processor's coordinate conversion, rebuilt edge delivery, no-loop behavior, and one-delta aggregation.
- Modify `pipeline/config.py`: add the explicit reconstruction enum.
- Modify `experiments/config.py` and `experiments/matrix.py`: freeze the legacy canonical mode tuple instead of iterating over every enum member.
- Modify `pipeline/artifacts.py`: add the immutable staged result and atomic staged writer.
- Modify `reconstruction/modes/base.py`, `reconstruction/modes/__init__.py`, and `reconstruction/registry.py`: expose the new typed mode and staged return contract.
- Modify `pipeline/runner.py` and `run_reconstruction.py`: unwrap/write staged results while preserving the primary return contract and report both output paths.
- Modify focused tests under `tests/reconstruction/`, `tests/experiments/`, `tests/evaluation/`, and top-level `tests/` for config, registry, runner, artifact, CLI, and architecture boundaries.
- Modify `README.md`: document reconstruction and ATE commands for Stage 1 and Stage 2.

---

### Task 1: Add the opt-in mode without expanding existing matrices

**Files:**
- Modify: `pipeline/config.py`
- Modify: `experiments/config.py`
- Modify: `experiments/matrix.py`
- Test: `tests/test_pipeline_config.py`
- Test: `tests/experiments/test_matrix.py`
- Test: `tests/experiments/test_ate_experiment.py`

**Interfaces:**
- Consumes: existing `ReconstructionMode`, `ExperimentConfig`, and `build_matrix(EvaluationKind)`.
- Produces: `ReconstructionMode.TRADITIONAL_SECOND_GLOBAL` with value `"traditional_second_global"`, plus `CANONICAL_RECONSTRUCTION_MODES: tuple[ReconstructionMode, ...]` limited to the three legacy modes.

- [ ] **Step 1: Write failing config and matrix tests**

Add these assertions:

```python
def test_second_global_mode_is_explicitly_selectable():
    loaded = load_pipeline_config(
        RECONSTRUCTION,
        (
            "reconstruction.mode=traditional_second_global",
            "loop.optimizer.implementation=python",
        ),
    )
    assert (
        loaded.config.reconstruction.mode
        is ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
    )


def test_ate_matrix_excludes_opt_in_second_global_mode():
    entries = build_matrix(EvaluationKind.ATE)
    assert len(entries) == 9
    assert all(
        entry.reconstruction_mode
        is not ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
        for entry in entries
    )
```

Keep the legacy `ExperimentConfig.reconstruction_modes` fixture equal to:

```python
("no_loop", "traditional", "corrected")
```

- [ ] **Step 2: Run tests and verify the enum failure**

Run:

```bash
pytest -q \
  tests/test_pipeline_config.py::test_second_global_mode_is_explicitly_selectable \
  tests/experiments/test_matrix.py::test_ate_matrix_excludes_opt_in_second_global_mode
```

Expected: FAIL because `ReconstructionMode.TRADITIONAL_SECOND_GLOBAL` and the YAML value do not exist.

- [ ] **Step 3: Add the enum and freeze canonical matrices**

Add to `pipeline/config.py`:

```python
class ReconstructionMode(str, Enum):
    NO_LOOP = "no_loop"
    TRADITIONAL = "traditional"
    CORRECTED = "corrected"
    TRADITIONAL_SECOND_GLOBAL = "traditional_second_global"
```

Add to `experiments/matrix.py`:

```python
CANONICAL_RECONSTRUCTION_MODES = (
    ReconstructionMode.NO_LOOP,
    ReconstructionMode.TRADITIONAL,
    ReconstructionMode.CORRECTED,
)
```

Use `CANONICAL_RECONSTRUCTION_MODES` in `build_matrix()` and use its `.value`
tuple in `ExperimentConfig.__post_init__` instead of `tuple(ReconstructionMode)`.
Leave `build_capability_matrix()` explicitly limited to its current three
modes.

- [ ] **Step 4: Run focused config and matrix tests**

Run:

```bash
pytest -q tests/test_pipeline_config.py tests/experiments/test_matrix.py tests/experiments/test_ate_experiment.py
```

Expected: PASS, with exactly nine canonical ATE entries and successful explicit
new-mode config parsing.

- [ ] **Step 5: Commit the opt-in configuration boundary**

```bash
git add pipeline/config.py experiments/config.py experiments/matrix.py \
  tests/test_pipeline_config.py tests/experiments/test_matrix.py \
  tests/experiments/test_ate_experiment.py
git commit -m "feat: register opt-in second global mode"
```

---

### Task 2: Materialize exact full Stage 1 windows

**Files:**
- Create: `reconstruction/modes/traditional_second_global.py`
- Create: `tests/reconstruction/test_traditional_second_global_mode.py`

**Interfaces:**
- Consumes: `TraditionalWindowState`, `LoopSolution`, `accumulate_sim3`, and `apply_sim3_to_pose`.
- Produces: immutable `MaterializedTraditionalWindow` and `materialize_traditional_stage1_windows(states: Sequence[TraditionalWindowState], solution: LoopSolution, *, apply_pose_sim3: Callable = apply_sim3_to_pose) -> tuple[MaterializedTraditionalWindow, ...]`.

- [ ] **Step 1: Write failing exact-materialization tests**

Create states with a non-unit relative scale, a non-zero translation, and an
anchor mask. Use the existing processor as the reference:

```python
def test_stage1_materialization_exactly_reproduces_traditional_aggregate():
    states = (
        traditional_state(0),
        traditional_state(1, relative_scale=2.0, anchor_scale=3.0),
    )
    solution = LoopSolution(
        optimized_transforms=(identity_sim3(), identity_sim3(2.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    expected = processor_fixture().aggregate(states, solution)

    windows = materialize_traditional_stage1_windows(states, solution)
    local_points = torch.cat(
        (windows[0].local_points, windows[1].local_points[1:]),
        dim=0,
    )
    camera_poses = torch.cat(
        (windows[0].camera_poses, windows[1].camera_poses[1:]),
        dim=0,
    )
    confidence = torch.cat(
        (windows[0].confidence, windows[1].confidence[1:]),
        dim=0,
    )
    global_points = torch.einsum(
        "nij,nhwj->nhwi",
        camera_poses,
        homogenize_points(local_points),
    )[..., :3]

    assert torch.equal(local_points, expected.local_points)
    assert torch.equal(camera_poses, expected.camera_poses)
    assert torch.equal(confidence, expected.confidence)
    assert torch.equal(global_points, expected.global_points)
```

Also assert the original state tensors remain unchanged and state/solution
length mismatch raises:

```python
with pytest.raises(ValueError, match="materialization.*count"):
    materialize_traditional_stage1_windows(states, bad_solution)
```

- [ ] **Step 2: Run the materialization test and verify import failure**

Run:

```bash
pytest -q tests/reconstruction/test_traditional_second_global_mode.py -k materialization
```

Expected: collection FAIL because the new module and materialization API are
missing.

- [ ] **Step 3: Implement the immutable materializer**

Define:

```python
@dataclass(frozen=True)
class MaterializedTraditionalWindow:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    stage1_absolute_sim3: Sim3
```

Implement the baseline transform loop without changing Traditional source:

```python
reference = identity_sim3("cpu")
windows = []
for state, relative in zip(states, solution.optimized_transforms, strict=True):
    absolute = accumulate_sim3(reference, relative)
    scale, rotation, translation = absolute
    points = state.local_points.clone()
    if state.anchor_scale_mask is None:
        points = torch.as_tensor(scale, device=points.device, dtype=points.dtype) * points
    else:
        points = (
            torch.as_tensor(reference[0], device=points.device, dtype=points.dtype)
            * state.anchor_scale_mask.to(points)
            * points
        )
    poses = apply_pose_sim3(
        state.camera_poses.clone(),
        scale,
        rotation.to(state.camera_poses),
        translation.to(state.camera_poses),
    )
    windows.append(
        MaterializedTraditionalWindow(
            window_index=state.window_index,
            frame_start=state.frame_start,
            frame_end=state.frame_end,
            local_points=points,
            camera_poses=poses,
            confidence=state.confidence.clone(),
            stage1_absolute_sim3=absolute,
        )
    )
    reference = absolute
return tuple(windows)
```

Validate transform count, finite outputs, unchanged tensor shapes, consecutive
window indices, and non-empty input. Keep overlap trimming out of this helper.

- [ ] **Step 4: Run materialization and existing Traditional tests**

Run:

```bash
pytest -q \
  tests/reconstruction/test_traditional_second_global_mode.py -k materialization \
  tests/reconstruction/test_traditional_mode.py \
  tests/test_traditional_loop_method.py
```

Expected: PASS. Confirm `git diff -- reconstruction/modes/traditional.py loop_closure/methods/traditional.py` prints no diff.

- [ ] **Step 5: Commit the Stage 1 materializer**

```bash
git add reconstruction/modes/traditional_second_global.py \
  tests/reconstruction/test_traditional_second_global_mode.py
git commit -m "feat: materialize traditional stage one windows"
```

---

### Task 3: Build and optimize the Stage 2 residual graph

**Files:**
- Create: `loop_closure/methods/second_global.py`
- Modify: `reconstruction/modes/traditional_second_global.py`
- Create: `tests/test_second_global_loop_method.py`
- Modify: `tests/reconstruction/test_traditional_second_global_mode.py`

**Interfaces:**
- Consumes: `MaterializedTraditionalWindow`, `ResidualAlignmentResult`, `align_post_anchor_window`, `CorrectedLoopProcessor`, `closed_form_inverse_sim3`, and `accumulate_sim3`.
- Produces: immutable `SecondGlobalWindowState`, `build_second_global_states(windows: Sequence[MaterializedTraditionalWindow], *, overlap: int, confidence_keep_ratio: float, residual_align: Callable = align_post_anchor_window, register_adjacent: Callable = register_adjacent_windows, apply_pose_sim3: Callable = apply_sim3_to_pose) -> tuple[tuple[SecondGlobalWindowState, ...], tuple[ResidualAlignmentResult, ...]]`, and `SecondGlobalLoopProcessor`.

- [ ] **Step 1: Write failing sequential residual-state tests**

Inject two non-commuting residual transforms and record call inputs. Define
these test-local helpers first:

```python
def residual_sim3(index: int):
    angle = 0.1 * index
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return 1.0 + index, rotation, torch.tensor([float(index), 0.0, 0.0])


def assert_sim3_equal(actual, expected):
    assert torch.as_tensor(actual[0]).item() == pytest.approx(
        torch.as_tensor(expected[0]).item()
    )
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])
```

```python
def test_second_global_adjacent_pass_uses_refined_predecessor():
    calls = []

    def residual_align(**values):
        calls.append(values)
        index = len(calls)
        sim3 = residual_sim3(index)
        return ResidualAlignmentResult(
            local_points=values["current_points"] * sim3[0],
            camera_poses=apply_sim3_to_pose(values["current_poses"], *sim3),
            sim3=sim3,
            correspondence_count=1,
            abs_log_scale=abs(math.log(float(sim3[0]))),
            rotation_rad=0.1 * index,
            translation_norm=float(index),
        )

    states, results = build_second_global_states(
        materialized_windows,
        overlap=1,
        confidence_keep_ratio=1.0,
        residual_align=residual_align,
    )

    assert calls[1]["previous_points"] is states[1].local_points
    assert_sim3_equal(
        states[2].sim3_edge,
        accumulate_sim3(
            closed_form_inverse_sim3(*states[1].sim3_abs),
            states[2].sim3_abs,
        ),
    )
    assert len(results) == len(states) - 1
```

Add one-window identity, wrong index order, invalid residual output shape, and
`N - 1` residual count assertions.

- [ ] **Step 2: Run the sequential-state tests and verify missing API failure**

Run:

```bash
pytest -q tests/reconstruction/test_traditional_second_global_mode.py -k second_global_states
```

Expected: FAIL because `SecondGlobalWindowState` and
`build_second_global_states` are missing.

- [ ] **Step 3: Implement Stage 2 state construction**

Define:

```python
@dataclass(frozen=True)
class SecondGlobalWindowState:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    sim3_abs: Sim3
    sim3_edge: Sim3 | None
```

Set the first `sim3_abs` to identity and `sim3_edge` to `None`. For each later
window, call `residual_align` with the previous built state and current
materialized window, assign `sim3_abs=result.sim3`, and compute:

```python
sim3_edge = accumulate_sim3(
    closed_form_inverse_sim3(*states[-1].sim3_abs),
    result.sim3,
)
```

Validate every transform and output shape before appending the state.

- [ ] **Step 4: Write failing Stage 2 loop processor tests**

Use deterministic evidence and optimizer fakes. Define test-local
`second_global_state(index, *, sim3_abs, sim3_edge)` with two frames of finite
points/poses/confidence, `FixedEvidence` whose `estimate()` returns constructor
alignments, and this exact processor factory:

```python
def processor_fixture(optimizer=None):
    optimizer_config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    return SecondGlobalLoopProcessor(
        optimizer_config,
        optimizer=optimizer,
    )
```

Then add:

```python
def test_second_global_processor_uses_residual_node_coordinates():
    states = (
        second_global_state(
            0,
            sim3_abs=identity_sim3(),
            sim3_edge=None,
        ),
        second_global_state(
            1,
            sim3_abs=identity_sim3(2.0),
            sim3_edge=identity_sim3(2.0),
        ),
    )
    constraint = processor_fixture().build_constraints(
        states,
        (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),),
        FixedEvidence(alignment_a, alignment_b),
    )[0]
    expected = build_local_loop_constraint(
        states[1].sim3_abs,
        states[0].sim3_abs,
        alignment_a,
        alignment_b,
    )
    assert_sim3_equal(constraint.measurement, expected)


def test_second_global_optimizer_receives_rebuilt_edges_and_constraints():
    processor = processor_fixture(recording_optimizer)
    solution = processor.optimize(states, constraints)
    assert recording_optimizer.edges == [state.sim3_edge for state in states[1:]]
    assert recording_optimizer.constraints == [
        (constraint.window_a, constraint.window_b, constraint.measurement)
        for constraint in constraints
    ]
    assert len(solution.optimized_transforms) == len(states)
```

Also verify no constraints do not invoke the optimizer and aggregation applies
`optimized_abs compose inverse(state.sim3_abs)` exactly once.

- [ ] **Step 5: Run processor tests and verify missing module failure**

Run:

```bash
pytest -q tests/test_second_global_loop_method.py
```

Expected: collection FAIL because `loop_closure.methods.second_global` does not
exist.

- [ ] **Step 6: Implement the named processor by reusing corrected math**

Create:

```python
from loop_closure.methods.corrected import CorrectedLoopProcessor
from pipeline.config import ReconstructionMode


class SecondGlobalLoopProcessor(CorrectedLoopProcessor):
    """Residual graph optimization for materialized Traditional geometry."""

    name = ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
```

Do not copy or alter optimizer mathematics. Keep focused tests against the
subclass to freeze the inherited coordinate conversion, edge delivery, no-loop
path, and one-delta aggregation contracts.

- [ ] **Step 7: Run all Stage 2 math tests**

Run:

```bash
pytest -q \
  tests/test_second_global_loop_method.py \
  tests/reconstruction/test_traditional_second_global_mode.py -k 'materialization or second_global_states' \
  tests/reconstruction/test_residual_alignment.py \
  tests/test_corrected_loop_method.py
```

Expected: PASS.

- [ ] **Step 8: Commit the Stage 2 residual graph math**

```bash
git add loop_closure/methods/second_global.py \
  reconstruction/modes/traditional_second_global.py \
  tests/test_second_global_loop_method.py \
  tests/reconstruction/test_traditional_second_global_mode.py
git commit -m "feat: build second global residual graph"
```

---

### Task 4: Orchestrate an exact Stage 1 and optimized Stage 2

**Files:**
- Modify: `pipeline/artifacts.py`
- Modify: `reconstruction/modes/base.py`
- Modify: `reconstruction/modes/traditional_second_global.py`
- Modify: `reconstruction/modes/__init__.py`
- Modify: `reconstruction/registry.py`
- Modify: `tests/reconstruction/test_traditional_second_global_mode.py`
- Modify: `tests/reconstruction/test_mode_registry.py`
- Modify: `tests/test_architecture_boundaries.py`

**Interfaces:**
- Consumes: `TraditionalReconstructionMode`, `TraditionalLoopProcessor`, `SecondGlobalLoopProcessor`, `ReconstructionContext`, and detector/evidence services.
- Produces: `StagedReconstructionArtifacts(stage1, stage2)`, `TraditionalSecondGlobalReconstructionMode.run(context) -> StagedReconstructionArtifacts`, and registry construction for the new enum.

- [ ] **Step 1: Write failing staged-result and registry tests**

Add to artifact tests or the new mode test:

```python
def test_staged_result_requires_exact_stage_modes():
    result = StagedReconstructionArtifacts(
        stage1=make_artifact(reconstruction_mode=ReconstructionMode.TRADITIONAL),
        stage2=make_artifact(
            reconstruction_mode=ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
        ),
    )
    assert result.primary is result.stage2

    with pytest.raises(ValueError, match="stage1.*traditional"):
        StagedReconstructionArtifacts(
            stage1=make_artifact(reconstruction_mode=ReconstructionMode.NO_LOOP),
            stage2=result.stage2,
        )
```

Extend the registry parametrization:

```python
(
    ReconstructionMode.TRADITIONAL_SECOND_GLOBAL,
    TraditionalSecondGlobalReconstructionMode,
)
```

and include the new mode in the complete-loop-services rejection test.

- [ ] **Step 2: Run staged-result and registry tests and verify failures**

Run:

```bash
pytest -q \
  tests/reconstruction/test_mode_registry.py \
  tests/reconstruction/test_traditional_second_global_mode.py -k staged
```

Expected: FAIL because the result and mode classes are missing.

- [ ] **Step 3: Add the staged artifact contract**

In `pipeline/artifacts.py` define:

```python
@dataclass(frozen=True)
class StagedReconstructionArtifacts:
    stage1: ReconstructionArtifact
    stage2: ReconstructionArtifact

    def __post_init__(self) -> None:
        if self.stage1.reconstruction_mode is not ReconstructionMode.TRADITIONAL:
            raise ValueError("stage1 artifact must use traditional mode")
        if (
            self.stage2.reconstruction_mode
            is not ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
        ):
            raise ValueError(
                "stage2 artifact must use traditional_second_global mode"
            )
        if self.stage1.frame_ids != self.stage2.frame_ids:
            raise ValueError("staged artifact frame_ids must match")
        if self.stage1.prediction_key != self.stage2.prediction_key:
            raise ValueError("staged artifact prediction keys must match")

    @property
    def primary(self) -> ReconstructionArtifact:
        return self.stage2
```

Update `ReconstructionModeRunner.run` to return
`ReconstructionArtifact | StagedReconstructionArtifacts`.

- [ ] **Step 4: Write the failing end-to-end mode characterization**

Use fresh but identical prediction generators for standalone Traditional and
the new mode. In the new test module, define `_predictions()` from
`iter_window_predictions(LiteralProvider(), SPECS, images, "cpu")` and define
`_context(predictions, mode)` by copying the focused `ReconstructionContext`
fixture fields from `tests/reconstruction/test_traditional_mode.py`, with only
`reconstruction_mode=mode` varying. Inject a detector that records one call,
evidence whose call log distinguishes Stage 1 and Stage 2 states,
deterministic first/second processors, and deterministic residual alignment:

```python
def test_new_mode_preserves_stage1_and_rebuilds_stage2_from_corrected_geometry():
    baseline = standalone_traditional_mode.run(
        _context(_predictions(), ReconstructionMode.TRADITIONAL)
    )
    staged = second_global_mode.run(
        _context(
            _predictions(),
            ReconstructionMode.TRADITIONAL_SECOND_GLOBAL,
        )
    )

    for name in ("local_points", "global_points", "camera_poses", "confidence"):
        assert torch.equal(getattr(staged.stage1, name), getattr(baseline, name))
    assert detector.call_count == 1
    assert evidence.stage1_call_count == evidence.stage2_call_count == 1
    assert residual_inputs[0]["current_points"][..., 2].tolist() == [
        [[[3.0]]],
        [[[3.0]]],
    ]
    assert staged.stage2.reconstruction_mode is (
        ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
    )
    expected_global = torch.einsum(
        "nij,nhwj->nhwi",
        staged.stage2.camera_poses,
        homogenize_points(staged.stage2.local_points),
    )[..., :3]
    assert torch.equal(staged.stage2.global_points, expected_global)
```

Add a second test where Stage 1 evidence raises `ValueError` and Stage 2
evidence succeeds, proving Stage 2 receives the original detector candidate
rather than only accepted Stage 1 constraints.

- [ ] **Step 5: Run the mode characterization and verify missing orchestration**

Run:

```bash
pytest -q tests/reconstruction/test_traditional_second_global_mode.py -k 'preserves_stage1 or original_detector_candidate'
```

Expected: FAIL because `TraditionalSecondGlobalReconstructionMode.run` is not
implemented.

- [ ] **Step 6: Implement recording wrappers and two-stage orchestration**

Implement private delegating wrappers:

```python
class _RecordingDetector:
    def __init__(self, delegate):
        self.delegate = delegate
        self.candidates: tuple[LoopCandidate, ...] | None = None

    def detect(self, manifest, images):
        self.candidates = self.delegate.detect(manifest, images)
        return self.candidates


class _RecordingTraditionalProcessor:
    def __init__(self, delegate):
        self.delegate = delegate
        self.constraints: tuple[LoopConstraint, ...] | None = None
        self.solution: LoopSolution | None = None

    def build_constraints(self, states, candidates, evidence):
        result = self.delegate.build_constraints(states, candidates, evidence)
        self.constraints = tuple(result)
        return result

    def optimize(self, states, constraints):
        self.solution = self.delegate.optimize(states, constraints)
        return self.solution

    def aggregate(self, states, solution):
        return self.delegate.aggregate(states, solution)
```

In `run()`:

1. validate the new context mode;
2. replace it with `ReconstructionMode.TRADITIONAL` only for the delegated
   Stage 1 call;
3. validate recorded trace, candidates, constraints, and solution;
4. materialize Stage 1 windows;
5. build Stage 2 states and residual summaries;
6. rebuild constraints from the recorded candidate tuple;
7. optimize and aggregate Stage 2;
8. construct Stage 2 diagnostics with explicit Stage 1/Stage 2 numeric keys;
9. return `StagedReconstructionArtifacts(stage1, stage2)`.

Use separate processor instances with the same `OptimizerConfig`; production
constructors may accept `stage1_processor` and `stage2_processor` overrides for
deterministic tests.

Measure sub-stages with `time.perf_counter()` and construct the Stage 2
diagnostics from this exact key set:

```python
stage1_scalars = stage1.diagnostics.mode_scalars
stage2_scalars = {
    "window_count": len(second_states),
    "stage1_candidate_count": stage1.diagnostics.candidate_count,
    "stage1_constraint_count": stage1.diagnostics.constraint_count,
    "stage1_used_no_loop_path": int(
        stage1_scalars.get("used_no_loop_path", 0)
    ),
    "stage2_candidate_count": len(candidates),
    "stage2_constraint_count": len(constraints),
    "stage2_used_no_loop_path": int(solution.used_no_loop_path),
    **{
        f"stage2_{name}": value
        for name, value in summarize_residual_alignments(
            residual_results,
            len(second_states),
        ).items()
    },
    "stage2_max_abs_log_scale_delta": aggregate.mode_scalars[
        "max_abs_log_scale_delta"
    ],
    "stage2_mean_abs_log_scale_delta": aggregate.mode_scalars[
        "mean_abs_log_scale_delta"
    ],
}
```

Set top-level `candidate_count=len(candidates)` and
`constraint_count=len(constraints)`. Use timing keys
`stage1_reconstruction`, `stage1_materialization`,
`stage2_adjacent_residual`, `stage2_constraints_optimization`, and
`stage2_aggregation`. Reject an absent recording, a trace/solution count
mismatch, anything other than `len(states) - 1` residual results, and an
aggregate frame count different from `len(context.frame_ids)`.

- [ ] **Step 7: Register and export the new mode**

Import and return `TraditionalSecondGlobalReconstructionMode` in
`reconstruction/registry.py` for the new enum. Export the mode and state types
from `reconstruction/modes/__init__.py`. Extend architecture-boundary tests so
the new reconstruction module may import loop method APIs but evaluators still
cannot import reconstruction internals.

- [ ] **Step 8: Run mode, registry, Traditional, and architecture tests**

Run:

```bash
pytest -q \
  tests/reconstruction/test_traditional_second_global_mode.py \
  tests/reconstruction/test_mode_registry.py \
  tests/reconstruction/test_traditional_mode.py \
  tests/test_traditional_loop_method.py \
  tests/test_architecture_boundaries.py
git diff --exit-code -- reconstruction/modes/traditional.py loop_closure/methods/traditional.py
```

Expected: PASS and the final diff command exits 0.

- [ ] **Step 9: Commit the new reconstruction mode**

```bash
git add pipeline/artifacts.py reconstruction/modes/base.py \
  reconstruction/modes/traditional_second_global.py \
  reconstruction/modes/__init__.py reconstruction/registry.py \
  tests/reconstruction/test_traditional_second_global_mode.py \
  tests/reconstruction/test_mode_registry.py tests/test_architecture_boundaries.py
git commit -m "feat: orchestrate traditional second global optimization"
```

---

### Task 5: Atomically persist and report both artifacts

**Files:**
- Modify: `pipeline/artifacts.py`
- Modify: `pipeline/runner.py`
- Modify: `run_reconstruction.py`
- Modify: `tests/test_reconstruction_artifacts.py`
- Modify: `tests/test_pipeline_runner.py`
- Modify: `tests/test_run_reconstruction_cli.py`
- Modify: `tests/evaluation/test_evaluate_ate_cli.py`

**Interfaces:**
- Consumes: `StagedReconstructionArtifacts`, `write_reconstruction_artifact`, and existing artifact metadata.
- Produces: `write_staged_reconstruction_artifacts(staged, output_dir, **metadata) -> Mapping[str, Path]`, `PipelineRunner.stage_artifact_dirs`, root Stage 2 output, and nested `stage1/` output.

- [ ] **Step 1: Write failing staged-writer tests**

Add this test-local factory and tests that round-trip both outputs:

```python
def make_staged_artifacts():
    return StagedReconstructionArtifacts(
        stage1=make_artifact(
            reconstruction_mode=ReconstructionMode.TRADITIONAL,
        ),
        stage2=make_artifact(
            reconstruction_mode=(
                ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
            ),
        ),
    )


def test_staged_writer_atomically_writes_two_standard_artifacts(tmp_path):
    staged = make_staged_artifacts()
    paths = write_staged_reconstruction_artifacts(
        staged,
        tmp_path / "run",
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )
    assert paths == {
        "stage1": tmp_path / "run" / "stage1",
        "stage2": tmp_path / "run",
    }
    assert_artifacts_equal(load_reconstruction_artifact(paths["stage1"]), staged.stage1)
    assert_artifacts_equal(load_reconstruction_artifact(paths["stage2"]), staged.stage2)
```

Inject a writer that raises on Stage 1 and assert the destination does not
exist. Add overwrite rejection for an existing root.

- [ ] **Step 2: Run staged-writer tests and verify missing writer failure**

Run:

```bash
pytest -q tests/test_reconstruction_artifacts.py -k staged
```

Expected: FAIL because `write_staged_reconstruction_artifacts` is missing.

- [ ] **Step 3: Implement the temporary-tree staged writer**

Build a temporary container next to the requested output. Inside it, call the
existing writer for Stage 2 at `container/artifact`, then for Stage 1 at
`container/artifact/stage1`, and finally rename `container/artifact` to the
requested output:

```python
with tempfile.TemporaryDirectory(
    dir=output.parent,
    prefix=f".{output.name}.staged.",
) as temporary:
    staged_root = Path(temporary) / "artifact"
    write_reconstruction_artifact(staged.stage2, staged_root, **metadata)
    write_reconstruction_artifact(
        staged.stage1,
        staged_root / "stage1",
        **metadata,
    )
    staged_root.replace(output)
return MappingProxyType({"stage1": output / "stage1", "stage2": output})
```

Create `output.parent` only when needed, refuse an existing `output`, and clean
the temporary tree automatically on every exception.

- [ ] **Step 4: Write failing PipelineRunner and CLI tests**

Create a fake mode returning `StagedReconstructionArtifacts`; assert:

```python
result = runner.run()
assert result is staged.stage2
assert runner.artifact_dir == expected_root
assert dict(runner.stage_artifact_dirs) == {
    "stage1": expected_root / "stage1",
    "stage2": expected_root,
}
assert events.values.count("write_staged_artifacts") == 1
assert events.values.count("write_artifact") == 0
```

Extend the CLI fake runner with the stage mapping and assert stdout contains:

```text
stage1_artifact_dir=results/artifact/stage1
stage2_artifact_dir=results/artifact
```

Add an ATE CLI test that loads and evaluates both paths through the existing
CLI without any stage-specific evaluator branch.

- [ ] **Step 5: Run runner/CLI tests and verify staged handling failure**

Run:

```bash
pytest -q \
  tests/test_pipeline_runner.py -k staged \
  tests/test_run_reconstruction_cli.py \
  tests/evaluation/test_evaluate_ate_cli.py -k staged
```

Expected: FAIL because runner dependencies and CLI output do not understand
staged results.

- [ ] **Step 6: Implement runner unwrapping and path reporting**

Add `write_staged_artifacts` to `PipelineDependencies`. In `PipelineRunner`,
initialize an immutable empty `stage_artifact_dirs` mapping. After mode
execution:

```python
if isinstance(result, StagedReconstructionArtifacts):
    artifact = result.primary
    staged = replace(
        result,
        stage2=replace(
            artifact,
            diagnostics=_with_reconstruction_timing(
                artifact,
                reconstruction_ms,
            ).diagnostics,
        ),
    )
    paths = dependencies.write_staged_artifacts(staged, output_dir, **metadata)
    self.stage_artifact_dirs = MappingProxyType(dict(paths))
    self.artifact_dir = paths["stage2"]
else:
    artifact = _with_reconstruction_timing(result, reconstruction_ms)
    self.artifact_dir = dependencies.write_artifact(artifact, output_dir, **metadata)
    self.stage_artifact_dirs = MappingProxyType({"stage2": self.artifact_dir})
return artifact
```

Define this runner-local helper rather than duplicating `dataclasses.replace`:

```python
def _with_reconstruction_timing(
    artifact: ReconstructionArtifact,
    reconstruction_ms: float,
) -> ReconstructionArtifact:
    return replace(
        artifact,
        diagnostics=replace(
            artifact.diagnostics,
            stage_timings_ms={
                **dict(artifact.diagnostics.stage_timings_ms),
                "reconstruction": reconstruction_ms,
            },
        ),
    )
```

Use one metadata dictionary and one `git_commit()` call. Preserve the Stage 1
diagnostics exactly; apply runner-level total reconstruction timing only to
Stage 2.

Update `run_reconstruction.py` to append the two explicit stage paths only
when `stage1` is present.

- [ ] **Step 7: Run artifact, runner, CLI, and experiment-repository tests**

Run:

```bash
pytest -q \
  tests/test_reconstruction_artifacts.py \
  tests/test_pipeline_runner.py \
  tests/test_run_reconstruction_cli.py \
  tests/evaluation/test_evaluate_ate_cli.py \
  tests/experiments/test_ate_experiment.py
```

Expected: PASS. The artifact repository continues to recognize the Stage 2
root because its existing required files remain at the root.

- [ ] **Step 8: Commit staged artifact persistence**

```bash
git add pipeline/artifacts.py pipeline/runner.py run_reconstruction.py \
  tests/test_reconstruction_artifacts.py tests/test_pipeline_runner.py \
  tests/test_run_reconstruction_cli.py \
  tests/evaluation/test_evaluate_ate_cli.py
git commit -m "feat: persist staged global optimization artifacts"
```

---

### Task 6: Document cloud execution and verify the complete feature

**Files:**
- Modify: `README.md`
- Create: `docs/traditional-second-global-cloud-validation.md`
- Modify: `tests/test_cloud_validation_commands.py`

**Interfaces:**
- Consumes: the new mode, root Stage 2 artifact, nested Stage 1 artifact, and existing `evaluate_ate.py` CLI.
- Produces: copy-paste cloud clone/setup/reconstruction/dual-ATE commands and a final regression record.

- [ ] **Step 1: Write the failing documentation command test**

Require the validation guide to contain the exact public mode and both artifact
paths:

```python
def test_second_global_cloud_guide_has_dual_ate_commands():
    text = Path(
        "docs/traditional-second-global-cloud-validation.md"
    ).read_text(encoding="utf-8")
    assert "--branch codex/keyframe-selection" in text
    assert "reconstruction.mode=traditional_second_global" in text
    assert "--artifact \"$RESULT_DIR/stage1\"" in text
    assert "--artifact \"$RESULT_DIR\"" in text
    assert "cmp_stage1_trajectory.py" not in text
```

- [ ] **Step 2: Run the documentation test and verify missing guide failure**

Run:

```bash
pytest -q tests/test_cloud_validation_commands.py -k second_global
```

Expected: FAIL because the guide is missing.

- [ ] **Step 3: Add concise README and cloud validation commands**

Document:

```bash
git clone --recursive --branch codex/keyframe-selection \
  https://github.com/Cjuicy/LASER.git LASER-Improved
cd LASER-Improved
conda create -n laser-second-global python=3.11 -y
conda activate laser-second-global
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e viser
bash scripts/download_weights.sh
```

Document reconstruction:

```bash
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set input.image_dir=/data/sequence/images \
  --set output.scene_name=scene-name \
  --set segmentation.method=geometry \
  --set reconstruction.mode=traditional_second_global \
  --set window.size=20 \
  --set window.overlap=5
```

Document Stage 1 and Stage 2 ATE commands with separate output directories and
the exact same ground truth/config arguments. Include a Python tensor equality
one-liner that uses `torch.load(path, weights_only=True)["camera_poses"]` and
`torch.equal` to compare a standalone Traditional trajectory with the nested
Stage 1 trajectory.

- [ ] **Step 4: Run documentation and CLI help checks**

Run:

```bash
pytest -q tests/test_cloud_validation_commands.py
python run_reconstruction.py --help
python evaluate_ate.py --help
python run_experiment_matrix.py --help
```

Expected: all commands exit 0.

- [ ] **Step 5: Run focused feature regression**

Run:

```bash
pytest -q \
  tests/reconstruction/test_traditional_second_global_mode.py \
  tests/test_second_global_loop_method.py \
  tests/test_reconstruction_artifacts.py \
  tests/test_pipeline_runner.py \
  tests/test_pipeline_config.py \
  tests/reconstruction/test_mode_registry.py \
  tests/experiments/test_matrix.py \
  tests/evaluation/test_evaluate_ate_cli.py
```

Expected: PASS.

- [ ] **Step 6: Run the full suite from a clean command invocation**

Run:

```bash
pytest -q
```

Expected: 0 failures. Record the exact pass count and warnings in the final
handoff.

- [ ] **Step 7: Verify source invariants and working-tree hygiene**

Run:

```bash
git diff --exit-code 9119e339e9f027354b88501e1a588f34627bbbd7 \
  -- reconstruction/modes/traditional.py loop_closure/methods/traditional.py
git diff --check
git status --short
```

Expected: the first two commands exit 0. `git status --short` lists only the
planned documentation changes before the final documentation commit.

- [ ] **Step 8: Commit documentation**

```bash
git add README.md docs/traditional-second-global-cloud-validation.md \
  tests/test_cloud_validation_commands.py
git commit -m "docs: add second global cloud validation"
```

- [ ] **Step 9: Re-run final verification after the last commit**

Run:

```bash
pytest -q
python run_reconstruction.py --help
python evaluate_ate.py --help
python run_experiment_matrix.py --help
git status --short --branch
```

Expected: all tests and help commands exit 0, and the branch is clean and ahead
of `origin/codex/keyframe-selection` only by the planned feature commits.
