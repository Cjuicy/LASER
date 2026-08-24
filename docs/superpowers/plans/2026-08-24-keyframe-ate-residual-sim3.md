# Keyframe-Aware ATE Residual Sim(3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propagate keyframe/anchor geometry into no-loop and corrected trajectories through a shared post-anchor residual Sim(3), then evaluate each reconstruction artifact with every scene-supported evaluator without duplicating reconstruction.

**Architecture:** Add one residual-alignment component called after anchor propagation in `no_loop` and `corrected`. Corrected rebuilds its absolute transforms and sequential edges from the residual result, while traditional stays untouched; a version-2 capability experiment schema and evaluator bundle then schedule and resume ATE/point-cloud evaluation independently.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, OmegaConf, pytest, existing LASER reconstruction artifacts and evaluators.

**Spec:** `docs/superpowers/specs/2026-08-24-keyframe-ate-residual-sim3-design.md`

## Global Constraints

- Do not modify `reconstruction/modes/traditional.py` or `loop_closure/methods/traditional.py`.
- Run residual registration only for non-first `no_loop` and `corrected` windows when `anchor_propagation.enabled=true`.
- Compose refined absolutes as `residual ∘ coarse` and corrected edges as `inverse(previous_refined_abs) ∘ current_refined_abs`.
- Apply residual scale to local points and full residual Sim(3) to camera poses.
- Apply the corrected optimizer delta exactly once after residual refinement.
- Permit point-cloud evaluation only for `no_loop`.
- Force traditional keyframe refinement off and schedule traditional only for ATE-capable scenes.
- Missing capability is `skipped`; declared invalid input/evaluator error is `failed`; reconstruction error makes scheduled evaluators `blocked`.
- Preserve a sibling evaluator's passed output after another evaluator fails.
- Keep version-1 experiment configs, artifact tensor files/loaders, and PI3 prediction cache identity unchanged.
- Diagnostics may add only finite numeric `mode_scalars`.

---

## File Map

**Create**

- `reconstruction/residual_alignment.py` — residual estimation, strict validation, application, metrics, summary scalars.
- `tests/reconstruction/test_residual_alignment.py` — residual mathematical contract.
- `experiments/evaluation_bundle.py` — evaluator identity, terminal status, atomic record, isolated execution, resume.
- `tests/experiments/test_evaluation_bundle.py` — isolation/failure/resume contract.
- `tests/experiments/test_keyframe_metrics_experiment.py` — version-2 config/matrix/runner integration.
- `tests/test_run_experiment_matrix_cli.py` — capability CLI status.
- `configs/experiments/keyframe_metrics_matrix.yaml` — checked-in version-2 example.

**Modify**

- `reconstruction/modes/no_loop.py`
- `reconstruction/modes/corrected.py`
- `tests/reconstruction/test_no_loop_mode.py`
- `tests/reconstruction/test_corrected_mode.py`
- `tests/test_corrected_loop_method.py`
- `experiments/config.py`
- `experiments/matrix.py`
- `experiments/ate.py`
- `experiments/pointcloud.py`
- `experiments/runner.py`
- `run_experiment_matrix.py`
- `tests/experiments/test_ate_experiment.py`
- `tests/experiments/test_pointcloud_experiment.py`
- `tests/experiments/test_matrix.py`
- `docs/reconstruction-evaluation-cloud-validation.md`
- `tests/test_cloud_validation_commands.py`

---

### Task 1: Shared post-anchor residual Sim(3)

**Files:**
- Create: `reconstruction/residual_alignment.py`
- Create: `tests/reconstruction/test_residual_alignment.py`

**Interfaces:**
- Consumes: `register_adjacent_windows`, `apply_sim3_to_pose`, `mutual_confidence_mask`, `Sim3`.
- Produces: `ResidualAlignmentResult`, `align_post_anchor_window`, and `summarize_residual_alignments`.

- [ ] **Step 1: Write the failing application test**

Create a literal one-frame overlap and inject scale 2, a 90-degree z rotation, and translation `[3,4,0]`:

```python
def test_post_anchor_residual_scales_points_and_transforms_poses():
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([3.0, 4.0, 0.0])
    result = align_post_anchor_window(
        previous_points=_points(1.0),
        previous_poses=_poses(0.0),
        previous_confidence=torch.ones((1, 1, 1)),
        current_points=_points(5.0),
        current_poses=_poses(2.0),
        current_confidence=torch.ones((1, 1, 1)),
        overlap=1,
        confidence_keep_ratio=1.0,
        register_adjacent=lambda *args: (2.0, rotation, translation),
    )
    assert torch.equal(result.local_points, _points(10.0))
    assert torch.allclose(
        result.camera_poses[:, :3, 3],
        torch.tensor([[3.0, 8.0, 0.0]]),
    )
    assert result.correspondence_count == 1
    assert result.rotation_rad == pytest.approx(math.pi / 2)
    assert result.translation_norm == pytest.approx(5.0)
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/reconstruction/test_residual_alignment.py -q`

Expected: collection fails because `reconstruction.residual_alignment` does not exist.

- [ ] **Step 3: Implement the exact result and function signatures**

```python
@dataclass(frozen=True)
class ResidualAlignmentResult:
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    sim3: Sim3
    correspondence_count: int
    abs_log_scale: float
    rotation_rad: float
    translation_norm: float


def align_post_anchor_window(
    *,
    previous_points: torch.Tensor,
    previous_poses: torch.Tensor,
    previous_confidence: torch.Tensor,
    current_points: torch.Tensor,
    current_poses: torch.Tensor,
    current_confidence: torch.Tensor,
    overlap: int,
    confidence_keep_ratio: float,
    register_adjacent: Callable = register_adjacent_windows,
    apply_pose_sim3: Callable = apply_sim3_to_pose,
) -> ResidualAlignmentResult:
    mask = mutual_confidence_mask(
        previous_confidence,
        current_confidence,
        overlap,
        confidence_keep_ratio,
        context="post-anchor residual registration",
    )
    correspondence_count = int(torch.count_nonzero(mask).item())
    if correspondence_count == 0:
        raise ValueError("post-anchor residual registration has no correspondences")
    sim3 = register_adjacent(
        previous_points[-overlap:],
        current_points[:overlap],
        previous_poses[-overlap:],
        current_poses[:overlap],
        mask,
    )
    validate_sim3(sim3, context="post-anchor residual Sim(3)")
    scale, rotation, translation = sim3
    scale_tensor = torch.as_tensor(scale, device=current_points.device, dtype=current_points.dtype)
    local_points = scale_tensor * current_points
    camera_poses = apply_pose_sim3(
        current_poses,
        scale,
        rotation.to(current_poses),
        translation.to(current_poses),
    )
    if local_points.shape != current_points.shape:
        raise ValueError("post-anchor residual changed point shape")
    if camera_poses.shape != current_poses.shape:
        raise ValueError("post-anchor residual changed pose shape")
    return _validated_result(
        local_points,
        camera_poses,
        sim3,
        correspondence_count,
    )
```

Implement `_validated_result` to require unchanged shapes, finite outputs, `R.T @ R` within `atol=rtol=1e-4` of identity, determinant within `1e-4` of `+1`, and finite Python metrics. Compute angle with `acos(clamp((trace(R)-1)/2,-1,1))`.

Implement:

```python
def summarize_residual_alignments(
    results: Sequence[ResidualAlignmentResult],
    window_count: int,
) -> dict[str, int | float]:
    if window_count < len(results):
        raise ValueError("residual result count exceeds window count")
    scales = [item.abs_log_scale for item in results]
    rotations = [item.rotation_rad for item in results]
    translations = [item.translation_norm for item in results]
    return {
        "residual_applied_window_count": len(results),
        "residual_skipped_window_count": window_count - len(results),
        "max_abs_log_residual_scale": max(scales, default=0.0),
        "mean_abs_log_residual_scale": sum(scales) / len(scales) if scales else 0.0,
        "max_residual_rotation_rad": max(rotations, default=0.0),
        "mean_residual_rotation_rad": sum(rotations) / len(rotations) if rotations else 0.0,
        "max_residual_translation_norm": max(translations, default=0.0),
        "mean_residual_translation_norm": (
            sum(translations) / len(translations) if translations else 0.0
        ),
    }
```

- [ ] **Step 4: Add invalid-transform tests and run GREEN**

Parametrize scale zero, reflected rotation, non-orthonormal rotation, NaN component, shape mismatch, and zero correspondences. Each must raise `ValueError` containing `post-anchor residual`.

Run: `pytest tests/reconstruction/test_residual_alignment.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add reconstruction/residual_alignment.py tests/reconstruction/test_residual_alignment.py
git commit -m "feat: add post-anchor residual alignment"
```

---

### Task 2: Propagate residual points/poses into no-loop artifacts

**Files:**
- Modify: `reconstruction/modes/no_loop.py`
- Modify: `tests/reconstruction/test_no_loop_mode.py`

**Interfaces:**
- Consumes: Task 1 public API.
- Produces: residual-refined `local_points`, `camera_poses`, `global_points`, serialized trajectory/pointmap tensors, and numeric residual diagnostics.

- [ ] **Step 1: Write the failing ordering/artifact test**

Inject `residual_align` that asserts current depth already contains the anchor scale, doubles current points, and adds x translation 9 to poses:

```python
def residual_align(**values):
    observed_anchor_depths.append(float(values["current_points"][0, 0, 0, 2]))
    poses = values["current_poses"].clone()
    poses[:, 0, 3] += 9.0
    return ResidualAlignmentResult(
        local_points=2.0 * values["current_points"],
        camera_poses=poses,
        sim3=identity_sim3(2.0),
        correspondence_count=1,
        abs_log_scale=math.log(2.0),
        rotation_rad=0.0,
        translation_norm=9.0,
    )
```

Assert `observed_anchor_depths == [3.0, 4.0]`, finalized local depths are
`[1.0, 1.0, 6.0, 8.0]`, finalized pose x translations are
`[0.0, 0.0, 9.0, 9.0]`, those values appear in `trajectory.pt` and
`pointmap.pt`, and diagnostics report applied `2` / skipped `1`.

- [ ] **Step 2: Run RED**

Run: `pytest tests/reconstruction/test_no_loop_mode.py::test_no_loop_residual_runs_after_anchor_and_reaches_both_artifacts -q`

Expected: constructor rejects `residual_align`.

- [ ] **Step 3: Integrate after anchor and before trim/state publication**

Add constructor dependency `residual_align: Callable = align_post_anchor_window` and collect `residual_results`. Use this exact call after anchor scaling:

```python
residual = self._residual_align(
    previous_points=previous["local_points"],
    previous_poses=previous["camera_poses"],
    previous_confidence=previous["confidence"],
    current_points=local_points,
    current_poses=camera_poses,
    current_confidence=confidence,
    overlap=overlap,
    confidence_keep_ratio=context.registration_config.confidence_keep_ratio,
    register_adjacent=self._register_adjacent,
    apply_pose_sim3=self._apply_pose_sim3,
)
local_points = residual.local_points
camera_poses = residual.camera_poses
residual_results.append(residual)
```

Run it only inside `previous is not None and context.anchor_config.enabled`. Publish finalized tensors as `previous`. Merge `summarize_residual_alignments(residual_results, window_count)` into `mode_scalars`.

- [ ] **Step 4: Add anchor-disabled parity and serialized consistency tests**

Replace `anchor_config.enabled` with false, inject a residual callable that raises if called, and assert exact pre-change points/poses. Load both serialized tensor files and compare with the returned artifact.

Run: `pytest tests/reconstruction/test_no_loop_mode.py tests/reconstruction/test_residual_alignment.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add reconstruction/modes/no_loop.py tests/reconstruction/test_no_loop_mode.py
git commit -m "feat: propagate residual alignment into no-loop ATE"
```

---

### Task 3: Build corrected transforms/edges after residual alignment

**Files:**
- Modify: `reconstruction/modes/corrected.py`
- Modify: `tests/reconstruction/test_corrected_mode.py`
- Modify: `tests/test_corrected_loop_method.py`

**Interfaces:**
- Consumes: Task 1 API plus `accumulate_sim3` and `closed_form_inverse_sim3`.
- Produces: refined `CorrectedWindowState.sim3_abs`, refined `sim3_edge`, refined loop constraints, and a single final optimizer delta.

- [ ] **Step 1: Write the failing composition/edge test**

Use two windows with coarse scale 2, anchor scale 3, residual scale 4. In an inspecting processor assert:

```python
assert float(torch.as_tensor(states[1].sim3_abs[0])) == 8.0
assert float(torch.as_tensor(states[1].sim3_edge[0])) == 8.0
return LoopSolution(
    optimized_transforms=tuple(state.sim3_abs for state in states),
    constraints=(),
    used_no_loop_path=True,
)
```

Assert the finalized non-overlap point includes coarse × anchor × residual and no-constraint aggregation preserves the residual-refined pose.

- [ ] **Step 2: Run RED**

Run: `pytest tests/reconstruction/test_corrected_mode.py::test_corrected_composes_residual_after_coarse_and_rebuilds_edge -q`

Expected: constructor rejects `residual_align` and state transforms remain coarse.

- [ ] **Step 3: Move edge creation after residual finalization**

Rename the pre-segmentation transform variable to `coarse_sim3_abs`. After anchor:

```python
sim3_abs = coarse_sim3_abs
if states and context.anchor_config.enabled:
    residual = self._residual_align(
        previous_points=states[-1].local_points,
        previous_poses=states[-1].camera_poses,
        previous_confidence=states[-1].confidence,
        current_points=local_points,
        current_poses=camera_poses,
        current_confidence=confidence,
        overlap=overlap,
        confidence_keep_ratio=context.registration_config.confidence_keep_ratio,
        register_adjacent=self._register_adjacent,
        apply_pose_sim3=self._apply_pose_sim3,
    )
    local_points = residual.local_points
    camera_poses = residual.camera_poses
    sim3_abs = accumulate_sim3(residual.sim3, coarse_sim3_abs)
    residual_results.append(residual)

sim3_edge = None
if states:
    sim3_edge = accumulate_sim3(
        closed_form_inverse_sim3(*states[-1].sim3_abs),
        sim3_abs,
    )
```

Validate refined absolute/edge, store them, merge residual summaries, and add `refined_edge_count=len(states)-1`.

- [ ] **Step 4: Prove refined constraints and optimizer inputs**

In `tests/test_corrected_loop_method.py` construct states with explicit refined absolutes and assert the exact local constraint formula. In mode tests capture processor states and assert optimized edges equal refined edges, not coarse edges.

Run: `pytest tests/reconstruction/test_corrected_mode.py tests/test_corrected_loop_method.py -q`

Expected: all tests pass.

- [ ] **Step 5: Prove anchor-disabled and one-delta invariants**

Add tests that residual is never called when anchor is disabled, no-constraint output retains refined poses, and an optimizer scale is multiplied exactly once after coarse/anchor/residual.

Run: `pytest tests/reconstruction/test_corrected_mode.py tests/reconstruction/test_no_loop_mode.py tests/test_corrected_loop_method.py -q`

Expected: all tests pass.

- [ ] **Step 6: Verify traditional is untouched and commit**

Run:

```bash
git diff 02bca0d22c044e0553691e415b41866cf64a1a49 --exit-code -- \
  reconstruction/modes/traditional.py loop_closure/methods/traditional.py
```

Expected: exit 0, no diff.

Commit:

```bash
git add reconstruction/modes/corrected.py tests/reconstruction/test_corrected_mode.py tests/test_corrected_loop_method.py
git commit -m "feat: refine corrected loop edges after anchors"
```

---

### Task 4: Version-2 capability config and exact matrix

**Files:**
- Modify: `experiments/config.py`
- Modify: `experiments/matrix.py`
- Modify: `experiments/ate.py`
- Modify: `experiments/pointcloud.py`
- Modify: `tests/experiments/test_ate_experiment.py`
- Modify: `tests/experiments/test_pointcloud_experiment.py`
- Modify: `tests/experiments/test_matrix.py`
- Create: `tests/experiments/test_keyframe_metrics_experiment.py`

**Interfaces:**
- Produces: `EvaluationInputConfig`, `CapabilityExperimentConfig`, `CapabilityMatrixEntry`, `build_capability_matrix`, `evaluate_ate_inputs`, `evaluate_pointcloud_inputs`.
- Preserves: version-1 `ExperimentConfig`, `MatrixEntry`, `build_matrix`, and output payloads.

- [ ] **Step 1: Write failing version-2 config tests**

Use this exact schema:

```yaml
version: 2
reconstruction_config: configs/reconstruction/pi3_laser.yaml
output_root: outputs/experiments/keyframe-metrics
dataset:
  name: fixture
  sequence: sequence-0
  trajectory_ground_truth:
    path: data/groundtruth.txt
    format: tum
  pointcloud_ground_truth:
    path: data/groundtruth_pointmaps.npz
evaluation:
  trajectory_config: configs/evaluation/ate.yaml
  pointcloud_config: configs/evaluation/pointcloud.yaml
matrix:
  segmentation: [depth, geometry, atomic]
  refinement: [false, true]
```

Assert at least one declaration is required; trajectory needs format/config; point cloud needs config; unknown fields fail.

- [ ] **Step 2: Run RED**

Run: `pytest tests/experiments/test_keyframe_metrics_experiment.py -q`

Expected: version 2 is rejected.

- [ ] **Step 3: Implement normalized version-2 types/parser**

```python
@dataclass(frozen=True)
class EvaluationInputConfig:
    kind: EvaluationKind
    config_path: str
    ground_truth_path: str
    ground_truth_format: str | None = None


@dataclass(frozen=True)
class CapabilityExperimentConfig:
    version: int
    reconstruction_config: str
    output_root: str
    dataset_name: str
    sequence: str
    evaluator_inputs: Mapping[EvaluationKind, EvaluationInputConfig]
    segmentation_methods: tuple[str, ...]
    refinement_states: tuple[bool, ...]

    @property
    def entries(self):
        from .matrix import build_capability_matrix
        return build_capability_matrix(self)
```

Freeze `evaluator_inputs` with `MappingProxyType`. Require canonical segmentation order and exact refinement `(False, True)`. Dispatch parser by top-level version and leave version-1 parsing unchanged.

- [ ] **Step 4: Write failing capability-matrix tests**

For each segmentation method require:

- dual/ATE-only: no-loop off/on, traditional baseline once, corrected off/on;
- pointcloud-only: no-loop off/on only.

Require evaluator tuples: dual no-loop `(ATE, POINTCLOUD)`; ATE loop modes `(ATE,)`; pointcloud-only no-loop `(POINTCLOUD,)`.

- [ ] **Step 5: Implement capability entries/matrix**

```python
@dataclass(frozen=True)
class CapabilityMatrixEntry:
    segmentation_method: SegmentationMethod
    reconstruction_mode: ReconstructionMode
    window_reference_enabled: bool
    evaluator_kinds: tuple[EvaluationKind, ...]

    @property
    def name(self) -> str:
        if self.reconstruction_mode is ReconstructionMode.TRADITIONAL:
            return f"{self.segmentation_method.value}-traditional-baseline"
        state = "on" if self.window_reference_enabled else "off"
        return (
            f"{self.segmentation_method.value}-"
            f"{self.reconstruction_mode.value}-wr-{state}"
        )
```

`build_capability_matrix` derives modes from capabilities, never schedules point cloud for loop modes, forces traditional false, and rejects duplicate names.

- [ ] **Step 6: Extract evaluator-input functions**

Add exact keyword-only functions:

```python
def evaluate_ate_inputs(
    artifact_dir: str | Path,
    *,
    ground_truth: str | Path,
    ground_truth_format: str,
    evaluation_config: str | Path,
    output_dir: str | Path,
) -> Path:
    estimate = load_trajectory_estimate(artifact_dir)
    truth = load_ground_truth_trajectory(ground_truth, ground_truth_format)
    config = load_trajectory_evaluation_config(evaluation_config)
    metrics = evaluate_trajectory(estimate, truth, config)
    return write_trajectory_metrics(
        metrics,
        output_dir,
        artifact_manifest_sha256=_artifact_manifest_sha256(artifact_dir),
    )
```

`evaluate_ate_artifact` delegates to it. Add:

```python
def evaluate_pointcloud_inputs(
    artifact_dir: str | Path,
    *,
    dataset_name: str,
    sequence: str,
    ground_truth: str | Path,
    evaluation_config: str | Path,
    output_dir: str | Path,
) -> Path:
    estimate = load_pointmap_estimate(artifact_dir)
    config = load_pointcloud_evaluation_config(evaluation_config)
    with np.load(ground_truth, allow_pickle=False) as data:
        truth = PointCloudGroundTruth(data["point_maps"], data["valid_mask"])
    evaluation = evaluate_point_maps(estimate, truth, config)
    result = SequencePointCloudResult(
        dataset_name,
        sequence,
        len(estimate.frame_ids),
        evaluation.primary,
        asdict(evaluation.diagnostics),
    )
    summary = aggregate_pointcloud_results(
        dataset_name,
        (result,),
        expected_sequences=(sequence,),
    )
    return write_pointcloud_results(summary, (result,), output_dir)
```

`evaluate_pointcloud_artifact` delegates to it. Copy current evaluator bodies into these functions without changing metric payloads.

- [ ] **Step 7: Run GREEN and commit**

Run:

`pytest tests/experiments/test_matrix.py tests/experiments/test_ate_experiment.py tests/experiments/test_pointcloud_experiment.py tests/experiments/test_keyframe_metrics_experiment.py -q`

Expected: all pass; v1 counts/names remain 9 ATE and 3 point-cloud.

Commit:

```bash
git add experiments/config.py experiments/matrix.py experiments/ate.py experiments/pointcloud.py \
  tests/experiments/test_ate_experiment.py tests/experiments/test_pointcloud_experiment.py \
  tests/experiments/test_matrix.py tests/experiments/test_keyframe_metrics_experiment.py
git commit -m "feat: add keyframe evaluation capability matrix"
```

---

### Task 5: Independent evaluator status, failure, and resume

**Files:**
- Create: `experiments/evaluation_bundle.py`
- Create: `tests/experiments/test_evaluation_bundle.py`

**Interfaces:**
- Consumes: Task 4 config/entry/input functions.
- Produces: `EvaluatorStatus`, `EvaluatorIdentity`, `EvaluatorError`, `EvaluatorRecord`, `EvaluationBundleRunner.run`, `EvaluationBundleRunner.blocked`.

- [ ] **Step 1: Write failing record validation tests**

Require passed records to have 64-char identity/output, failed/blocked to have an error, and skipped to have no identity/output/error.

Run: `pytest tests/experiments/test_evaluation_bundle.py -q`

Expected: module missing.

- [ ] **Step 2: Implement terminal types and identity**

```python
class EvaluatorStatus(str, Enum):
    PASSED = "passed"
    SKIPPED = "skipped"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class EvaluatorIdentity:
    artifact_manifest_sha256: str
    evaluator_config_sha256: str
    ground_truth_sha256: str
    source_revision: str

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
```

Add validated `EvaluatorError(exception_type, message)` and `EvaluatorRecord(kind,status,identity_digest,output_path,error)`. Implement strict atomic JSON write/read using temp file, flush/fsync, replace, and unknown-field rejection.

- [ ] **Step 3: Write failing dual evaluator isolation/resume test**

First run: fake ATE writes output; fake point-cloud raises `ValueError("shape mismatch")`. Assert statuses passed/failed and ATE output remains. Retry with working point-cloud: ATE call count stays 1; point-cloud count becomes 2; both pass.

- [ ] **Step 4: Implement bundle execution**

`EvaluationBundleRunner.run` iterates canonical `(ATE, POINTCLOUD)`:

1. undeclared/unsupported → write `skipped`;
2. hash artifact manifest, evaluator config, GT, source revision;
3. reuse passed record only on exact identity and existing output;
4. invoke matching evaluator under `output_root / kind.value`;
5. catch ordinary exceptions, sanitize message, write `failed`, continue;
6. write `passed` only after returned output exists.

`blocked` writes blocked scheduled records and skipped unsupported records. Do not catch `KeyboardInterrupt` or `SystemExit`.

- [ ] **Step 5: Add declared-missing, blocked, and invalidation tests**

Prove missing declared GT fails; omitted capability skips; reconstruction error blocks; changed config/GT/source invalidates only the matching evaluator; messages redact URL user-info/control characters.

Run: `pytest tests/experiments/test_evaluation_bundle.py -q`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/evaluation_bundle.py tests/experiments/test_evaluation_bundle.py
git commit -m "feat: run evaluators as resumable bundles"
```

---

### Task 6: Capability runner, keep-going summary, and CLI

**Files:**
- Modify: `experiments/runner.py`
- Modify: `run_experiment_matrix.py`
- Modify: `tests/experiments/test_keyframe_metrics_experiment.py`
- Create: `tests/test_run_experiment_matrix_cli.py`
- Create: `configs/experiments/keyframe_metrics_matrix.yaml`

**Interfaces:**
- Consumes: Tasks 4–5.
- Produces: `CapabilityExperimentRunRecord`, `run_capability_matrix`, version-aware `run_matrix`, `evaluation_summary.json`, capability CLI exit status.

- [ ] **Step 1: Write failing one-artifact/two-evaluator test**

Inject a fake pipeline writer and bundle runner. Assert a dual no-loop entry reconstructs once, both evaluators receive the same artifact path, and evaluator kind never affects reconstruction identity.

Run focused test; expected failure because runner supports one evaluator only.

- [ ] **Step 2: Implement record and version-aware dispatch**

```python
@dataclass(frozen=True)
class CapabilityExperimentRunRecord:
    entry: CapabilityMatrixEntry
    reconstruction_identity: str | None
    artifact_dir: Path | None
    evaluations: tuple[EvaluatorRecord, ...]
    prediction_cache_mode: PredictionCacheMode

    @property
    def succeeded(self) -> bool:
        return all(
            item.status in {EvaluatorStatus.PASSED, EvaluatorStatus.SKIPPED}
            for item in self.evaluations
        )
```

Rename current body `_run_legacy_matrix`. `run_matrix` dispatches by config type. Capability entry overrides are exact:

```python
values.extend(
    (
        f"segmentation.method={entry.segmentation_method.value}",
        "segmentation.window_reference.enabled="
        f"{str(entry.window_reference_enabled).lower()}",
        f"reconstruction.mode={entry.reconstruction_mode.value}",
    )
)
```

Reject traditional entries whose refinement is true.

- [ ] **Step 3: Implement keep-going reconstruction and summary**

For each entry, use `ArtifactRepository` once. On success call one bundle. On reconstruction exception call `bundle.blocked` and continue. Atomically write `evaluation_summary.json` with entry, reconstruction identity/path, every evaluator status/output/error, and overall success.

- [ ] **Step 4: Add exact counts/failure tests**

Require dual or ATE-only config: 15 variants; pointcloud-only: 6; traditional: exactly 3 false-refinement variants. Prove one reconstruction failure blocks its evaluators but later entries run. Prove failed point cloud preserves passed ATE.

Add one CPU end-to-end test that writes a three-frame no-loop artifact, a TUM
trajectory truth file, and a point-map NPZ; run the real ATE and point-cloud
input functions through the bundle, write `evaluation_summary.json`, rerun,
and assert both passed evaluators are resumed without another evaluator call.

- [ ] **Step 5: Implement CLI summary/exit**

Preserve v1 output. For v2 print:

`entries=<N> evaluators=<N> passed=<N> skipped=<N> failed=<N> blocked=<N>`

Return 1 for any failed/blocked record, otherwise 0. Add monkeypatched CLI tests.

- [ ] **Step 6: Add checked-in version-2 config**

Create `configs/experiments/keyframe_metrics_matrix.yaml` using Task 4 schema, relative fixture paths, both declarations, canonical segmentation/refinement, output `outputs/experiments/keyframe-metrics`.

- [ ] **Step 7: Run GREEN and commit**

Run:

`pytest tests/experiments/test_keyframe_metrics_experiment.py tests/experiments/test_evaluation_bundle.py tests/test_run_experiment_matrix_cli.py -q`

Expected: all pass.

Commit:

```bash
git add experiments/runner.py run_experiment_matrix.py configs/experiments/keyframe_metrics_matrix.yaml \
  tests/experiments/test_keyframe_metrics_experiment.py tests/test_run_experiment_matrix_cli.py
git commit -m "feat: execute capability-aware keyframe metrics"
```

---

### Task 7: Documentation and complete verification

**Files:**
- Modify: `docs/reconstruction-evaluation-cloud-validation.md`
- Modify: `tests/test_cloud_validation_commands.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: reproducible dual/ATE-only/pointcloud-only workflow and fresh acceptance evidence.

- [ ] **Step 1: Write failing documentation assertions**

Require document strings: `keyframe_metrics_matrix.yaml`, `trajectory_ground_truth`, `pointcloud_ground_truth`, `evaluation_summary.json`. Keep executing documented help commands.

Run: `pytest tests/test_cloud_validation_commands.py -q`

Expected: missing documentation assertions fail.

- [ ] **Step 2: Document exact workflows/statuses**

Document dual capability, removing point-cloud declaration for ATE-only, removing trajectory declaration for pointcloud-only, traditional forced keyframe off, output layout, passed/skipped/failed/blocked, and rerun/resume. Do not include credentials or host-specific paths.

- [ ] **Step 3: Run docs/CLI smoke tests and commit**

```bash
pytest tests/test_cloud_validation_commands.py tests/test_run_experiment_matrix_cli.py -q
python run_reconstruction.py --help
python evaluate_ate.py --help
python evaluate_pointcloud.py --help
python run_experiment_matrix.py --help
git add docs/reconstruction-evaluation-cloud-validation.md tests/test_cloud_validation_commands.py
git commit -m "docs: add keyframe ATE validation workflow"
```

Expected: tests pass; every CLI exits 0; commit succeeds.

- [ ] **Step 4: Run focused acceptance suite**

```bash
pytest -q \
  tests/reconstruction/test_residual_alignment.py \
  tests/reconstruction/test_no_loop_mode.py \
  tests/reconstruction/test_corrected_mode.py \
  tests/reconstruction/test_traditional_mode.py \
  tests/test_corrected_loop_method.py \
  tests/test_traditional_loop_method.py \
  tests/experiments/test_matrix.py \
  tests/experiments/test_keyframe_metrics_experiment.py \
  tests/experiments/test_evaluation_bundle.py \
  tests/test_run_experiment_matrix_cli.py
```

Expected: zero failures.

- [ ] **Step 5: Run full regression**

If extension imports are absent, first run `python setup.py build_ext --inplace`.

Run: `pytest -q`

Expected: zero failures.

- [ ] **Step 6: Verify protected files/cache/source hygiene**

```bash
git diff 02bca0d22c044e0553691e415b41866cf64a1a49 --exit-code -- \
  reconstruction/modes/traditional.py loop_closure/methods/traditional.py
pytest -q tests/test_prediction_fingerprint.py tests/test_prediction_store.py tests/test_prediction_provider.py
git diff --check
git status --short
```

Expected: no traditional diff, cache tests pass, no whitespace errors, no unintended generated files. Record exact test counts, CLI exits, protected-file result, and final commit in the handoff; if a fix changes source, repeat the affected red/green cycle and Steps 4–6.
