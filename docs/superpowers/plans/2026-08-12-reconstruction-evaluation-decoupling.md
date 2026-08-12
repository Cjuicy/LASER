# LASER Reconstruction and Evaluation Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three algorithm-preserving LASER reconstruction modes with selectable Depth/Geometry/Atomic segmentation, typed reusable artifacts, a nine-case ATE matrix, and a no-loop-only three-case point-cloud matrix whose evaluators never reconstruct data.

**Architecture:** A shared ordinary PI3 prediction stream feeds one of three complete reconstruction modes: no-loop incremental, Traditional deferred, or Corrected online. The modes share component interfaces but retain their existing state-transition order, emit one `ReconstructionArtifact`, and are followed by artifact-only trajectory or point-cloud evaluators. Experiment planning owns the valid matrices and reuses reconstruction artifacts independently of evaluation type.

**Tech Stack:** Python 3.11, PyTorch, NumPy 1.26.4, OmegaConf/Hydra, SciPy, scikit-image, Open3D, evo, pytest, Cython, Git.

## Global Constraints

- Public segmentation methods remain exactly `depth`, `geometry`, and `atomic`.
- Public reconstruction modes are exactly `no_loop`, `traditional`, and `corrected`.
- Traditional must retain deferred adjacent Sim(3)/anchor-scale application, post-window loop detection, Traditional constraints, optimization, and final aggregation order.
- Corrected must retain online adjacent Sim(3)/anchor-scale application, post-window loop detection, local-constraint conversion, sequential-edge optimization, and one-time correction delta application.
- no-loop must retain LASER Table 4 incremental assembly order and must never construct or call loop components.
- ATE supports `depth|geometry|atomic × no_loop|traditional|corrected`; point-cloud evaluation supports `depth|geometry|atomic × no_loop` only.
- `window.size > window.overlap >= 1`; neither evaluator may lock window size or overlap.
- Ordinary PI3 prediction cache identity must remain independent of segmentation, reconstruction mode, anchor, loop, and evaluation settings.
- Evaluators may import artifact views and metric/data helpers only; they may not import reconstruction, segmentation, anchor propagation, prediction providers, or loop closure.
- The LASER `AnchorPropagator` mathematical implementation remains unchanged and is not copied.
- Existing command/config/internal-API compatibility is not required; do not add aliases or adapters for retired fields.
- Existing untracked generated files `inference_engine/utils/_segmentation_cy.cpp` and `inference_engine/utils/fast_seg.cpp` are outside this work and must never be staged.
- Every production change follows RED → verify expected failure → GREEN → full focused regression → commit.
- Final cloud instructions must reference the actually pushed `codex/laser-paper-pointmap-eval` branch and its verified commit.

---

## File Responsibility Map

Create these focused modules:

```text
pipeline/artifacts.py
    Typed reconstruction artifact/views plus validated atomic writer/loader.

reconstruction/prediction_stream.py
    Canonical WindowSpec-to-WindowPrediction adapter over ordinary PI3 provider.
reconstruction/shared.py
    Shared validation, confidence-mask, segmentation-graph, Sim(3), and overlap helpers.
reconstruction/registry.py
    Exact ReconstructionMode-to-runner registry.
reconstruction/modes/base.py
    ReconstructionContext, protocol, and mode-neutral diagnostic contracts.
reconstruction/modes/no_loop.py
    LASER incremental no-loop state machine.
reconstruction/modes/traditional.py
    Traditional deferred state machine and final loop aggregation.
reconstruction/modes/corrected.py
    Corrected online state machine and one-time loop delta aggregation.

loop_closure/types.py
    Sim3, LoopCandidate, LoopConstraint, and LoopSolution validation.
loop_closure/detection.py
    LoopDetector interface/adapter over existing SALAD detection.
loop_closure/evidence.py
    Mode-neutral loop-window protocol and joint A/B evidence provider.
loop_closure/methods/traditional.py
    Traditional constraint, optimizer, and aggregate math without a WindowEngine.
loop_closure/methods/corrected.py
    Corrected constraint, optimizer, and delta math without a WindowEngine.

evaluation/trajectory/{config,evaluator,results}.py
    Artifact-only ATE/RPE configuration, math, and result persistence.
evaluation/pointcloud/{config,evaluator,geometry_metrics,results}.py
    Artifact-only point-cloud protocol, metrics, and result persistence.

experiments/{config,matrix,ate,pointcloud}.py
    Dataset orchestration, valid matrix expansion, artifact reuse, and evaluator dispatch.

run_reconstruction.py
evaluate_ate.py
evaluate_pointcloud.py
run_experiment_matrix.py
    New public commands with no legacy aliases.
```

Modify these existing modules:

```text
pipeline/config.py
    Version-2 schema with ReconstructionMode and optional loop configuration.
pipeline/preflight.py
    Mode-sensitive checkpoint validation.
pipeline/runner.py
    Assemble provider, segmenter, anchor, mode, and artifact writer only.
pipeline/diagnostics.py
    Consume mode-neutral diagnostics instead of WindowCache/LoopSolution.
inference_engine/segmentation/{confidence,depth}.py
    Explicit Depth confidence quantile rule; one strategy implementation.
loop_closure/constraint_estimation.py
    Accept the mode-neutral LoopWindow protocol.
inference_engine/prediction_cache/fingerprint.py
    Keep and test the mode-independent ordinary cache contract.
README.md
    New architecture, commands, matrices, and cloud entry points.
```

Delete only after replacements pass:

```text
mv_recon/eval.py
mv_recon/paper_streaming.py
mv_recon/protocol.py
mv_recon/loop_experiment.py
mv_recon/compare_loop_results.py
configs/evaluation/mv_recon_laser_nrgbd_depth_loop_traditional.yaml
tests/mv_recon/test_eval_orchestration.py
tests/mv_recon/test_paper_streaming.py
tests/mv_recon/test_protocol.py
tests/mv_recon/test_loop_experiment.py
tests/mv_recon/test_compare_loop_results.py
```

Move, do not rewrite, pure point-cloud metric/result behavior from `mv_recon/geometry_metrics.py`, `mv_recon/results.py`, and the necessary dataset-plan/hash helpers in `mv_recon/protocol.py`; delete the originals after import and numerical parity tests pass.

---

### Task 1: Freeze Existing Algorithm Behavior Before Moving It

**Files:**
- Create: `tests/reconstruction/test_legacy_mode_characterization.py`
- Create: `tests/reconstruction/fixtures.py`
- Read: `tests/mv_recon/test_paper_streaming.py`
- Read: `tests/test_traditional_loop_method.py`
- Read: `tests/test_corrected_loop_method.py`

**Interfaces:**
- Consumes: current `reconstruct_incremental_point_maps`, `TraditionalWindowEngine`/`TraditionalLoopClosureStrategy`, and `CorrectedWindowEngine`/`CorrectedLoopClosureStrategy`.
- Produces: hand-derived literal characterization fixtures reused unchanged when imports switch to new modes in Tasks 5, 7, and 8.

- [ ] **Step 1: Add complete deterministic fixtures**

Create `tests/reconstruction/fixtures.py` with real tensors and literal expectations:

```python
from dataclasses import dataclass

import numpy as np
import torch

from inference_engine.prediction_cache.types import WindowSpec
from inference_engine.segmentation.base import SegmentationResult


SPECS = (
    WindowSpec(0, 0, 2),
    WindowSpec(1, 1, 3),
    WindowSpec(2, 2, 4),
)


class LiteralProvider:
    def get(self, spec, images):
        count = spec.frame_count
        points = torch.zeros((1, count, 1, 1, 3), dtype=torch.float32)
        points[..., 2] = 1.0
        return {
            "local_points": points,
            "camera_poses": torch.eye(4).repeat(1, count, 1, 1),
            "conf": torch.arange(count, dtype=torch.float32).view(1, count, 1, 1),
            "images": images.unsqueeze(0),
        }


class OneRegionSegmenter:
    def segment(self, points, confidence, images):
        assert points.shape[0] == confidence.shape[0] == images.shape[0]
        return [
            SegmentationResult(
                labels=np.zeros(points.shape[1:3], dtype=np.intp),
                diagnostics={"method": "depth", "region_count": 1},
            )
            for _ in points
        ]


@dataclass
class SequencedAnchor:
    scales: tuple[float, ...] = (5.0, 7.0)
    calls: list[tuple[float, float]] | None = None

    def __post_init__(self):
        self.calls = []

    def propagate(self, source, target, *unused):
        self.calls.append((float(source[-1, 0, 0, 2]), float(target[0, 0, 0, 2])))
        scale = self.scales[len(self.calls) - 1]
        return torch.full((*target.shape[:-1], 1), scale)
```

- [ ] **Step 2: Add characterization tests with independently derived values**

Create `tests/reconstruction/test_legacy_mode_characterization.py` and assert:

```python
def test_legacy_no_loop_uses_corrected_predecessor_for_next_registration():
    # Registration scales 2 then 3; anchor scales 5 then 7.
    # The third registration must see 1 * 2 * 5 == 10 from window two.
    assert registration_sources == [(1.0, 1.0), (10.0, 1.0)]
    assert anchor.calls == [(1.0, 2.0), (10.0, 3.0)]
    assert result.local_points[:, 0, 0, 2].tolist() == [1.0, 1.0, 10.0, 21.0]


def test_legacy_traditional_records_scale_before_final_application():
    assert second_cache.local_points[:, 0, 0, 2].tolist() == [1.0, 1.0]
    assert second_cache.anchor_scale_mask[:, 0, 0, 0].tolist() == [3.0, 3.0]
    assert aggregated.payload["local_points"][:, 0, 0, 2].tolist() == [1.0, 1.0, 6.0]


def test_legacy_corrected_applies_online_scale_once():
    assert second_cache.local_points[:, 0, 0, 2].tolist() == [6.0, 6.0]
    assert aggregate.payload["local_points"][:, 0, 0, 2].tolist() == [1.0, 1.0, 6.0]
```

Use the existing real classes and the complete fixture structures from their current tests. Do not assert calls on mocks; assert stored/final tensors and the actual predecessor values received by registration.

- [ ] **Step 3: Run the characterization suite**

Run:

```bash
pytest -q \
  tests/reconstruction/test_legacy_mode_characterization.py \
  tests/mv_recon/test_paper_streaming.py \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py
```

Expected: PASS. These characterize existing behavior; no production behavior is added in this task.

- [ ] **Step 4: Perform the mutation check**

Temporarily change each literal test fixture once in the test working copy—registration scale `2.0→4.0`, Traditional deferred flag `False→True`, and Corrected delta scale `1.0→2.0`—and confirm the relevant test fails. Restore the fixture edits and rerun the command from Step 3 to PASS.

- [ ] **Step 5: Commit the regression guard**

```bash
git add tests/reconstruction/fixtures.py tests/reconstruction/test_legacy_mode_characterization.py
git commit -m "test: characterize reconstruction mode ordering"
```

---

### Task 2: Replace Loop Enable/Method With the Version-2 Reconstruction Schema

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/preflight.py`
- Create: `configs/reconstruction/pi3_laser.yaml`
- Create: `configs/reconstruction/pi3_laser_no_loop.yaml`
- Modify: `tests/test_pipeline_config.py`
- Modify: `tests/test_pipeline_runner.py`

**Interfaces:**
- Produces: `ReconstructionMode`, `ReconstructionConfig`, top-level `RegistrationConfig`, `PipelineConfig.reconstruction`, and `PipelineConfig.loop: LoopConfig | None`.
- Consumed by: every mode, preflight, runner, matrix planner, artifact manifest.

- [ ] **Step 1: Write failing schema and preflight tests**

Add these behavior tests:

```python
def test_version_two_selects_exact_reconstruction_mode():
    loaded = load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml",
        ("reconstruction.mode=traditional",),
    )
    assert loaded.config.version == 2
    assert loaded.config.reconstruction.mode is ReconstructionMode.TRADITIONAL
    assert loaded.config.registration.confidence_keep_ratio == pytest.approx(0.5)


def test_retired_loop_enable_and_method_are_rejected(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        Path("configs/reconstruction/pi3_laser.yaml").read_text()
        + "\nloop:\n  enabled: false\n  method: traditional\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown configuration field"):
        load_pipeline_config(path)


def test_no_loop_config_may_omit_loop_section():
    loaded = load_pipeline_config("configs/reconstruction/pi3_laser_no_loop.yaml")
    assert loaded.config.reconstruction.mode is ReconstructionMode.NO_LOOP
    assert loaded.config.loop is None


def test_loop_mode_requires_loop_configuration():
    with pytest.raises(ValueError, match="traditional requires loop configuration"):
        load_pipeline_config(
            "configs/reconstruction/pi3_laser_no_loop.yaml",
            ("reconstruction.mode=traditional",),
        )


def test_no_loop_preflight_does_not_require_loop_checkpoints(tmp_path):
    config = no_loop_config_with_existing_model_checkpoint(tmp_path)
    validate_preflight(config, three_frame_manifest(tmp_path), cuda_available=False)
```

Also parameterize the window validation over `(10,5)`, `(20,5)`, `(20,10)` and reject `(5,5)`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest -q \
  tests/test_pipeline_config.py \
  tests/test_pipeline_runner.py -k "version_two or retired_loop or no_loop_config or loop_mode_requires or no_loop_preflight or window"
```

Expected: FAIL because `ReconstructionMode`, version 2, top-level registration, and optional loop configuration do not exist.

- [ ] **Step 3: Implement the exact schema**

Add to `pipeline/config.py`:

```python
class ReconstructionMode(str, Enum):
    NO_LOOP = "no_loop"
    TRADITIONAL = "traditional"
    CORRECTED = "corrected"


class ConfidenceQuantileMethod(str, Enum):
    HIGHER = "higher"
    NEAREST = "nearest"


@dataclass(frozen=True)
class ReconstructionConfig:
    mode: ReconstructionMode = MISSING


@dataclass(frozen=True)
class RegistrationConfig:
    confidence_keep_ratio: float = MISSING


@dataclass(frozen=True)
class LoopConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
```

Add `confidence_quantile_method: ConfidenceQuantileMethod` to `SegmentationConfig`. `higher` preserves the existing ATE/strategy behavior; the point-cloud Table 4 experiment will explicitly override it to `nearest`. This removes the hidden evaluator-only `PaperDepthSegmentationStrategy` while preserving both numerical protocols through one strategy implementation.

Set `PipelineConfig` fields in this order after anchor propagation:

```python
registration: RegistrationConfig = field(default_factory=RegistrationConfig)
reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
loop: LoopConfig | None = None
```

Normalize the two new enums, require `version == 2`, validate registration for every mode, and validate loop detection/constraint/optimizer only when mode is Traditional or Corrected. When a loop mode has `loop is None`, raise the exact mode-specific error before any file access.

- [ ] **Step 4: Add canonical reconstruction configs**

`configs/reconstruction/pi3_laser.yaml` contains all loop fields and defaults to `reconstruction.mode: corrected`. `configs/reconstruction/pi3_laser_no_loop.yaml` contains the same PI3/window/segmentation/anchor/registration fields, sets `reconstruction.mode: no_loop`, and omits `loop` entirely. Both use version 2 and legal 20/5 windows.

- [ ] **Step 5: Make preflight mode-sensitive**

Replace `if config.loop.enabled` with:

```python
if config.reconstruction.mode is not ReconstructionMode.NO_LOOP:
    assert config.loop is not None
    required_files.extend(
        (
            ("loop.detection.salad_checkpoint", config.loop.detection.salad_checkpoint),
            ("loop.detection.dino_checkpoint", config.loop.detection.dino_checkpoint),
        )
    )
```

Do not access `config.loop` on the no-loop branch.

- [ ] **Step 6: Verify GREEN and full config regression**

Run:

```bash
pytest -q tests/test_pipeline_config.py tests/test_pipeline_runner.py -k "config or preflight or window"
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/config.py pipeline/preflight.py configs/reconstruction tests/test_pipeline_config.py tests/test_pipeline_runner.py
git commit -m "refactor: define explicit reconstruction modes"
```

---

### Task 3: Introduce Validated Reconstruction Artifacts and Narrow Views

**Files:**
- Create: `pipeline/artifacts.py`
- Create: `tests/test_reconstruction_artifacts.py`
- Modify: `pipeline/__init__.py`

**Interfaces:**
- Produces: `ReconstructionDiagnostics`, `ReconstructionArtifact`, `TrajectoryEstimate`, `PointMapEstimate`, `write_reconstruction_artifact()`, `load_reconstruction_artifact()`, `load_trajectory_estimate()`, and `load_pointmap_estimate()`.
- Consumed by: all modes, pipeline runner, both evaluators, and experiments.

- [ ] **Step 1: Write failing validation and persistence tests**

```python
def test_artifact_requires_one_consistent_finite_frame_axis():
    with pytest.raises(ValueError, match="frame count"):
        make_artifact(frame_ids=(0, 1), camera_poses=torch.eye(4).repeat(3, 1, 1))


def test_artifact_views_expose_only_evaluator_fields():
    artifact = make_artifact()
    assert artifact.trajectory == TrajectoryEstimate(
        frame_ids=(0, 1, 2),
        camera_poses=artifact.camera_poses,
    )
    assert artifact.pointmap.reconstruction_mode is ReconstructionMode.NO_LOOP
    assert not hasattr(artifact.trajectory, "global_points")
    assert not hasattr(artifact.pointmap, "camera_poses")


def test_artifact_round_trip_validates_tensor_digest(tmp_path):
    output = write_reconstruction_artifact(
        make_artifact(),
        tmp_path / "run",
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )
    assert load_reconstruction_artifact(output) == make_artifact()
    (output / "pointmap.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="pointmap.pt digest mismatch"):
        load_pointmap_estimate(output)
```

Use `torch.equal` helpers for dataclass tensor comparison rather than relying on tensor `==`.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_reconstruction_artifacts.py
```

Expected: import failure because `pipeline.artifacts` does not exist.

- [ ] **Step 3: Implement typed contracts and validation**

Use schema version `1`. Validate:

```python
local_points:  (N,H,W,3)
global_points: (N,H,W,3)
camera_poses:  (N,4,4)
confidence:    (N,H,W)
len(frame_ids) == N
frame_ids are unique non-negative integers
prediction_key is a non-empty string
all tensors are CPU floating tensors containing finite values
```

`ReconstructionDiagnostics` contains immutable mappings/tuples for stage timings, segmentation summaries, candidate count, constraint count, and mode-specific scalar diagnostics. It must not contain live graph vertices, models, providers, or tensors.

- [ ] **Step 4: Implement atomic persistence**

Write tensors to temporary sibling files, `fsync`, then `Path.replace()`. Write `manifest.json` last. For each tensor record filename, shape, dtype, and SHA256. Loader reads and validates manifest schema/digests before `torch.load(weights_only=True)`, then constructs the typed artifact so shape/finite validation runs again.

`load_trajectory_estimate()` reads only `trajectory.pt` plus manifest; `load_pointmap_estimate()` reads only `pointmap.pt`, `confidence.pt`, and manifest.

- [ ] **Step 5: Run GREEN**

```bash
pytest -q tests/test_reconstruction_artifacts.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/artifacts.py pipeline/__init__.py tests/test_reconstruction_artifacts.py
git commit -m "feat: add typed reconstruction artifacts"
```

---

### Task 4: Extract the Shared Ordinary Prediction Stream

**Files:**
- Create: `reconstruction/__init__.py`
- Create: `reconstruction/prediction_stream.py`
- Create: `tests/reconstruction/test_prediction_stream.py`
- Modify: `inference_engine/prediction_cache/provider.py`

**Interfaces:**
- Consumes: `OrdinaryPredictionProvider.get(spec, images)` and canonical `WindowSpec` tuples.
- Produces: `WindowPrediction.from_mapping(mapping, spec, prediction_key, process_device)` and `iter_window_predictions(provider, specs, images, process_device)`.

- [ ] **Step 1: Write failing real-stream tests**

```python
def test_stream_yields_canonical_typed_predictions_once():
    provider = CompleteRecordingProvider(prediction_key="p" * 64)
    predictions = tuple(
        iter_window_predictions(
            provider,
            SPECS,
            torch.zeros((4, 3, 2, 2)),
            process_device="cpu",
        )
    )
    assert [item.spec for item in predictions] == list(SPECS)
    assert provider.specs == list(SPECS)
    assert all(item.prediction_key == "p" * 64 for item in predictions)
    assert predictions[0].local_points.shape == (2, 2, 2, 3)


def test_stream_rejects_nonfinite_prediction_with_window_context():
    provider = NonfiniteCompleteProvider(prediction_key="p" * 64)
    with pytest.raises(ValueError, match="window 1 frames 1:3.*local_points"):
        tuple(iter_window_predictions(provider, SPECS, images, "cpu"))
```

The test provider must return all real fields: batched local points, camera poses, confidence, and images; do not use a partial mapping.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/reconstruction/test_prediction_stream.py
```

Expected: import failure for `reconstruction.prediction_stream`.

- [ ] **Step 3: Implement `WindowPrediction`**

```python
@dataclass(frozen=True)
class WindowPrediction:
    spec: WindowSpec
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    images: torch.Tensor
    prediction_key: str
```

Implement `WindowPrediction.from_mapping(mapping: Mapping[str, object], spec: WindowSpec, prediction_key: str, process_device: str) -> WindowPrediction`. It squeezes exactly one model batch dimension, moves tensors to `process_device`, clones no inputs unnecessarily, and attaches window index/frame range to every validation error.

- [ ] **Step 4: Implement canonical iteration**

Validate the images tensor against the last `WindowSpec`, slice each exact frame range, call `provider.get` once per spec, and yield in order. Read `prediction_key` from `provider.store.fingerprint.key`; add a read-only `prediction_key` property to `OrdinaryPredictionProvider` so tests and alternative providers expose the same contract.

- [ ] **Step 5: Preserve cache identity behavior**

Run:

```bash
pytest -q \
  tests/reconstruction/test_prediction_stream.py \
  tests/test_prediction_provider.py \
  tests/test_prediction_cache_matrix.py \
  tests/test_prediction_cache_method_parity.py \
  tests/test_prediction_fingerprint.py
```

Expected: PASS; ordinary key remains unchanged across segmentation and reconstruction mode changes.

- [ ] **Step 6: Commit**

```bash
git add reconstruction inference_engine/prediction_cache/provider.py tests/reconstruction/test_prediction_stream.py
git commit -m "refactor: expose shared PI3 prediction stream"
```

---

### Task 5: Implement the no-loop Incremental Mode From the Table 4 Path

**Files:**
- Create: `reconstruction/modes/__init__.py`
- Create: `reconstruction/modes/base.py`
- Create: `reconstruction/shared.py`
- Create: `reconstruction/modes/no_loop.py`
- Create: `tests/reconstruction/test_no_loop_mode.py`
- Modify: `inference_engine/segmentation/confidence.py`
- Modify: `inference_engine/segmentation/depth.py`
- Modify: `tests/reconstruction/test_legacy_mode_characterization.py`

**Interfaces:**
- Produces: `ReconstructionContext`, `ReconstructionModeRunner`, `NoLoopReconstructionMode.run(context) -> ReconstructionArtifact`.
- Consumes: `WindowPrediction`, selected `SegmentationStrategy`, the unchanged `AnchorPropagator`, registration config, window schedule, and artifact type.

- [ ] **Step 1: Switch the no-loop characterization import and watch it fail**

Replace the legacy function import in the characterization test with:

```python
from reconstruction.modes.no_loop import NoLoopReconstructionMode
```

Add a boundary test:

```python
def test_no_loop_dependencies_contain_no_loop_services():
    parameters = inspect.signature(NoLoopReconstructionMode).parameters
    assert "loop_detector" not in parameters
    assert "constraint_estimator" not in parameters
    assert "optimizer" not in parameters
```

Run:

```bash
pytest -q tests/reconstruction/test_no_loop_mode.py tests/reconstruction/test_legacy_mode_characterization.py -k no_loop
```

Expected: FAIL because the new mode does not exist.

- [ ] **Step 2: Implement mode-neutral context**

```python
@dataclass(frozen=True)
class ReconstructionContext:
    predictions: Iterable[WindowPrediction]
    frame_ids: tuple[int, ...]
    segmentation_strategy: SegmentationStrategy
    anchor_propagator: AnchorPropagator
    segmentation_config: SegmentationConfig
    anchor_config: AnchorPropagationConfig
    registration_config: RegistrationConfig
    window_config: WindowConfig
    reconstruction_mode: ReconstructionMode
```

The mode protocol has one method `run(context) -> ReconstructionArtifact`. Do not include loop services in this common context; loop modes receive them in their own constructors.

- [ ] **Step 3: Make Depth quantile selection explicit in the shared strategy**

Change `select_numpy_top_confidence_mask(confidence, keep_ratio, method)` to accept only `higher` or `nearest`, and route `SegmentationConfig.confidence_quantile_method.value` through `DepthSegmentationStrategy`. Preserve `higher` as the ATE default; set the future point-cloud matrix override to `nearest`.

Add literal test input `[0,1,2,3,4,5]` proving that `higher` selects `3..5` and `nearest` selects `2..5`. This test catches accidental reintroduction of the hidden evaluator-only depth strategy.

- [ ] **Step 4: Migrate the incremental state machine**

Move the behavior of `reconstruct_incremental_point_maps` into `NoLoopReconstructionMode.run` with this exact order for every `WindowPrediction`: establish or reuse the first-window intrinsic; form current/previous overlap confidence masks; register against the already corrected predecessor; immediately apply the adjacent Sim(3); segment the transformed current points; build temporal graphs from those segmentation results; call the unchanged `AnchorPropagator.propagate(previous_points_numpy, current_points_numpy, previous_graph, current_graph, overlap)`; immediately multiply the current points by its returned layer-scale mask; append only the non-overlap frames; and store the corrected current points/poses/graph as the next predecessor.

Return a `ReconstructionArtifact` with `reconstruction_mode=NO_LOOP`. Do not import any `loop_closure` module. Keep the first-window intrinsic and confidence quantile behavior selected by configuration.

- [ ] **Step 5: Verify GREEN and exact legacy outputs**

```bash
pytest -q \
  tests/reconstruction/test_no_loop_mode.py \
  tests/reconstruction/test_legacy_mode_characterization.py -k no_loop \
  tests/test_anchor_propagation_contract.py \
  tests/test_segmentation_strategies.py \
  tests/test_depth_segmentation_stages.py
```

Expected: PASS, including literal `[1,1,10,21]` point depths and the corrected predecessor `[1,10]` registration sources.

- [ ] **Step 6: Commit**

```bash
git add reconstruction/modes reconstruction/shared.py inference_engine/segmentation tests/reconstruction
git commit -m "feat: add incremental no-loop reconstruction mode"
```

---

### Task 6: Separate Loop Detection, Evidence, and Pure Loop Types From Engines

**Files:**
- Create: `loop_closure/types.py`
- Create: `loop_closure/detection.py`
- Create: `loop_closure/evidence.py`
- Modify: `loop_closure/constraint_estimation.py`
- Modify: `loop_closure/methods/shared.py`
- Modify: `loop_closure/__init__.py`
- Create: `tests/test_loop_component_boundaries.py`
- Modify: `tests/test_loop_constraint_estimation.py`

**Interfaces:**
- Produces: `Sim3`, `LoopCandidate`, `LoopConstraint`, `LoopSolution`, `LoopWindow`, `LoopDetector.detect()`, and `LoopEvidenceProvider.estimate()`.
- Consumed by: Traditional and Corrected modes/math in Tasks 7 and 8.

- [ ] **Step 1: Write failing mode-neutral loop component tests**

```python
def test_loop_detector_returns_typed_candidates_from_manifest(tmp_path):
    backend = CompleteSaladBackend(
        descriptors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        candidates=((12, 2, 0.8),),
    )
    detector = SaladLoopDetector(
        config,
        output_path=tmp_path / "candidates.txt",
        backend=backend,
    )
    result = detector.detect(thirteen_frame_manifest, thirteen_frame_images)
    assert result == (LoopCandidate(frame_a=12, frame_b=2, similarity=0.8),)


def test_joint_evidence_accepts_mode_neutral_loop_window():
    window = CompleteLoopWindow(
        window_index=0,
        frame_start=0,
        frame_end=2,
        local_points=torch.ones((2, 1, 1, 3)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
    )
    other = CompleteLoopWindow(
        window_index=1,
        frame_start=2,
        frame_end=4,
        local_points=torch.ones((2, 1, 1, 3)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
    )
    candidate = LoopCandidate(frame_a=2, frame_b=0, similarity=0.8)
    alignment_a, alignment_b = provider.estimate(other, window, candidate)
    validate_sim3(alignment_a)
    validate_sim3(alignment_b)
```

The complete fake backend must mirror descriptor arrays, frame indices, similarities, and output metadata used by real detection.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_loop_component_boundaries.py tests/test_loop_constraint_estimation.py
```

Expected: import failures for the new modules/protocols.

- [ ] **Step 3: Move pure types without changing validation**

Move `Sim3`, `validate_sim3`, `LoopCandidate`, `LoopConstraint`, and `LoopSolution` from `loop_closure/methods/base.py` into `loop_closure/types.py`. Update imports without changing their validation or field names.

- [ ] **Step 4: Define the mode-neutral evidence boundary**

```python
@runtime_checkable
class LoopWindow(Protocol):
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor


class LoopEvidenceProvider(Protocol):
    def estimate(
        self,
        window_a: LoopWindow,
        window_b: LoopWindow,
        candidate: LoopCandidate,
    ) -> tuple[Sim3, Sim3]:
        raise NotImplementedError
```

Wrap the current `JointAlignmentEstimator`; keep centered frame selection, joint PI3 forward kind, confidence intersection, and adjacent registration unchanged.

- [ ] **Step 5: Wrap detection as its own component**

Define a `LoopDetector` protocol with `detect(manifest: ImageManifest, images: torch.Tensor) -> tuple[LoopCandidate, ...]`. Implement `SaladLoopDetector(config, output_path, backend)` as the adapter over the existing SALAD candidate implementation; keep the output path and backend in its constructor, and return the typed immutable tuple from `detect`. It must not accept segmentation, anchor, window state, or evaluator objects.

- [ ] **Step 6: Verify GREEN and existing loop contracts**

```bash
pytest -q \
  tests/test_loop_component_boundaries.py \
  tests/test_loop_constraint_estimation.py \
  tests/test_loop_method_contracts.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add loop_closure tests/test_loop_component_boundaries.py tests/test_loop_constraint_estimation.py tests/test_loop_method_contracts.py
git commit -m "refactor: separate loop detection and evidence"
```

---

### Task 7: Migrate Traditional Deferred Reconstruction Without a WindowEngine

**Files:**
- Create: `reconstruction/modes/traditional.py`
- Modify: `loop_closure/methods/traditional.py`
- Modify: `loop_closure/methods/registry.py`
- Create: `tests/reconstruction/test_traditional_mode.py`
- Modify: `tests/reconstruction/test_legacy_mode_characterization.py`
- Modify: `tests/test_traditional_loop_method.py`

**Interfaces:**
- Produces: `TraditionalWindowState`, `TraditionalReconstructionMode.run(context)`, and pure `TraditionalLoopProcessor` methods.
- Consumes: shared prediction stream/context, `LoopDetector`, `LoopEvidenceProvider`, `Sim3LoopOptimizer`, selected segmentation, and unchanged anchor propagator.

- [ ] **Step 1: Write failing deferred-order tests**

```python
def test_traditional_defers_all_recorded_corrections_until_aggregate():
    mode = TraditionalReconstructionMode(detector, evidence, optimizer)
    artifact = mode.run(context_with_registration_scale_2_and_anchor_scale_3())
    assert detector.predecessor_depths_seen == [(1.0, 1.0)]
    assert mode.trace[1].observation_local_points[:, 0, 0, 2].tolist() == [1.0, 1.0]
    assert mode.trace[1].anchor_scale_mask[:, 0, 0, 0].tolist() == [3.0, 3.0]
    assert artifact.local_points[:, 0, 0, 2].tolist() == [1.0, 1.0, 6.0]


def test_traditional_detects_only_after_all_windows_are_recorded():
    assert event_log == [
        "window:0", "window:1", "window:2", "detect", "evidence", "optimize", "aggregate"
    ]
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/reconstruction/test_traditional_mode.py tests/reconstruction/test_legacy_mode_characterization.py -k traditional
```

Expected: FAIL because the new Traditional mode does not exist.

- [ ] **Step 3: Implement immutable Traditional window state**

```python
@dataclass(frozen=True)
class TraditionalWindowState:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    segmentation_labels: tuple[np.ndarray, ...]
    anchor_scale_mask: torch.Tensor | None
    relative_sim3: Sim3
    segmentation_diagnostics: tuple[Mapping[str, object], ...]
```

It satisfies `LoopWindow` structurally and has no `loop_method`, `tag`, or anonymous `loop_state` dictionary.

- [ ] **Step 4: Migrate the window state machine exactly**

For each prediction, compute confidence masks, relative adjacent Sim(3), segmentation graph, and anchor scale mask exactly as `TraditionalWindowEngine`; never mutate points/poses with those corrections during the window loop. Append the typed state. After all states exist, call detector exactly once, build evidence/constraints, optimize, then aggregate.

- [ ] **Step 5: Strip engine ownership from Traditional loop math**

Rename `TraditionalLoopClosureStrategy` to `TraditionalLoopProcessor`; delete `create_window_engine`. Keep `compute_sim3_ab`, candidate mapping, constraint deduplication, optimizer input construction, accumulated transform order, anchor-scale application, and overlap trimming unchanged. Change its input type from `WindowCache` to `Sequence[TraditionalWindowState]` through the `LoopWindow` boundary where possible.

- [ ] **Step 6: Verify literal and full Traditional regressions**

```bash
pytest -q \
  tests/reconstruction/test_traditional_mode.py \
  tests/reconstruction/test_legacy_mode_characterization.py -k traditional \
  tests/test_traditional_loop_method.py \
  tests/test_loop_method_contracts.py
```

Expected: PASS with the same recorded raw states and final tensor values as Task 1.

- [ ] **Step 7: Commit**

```bash
git add reconstruction/modes/traditional.py loop_closure/methods tests/reconstruction tests/test_traditional_loop_method.py tests/test_loop_method_contracts.py
git commit -m "refactor: migrate traditional reconstruction mode"
```

---

### Task 8: Migrate Corrected Online Reconstruction Without a WindowEngine

**Files:**
- Create: `reconstruction/modes/corrected.py`
- Modify: `loop_closure/methods/corrected.py`
- Modify: `loop_closure/methods/registry.py`
- Create: `tests/reconstruction/test_corrected_mode.py`
- Modify: `tests/reconstruction/test_legacy_mode_characterization.py`
- Modify: `tests/test_corrected_loop_method.py`

**Interfaces:**
- Produces: `CorrectedWindowState`, `CorrectedReconstructionMode.run(context)`, and pure `CorrectedLoopProcessor` methods.
- Consumes: same shared components as Traditional, while retaining online correction state.

- [ ] **Step 1: Write failing online-order and one-delta tests**

```python
def test_corrected_uses_corrected_window_as_next_registration_source():
    artifact = mode.run(three_window_context(registration_scales=(2.0, 3.0), anchor_scales=(5.0, 7.0)))
    assert registration_sources == [(1.0, 1.0), (10.0, 1.0)]
    assert artifact.local_points[:, 0, 0, 2].tolist() == [1.0, 1.0, 10.0, 21.0]


def test_corrected_applies_optimized_original_delta_exactly_once():
    artifact = mode_with_optimized_absolute_scale(4.0).run(context_with_original_absolute_scale(2.0))
    assert artifact.local_points[-1, 0, 0, 2].item() == pytest.approx(12.0)
    # Online state is depth 6; delta is 4/2 == 2; 6*2 == 12, not 24.
```

Add event order ending in `detect,evidence,optimize,apply_delta` after all windows.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/reconstruction/test_corrected_mode.py tests/reconstruction/test_legacy_mode_characterization.py -k corrected
```

Expected: FAIL because the new Corrected mode does not exist.

- [ ] **Step 3: Implement immutable Corrected state**

```python
@dataclass(frozen=True)
class CorrectedWindowState:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    segmentation_labels: tuple[np.ndarray, ...]
    anchor_scale_mask: torch.Tensor | None
    sim3_abs: Sim3
    sim3_edge: Sim3 | None
    segmentation_diagnostics: tuple[Mapping[str, object], ...]
```

- [ ] **Step 4: Migrate online window processing exactly**

Apply adjacent absolute Sim(3) before segmentation/anchor, compute `sim3_edge` using the inverse of the previous absolute transform, apply anchor scale immediately, and store the corrected points/poses as the next predecessor. Detect loops only after all states are complete.

- [ ] **Step 5: Strip engine ownership from Corrected loop math**

Rename to `CorrectedLoopProcessor`; delete `create_window_engine`. Preserve `build_local_loop_constraint`, edge optimizer input, optimized absolute reconstruction, `optimized_abs * inverse(original_abs)` delta convention, and one-time delta application.

- [ ] **Step 6: Verify GREEN and all Corrected contracts**

```bash
pytest -q \
  tests/reconstruction/test_corrected_mode.py \
  tests/reconstruction/test_legacy_mode_characterization.py -k corrected \
  tests/test_corrected_loop_method.py \
  tests/test_loop_method_contracts.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add reconstruction/modes/corrected.py loop_closure/methods tests/reconstruction tests/test_corrected_loop_method.py tests/test_loop_method_contracts.py
git commit -m "refactor: migrate corrected reconstruction mode"
```

---

### Task 9: Build the Mode Registry, Pipeline Runner, Artifact Writer, and Reconstruction CLI

**Files:**
- Create: `reconstruction/registry.py`
- Rewrite: `pipeline/runner.py`
- Modify: `pipeline/diagnostics.py`
- Create: `run_reconstruction.py`
- Create: `tests/reconstruction/test_mode_registry.py`
- Rewrite: `tests/test_pipeline_runner.py`
- Create: `tests/test_run_reconstruction_cli.py`

**Interfaces:**
- Produces: `build_reconstruction_mode(mode, services)`, `PipelineRunner.run() -> ReconstructionArtifact`, `run_from_config()`, and CLI exit behavior.
- Consumes: config v2, ordinary prediction stream, selected segmentation/anchor, mode runners, artifact writer.

- [ ] **Step 1: Write failing dispatch and no-loop isolation tests**

```python
@pytest.mark.parametrize(
    ("name", "expected"),
    [("no_loop", NoLoopReconstructionMode), ("traditional", TraditionalReconstructionMode), ("corrected", CorrectedReconstructionMode)],
)
def test_registry_builds_exact_mode(name, expected):
    assert isinstance(build_reconstruction_mode(ReconstructionMode(name), services), expected)


def test_runner_no_loop_never_builds_loop_services(recording_dependencies):
    result = PipelineRunner(no_loop_loaded_config, dependencies=recording_dependencies).run()
    assert result.reconstruction_mode is ReconstructionMode.NO_LOOP
    assert recording_dependencies.events == [
        "preflight", "images", "fingerprint", "store", "model", "provider", "segmenter", "anchor", "no_loop", "write_artifact"
    ]


def test_runner_loop_modes_detect_after_prediction_stream_is_exhausted():
    assert events[-4:] == ["prediction_stream:end", "detect", "optimize", "write_artifact"]
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/reconstruction/test_mode_registry.py tests/test_pipeline_runner.py tests/test_run_reconstruction_cli.py
```

Expected: FAIL on the missing registry/new CLI and old runner contract.

- [ ] **Step 3: Implement exact registry**

Use an enum-keyed factory mapping with no fallback. no-loop factory receives no loop services; loop factories require non-None `LoopDetector`, `LoopEvidenceProvider`, and optimizer configuration.

- [ ] **Step 4: Rewrite runner orchestration**

Runner order:

```text
manifest → preflight → images/specs → fingerprint/store/model/provider
→ prediction stream → segmentation strategy → AnchorPropagator
→ mode registry/run → diagnostics → artifact writer
```

Remove all direct calls to `loop_strategy.create_window_engine`, `run_windows`, empty-constraint optimize, anonymous `ReconstructionResult`, and evaluator-related resizing. `PipelineRunner.run()` returns the typed artifact and writes the run directory.

- [ ] **Step 5: Make diagnostics mode-neutral**

Replace `WindowCache`/`LoopSolution` inputs with `ReconstructionDiagnostics`. Always report segmentation, mode, prediction-cache, timings, and frame/window coverage; only loop modes report candidates/constraints/optimizer scalars.

- [ ] **Step 6: Implement the public CLI**

`run_reconstruction.py` accepts only:

```text
--config PATH
--set KEY=VALUE  (repeatable)
```

It prints the resolved mode, segmentation, window/overlap, config hash, prediction key, and artifact directory. Invalid legacy flags fail argparse.

- [ ] **Step 7: Run GREEN and prediction regressions**

```bash
pytest -q \
  tests/reconstruction/test_mode_registry.py \
  tests/test_pipeline_runner.py \
  tests/test_run_reconstruction_cli.py \
  tests/test_prediction_cache_matrix.py \
  tests/test_prediction_cache_method_parity.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add reconstruction/registry.py pipeline/runner.py pipeline/diagnostics.py run_reconstruction.py tests/reconstruction tests/test_pipeline_runner.py tests/test_run_reconstruction_cli.py
git commit -m "feat: dispatch typed reconstruction pipeline"
```

---

### Task 10: Create an Artifact-Only Trajectory Evaluator

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/trajectory/__init__.py`
- Create: `evaluation/trajectory/config.py`
- Create: `evaluation/trajectory/evaluator.py`
- Create: `evaluation/trajectory/results.py`
- Create: `configs/evaluation/ate.yaml`
- Create: `evaluate_ate.py`
- Create: `tests/evaluation/test_trajectory_evaluator.py`
- Create: `tests/evaluation/test_evaluate_ate_cli.py`
- Read: `eval/vo_eval.py`

**Interfaces:**
- Consumes: `TrajectoryEstimate`, GT trajectory path/format, and `TrajectoryEvaluationConfig`.
- Produces: `TrajectoryMetrics(ate_rmse_m, rpe_translation_rmse_m, rpe_rotation_rmse_deg)` and atomic JSON output.

- [ ] **Step 1: Write failing numerical and narrow-load tests**

```python
def test_sim3_equivalent_trajectory_has_zero_ate_and_rpe():
    estimate = trajectory_estimate(translations=[(10, 0, 0), (12, 0, 0), (14, 0, 0)])
    ground_truth = ground_truth_trajectory(translations=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    result = evaluate_trajectory(estimate, ground_truth, default_ate_config())
    assert result.ate_rmse_m == pytest.approx(0.0, abs=1e-9)
    assert result.rpe_translation_rmse_m == pytest.approx(0.0, abs=1e-9)
    assert result.rpe_rotation_rmse_deg == pytest.approx(0.0, abs=1e-9)


def test_ate_cli_loads_trajectory_view_only(monkeypatch):
    monkeypatch.setattr(cli, "load_trajectory_estimate", recording_loader)
    argv = [
        "--artifact", str(artifact_dir),
        "--config", "configs/evaluation/ate.yaml",
        "--ground-truth", str(ground_truth_path),
        "--ground-truth-format", "tum",
        "--output", str(output_dir),
    ]
    assert cli.main(argv) == 0
    assert recording_loader.loaded_files == {"manifest.json", "trajectory.pt"}
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/evaluation/test_trajectory_evaluator.py tests/evaluation/test_evaluate_ate_cli.py
```

Expected: import failure for `evaluation.trajectory`.

- [ ] **Step 3: Implement typed config and pure evo math**

`configs/evaluation/ate.yaml` contains only:

```yaml
version: 1
alignment: sim3
rpe_delta_frames: 1
```

Move the `evo.main_ape`/`main_rpe` calls and TUM conversion needed from `eval/vo_eval.py` into pure functions that do not load models, images, point maps, or reconstruction configuration. Preserve `align=True`, `correct_scale=True`, translation-part APE/RPE, rotation-angle-degrees RPE, delta 1 frame, tolerance 0.01, and all-pairs behavior.

- [ ] **Step 4: Implement CLI and result writer**

CLI arguments:

```text
--artifact DIR
--config configs/evaluation/ate.yaml
--ground-truth PATH
--ground-truth-format sintel|replica|tum|tartanair
--output DIR
```

Write `trajectory_metrics.json` atomically with artifact manifest digest and all three metrics.

- [ ] **Step 5: Verify GREEN and old metric parity**

```bash
pytest -q tests/evaluation/test_trajectory_evaluator.py tests/evaluation/test_evaluate_ate_cli.py
```

Also run the literal trajectory through both the new evaluator and current `eval.vo_eval.eval_metrics`; assert absolute differences below `1e-9` in the parity test.

- [ ] **Step 6: Commit**

```bash
git add evaluation/trajectory evaluation/__init__.py configs/evaluation/ate.yaml evaluate_ate.py tests/evaluation
git commit -m "feat: add artifact-only trajectory evaluation"
```

---

### Task 11: Migrate Point-Cloud Metrics Into an Artifact-Only Evaluator

**Files:**
- Create: `evaluation/pointcloud/__init__.py`
- Create: `evaluation/pointcloud/config.py`
- Create: `evaluation/pointcloud/evaluator.py`
- Create: `evaluation/pointcloud/geometry_metrics.py`
- Create: `evaluation/pointcloud/results.py`
- Create: `configs/evaluation/pointcloud.yaml`
- Create: `evaluate_pointcloud.py`
- Create: `tests/evaluation/test_pointcloud_evaluator.py`
- Create: `tests/evaluation/test_pointcloud_results.py`
- Create: `tests/evaluation/test_evaluate_pointcloud_cli.py`
- Read: `mv_recon/geometry_metrics.py`
- Read: `mv_recon/results.py`

**Interfaces:**
- Consumes: `PointMapEstimate`, `PointCloudGroundTruth(point_maps, valid_mask)`, and `PointCloudEvaluationConfig`.
- Produces: existing `PrimaryMetrics`, diagnostics, dataset aggregation, and atomic JSON/CSV outputs.

- [ ] **Step 1: Write failing purity, mode, and numerical tests**

```python
def test_pointcloud_rejects_loop_artifact_before_backend_construction():
    estimate = pointmap_estimate(mode=ReconstructionMode.TRADITIONAL)
    with pytest.raises(ValueError, match="requires no_loop artifact"):
        evaluate_point_maps(estimate, ground_truth, config, backend_factory=forbidden_backend)


def test_identical_point_maps_keep_existing_primary_metrics():
    result = evaluate_point_maps(no_loop_estimate, identical_ground_truth, config, backend=identity_backend)
    assert result.primary.accuracy_mean_m == pytest.approx(0.0)
    assert result.primary.completion_mean_m == pytest.approx(0.0)
    assert result.primary.normal_consistency_mean == pytest.approx(1.0)


def test_pointcloud_cli_loads_no_trajectory_tensor(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifact"
    ground_truth_path = tmp_path / "ground-truth.npz"
    output_dir = tmp_path / "result"
    monkeypatch.setattr(cli, "load_pointmap_estimate", loader)
    argv = [
        "--artifact", str(artifact_dir),
        "--config", "configs/evaluation/pointcloud.yaml",
        "--ground-truth", str(ground_truth_path),
        "--dataset", "fixture",
        "--sequence", "sequence-0",
        "--output", str(output_dir),
    ]
    assert cli.main(argv) == 0
    assert loader.loaded_files == {"manifest.json", "pointmap.pt", "confidence.pt"}
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/evaluation/test_pointcloud_evaluator.py tests/evaluation/test_pointcloud_results.py tests/evaluation/test_evaluate_pointcloud_cli.py
```

Expected: import failures for the new point-cloud package.

- [ ] **Step 3: Move metric code with import-only changes first**

Move crop, Umeyama, ICP backend, directional nearest-neighbour metrics, Accuracy, Completion, Normal Consistency, Chamfer, Precision, Recall, and F-score definitions. Change imports from `mv_recon.*` to `evaluation.pointcloud.*`; do not change formulas/constants in this step.

- [ ] **Step 4: Replace the coupled protocol**

`configs/evaluation/pointcloud.yaml` contains only:

```yaml
version: 1
center_crop_size: 224
alignment: umeyama_sim3_then_icp
icp_type: point_to_point
icp_threshold_m: 0.1
normal_estimation: open3d_default
fscore_thresholds_m: [0.01, 0.02, 0.05]
```

No pipeline overrides, model, cache, segmentation, anchor, registration, window, or loop fields are accepted.

- [ ] **Step 5: Move result storage and implement CLI**

Preserve coverage checking, finite JSON values, atomic writes, primary mean/median aggregation, paper references, and extended diagnostics. CLI arguments:

```text
--artifact DIR
--config configs/evaluation/pointcloud.yaml
--ground-truth NPZ
--dataset NAME
--sequence NAME
--output DIR
```

The NPZ contains complete `point_maps` `(N,H,W,3)` and boolean `valid_mask` `(N,H,W)` arrays.

- [ ] **Step 6: Verify old/new parity**

```bash
pytest -q \
  tests/evaluation/test_pointcloud_evaluator.py \
  tests/evaluation/test_pointcloud_results.py \
  tests/evaluation/test_evaluate_pointcloud_cli.py \
  tests/mv_recon/test_geometry_metrics.py \
  tests/mv_recon/test_results.py
```

Expected: PASS with identical primary/diagnostic literals for the existing fixtures.

- [ ] **Step 7: Commit**

```bash
git add evaluation/pointcloud configs/evaluation/pointcloud.yaml evaluate_pointcloud.py tests/evaluation
git commit -m "feat: add artifact-only pointcloud evaluation"
```

---

### Task 12: Implement Valid ATE and Point-Cloud Experiment Matrices With Artifact Reuse

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/config.py`
- Create: `experiments/matrix.py`
- Create: `experiments/ate.py`
- Create: `experiments/pointcloud.py`
- Create: `configs/experiments/ate_matrix.yaml`
- Create: `configs/experiments/pointcloud_matrix.yaml`
- Create: `run_experiment_matrix.py`
- Create: `tests/experiments/test_matrix.py`
- Create: `tests/experiments/test_ate_experiment.py`
- Create: `tests/experiments/test_pointcloud_experiment.py`

**Interfaces:**
- Produces: `EvaluationKind`, `MatrixEntry`, `build_matrix()`, `reconstruction_identity()`, and `run_matrix()`.
- Consumes: reconstruction config/runner, artifact store, dataset adapters, and artifact-only evaluators.

- [ ] **Step 1: Write failing exact-matrix tests**

```python
def test_ate_matrix_is_exact_three_by_three_product():
    entries = build_matrix(EvaluationKind.ATE)
    assert {(e.segmentation_method.value, e.reconstruction_mode.value) for e in entries} == {
        (seg, mode)
        for seg in ("depth", "geometry", "atomic")
        for mode in ("no_loop", "traditional", "corrected")
    }
    assert len(entries) == 9


def test_pointcloud_matrix_is_exact_three_no_loop_entries():
    entries = build_matrix(EvaluationKind.POINTCLOUD)
    assert [(e.segmentation_method.value, e.reconstruction_mode.value) for e in entries] == [
        ("depth", "no_loop"), ("geometry", "no_loop"), ("atomic", "no_loop")
    ]


def test_pointcloud_matrix_rejects_loop_mode_before_runner_call():
    with pytest.raises(ValueError, match="pointcloud matrix only supports no_loop"):
        run_matrix(config_with_pointcloud_traditional, runner=forbidden_runner)


def test_identical_reconstruction_identity_reuses_artifact():
    run_matrix(ate_then_pointcloud_same_no_loop_config, runner=recording_runner)
    assert recording_runner.calls == 1
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/experiments
```

Expected: import failure for `experiments.matrix`.

- [ ] **Step 3: Implement strict matrix config**

Each experiment config names:

```text
version
evaluation kind
reconstruction config path
evaluation config path
dataset adapter/config
output root
matrix segmentation values
matrix reconstruction-mode values
```

Reject unknown keys, duplicate values, non-canonical ordering, and invalid products before loading images/models.

- [ ] **Step 4: Implement reconstruction identity and cache policy**

Hash only resolved reconstruction config, input manifest digest, checkpoint digest, and code/schema versions. Evaluation kind/config is excluded. A complete matching artifact directory is reused; incomplete or digest-mismatched directories fail rather than silently overwrite. For a matrix requested with ordinary prediction cache `refresh`, only the first entry refreshes and later entries use `readonly`; other cache modes remain unchanged.

- [ ] **Step 5: Implement ATE and point-cloud dataset orchestration**

ATE adapters preserve existing Sintel/Replica/TUM/KITTI trajectory formats and frame ordering. Point-cloud adapters reuse the current 7-Scenes/NeuralRGBD sequence maps, exact frame IDs, GT hashes, and per-sequence coverage validation, but live under `experiments/pointcloud.py`; evaluator modules never import datasets or reconstruction.

Point-cloud reconstruction overrides `segmentation.confidence_quantile_method=nearest` for the LASER Table 4 profile. ATE uses the reconstruction config default `higher`. Both allow command-line window overrides.

- [ ] **Step 6: Implement CLI and dry-run**

`run_experiment_matrix.py` accepts `--config`, repeated `--set`, and `--dry-run`. Dry-run prints every resolved entry, window/overlap, reconstruction identity, expected artifact directory, and cache mode without constructing models.

- [ ] **Step 7: Run GREEN and matrix/window tests**

```bash
pytest -q tests/experiments tests/test_prediction_cache_matrix.py
python run_experiment_matrix.py --config configs/experiments/ate_matrix.yaml --dry-run --set window.size=10 --set window.overlap=5
python run_experiment_matrix.py --config configs/experiments/ate_matrix.yaml --dry-run --set window.size=20 --set window.overlap=5
python run_experiment_matrix.py --config configs/experiments/pointcloud_matrix.yaml --dry-run --set window.size=20 --set window.overlap=10
```

Expected: tests PASS; dry runs report 9, 9, and 3 valid entries respectively and exit 0.

- [ ] **Step 8: Commit**

```bash
git add experiments configs/experiments run_experiment_matrix.py tests/experiments
git commit -m "feat: add reconstruction evaluation matrices"
```

---

### Task 13: Enforce Architecture Boundaries and Remove the Coupled Paths

**Files:**
- Create: `tests/test_architecture_boundaries.py`
- Modify: `inference_engine/__init__.py`
- Modify: `loop_closure/methods/__init__.py`
- Modify: `utils/interfaces.py`
- Modify: `eval_launch.py`
- Delete: `loop_closure/methods/base.py` after all imports move to `loop_closure/types.py`
- Delete: `inference_engine/streaming_window_engine.py` after all mode/runner imports are gone
- Delete: files listed in the deletion section above
- Delete: `mv_recon/geometry_metrics.py`, `mv_recon/results.py`, and `mv_recon/compare_results.py` after new parity tests own their behavior
- Delete: old LASER point-map evaluation/comparison configs superseded by `configs/evaluation/pointcloud.yaml` and `configs/experiments/pointcloud_matrix.yaml`
- Modify: affected tests/imports discovered by the exact `rg` commands below

**Interfaces:**
- Produces: mechanically enforced dependency direction and one formal execution path.
- Consumes: all replacement modules from Tasks 2–12.

- [ ] **Step 1: Write failing AST dependency tests**

```python
FORBIDDEN = {
    "evaluation": ("reconstruction", "inference_engine.segmentation", "inference_engine.anchor_propagation", "loop_closure"),
    "reconstruction.modes.no_loop": ("loop_closure",),
}


def test_forbidden_import_edges_are_absent():
    violations = collect_forbidden_imports(FORBIDDEN)
    assert violations == []


def test_evaluator_signatures_accept_views_not_artifacts_or_pipeline_config():
    assert tuple(inspect.signature(evaluate_trajectory).parameters) == ("estimate", "ground_truth", "config")
    assert tuple(inspect.signature(evaluate_point_maps).parameters)[:3] == ("estimate", "ground_truth", "config")
```

The AST helper parses `Import` and `ImportFrom` nodes and reports exact file/line/module; it must not grep prose or source text.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_architecture_boundaries.py
```

Expected: FAIL while old evaluator/reconstruction imports still exist.

- [ ] **Step 3: Resolve every runtime reference before deleting files**

Run:

```bash
rg -n "mv_recon\.(eval|paper_streaming|protocol|loop_experiment|compare_loop_results|geometry_metrics|results)|TraditionalWindowEngine|CorrectedWindowEngine|StreamingWindowEngine|WindowCache|ReconstructionResult|loop\.enabled|loop\.method" \
  --glob '*.py' --glob '*.yaml' --glob '*.md'
```

Update runtime imports to new modules. `utils/interfaces.py` may expose only thin artifact-view utilities or be removed from the reconstruction path. Remove streaming-model selection branches from `eval_launch.py`; the new commands own LASER reconstruction/ATE. Do not change unrelated vanilla PI3 depth evaluation.

- [ ] **Step 4: Delete the old paths**

Use `git rm` for the exact files enumerated above. Do not remove dataset loaders, `mv_recon/eval_outdoor.py`, or unrelated depth evaluation unless `rg` proves they are superseded and their tests have a replacement.

- [ ] **Step 5: Verify boundaries and absence through behavior**

Run:

```bash
pytest -q tests/test_architecture_boundaries.py tests/reconstruction tests/evaluation tests/experiments
python run_reconstruction.py --help
python evaluate_ate.py --help
python evaluate_pointcloud.py --help
python run_experiment_matrix.py --help
```

Expected: PASS/exit 0. The AST test, public imports, and CLI execution prove the new boundaries; do not add tests that merely assert deleted filenames are absent.

- [ ] **Step 6: Run the complete CPU suite**

```bash
pytest -q
```

Expected: PASS with zero failures. If GPU/data-only tests are marked/skipped by existing conventions, report exact skip counts; do not convert failures into skips.

- [ ] **Step 7: Commit**

```bash
git add -u
git add tests/test_architecture_boundaries.py inference_engine loop_closure utils eval_launch.py
git commit -m "refactor: remove coupled reconstruction evaluators"
```

Before committing, confirm `git status --short` does not stage the two generated `.cpp` files.

---

### Task 14: Document and Verify the Cloud Workflow, Then Push the Tested Branch

**Files:**
- Modify: `README.md`
- Create: `docs/reconstruction-evaluation-cloud-validation.md`
- Create: `tests/test_cloud_validation_commands.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: copy-paste cloud clone/setup/test/matrix commands tied to the pushed branch.
- Consumes: final public CLIs/configs and actual repository branch.

- [ ] **Step 1: Add a failing executable-command test**

Do not test prose with string matching. Parse fenced `bash` blocks tagged with `# smoke-test`, substitute a temporary dataset/checkpoint fixture for documented environment variables, and execute only the documented dry-run/help commands:

```python
def test_documented_smoke_commands_execute(tmp_path):
    commands = load_tagged_bash_commands("docs/reconstruction-evaluation-cloud-validation.md")
    results = run_commands(commands, cwd=REPOSITORY_ROOT, env=fixture_environment(tmp_path))
    assert [(item.command, item.returncode) for item in results] == [
        (item.command, 0) for item in results
    ]
```

Run:

```bash
pytest -q tests/test_cloud_validation_commands.py
```

Expected: FAIL because the cloud document does not exist.

- [ ] **Step 2: Write exact cloud clone and setup commands**

Document:

```bash
git clone --recursive --branch codex/laser-paper-pointmap-eval https://github.com/Cjuicy/LASER.git
cd LASER
git submodule update --init --recursive
conda create -n laser-decoupled python=3.11 -y
conda activate laser-decoupled
pip install -r requirements.txt
python setup.py build_ext --inplace
bash scripts/download_weights.sh
```

Document expected weight paths and dataset roots explicitly. Add `*.cpp` generated by Cython to `.gitignore` only if repository policy confirms both generated files are build outputs and no tracked `.cpp` source matches that rule; otherwise add the two exact generated paths.

- [ ] **Step 3: Document verified test and dry-run commands**

Include:

```bash
pytest -q
python run_experiment_matrix.py --config configs/experiments/ate_matrix.yaml --dry-run
python run_experiment_matrix.py --config configs/experiments/pointcloud_matrix.yaml --dry-run
```

Include single-run examples for all three modes, the complete ATE/point-cloud matrix commands, and window examples 10/5, 20/5, and 20/10. State artifact, trajectory result, and point-cloud result directories.

- [ ] **Step 4: Run documentation command tests and full verification**

Run fresh:

```bash
pytest -q tests/test_cloud_validation_commands.py
pytest -q
python run_experiment_matrix.py --config configs/experiments/ate_matrix.yaml --dry-run
python run_experiment_matrix.py --config configs/experiments/pointcloud_matrix.yaml --dry-run
git diff --check
```

Expected: all commands exit 0; full pytest has zero failures; dry runs report exactly 9 and 3 entries.

- [ ] **Step 5: Commit docs and final verification support**

```bash
git add README.md docs/reconstruction-evaluation-cloud-validation.md tests/test_cloud_validation_commands.py .gitignore
git commit -m "docs: add cloud reconstruction validation workflow"
```

- [ ] **Step 6: Inspect scope and push the tested branch**

```bash
git status --short
git log --oneline origin/codex/laser-paper-pointmap-eval..HEAD
git diff --stat origin/codex/laser-paper-pointmap-eval...HEAD
git push origin codex/laser-paper-pointmap-eval
git ls-remote --heads origin codex/laser-paper-pointmap-eval
```

Expected:

- only the known untracked generated Cython `.cpp` files remain locally, or they are ignored by the committed exact ignore rules;
- push succeeds;
- `ls-remote` returns the same commit as local `HEAD`.

- [ ] **Step 7: Record final handoff evidence**

The final response must report:

- pushed branch and exact commit;
- full pytest pass/fail/skip counts from Step 4;
- ATE dry-run count 9 and point-cloud dry-run count 3;
- clickable cloud validation document;
- copy-paste HTTPS clone command;
- any GPU/full-dataset validation not executed locally, stated explicitly rather than implied.

---

## Plan Self-Review Checklist

Before execution begins, verify:

- Every design requirement maps to at least one task and one behavioral test.
- `ReconstructionMode`, quantile field, artifact/view names, and mode runner signatures are consistent in every task.
- Traditional and Corrected state fields match the equations consumed by their loop processors.
- No evaluator task imports or constructs reconstruction components.
- Point-cloud mode rejection occurs before Open3D/backend construction.
- Window flexibility is tested at 10/5, 20/5, and 20/10.
- The point-cloud Table 4 `nearest` quantile is explicit while ATE retains `higher`.
- Old paths are deleted only after numerical parity tests pass.
- No step stages the existing generated Cython `.cpp` files.
- The final push and `ls-remote` evidence make the cloud clone command truthful.
