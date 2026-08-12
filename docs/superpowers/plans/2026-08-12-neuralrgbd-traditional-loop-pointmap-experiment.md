# NeuralRGBD Traditional-Loop Point-Map Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable, explicitly non-paper NeuralRGBD point-map profile that runs LASER's real traditional loop detection, joint constraint estimation, Sim(3) optimization, and aggregation, then compares its metrics with the preserved no-loop Depth result.

**Architecture:** Extend the strict evaluator with a third `experiment` protocol mode and a mode-specific point-map assembly identity. Existing paper/comparison runs keep the evaluator-local incremental replay; only the experiment dispatches to a focused adapter around the existing pipeline loop components. The result identity includes SALAD/DINO content hashes, and a separate comparison reader reports traditional-loop minus no-loop deltas while exposing that the two assemblies differ.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, PyTorch, existing LASER pipeline/loop strategies, Open3D metrics, pytest.

## Global Constraints

- The paper and Depth/Geometry/Atomic comparison profiles remain immutable and keep loop closure disabled.
- The experiment dataset is exactly the nine fixed NeuralRGBD kf10 sequences.
- Window size is 20 selected frames, overlap is 5 selected frames, and dataset keyframe interval is 10 source frames.
- Segmentation is Depth/Felzenszwalb `300 / 1.1 / 500`; confidence ratios are `0.5`; temporal IoU is `0.3`; anchor propagation is enabled at IoU `0.4`.
- Loop closure is enabled with `LoopMethod.TRADITIONAL`; the existing SALAD detector, joint Pi3 estimator, optimizer, and aggregation implementations are reused without modification.
- Geometry evaluation remains 224 center crop, Umeyama Sim(3), point-to-point ICP at `0.1 m`, Open3D default normals, and F-score thresholds `0.01 / 0.02 / 0.05 m`.
- The result is labeled `experiment` with assembly `pipeline-loop-aggregate-v1`; it must never be labeled as a paper baseline.
- Pi3, SALAD, and DINO checkpoints must be local files; their SHA256 values participate in experiment resume identity.
- Existing ordinary Pi3 prediction-cache semantics remain unchanged. Loop descriptors, candidates, constraints, and joint predictions do not enter that cache.
- Do not stage generated `inference_engine/utils/_segmentation_cy.cpp` or `inference_engine/utils/fast_seg.cpp`.

## File Structure

- Create `configs/evaluation/mv_recon_laser_nrgbd_depth_loop_traditional.yaml`: locked Hydra experiment profile.
- Modify `mv_recon/protocol.py`: experiment mode, assembly identity, dataset and pipeline locks.
- Modify `mv_recon/results.py`: auxiliary checkpoint hashes in resumable run identity.
- Create `mv_recon/loop_experiment.py`: adapter around existing traditional loop components.
- Modify `mv_recon/eval.py`: mode dispatch, checkpoint hashing, and provenance.
- Create `mv_recon/compare_loop_results.py`: no-loop versus traditional-loop comparison CLI.
- Modify `mv_recon/README.md`: cloud preparation, smoke/full runs, and comparison.
- Modify/create matching tests under `tests/mv_recon/`.

---

### Task 1: Lock the experiment protocol and content-address all checkpoints

**Files:**
- Create: `configs/evaluation/mv_recon_laser_nrgbd_depth_loop_traditional.yaml`
- Modify: `mv_recon/protocol.py`
- Modify: `mv_recon/results.py`
- Modify: `tests/mv_recon/test_protocol.py`
- Modify: `tests/mv_recon/test_results.py`

**Interfaces:**
- Produces: `EXPERIMENT_PROTOCOL_NAME: str`
- Produces: `PAPER_POINTMAP_ASSEMBLY: str`
- Produces: `LOOP_EXPERIMENT_POINTMAP_ASSEMBLY: str`
- Produces: `pointmap_assembly_for_mode(mode: str) -> str`
- Extends: `RunIdentity.auxiliary_checkpoint_sha256: Mapping[str, str]`
- Produces: `ResolvedEvaluationProtocol.pointmap_assembly: str`

- [ ] **Step 1: Write failing profile and protocol tests**

Add to `tests/mv_recon/test_protocol.py`:

```python
def _experiment_config(tmp_path: Path):
    return _profile_config(
        tmp_path,
        "mv_recon_laser_nrgbd_depth_loop_traditional",
    )


def test_traditional_loop_experiment_profile_is_fully_locked(tmp_path):
    resolved = resolve_evaluation_protocol(_experiment_config(tmp_path), ROOT)
    pipeline = resolved.pipeline.config
    assert resolved.protocol.mode == "experiment"
    assert resolved.protocol.name == "laser_neuralrgbd_pointmap_loop_experiment"
    assert resolved.datasets == ("NRGBD-dense",)
    assert resolved.pointmap_assembly == "pipeline-loop-aggregate-v1"
    assert (pipeline.window.size, pipeline.window.overlap) == (20, 5)
    assert pipeline.segmentation.method.value == "depth"
    assert pipeline.anchor_propagation.enabled is True
    assert pipeline.loop.enabled is True
    assert pipeline.loop.method.value == "traditional"
    assert pipeline.prediction_cache.mode.value == "auto"


@pytest.mark.parametrize(
    "override",
    (
        "loop.enabled=false",
        "loop.method=corrected",
        "segmentation.method=atomic",
        "window.size=10",
    ),
)
def test_traditional_loop_experiment_rejects_drift(tmp_path, override):
    config = _experiment_config(tmp_path)
    config.protocol.pipeline_overrides.append(override)
    with pytest.raises(ValueError, match="experiment protocol drift"):
        resolve_evaluation_protocol(config, ROOT)


def test_paper_and_comparison_keep_incremental_assembly(tmp_path):
    paper = resolve_evaluation_protocol(_root_config(tmp_path), ROOT)
    depth = resolve_evaluation_protocol(
        _profile_config(tmp_path, "mv_recon_laser_nrgbd_depth"), ROOT
    )
    assert paper.pointmap_assembly == "laser-incremental-global-map-v1"
    assert depth.pointmap_assembly == "laser-incremental-global-map-v1"
```

Add to `tests/mv_recon/test_results.py`:

```python
def test_resume_identity_includes_auxiliary_checkpoint_hashes(tmp_path):
    identity = _identity(
        auxiliary_checkpoint_sha256={"salad": "a" * 64, "dino": "b" * 64}
    )
    store = ResultStore(tmp_path / "run", identity, resume=False)
    _initialize(store)
    changed = replace(
        identity,
        auxiliary_checkpoint_sha256={"salad": "c" * 64, "dino": "b" * 64},
    )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        ResultStore(tmp_path / "run", changed, resume=True)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=. pytest -q \
  tests/mv_recon/test_protocol.py::test_traditional_loop_experiment_profile_is_fully_locked \
  tests/mv_recon/test_protocol.py::test_traditional_loop_experiment_rejects_drift \
  tests/mv_recon/test_protocol.py::test_paper_and_comparison_keep_incremental_assembly \
  tests/mv_recon/test_results.py::test_resume_identity_includes_auxiliary_checkpoint_hashes
```

Expected: FAIL because the profile, experiment mode, resolved assembly field,
and auxiliary identity field do not exist.

- [ ] **Step 3: Add the locked Hydra profile**

Create `configs/evaluation/mv_recon_laser_nrgbd_depth_loop_traditional.yaml`:

```yaml
# @package _global_
defaults:
  - mv_recon_laser_nrgbd_depth
  - _self_
name: mv_recon_laser_nrgbd_depth_loop_traditional
protocol:
  name: laser_neuralrgbd_pointmap_loop_experiment
  mode: experiment
  prediction_cache_mode: auto
  pipeline_overrides:
    - window.size=20
    - window.overlap=5
    - segmentation.method=depth
    - segmentation.confidence_keep_ratio=0.5
    - segmentation.depth_merge_threshold=0.1
    - segmentation.temporal_iou_threshold=0.3
    - segmentation.felzenszwalb.scale=300
    - segmentation.felzenszwalb.sigma=1.1
    - segmentation.felzenszwalb.min_size=500
    - segmentation.geometry.normal_method=cross
    - segmentation.geometry.normal_threshold_degrees=20.0
    - segmentation.atomic.split_mode=conservative
    - segmentation.atomic.split_score_threshold=0.10
    - anchor_propagation.enabled=true
    - anchor_propagation.correspondence_iou_threshold=0.4
    - loop.enabled=true
    - loop.method=traditional
    - loop.registration.confidence_keep_ratio=0.5
```

- [ ] **Step 4: Implement protocol mode and assembly resolution**

In `mv_recon/protocol.py`, add:

```python
EXPERIMENT_PROTOCOL_NAME = "laser_neuralrgbd_pointmap_loop_experiment"
PAPER_POINTMAP_ASSEMBLY = "laser-incremental-global-map-v1"
LOOP_EXPERIMENT_POINTMAP_ASSEMBLY = "pipeline-loop-aggregate-v1"
PROTOCOL_MODES = frozenset({"paper", "comparison", "experiment"})


def pointmap_assembly_for_mode(mode: str) -> str:
    if mode in {"paper", "comparison"}:
        return PAPER_POINTMAP_ASSEMBLY
    if mode == "experiment":
        return LOOP_EXPERIMENT_POINTMAP_ASSEMBLY
    raise ValueError(f"unsupported evaluation mode: {mode!r}")
```

Add `pointmap_assembly: str` to `ResolvedEvaluationProtocol`; include the
selected value in resolved/identity YAML. Refactor `_validate_protocol_locks`
so common Depth and metric locks are shared. Experiment mode must require the
experiment protocol name, Depth, auto prediction cache, enabled loop, and
traditional loop method. Paper/comparison continue requiring disabled loop and
traditional method. `_validate_datasets` must require exactly
`("NRGBD-dense",)` for experiment mode.

- [ ] **Step 5: Extend result identity compatibly**

In `mv_recon/results.py`, import `field` and change the dataclass to:

```python
@dataclass(frozen=True)
class RunIdentity:
    protocol_identity_sha256: str
    pipeline_sha256: str
    checkpoint_sha256: str
    sequence_map_sha256: Mapping[str, str]
    auxiliary_checkpoint_sha256: Mapping[str, str] = field(default_factory=dict)
    metric_version: str = METRIC_SCHEMA_VERSION
```

In `_identity_from_payload`, accept a missing auxiliary field as `{}` so the
preserved no-loop result remains readable; reject a present non-mapping value.

- [ ] **Step 6: Run tests and verify GREEN**

```bash
PYTHONPATH=. pytest -q tests/mv_recon/test_protocol.py tests/mv_recon/test_results.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add configs/evaluation/mv_recon_laser_nrgbd_depth_loop_traditional.yaml \
  mv_recon/protocol.py mv_recon/results.py \
  tests/mv_recon/test_protocol.py tests/mv_recon/test_results.py
git commit -m "feat: define traditional-loop point-map experiment"
```

---

### Task 2: Run the real traditional loop pipeline from point-map evaluation

**Files:**
- Create: `mv_recon/loop_experiment.py`
- Create: `tests/mv_recon/test_loop_experiment.py`

**Interfaces:**
- Produces: `LoopPointMapResult`
- Produces: `LoopExperimentDependencies`
- Produces: `reconstruct_traditional_loop_point_maps(*, engine, images, manifest, artifact_dir, dependencies=None) -> LoopPointMapResult`
- Consumes: prepared engine fields `pipeline_config`, `prediction_store`, `window_specs`, `loop_strategy`, and `model_handle`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/mv_recon/test_loop_experiment.py` with two minimal real
`WindowCache` values and lightweight external-stage fakes. Assert the main
test calls candidate detection, constructs the supplied joint estimator,
builds one constraint, optimizes once, aggregates once, returns `(N,H,W,3)`
points and `(N,H,W)` confidence, and records candidate/constraint/rejection and
fallback diagnostics. Add a zero-candidate test asserting no joint estimator
is constructed and `used_no_loop_path` is true. Add validation tests for an
inactive loop, non-traditional method, non-finite tensors, mismatched frame
counts, and missing aggregate payload keys.

The main assertion block must include:

```python
assert state.constraint_estimator == "joint-estimator"
assert state.optimize_calls == 1
assert state.aggregate_calls == 1
assert result.points.shape == (3, 1, 1, 3)
assert result.confidence.shape == (3, 1, 1)
assert result.diagnostics["candidate_count"] == 1
assert result.diagnostics["constraint_count"] == 1
assert result.diagnostics["rejected_candidate_count"] == 0
assert result.diagnostics["used_no_loop_path"] is False
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
PYTHONPATH=. pytest -q tests/mv_recon/test_loop_experiment.py
```

Expected: collection FAIL because `mv_recon.loop_experiment` does not exist.

- [ ] **Step 3: Implement the focused adapter**

Create these public types in `mv_recon/loop_experiment.py`:

```python
@dataclass(frozen=True)
class LoopPointMapResult:
    points: torch.Tensor
    confidence: torch.Tensor
    ordinary_prediction_key: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class LoopExperimentDependencies:
    run_windows: Callable = run_windows
    detect_loop_candidates: Callable = detect_loop_candidates
    build_constraint_estimator: Callable = JointAlignmentEstimator
    collect_prediction_diagnostics: Callable = collect_prediction_diagnostics
```

The implementation must execute:

```python
with store.entry_lock():
    caches = selected.run_windows(
        engine, manifest, images, engine.window_specs, config
    )
candidates = selected.detect_loop_candidates(
    config.loop.detection,
    manifest,
    Path(artifact_dir) / "loop_candidates.json",
)
estimator = (
    selected.build_constraint_estimator(
        model=engine.model_handle,
        images=images,
        manifest=manifest,
        chunk_size=config.loop.constraint.chunk_size,
        confidence_keep_ratio=config.loop.registration.confidence_keep_ratio,
    )
    if candidates
    else None
)
constraints = engine.loop_strategy.build_constraints(
    caches, candidates, constraint_estimator=estimator
)
solution = engine.loop_strategy.optimize(caches, constraints)
aggregation = engine.loop_strategy.aggregate(caches, solution)
```

Validate the method/config and aggregate shapes/finiteness. Merge ordinary
prediction diagnostics with `candidate_count`, `constraint_count`,
`rejected_candidate_count`, `used_no_loop_path`, and `loop_method`. The
artifact directory is outside the ordinary prediction cache.

- [ ] **Step 4: Run adapter tests and verify GREEN**

```bash
PYTHONPATH=. pytest -q tests/mv_recon/test_loop_experiment.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add mv_recon/loop_experiment.py tests/mv_recon/test_loop_experiment.py
git commit -m "feat: run traditional loop point-map assembly"
```

---

### Task 3: Dispatch experiment inference and record auditable provenance

**Files:**
- Modify: `mv_recon/eval.py`
- Modify: `tests/mv_recon/test_paper_streaming.py`
- Modify: `tests/mv_recon/test_eval_orchestration.py`

**Interfaces:**
- Consumes: `reconstruct_traditional_loop_point_maps`
- Produces: loop counts in `InferenceOutput.cache_diagnostics`
- Produces: assembly, loop method, and auxiliary hashes in the manifest

- [ ] **Step 1: Write failing dispatch and preflight tests**

In `tests/mv_recon/test_paper_streaming.py`, add a dispatch test that uses the
existing prepared-engine fixture, sets the Hydra mode to `experiment`, stubs
`reconstruct_traditional_loop_point_maps`, forbids
`reconstruct_incremental_point_maps`, and asserts returned loop diagnostics.
Keep the existing paper test and explicitly forbid the loop adapter there.

The experiment branch assertion must be:

```python
output = eval_module.run_streaming_inference(
    image_paths,
    model,
    OmegaConf.create(
        {"protocol": {"mode": "experiment"}, "output_dir": str(tmp_path)}
    ),
    (2, 3),
)
assert len(loop_calls) == 1
assert output.cache_diagnostics["constraint_count"] == 1
```

In `tests/mv_recon/test_eval_orchestration.py`, add a preflight test that
composes the experiment profile, creates local Pi3/SALAD/DINO files, overrides
the loop checkpoint paths, and asserts no model construction plus:

```python
assert set(manifest["auxiliary_checkpoint_sha256"]) == {"salad", "dino"}
assert manifest["evaluation_mode"] == "experiment"
assert manifest["pointmap_assembly"] == "pipeline-loop-aggregate-v1"
assert manifest["loop_enabled"] is True
assert manifest["loop_method"] == "traditional"
```

Add missing SALAD and missing DINO cases that raise `FileNotFoundError` before
model construction.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=. pytest -q \
  tests/mv_recon/test_paper_streaming.py \
  tests/mv_recon/test_eval_orchestration.py
```

Expected: FAIL because experiment dispatch and auxiliary provenance are absent.

- [ ] **Step 3: Add mode-aware inference dispatch**

In `mv_recon/eval.py`, preprocess images and prepare the engine once. Branch on
`str(hydra_cfg.protocol.mode)`:

```python
if mode == "experiment":
    loop_result = reconstruct_traditional_loop_point_maps(
        engine=engine,
        images=images,
        manifest=manifest,
        artifact_dir=(
            Path(str(hydra_cfg.output_dir))
            / "loop_artifacts"
            / engine.prediction_store.fingerprint.key
        ),
    )
    points = loop_result.points
    confidence = loop_result.confidence
    prediction_key = loop_result.ordinary_prediction_key
    diagnostics = dict(loop_result.diagnostics)
else:
    # Keep the current paper-compatible incremental replay byte-for-byte.
```

Resize the selected `points` once after the branch. Return branch-selected
confidence, key, and diagnostics.

- [ ] **Step 4: Hash auxiliary checkpoint contents**

Add this helper in `mv_recon/eval.py`:

```python
def _auxiliary_checkpoint_digests(
    resolved: ResolvedEvaluationProtocol,
    repository_root: Path,
    digest: Callable[[Path], str],
) -> dict[str, str]:
    if resolved.protocol.mode != "experiment":
        return {}
    detection = resolved.pipeline.config.loop.detection
    paths = {
        "salad": Path(detection.salad_checkpoint),
        "dino": Path(detection.dino_checkpoint),
    }
    result = {}
    for label, path in paths.items():
        selected = path if path.is_absolute() else repository_root / path
        selected = selected.resolve()
        if not selected.is_file():
            raise FileNotFoundError(
                f"loop {label} checkpoint does not exist or is not a file: "
                f"{selected}"
            )
        result[label] = digest(selected)
    return result
```

Call it before creating `ResultStore`. Add the mapping to `RunIdentity` and the
manifest. Replace the manifest's global assembly constant with
`resolved.pointmap_assembly`, then add:

```python
"loop_enabled": resolved.pipeline.config.loop.enabled,
"loop_method": resolved.pipeline.config.loop.method.value,
"auxiliary_checkpoint_sha256": dict(auxiliary_checkpoint_sha256),
```

- [ ] **Step 5: Preserve per-sequence loop diagnostics**

No `SequenceResult` schema change is required: `run_sequence` already stores
the complete `InferenceOutput.cache_diagnostics`. Scalar experiment values
therefore appear in `sequences.csv` as `cache_candidate_count`,
`cache_constraint_count`, `cache_rejected_candidate_count`,
`cache_used_no_loop_path`, and `cache_joint_forward_count`.

- [ ] **Step 6: Run evaluator tests and verify GREEN**

```bash
PYTHONPATH=. pytest -q \
  tests/mv_recon/test_paper_streaming.py \
  tests/mv_recon/test_eval_orchestration.py \
  tests/mv_recon/test_protocol.py \
  tests/mv_recon/test_results.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add mv_recon/eval.py \
  tests/mv_recon/test_paper_streaming.py \
  tests/mv_recon/test_eval_orchestration.py
git commit -m "feat: dispatch point-map loop experiments"
```

---

### Task 4: Compare traditional loop with the preserved no-loop Depth run

**Files:**
- Create: `mv_recon/compare_loop_results.py`
- Create: `tests/mv_recon/test_compare_loop_results.py`

**Interfaces:**
- Produces: `compare_loop_run_directories(no_loop_dir: Path, traditional_loop_dir: Path, output_dir: Path) -> dict[str, object]`
- Produces CLI flags: `--no-loop-run`, `--traditional-loop-run`, `--output-dir`
- Consumes strict JSON/metric helpers from `mv_recon.compare_results`

- [ ] **Step 1: Write failing comparison tests**

Create fixtures based on `tests/mv_recon/test_compare_results.py::_write_run`.
The no-loop fixture uses `comparison` plus
`laser-incremental-global-map-v1`; the experiment fixture uses `experiment`
plus `pipeline-loop-aggregate-v1`, enabled traditional loop, and per-sequence
loop diagnostics.

Add a success test asserting:

```python
report = compare_loop_run_directories(no_loop, loop, tmp_path / "comparison")
assert report["assemblies"] == {
    "no_loop": "laser-incremental-global-map-v1",
    "traditional_loop": "pipeline-loop-aggregate-v1",
}
assert report["interpretation_warning"].startswith("Point-map assembly differs")
assert report["methods"][1]["delta_to_no_loop"]["accuracy_mean_m"] == (
    pytest.approx(-0.001)
)
assert (tmp_path / "comparison" / "loop_comparison.json").is_file()
assert (tmp_path / "comparison" / "loop_comparison.csv").is_file()
```

Add parametrized rejection tests for checkpoint, sequence map, ordinary key,
input manifest, GT, runtime, coverage, wrong loop method, absent constraint
diagnostics, and inconsistent candidate/constraint/rejection totals.

- [ ] **Step 2: Run new tests and verify RED**

```bash
PYTHONPATH=. pytest -q tests/mv_recon/test_compare_loop_results.py
```

Expected: collection FAIL because `mv_recon.compare_loop_results` is absent.

- [ ] **Step 3: Implement the strict reader**

Create `mv_recon/compare_loop_results.py`. Import the finite JSON readers,
metric flattening, macro diagnostic calculation, CSV writer, and fixed
NeuralRGBD sequence loader from `mv_recon.compare_results`; do not duplicate
metric formulas.

Require:

```python
NO_LOOP_MODE = "comparison"
NO_LOOP_ASSEMBLY = "laser-incremental-global-map-v1"
LOOP_MODE = "experiment"
LOOP_ASSEMBLY = "pipeline-loop-aggregate-v1"
```

Both inputs must be successful nine-sequence `subset` results with Depth,
identical metric/checkpoint/map/input/GT/ordinary/runtime identities, threshold
lists, window 20/5, segmentation, anchor, geometry, and auto cache settings.
Git commits are recorded but need not match because the preserved no-loop run
predates this feature.

For every experiment sequence require nonnegative integer `candidate_count`,
`constraint_count`, and `rejected_candidate_count`, boolean
`used_no_loop_path`, and:

```python
candidate_count == constraint_count + rejected_candidate_count
```

Zero candidates are valid. Emit raw metrics, loop-minus-no-loop deltas,
directions, loop diagnostic totals, both commits, and both assemblies. Include
this exact warning:

```text
Point-map assembly differs between runs; deltas combine traditional loop closure with the pipeline's delayed Sim(3) aggregation semantics and are not a pure loop-only causal estimate.
```

- [ ] **Step 4: Add the CLI and atomic outputs**

```python
parser.add_argument("--no-loop-run", type=Path, required=True)
parser.add_argument("--traditional-loop-run", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
```

Write `loop_comparison.json` and `loop_comparison.csv`. Exit zero only after
both are written.

- [ ] **Step 5: Run comparison tests and verify GREEN**

```bash
PYTHONPATH=. pytest -q \
  tests/mv_recon/test_compare_loop_results.py \
  tests/mv_recon/test_compare_results.py
```

Expected: PASS, including all existing three-method comparison tests.

- [ ] **Step 6: Commit Task 4**

```bash
git add mv_recon/compare_loop_results.py \
  tests/mv_recon/test_compare_loop_results.py
git commit -m "feat: compare traditional-loop point maps"
```

---

### Task 5: Document, verify, push, and hand off cloud commands

**Files:**
- Modify: `mv_recon/README.md`

**Interfaces:**
- Documents profile: `mv_recon_laser_nrgbd_depth_loop_traditional`
- Documents comparison CLI: `python mv_recon/compare_loop_results.py`

- [ ] **Step 1: Add exact preparation and preflight commands**

Document:

```bash
cd /root/autodl-tmp/LASER-pointmap-eval
conda activate vggt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
test -f weights/model.safetensors
test -f weights/dino_salad.ckpt
test -f weights/dinov2_vitb14_pretrain.pth

python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth_loop_traditional \
  protocol.preflight_only=true \
  output_dir=outputs/pointmap/nrgbd_depth_loop_traditional_preflight
```

- [ ] **Step 2: Add smoke and diagnostic inspection commands**

Document:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth_loop_traditional \
  protocol.max_sequences=1 \
  output_dir=outputs/pointmap/nrgbd_depth_loop_traditional_smoke

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/pointmap/nrgbd_depth_loop_traditional_smoke")
manifest = json.loads((root / "protocol_manifest.json").read_text())
results = json.loads((root / "results.json").read_text())
print("mode:", manifest["evaluation_mode"])
print("assembly:", manifest["pointmap_assembly"])
print("loop:", manifest["loop_method"], manifest["loop_enabled"])
for sequence in results["sequences"]:
    diagnostics = sequence["cache_diagnostics"]
    print(
        sequence["sequence"],
        "candidates=", diagnostics["candidate_count"],
        "constraints=", diagnostics["constraint_count"],
        "joint_forward=", diagnostics["joint_forward_count"],
        "no_loop_fallback=", diagnostics["used_no_loop_path"],
    )
PY
```

Explain: the experiment path is proven even if breakfast_room has zero valid
loops. A real loop-consumed sequence has candidates `> 0`, constraints `> 0`,
joint forward `> 0`, and fallback `False`; if the smoke has none, continue to
the full fixed-threshold run rather than tuning thresholds.

- [ ] **Step 3: Add persistent full-run and comparison commands**

Document:

```bash
mkdir -p outputs/logs
screen -dmS nrgbd_depth_loop_traditional bash -lc '
cd /root/autodl-tmp/LASER-pointmap-eval
conda run -n vggt python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth_loop_traditional \
  output_dir=outputs/pointmap/nrgbd_depth_loop_traditional \
  > outputs/logs/nrgbd_depth_loop_traditional.log 2>&1
'

screen -ls
tail -n 80 outputs/logs/nrgbd_depth_loop_traditional.log

python mv_recon/compare_loop_results.py \
  --no-loop-run outputs/pointmap/nrgbd_depth_replay \
  --traditional-loop-run outputs/pointmap/nrgbd_depth_loop_traditional \
  --output-dir outputs/pointmap/nrgbd_loop_comparison
cat outputs/pointmap/nrgbd_loop_comparison/loop_comparison.csv
python -m json.tool \
  outputs/pointmap/nrgbd_loop_comparison/loop_comparison.json
```

State that the current cloud no-loop directory is
`outputs/pointmap/nrgbd_depth_replay`; a fresh clone must first produce a full
no-loop Depth run and pass its directory to `--no-loop-run`.

- [ ] **Step 4: Run fresh full verification**

```bash
PYTHONPATH=. pytest -q
git diff --check
python -m py_compile \
  mv_recon/protocol.py \
  mv_recon/results.py \
  mv_recon/loop_experiment.py \
  mv_recon/eval.py \
  mv_recon/compare_loop_results.py
git status --short
```

Expected: all tests pass; diff and compilation exit zero; only the two known
generated Cython `.cpp` files may remain untracked.

- [ ] **Step 5: Commit documentation**

```bash
git add mv_recon/README.md
git commit -m "docs: add traditional-loop point-map commands"
```

- [ ] **Step 6: Push and verify the remote commit**

```bash
git push origin codex/laser-paper-pointmap-eval
git rev-parse HEAD
git rev-parse origin/codex/laser-paper-pointmap-eval
```

Expected: both hashes match. Hand off clone/pull, preflight, smoke, full-run,
diagnostic, and comparison commands only after this check.

---

## Acceptance Checklist

- [ ] Existing paper/comparison tests still prove loop closure is off.
- [ ] Experiment protocol rejects dataset/window/segmentation/loop drift.
- [ ] Experiment preflight hashes Pi3, SALAD, and DINO without model construction.
- [ ] A changed SALAD or DINO file invalidates resume identity.
- [ ] Experiment inference calls real candidate detection, joint constraint estimation, traditional optimization, and aggregation.
- [ ] Experiment mode never calls paper-compatible incremental replay.
- [ ] Candidate/constraint/joint-forward/fallback diagnostics exist per sequence.
- [ ] Comparison rejects incomparable inputs and warns about assembly semantics.
- [ ] Local full suite, diff check, and Python compilation pass freshly.
- [ ] Remote branch hash equals the verified local commit before command handoff.
