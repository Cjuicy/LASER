# Layer-Atomic Split Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate `layer_atomic_split` into the diagnostics branch and produce a strict three-method KITTI diagnostic pipeline that localizes segmentation changes and ATE/RPE regressions.

**Architecture:** Preserve the production split implementation as the single source of truth, expose a staged result for read-only diagnostics, and run `depth`, `geometry_baseline`, and `layer_atomic_split` through the existing two-pass orchestration. Pass 1 records aligned trajectory and scalar split evidence; Pass 2 reruns selected sequences and renders pre/post split evidence on the same frame union for all methods.

**Tech Stack:** Python 3.11, NumPy, SciPy, scikit-image, PyTorch, OpenCV, pytest, static HTML/PNG/PLY diagnostics.

## Global Constraints

- Target branch is `codex/segmentation-diagnostics`; source branch is `codex/auto-post-merge-split`.
- Official configurations are exactly `depth`, `geometry_baseline`, and `layer_atomic_split` in that order.
- `layer_atomic_split` is fixed to `normal_method=cross`, `split_score_thresh=0.10`, and `split_aux_confirmation=true`.
- Old `layer_atomic` labels may exist only as `pre_split_labels`; they are not a fourth trajectory configuration.
- `geometry_legacy_reference` is removed from execution, ranking, selection, cases, report, verifier, and cloud instructions.
- All trajectory errors use one full-sequence Sim(3) alignment; RPE remains `delta=1 frame, all_pairs=True`.
- Pass 2 reruns each selected sequence from frame zero and persists dense artifacts only for the shared selected-frame union.
- Missing metrics are `null` plus a reason; no fabricated zeros, scale maps, or causal claims.
- Preserve 40/50 GiB warning/hard storage limits, 10 GiB free-space reserve, atomic writes, checksums, ownership markers, and resumability.
- Do not add parameter sweeps, loop closure, new dependencies, or segmentation ground-truth claims.

---

### Task 1: Integrate the production post-merge split implementation

**Files:**
- Create: `inference_engine/utils/post_merge_split.py`
- Create: `scripts/evaluate_post_merge_split.py`
- Modify: `inference_engine/utils/layer_atomic_geometry.py`
- Modify: `inference_engine/utils/lsa.py`
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `inference_engine/streaming_window_engine_lc.py`
- Modify: `demo.py`
- Modify: `demo_lc.py`
- Modify: `eval_launch.py`
- Modify: `scripts/verify_segmentation_modes.py`
- Modify: `docs/unified-segmentation-cloud-validation.md`
- Test: `tests/test_post_merge_split.py`
- Test: `tests/test_post_merge_split_evaluation.py`
- Test: `tests/test_layer_atomic_split_integration.py`
- Test: `tests/test_segmentation_modes.py`
- Test: `tests/test_segmentation_engine_modes.py`
- Test: `tests/test_demo.py`
- Test: `tests/test_demo_lc.py`
- Test: `tests/test_segmentation_smoke_script.py`

**Interfaces:**
- Consumes: `merge_layer_atoms(...)`, `make_sp_graph(...)`, and the streaming engine's existing RGB window tensor.
- Produces: `refine_auto_regions(...)`, `segment_point_map_layer_atomic_split(...)`, and `segment_mode="layer_atomic_split"` production routing.

- [ ] **Step 1: Apply the already-tested source commits**

Run:

```bash
git cherry-pick c9f1ac0 07c70dd c4c2820
```

Expected: the three commits are applied or conflicts are reported only in files independently changed by diagnostics.

- [ ] **Step 2: Resolve integration conflicts without dropping diagnostics hooks**

Keep both the source branch's `layer_atomic_split` arguments and the diagnostics branch's `diagnostic_sink`, `diagnostic_context`, segmentation metadata, and no-loop-closure behavior. In `eval_launch.py`, retain one parser definition for:

```python
parser.add_argument(
    "--segment_mode",
    default="depth",
    choices=("depth", "geometry", "layer_atomic", "layer_atomic_split"),
)
```

Keep fixed split arguments passed into `StreamingWindowEngine` and `StreamingWindowEngineLC`:

```python
split_score_thresh=args.split_score_thresh,
split_aux_confirmation=args.split_aux_confirmation,
```

- [ ] **Step 3: Run the imported production tests**

Run:

```bash
pytest -q tests/test_post_merge_split.py tests/test_post_merge_split_evaluation.py tests/test_layer_atomic_split_integration.py tests/test_segmentation_modes.py tests/test_segmentation_engine_modes.py tests/test_demo.py tests/test_demo_lc.py tests/test_segmentation_smoke_script.py
```

Expected: all listed tests pass.

- [ ] **Step 4: Commit the resolved production integration**

```bash
git add demo.py demo_lc.py eval_launch.py inference_engine scripts tests docs/unified-segmentation-cloud-validation.md
git commit -m "feat: integrate layer atomic split method"
```

---

### Task 2: Expose production split stages and dense decision evidence

**Files:**
- Modify: `inference_engine/utils/post_merge_split.py`
- Modify: `inference_engine/utils/layer_atomic_geometry.py`
- Test: `tests/test_post_merge_split.py`
- Test: `tests/test_layer_atomic_split_integration.py`

**Interfaces:**
- Consumes: production `_merge_layer_atoms_with_metadata(...)` and `refine_auto_regions(...)` semantics from Task 1.
- Produces: `SplitTrace`, `LayerAtomicSplitResult`, `refine_auto_regions_with_trace(...)`, and `segment_point_map_layer_atomic_split_stages(...)`.

- [ ] **Step 1: Write failing trace tests**

Add tests that assert accepted and rejected parents produce compact evidence arrays:

```python
trace = pms.refine_auto_regions_with_trace(
    points, rgb, labels, atoms, np.ones(1),
    seg_min_size=20,
    normal_method="cross",
    split_score_thresh=0.10,
)
assert trace.labels.shape == labels.shape
assert trace.changed_mask.dtype == np.bool_
assert trace.parent_map.shape == labels.shape
assert trace.child_map.shape == labels.shape
assert trace.score_map.shape == labels.shape
assert trace.decision_map.shape == labels.shape
assert trace.diagnostics.split_accepted_count == 1
```

Add a staged/public parity test:

```python
stages = lag.segment_point_map_layer_atomic_split_stages(
    points,
    depth_merge_thresh=0.1,
    rgb_images=rgb,
    seg_min_size=20,
)
formal = lag.segment_point_map_layer_atomic_split(
    points,
    depth_merge_thresh=0.1,
    rgb_images=rgb,
    seg_min_size=20,
)
np.testing.assert_array_equal(stages.final_labels, formal)
np.testing.assert_array_equal(stages.split_trace.changed_mask, stages.pre_split_labels != stages.final_labels)
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```bash
pytest -q tests/test_post_merge_split.py tests/test_layer_atomic_split_integration.py
```

Expected: failure because the trace and staged interfaces do not exist.

- [ ] **Step 3: Implement focused result types and wrappers**

In `post_merge_split.py`, add:

```python
@dataclass(frozen=True)
class SplitTrace:
    labels: np.ndarray
    diagnostics: SplitDiagnostics
    changed_mask: np.ndarray
    parent_map: np.ndarray
    child_map: np.ndarray
    score_map: np.ndarray
    decision_map: np.ndarray
```

Move the current implementation body into `refine_auto_regions_with_trace(...)`, recording decision code `1` for accepted, `2` for no markers, `3` for small child, and `4` for low score. Keep compatibility:

```python
def refine_auto_regions(*args, **kwargs):
    trace = refine_auto_regions_with_trace(*args, **kwargs)
    return trace.labels, trace.diagnostics
```

In `layer_atomic_geometry.py`, add:

```python
@dataclass(frozen=True)
class LayerAtomicSplitResult:
    initial_labels: np.ndarray
    coarse_labels: np.ndarray
    pre_split_labels: np.ndarray
    final_labels: np.ndarray
    atom_labels: np.ndarray
    atom_scales: np.ndarray
    split_trace: SplitTrace
```

Implement `segment_point_map_layer_atomic_split_stages(...)` and make the public final-label function return `.final_labels` from it.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/test_post_merge_split.py tests/test_layer_atomic_split_integration.py tests/test_segmentation_modes.py
```

Expected: all pass.

- [ ] **Step 5: Commit staged split evidence**

```bash
git add inference_engine/utils/post_merge_split.py inference_engine/utils/layer_atomic_geometry.py tests/test_post_merge_split.py tests/test_layer_atomic_split_integration.py
git commit -m "feat: expose split diagnostic stages"
```

---

### Task 3: Replace the four-profile summary with a three-method contract

**Files:**
- Modify: `inference_engine/diagnostics/schema.py`
- Modify: `inference_engine/diagnostics/metrics.py`
- Modify: `inference_engine/diagnostics/orchestrator.py`
- Test: `tests/test_diagnostic_schema_storage.py`
- Test: `tests/test_diagnostic_trajectory.py`
- Test: `tests/test_diagnostic_orchestrator.py`

**Interfaces:**
- Consumes: `layer_atomic_split` engine mode from Task 1.
- Produces: schema `2.0`, the exact `DIAGNOSTIC_PROFILES` order, depth-based Stability Guard, and depth-to-geometry Recovery.

- [ ] **Step 1: Write failing profile and summary tests**

Assert the exact profile contract:

```python
assert list(DIAGNOSTIC_PROFILES) == [
    "depth",
    "geometry_baseline",
    "layer_atomic_split",
]
assert DIAGNOSTIC_PROFILES["layer_atomic_split"] == {
    "segment_mode": "layer_atomic_split",
    "geometry_seg_profile": "baseline_params",
    "normal_method": "cross",
    "split_score_thresh": 0.10,
    "split_aux_confirmation": True,
    "official": True,
}
```

Assert summary and guard values:

```python
summary = build_sequence_summary(results)
assert set(summary["official_aggregate"]) == {
    "depth", "geometry_baseline", "layer_atomic_split"
}
assert "legacy_reference" not in summary
assert summary["recovery"]["02"]["score"] == pytest.approx(0.5)
guard = evaluate_stability_guard(split_ates, depth_ates, expected_sequences=("00", "05", "09"))
assert guard["baseline_config"] == "depth"
```

- [ ] **Step 2: Verify contract tests fail**

Run:

```bash
pytest -q tests/test_diagnostic_schema_storage.py tests/test_diagnostic_trajectory.py tests/test_diagnostic_orchestrator.py
```

Expected: failures naming old profiles, schema `1.0`, or legacy summary fields.

- [ ] **Step 3: Implement the three-method contract**

Set:

```python
SCHEMA_VERSION = "2.0"
OFFICIAL_CONFIGS = ("depth", "geometry_baseline", "layer_atomic_split")
```

Define Recovery as:

```python
def recovery_score(depth_ate, split_ate, geometry_ate, *, eps=1e-9):
    denominator = float(depth_ate - geometry_ate)
    if not np.isfinite(denominator) or denominator <= eps:
        return {"valid": False, "score": None, "invalid_reason": "non_positive_recovery_gap"}
    score = (float(depth_ate) - float(split_ate)) / denominator
    return {"valid": bool(np.isfinite(score)), "score": float(score), "invalid_reason": None}
```

Build `summary["recovery"]` for 02/04/10 and evaluate `layer_atomic_split` against `depth` for the existing 3% mean, 0% median, and 10% 00/05/09 thresholds. Remove legacy reads and fields.

- [ ] **Step 4: Run contract tests**

Run:

```bash
pytest -q tests/test_diagnostic_schema_storage.py tests/test_diagnostic_trajectory.py tests/test_diagnostic_orchestrator.py
```

Expected: all pass.

- [ ] **Step 5: Commit the contract migration**

```bash
git add inference_engine/diagnostics/schema.py inference_engine/diagnostics/metrics.py inference_engine/diagnostics/orchestrator.py tests/test_diagnostic_schema_storage.py tests/test_diagnostic_trajectory.py tests/test_diagnostic_orchestrator.py
git commit -m "feat: compare three segmentation methods"
```

---

### Task 4: Trace `layer_atomic_split` through the official diagnostics path

**Files:**
- Modify: `inference_engine/diagnostics/segmentation.py`
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `inference_engine/diagnostics/orchestrator.py`
- Modify: `inference_engine/diagnostics/sink.py`
- Test: `tests/test_diagnostic_segmentation.py`
- Test: `tests/test_diagnostic_engine.py`
- Test: `tests/test_diagnostic_sink.py`

**Interfaces:**
- Consumes: `segment_point_map_layer_atomic_split_stages(...)` and `LayerAtomicSplitResult` from Task 2.
- Produces: scalar split metrics in Pass 1 and dense split arrays in selected Pass 2 frames with formal-label parity.

- [ ] **Step 1: Write a failing diagnostic parity test**

```python
stages = segment_point_map_layer_atomic_split_stages(
    points, .1, rgb_images=rgb, conf_map=confidence,
    top_conf_percentile=.5, seg_scale=2, seg_sigma=0, seg_min_size=2,
)
trace = trace_segmentation_frame(
    points,
    stages.final_labels,
    segment_mode="layer_atomic_split",
    rgb_image=rgb,
    conf_map=confidence,
    top_conf_percentile=.5,
    seg_scale=2,
    seg_sigma=0,
    seg_min_size=2,
    normal_method="cross",
    split_score_thresh=.10,
    split_aux_confirmation=True,
)
np.testing.assert_array_equal(trace["arrays"]["final_labels"], stages.final_labels)
assert "pre_split_labels" in trace["arrays"]
assert "changed_mask" in trace["arrays"]
assert trace["metrics"]["split"]["split_aux_confirmation"] is True
```

- [ ] **Step 2: Verify it fails**

Run:

```bash
pytest -q tests/test_diagnostic_segmentation.py tests/test_diagnostic_engine.py tests/test_diagnostic_sink.py
```

Expected: failure because diagnostic tracing does not accept split mode or RGB.

- [ ] **Step 3: Implement split-aware tracing**

Extend `trace_segmentation_frame(...)` with `rgb_image=None`, `split_score_thresh=.10`, and
`split_aux_confirmation=True`. For split mode, call the staged production function, require parity with
`formal_labels`, and emit:

```python
metrics["split"] = stages.split_trace.diagnostics.as_dict()
arrays.update(
    initial_labels=stages.initial_labels,
    coarse_labels=stages.coarse_labels,
    pre_split_labels=stages.pre_split_labels,
    final_labels=stages.final_labels,
    atom_labels=stages.atom_labels,
    atom_scales=stages.atom_scales,
    changed_mask=stages.split_trace.changed_mask,
    split_parent_map=stages.split_trace.parent_map,
    split_child_map=stages.split_trace.child_map,
    split_score_map=stages.split_trace.score_map,
    split_decision_map=stages.split_trace.decision_map,
)
```

Pass the window RGB frame and fixed split profile values through the engine's diagnostic observation call. Keep Pass 1 sink behavior scalar-only and Pass 2 dense-array filtering unchanged.

- [ ] **Step 4: Run focused tracing tests**

Run:

```bash
pytest -q tests/test_diagnostic_segmentation.py tests/test_diagnostic_engine.py tests/test_diagnostic_sink.py tests/test_segmentation_engine_modes.py
```

Expected: all pass.

- [ ] **Step 5: Commit official split tracing**

```bash
git add inference_engine/diagnostics/segmentation.py inference_engine/streaming_window_engine.py inference_engine/diagnostics/orchestrator.py inference_engine/diagnostics/sink.py tests/test_diagnostic_segmentation.py tests/test_diagnostic_engine.py tests/test_diagnostic_sink.py
git commit -m "feat: trace layer atomic split decisions"
```

---

### Task 5: Select improvement, degradation, split anomaly, and control intervals

**Files:**
- Modify: `inference_engine/diagnostics/selection.py`
- Modify: `inference_engine/diagnostics/orchestrator.py`
- Test: `tests/test_diagnostic_selection.py`
- Test: `tests/test_diagnostic_orchestrator.py`

**Interfaces:**
- Consumes: aligned per-frame errors and Pass 1 split scalar metrics.
- Produces: records with two regrets and deterministic selected interval reasons.

- [ ] **Step 1: Write failing selection tests**

Build synthetic windows with positive/negative regrets, a change point, a split anomaly, a no-effect split,
and a no-split motion-matched control. Assert reasons include:

```python
required = {
    "trajectory_degradation",
    "trajectory_improvement",
    "trajectory_change",
    "split_anomaly",
    "split_no_trajectory_effect",
    "matched_control",
}
assert required <= {reason for interval in selected for reason in interval.reasons}
```

Assert selection records expose:

```python
assert record["split_minus_depth_regret"] == pytest.approx(expected_depth_regret)
assert record["split_minus_geometry_regret"] == pytest.approx(expected_geometry_regret)
assert record["split_accepted_count"] >= 0
assert record["split_changed_pixel_ratio"] >= 0
```

- [ ] **Step 2: Verify selection tests fail**

Run:

```bash
pytest -q tests/test_diagnostic_selection.py tests/test_diagnostic_orchestrator.py
```

Expected: failures because old layer-atomic regret names and old reasons are still present.

- [ ] **Step 3: Implement split-aware records and deterministic reasons**

Replace old regret fields with the two exact names above. Aggregate split metrics only from
`layer_atomic_split` rows at each global window. Rank degradation by largest positive regret,
improvement by most negative regret, change point by largest adjacent regret delta, split anomaly by robust
z-score, and no-effect by high split activity plus absolute regret no greater than the sequence median.
Choose matched controls from zero-accepted-split windows by minimum normalized distance over speed, turn,
and confidence.

- [ ] **Step 4: Run selection tests**

Run:

```bash
pytest -q tests/test_diagnostic_selection.py tests/test_diagnostic_orchestrator.py
```

Expected: all pass and interval order is deterministic.

- [ ] **Step 5: Commit interval localization**

```bash
git add inference_engine/diagnostics/selection.py inference_engine/diagnostics/orchestrator.py tests/test_diagnostic_selection.py tests/test_diagnostic_orchestrator.py
git commit -m "feat: localize split trajectory effects"
```

---

### Task 6: Render split evidence and strict three-method reports

**Files:**
- Modify: `inference_engine/diagnostics/rendering.py`
- Modify: `inference_engine/diagnostics/report.py`
- Modify: `inference_engine/diagnostics/orchestrator.py`
- Test: `tests/test_diagnostic_rendering_report.py`
- Test: `tests/test_diagnostic_orchestrator.py`

**Interfaces:**
- Consumes: namespaced Pass 2 arrays and split-aware selection records.
- Produces: pre/post/changed/decision PNGs, three-method comparisons, CSV, overview timelines, and case pages.

- [ ] **Step 1: Write failing rendering and report tests**

Assert `render_case(...)` returns files for:

```python
required = {
    "pre_split_segments",
    "final_segments",
    "split_changed_regions",
    "split_scores",
    "split_decisions",
}
assert required <= set(rendered)
```

Assert generated HTML contains `layer_atomic_split`, both regret names, and the six case reason labels, while
excluding `geometry_legacy_reference` and any official method heading named exactly `layer_atomic`.

- [ ] **Step 2: Verify rendering/report tests fail**

Run:

```bash
pytest -q tests/test_diagnostic_rendering_report.py tests/test_diagnostic_orchestrator.py
```

Expected: missing split artifacts and old report names.

- [ ] **Step 3: Implement split renderers and report data**

Render `pre_split_labels` with the existing deterministic label palette, `changed_mask` as a red alpha overlay,
`split_score_map` with the heatmap helper, and `split_decision_map` with a fixed legend for accepted/no-markers/
small-child/low-score. Generalize method comparison and report colors:

```python
COLORS = {
    "depth": "#087f5b",
    "geometry_baseline": "#c92a2a",
    "layer_atomic_split": "#2459a9",
}
```

Write summary and case metrics using the two regret fields and selection reasons. Require all three config
directories before building a case.

- [ ] **Step 4: Run rendering/report tests**

Run:

```bash
pytest -q tests/test_diagnostic_rendering_report.py tests/test_diagnostic_orchestrator.py tests/test_diagnostic_rendering_report.py
```

Expected: all pass.

- [ ] **Step 5: Commit the report update**

```bash
git add inference_engine/diagnostics/rendering.py inference_engine/diagnostics/report.py inference_engine/diagnostics/orchestrator.py tests/test_diagnostic_rendering_report.py tests/test_diagnostic_orchestrator.py
git commit -m "feat: report split diagnostics"
```

---

### Task 7: Update the CPU verifier and cloud workflow

**Files:**
- Modify: `scripts/verify_segmentation_diagnostics.py`
- Modify: `scripts/run_segmentation_diagnostics.py`
- Modify: `docs/segmentation-diagnostics-cloud.md`
- Modify: `tests/test_segmentation_diagnostics_smoke.py`
- Modify: `tests/test_segmentation_smoke_script.py`

**Interfaces:**
- Consumes: complete three-method diagnostics and report interfaces.
- Produces: a no-weight CPU verification command plus KITTI 04 and KITTI 00–10 cloud commands.

- [ ] **Step 1: Write failing verifier assertions**

Update smoke tests to require the CLI description and synthetic output to contain
`depth`, `geometry_baseline`, and `layer_atomic_split`, and to reject old four-profile progress text.

- [ ] **Step 2: Verify smoke tests fail**

Run:

```bash
pytest -q tests/test_segmentation_diagnostics_smoke.py tests/test_segmentation_smoke_script.py
```

Expected: failure because verifier fixtures and documentation still use `layer_atomic`.

- [ ] **Step 3: Update verifier and documentation**

Make the CPU verifier exercise staged/public parity, split dense rendering, the two regrets, selection reasons,
three-method summary, schema 2.0, storage limits, and report generation. Update cloud progress examples to:

```text
[phase pass1 1/3] depth
[phase pass1 2/3] geometry_baseline
[phase pass1 3/3] layer_atomic_split
```

Keep the existing environment preparation, KITTI layout, `--dry-run`, `--resume`, KITTI 04 validation, and
KITTI 00–10 full-run commands. State the fixed split parameters in the experiment contract.

- [ ] **Step 4: Run verifier and smoke tests**

Run:

```bash
pytest -q tests/test_segmentation_diagnostics_smoke.py tests/test_segmentation_smoke_script.py
python scripts/verify_segmentation_diagnostics.py --output-dir /tmp/laser-split-diagnostics-verify
```

Expected: tests pass and verifier prints `[PASS]` for schema, parity, storage, selection, rendering, and report.

- [ ] **Step 5: Commit verifier and cloud instructions**

```bash
git add scripts/verify_segmentation_diagnostics.py scripts/run_segmentation_diagnostics.py docs/segmentation-diagnostics-cloud.md tests/test_segmentation_diagnostics_smoke.py tests/test_segmentation_smoke_script.py
git commit -m "docs: update split diagnostics workflow"
```

---

### Task 8: Full regression, audit, and delivery

**Files:**
- Modify only files required by failures that reproduce under the full suite.

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: a clean branch, passing tests, pushed remote branch, and exact cloud clone/run instructions.

- [ ] **Step 1: Run diagnostics and production focused regression**

Run:

```bash
pytest -q tests/test_post_merge_split.py tests/test_post_merge_split_evaluation.py tests/test_layer_atomic_split_integration.py tests/test_segmentation_modes.py tests/test_segmentation_engine_modes.py tests/test_diagnostic_schema_storage.py tests/test_diagnostic_trajectory.py tests/test_diagnostic_segmentation.py tests/test_diagnostic_selection.py tests/test_diagnostic_orchestrator.py tests/test_diagnostic_rendering_report.py tests/test_segmentation_diagnostics_smoke.py
```

Expected: all pass.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
pytest -q
```

Expected: all collected tests pass; environment-only skips are reported as skips, not failures.

- [ ] **Step 3: Run verifier and repository checks**

Run:

```bash
python scripts/verify_segmentation_diagnostics.py --output-dir /tmp/laser-split-diagnostics-final
git diff --check
git status --short --branch
```

Expected: verifier passes, diff check is empty, and no uncommitted files remain.

- [ ] **Step 4: Audit obsolete official configuration names**

Run:

```bash
rg -n 'geometry_legacy_reference|layer_atomic_minus_|layer_atomic_vs_geometry' inference_engine/diagnostics scripts/run_segmentation_diagnostics.py scripts/verify_segmentation_diagnostics.py docs/segmentation-diagnostics-cloud.md tests/test_diagnostic_*.py
```

Expected: no obsolete diagnostic configuration or regret field remains. References inside historical design and plan documents are allowed.

- [ ] **Step 5: Push the target branch**

Run:

```bash
git push origin codex/segmentation-diagnostics
```

Expected: remote `codex/segmentation-diagnostics` advances to the verified local HEAD.

- [ ] **Step 6: Deliver cloud commands**

Provide commands that clone the exact branch with `--single-branch`, create the environment, build the extension,
run the CPU verifier, run KITTI 04 first, and then run KITTI 00–10 with `--resume` after dry-run. State that real
ATE was not produced locally without KITTI, checkpoint, and GPU.
