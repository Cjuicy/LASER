# LASER Segmentation Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-command, two-pass KITTI Odometry diagnostics pipeline that compares three official LASER segmentation methods plus the geometry legacy reference without changing their default behavior.

**Architecture:** Add an optional, no-op-by-default diagnostic sink at the graph and scale-alignment boundaries. Pass 1 writes only pose shards and scalar JSONL metrics, a deterministic selector chooses intervals, and Pass 2 reruns complete sequences while persisting dense traces only for selected intervals. A master orchestrator runs four isolated configuration subprocesses sequentially, enforces a 50 GiB temporary budget, and builds offline CSV/JSON/PNG/PLY/HTML reports.

**Tech Stack:** Python 3.11+, PyTorch, NumPy 1.26, SciPy, scikit-image, OpenCV, trimesh, evo, pytest.

## Global Constraints

- Branch `codex/segmentation-diagnostics` is based on `codex/unified-segmentation-methods@98cce5f9f470599aca0cf5a6614f39409d929d58`.
- Diagnostics are disabled by default and must not change depth, geometry, layer-atomic, graph matching, scale propagation, Sim(3), caching, or loop-closure behavior.
- Official profiles are `depth`, `geometry_baseline`, and `layer_atomic`; `geometry_legacy_reference` is diagnostic-only.
- Official Felzenszwalb parameters remain `300 / 1.1 / 500`; geometry legacy remains explicit `200 / 1.0 / 300`.
- The target is non-loop-closure `StreamingWindowEngine` on KITTI Odometry 00–10.
- Four profiles run sequentially with the same checkpoint SHA-256 and fixed seed.
- Temporary hard limit is 50 GiB, warning threshold 40 GiB, and default free-space reserve 10 GiB.
- Pass 2 reruns complete sequences and saves dense traces only for the union of selected intervals.
- No external web assets are allowed in the static report.
- Use existing dependencies; do not add pandas, Jinja, Plotly, or a GUI/OpenGL requirement.

---

### Task 1: Versioned schemas, atomic storage, manifests, and disk budget

**Files:**
- Create: `inference_engine/diagnostics/__init__.py`
- Create: `inference_engine/diagnostics/schema.py`
- Create: `inference_engine/diagnostics/storage.py`
- Test: `tests/test_diagnostic_schema_storage.py`

**Interfaces:**
- Produces `DiagnosticContext`, `SelectedInterval`, `RunManifest`, `StorageBudget`, `RunLock`, `atomic_write_json`, `append_jsonl`, and `owned_temp_directory`.
- Later tasks use `DiagnosticContext.frame_id(local_index)`, `SelectedInterval.contains(frame_id)`, and `StorageBudget.enforce(estimated_bytes=0)`.

- [ ] **Step 1: Write failing schema and storage tests**

Test dataclass validation, JSON round-trip, interval containment, atomic replacement, JSONL appends, lock exclusion, ownership markers, warning/hard thresholds, and free-space reserve. Use byte-sized constructor overrides in tests so no large files are created:

```python
budget = StorageBudget(root=tmp_path, max_bytes=100, warn_bytes=80, min_free_bytes=0)
assert budget.state(used_bytes=79).level == "ok"
assert budget.state(used_bytes=80).level == "warning"
with pytest.raises(StorageLimitExceeded):
    budget.enforce(used_bytes=101)
```

- [ ] **Step 2: Run the tests and verify missing-module failures**

Run: `python -m pytest -q tests/test_diagnostic_schema_storage.py`
Expected: collection fails because `inference_engine.diagnostics` does not exist.

- [ ] **Step 3: Implement focused schema types**

Define immutable dataclasses with explicit validation:

```python
SCHEMA_VERSION = "1.0"

@dataclass(frozen=True)
class DiagnosticContext:
    run_id: str
    config_id: str
    sequence_id: str
    pass_id: int
    window_id: int
    frame_start: int

    def frame_id(self, local_index: int) -> int:
        return self.frame_start + local_index

@dataclass(frozen=True)
class SelectedInterval:
    sequence_id: str
    start_frame: int
    end_frame: int
    reasons: tuple[str, ...]
    score: float

    def contains(self, frame_id: int) -> bool:
        return self.start_frame <= frame_id <= self.end_frame
```

`RunManifest` stores status per `pass/config/sequence`, commit, checkpoint hash, config hash, dataset fingerprint, seed, environment, budget, and artifact schema version.

- [ ] **Step 4: Implement safe storage primitives**

`atomic_write_json` writes a sibling `.partial`, fsyncs, then uses `os.replace`. `append_jsonl` holds a process-local lock and fsyncs each record. `RunLock` uses exclusive file creation and records PID/hostname/run-id. `owned_temp_directory` writes `.laser-diagnostic-owner.json`; cleanup refuses directories without a matching run-id marker.

`StorageBudget` measures only the run-owned temp tree for the 50 GiB cap and checks filesystem free bytes separately. It reports `ok`, `warning`, or raises `StorageLimitExceeded` before a write.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest -q tests/test_diagnostic_schema_storage.py`
Expected: all pass.

Commit:

```bash
git add inference_engine/diagnostics tests/test_diagnostic_schema_storage.py
git commit -m "feat: add diagnostic schemas and storage guards"
```

---

### Task 2: Segmentation summaries, layer-atomic merge trace, and cross-method metrics

**Files:**
- Create: `inference_engine/diagnostics/segmentation.py`
- Create: `inference_engine/diagnostics/merge.py`
- Test: `tests/test_diagnostic_segmentation.py`
- Test: `tests/test_diagnostic_merge.py`

**Interfaces:**
- Consumes existing depth stages, geometry stages, `merge_layer_atoms`, and final label maps.
- Produces `summarize_labels(labels, initial_labels=None)`, `compare_labelings(left, right)`, `trace_segmentation_frame(...)`, and `analyze_layer_atomic_merge(...) -> LayerAtomicMergeTrace`.

- [ ] **Step 1: Write failing label-summary and comparison tests**

Use small hand-labelled arrays to pin segment count, LSR, Top-1/3/5, entropy, effective count, boundary ratio, partition validity, contingency, boundary disagreement, Variation of Information, and directional over-merge/over-split area. Missing values must be `None` with an `invalid_reason`, never zero.

- [ ] **Step 2: Verify tests fail because APIs are absent**

Run: `python -m pytest -q tests/test_diagnostic_segmentation.py`.

- [ ] **Step 3: Implement deterministic label metrics**

Relabel internally with `np.unique(..., return_inverse=True)`. Compute boundary masks from right/down differences, entropy from area probabilities, and Variation of Information from the contingency matrix. Return JSON-safe scalar dictionaries; arrays are returned separately only for Pass 2.

- [ ] **Step 4: Write failing layer-atomic trace tests**

Build synthetic atoms with same- and cross-coarse boundaries. Assert candidate counts, accepted counts, `G` quantiles, threshold margins, boundary mean/median/P90/P95, normal angle, component size after each union, 25/50/75% onset events, merge depth, longest chain, atoms-per-component, boundary-local scale mismatch, and invalid-pair reasons. Assert `trace.final_labels` exactly equals `merge_layer_atoms(...)`.

- [ ] **Step 5: Implement the read-only merge analyzer**

Recompute the existing atom scales, boundary codes, pair gaps, limits, and deterministic DSU order without calling or modifying the production DSU. Record an event after every accepted union. Reconstruct compact labels and raise `DiagnosticParityError` if they differ from the supplied formal final labels.

For normals, call `build_geometry_info_np(..., points=point_map, normal_method="cross")`. For boundary-local scale, compute internal neighbor spacing adjacent to each pair boundary and compare it with whole-atom scales.

- [ ] **Step 6: Implement traced stage collection without changing formal outputs**

`trace_segmentation_frame` receives the already-computed formal labels. It recomputes depth/geometry stages only when diagnostics are enabled, summarizes them, and verifies the recomputed final labels match the formal labels. Layer-atomic calls existing depth stages plus the read-only analyzer.

- [ ] **Step 7: Run tests and commit**

Run: `python -m pytest -q tests/test_diagnostic_segmentation.py tests/test_diagnostic_merge.py tests/test_geometry_segmentation.py tests/test_layer_atomic_geometry.py tests/test_layer_atomic_integration.py`.

Commit:

```bash
git add inference_engine/diagnostics tests/test_diagnostic_segmentation.py tests/test_diagnostic_merge.py
git commit -m "feat: trace segmentation and atomic merges"
```

---

### Task 3: Diagnostic sink, graph routing hooks, and scale/temporal observations

**Files:**
- Create: `inference_engine/diagnostics/sink.py`
- Create: `inference_engine/diagnostics/scale.py`
- Create: `inference_engine/diagnostics/temporal.py`
- Modify: `inference_engine/utils/lsa.py`
- Test: `tests/test_diagnostic_sink.py`
- Test: `tests/test_diagnostic_scale_temporal.py`
- Modify: `tests/test_segmentation_modes.py`

**Interfaces:**
- Produces `NullDiagnosticSink`, `FileDiagnosticSink`, `summarize_scale_observations`, and `summarize_temporal_graph`.
- Extends `make_sp_graph`, `refine_depth_segments`, and `align_adjacent_windows_depth_segments` with trailing optional `diagnostic_sink=None` and `diagnostic_context=None` parameters.
- Formal return values remain unchanged.

- [ ] **Step 1: Write failing sink selection and serialization tests**

Test Pass 1 scalar-only writes, Pass 2 selected-frame dense writes, non-selected frame suppression, bit-packed masks, uint16/uint32 labels, compressed NPZ, JSONL contexts, thread safety, and storage enforcement before each artifact.

- [ ] **Step 2: Verify sink tests fail**

Run: `python -m pytest -q tests/test_diagnostic_sink.py`.

- [ ] **Step 3: Implement sinks**

`NullDiagnosticSink` methods are no-ops. `FileDiagnosticSink` exposes:

```python
def emit_segmentation(context, local_index, metrics, arrays=None): ...
def snapshot_direct_anchors(context, graphs): ...
def emit_scale(context, metrics, arrays=None): ...
def emit_temporal(context, metrics, arrays=None): ...
def emit_inputs(context, images, point_maps, confidence): ...
def close(): ...
```

Pass 1 ignores `arrays`. Pass 2 writes arrays only when the global frame intersects selected intervals.

- [ ] **Step 4: Write failing scale/temporal formula and parity tests**

Construct small `Vertex` graphs with direct caches, propagated caches, fallback vertices, known IoUs, one-to-many, and many-to-one edges. Assert MAD/IQR/std of log scale, source count/area, anchor support, hop length, matched area, weighted IoU, churn, and lifetime.

Patch a recording sink into `align_adjacent_windows_depth_segments`; compare returned masks and every vertex cache against the untraced call.

- [ ] **Step 5: Implement observation-only hooks**

In `align_adjacent_windows_depth_segments`, snapshot target cache state immediately after `assign_overlap_window_depth_scale`, run the existing propagation loop unchanged, then emit summaries. Do not change iteration order or numeric expressions. In `make_sp_graph`, call `trace_segmentation_frame` after formal labels are computed and before `match_segmentation_seq`; emit temporal summaries after the graph is built.

- [ ] **Step 6: Run focused and routing regressions**

Run: `python -m pytest -q tests/test_diagnostic_sink.py tests/test_diagnostic_scale_temporal.py tests/test_segmentation_modes.py tests/test_segmentation_engine_modes.py`.

- [ ] **Step 7: Commit**

```bash
git add inference_engine/utils/lsa.py inference_engine/diagnostics tests/test_diagnostic_sink.py tests/test_diagnostic_scale_temporal.py tests/test_segmentation_modes.py
git commit -m "feat: observe segmentation scale and temporal state"
```

---

### Task 4: Streaming-engine metrics-only cache and selected input capture

**Files:**
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `inference_engine/streaming_window_engine_lc.py`
- Test: `tests/test_diagnostic_engine.py`
- Modify: `tests/test_segmentation_engine_modes.py`

**Interfaces:**
- `StreamingWindowEngine(..., diagnostic_sink=None, diagnostic_run_id=None, diagnostic_sequence_id=None, diagnostic_pass=0, cache_policy="full")`.
- Produces `parse_pose_cache_summary(remove_cache=True) -> dict[str, torch.Tensor]` with batched `extrinsic`.

- [ ] **Step 1: Write failing constructor and cache-policy tests**

Assert defaults preserve `cache_policy="full"`; invalid policies fail before model construction. In metrics-only mode, `_save_cache` writes only `camera_poses`, `window_id`, and `frame_start`, while `prev_window_cache` stays full in memory. Assert full mode writes the existing dictionary unchanged.

- [ ] **Step 2: Verify expected failures**

Run: `python -m pytest -q tests/test_diagnostic_engine.py`.

- [ ] **Step 3: Implement frame/window contexts and sink forwarding**

Frame start is `window_id * (window_size - overlap)`. `_build_segment_graph` passes `DiagnosticContext`; both normal and LC engines use the parent helper. After graph construction, emit selected RGB, confidence, and pre-refinement point maps. Pass the sink/context through `refine_depth_segments`.

- [ ] **Step 4: Implement metrics-only pose shards**

Persist one small shard per window using atomic writes. `parse_pose_cache_summary` trims overlap from later windows, concatenates camera poses, adds the batch dimension, and returns `{"extrinsic": poses}` without requiring local points, confidence, or images. Cleanup uses only run-owned directories.

- [ ] **Step 5: Verify engine and default-path parity**

Run: `python -m pytest -q tests/test_diagnostic_engine.py tests/test_segmentation_engine_modes.py tests/test_demo.py tests/test_demo_lc.py`.

- [ ] **Step 6: Commit**

```bash
git add inference_engine/streaming_window_engine.py inference_engine/streaming_window_engine_lc.py tests/test_diagnostic_engine.py tests/test_segmentation_engine_modes.py
git commit -m "feat: add bounded diagnostic engine caching"
```

---

### Task 5: Trajectory evaluation, Stability Guard, Recovery, and deterministic interval selection

**Files:**
- Create: `inference_engine/diagnostics/trajectory.py`
- Create: `inference_engine/diagnostics/metrics.py`
- Create: `inference_engine/diagnostics/selection.py`
- Test: `tests/test_diagnostic_trajectory.py`
- Test: `tests/test_diagnostic_selection.py`

**Interfaces:**
- Produces `evaluate_trajectory(pred, gt)`, `build_sequence_summary`, `evaluate_stability_guard`, `robust_zscore`, and `select_intervals(records, limit=48)`.

- [ ] **Step 1: Write failing synthetic trajectory tests**

Generate a known Sim(3)-transformed trajectory. Assert one global alignment recovers it, per-frame errors are computed from that one transform, ATE matches RMSE, and RPE uses `delta=1 frame`, `all_pairs=True`. Test missing/degenerate GT returns an invalid record rather than zeros.

- [ ] **Step 2: Implement trajectory evaluation**

Use evo association and alignment consistently with the existing evaluator. Save the evaluation signature in output:

```text
APE translation, align=True, correct_scale=True
RPE translation/rotation, delta=1 frame, all_pairs=True
```

- [ ] **Step 3: Write failing guard/recovery tests**

Assert default guard thresholds: mean regression <=3%, median <=0%, each of 00/05/09 <=10%, and all sequences valid. Test Recovery denominator guards.

- [ ] **Step 4: Implement sequence and group summaries**

Return JSON-safe official rankings, legacy reference fields, regret arrays, Recovery Gap, candidate Recovery Score, and explicit guard failure reasons.

- [ ] **Step 5: Write failing selector tests**

Test robust z-score fallbacks, trajectory Top-3, guard Top-2, change points, four metric families, matched controls, +/-2-window expansion, one-window-gap merging, union across configs, mandatory diversity, 48-interval cap, deterministic ties, and reason preservation.

- [ ] **Step 6: Implement deterministic selection**

Generate candidates by reason, merge intervals, satisfy mandatory coverage first, then rank remaining candidates with weights trajectory .40, merge/atom .20, scale .25, temporal .15. Controls use nearest normalized GT speed, turn magnitude, and confidence among below-median anomaly windows.

- [ ] **Step 7: Run tests and commit**

Run: `python -m pytest -q tests/test_diagnostic_trajectory.py tests/test_diagnostic_selection.py`.

Commit:

```bash
git add inference_engine/diagnostics tests/test_diagnostic_trajectory.py tests/test_diagnostic_selection.py
git commit -m "feat: select trajectory and segmentation diagnostic cases"
```

---

### Task 6: Deterministic renderers, PLY export, and two-level static report

**Files:**
- Create: `inference_engine/diagnostics/rendering.py`
- Create: `inference_engine/diagnostics/report.py`
- Test: `tests/test_diagnostic_rendering_report.py`

**Interfaces:**
- Produces `render_case(trace_paths, output_dir)`, `write_segment_ply`, and `build_report(run_dir) -> Path`.

- [ ] **Step 1: Write failing rendering tests**

Create a tiny synthetic trace and assert readable RGB/depth/confidence, stage, merge, component-growth, scale-source, scale-dispersion, temporal, top/side point-cloud PNGs; assert PLY vertex count/color fields and deterministic bytes for repeated runs.

- [ ] **Step 2: Implement headless deterministic rendering**

Use OpenCV and NumPy only. Label colors are derived from a stable integer hash; boundary colors are fixed; source map colors are direct teal, propagated purple, fallback gray. Use percentile-clipped legends with values stored in JSON. Project finite downsampled points to top/side planes without OpenGL.

- [ ] **Step 3: Write failing report tests**

Build a synthetic run tree. Assert overview has config/commit/checkpoint/budget, Stability Guard, Recovery, ATE/RPE heatmap, error timeline, selected ranking, correlations, incomplete warnings, and links to every case artifact. Assert case pages contain the approved segmentation -> merge -> scale -> trajectory layout and no external URLs.

- [ ] **Step 4: Implement CSV/JSON/SVG/HTML report**

Use stdlib `csv`, `json`, and escaped HTML. Draw charts as inline SVG. Compute Pearson/Spearman and lag 0-3 with sample counts and missing rates. Auto summaries use only evidence phrases such as “supports checking” and never claim causality.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_diagnostic_rendering_report.py`.

Commit:

```bash
git add inference_engine/diagnostics tests/test_diagnostic_rendering_report.py
git commit -m "feat: render segmentation diagnostic reports"
```

---

### Task 7: Checkpoint loading, KITTI worker, master orchestrator, resume, and CLI

**Files:**
- Create: `inference_engine/diagnostics/orchestrator.py`
- Create: `scripts/run_segmentation_diagnostics.py`
- Modify: `eval_launch.py`
- Test: `tests/test_diagnostic_orchestrator.py`
- Create: `tests/test_eval_launch.py`

**Interfaces:**
- Master CLI owns preflight, manifests, four sequential config subprocesses, Pass 1, selection, Pass 2, verification, cleanup, and report.
- Hidden worker mode loads one model per config and runs requested sequences sequentially.

- [ ] **Step 1: Write failing CLI/preflight tests**

Test defaults, exact four profiles, non-LC enforcement, dataset layout (`sequences/<id>/image_2`, `poses/<id>.txt`), shared checkpoint SHA-256, seed, frame counts, dry-run estimates, 50/40/10 GiB validation, and readable phase logs.

- [ ] **Step 2: Implement checkpoint and dataset preflight**

Load `.safetensors` or PyTorch checkpoints exactly as `demo.py` does. Store checkpoint SHA-256. Dataset fingerprint hashes sequence image relative paths, sizes, mtimes, and pose files without reading every image into memory.

- [ ] **Step 3: Write failing subprocess/resume tests**

Use a fake worker command to assert profiles are sequential, a failed profile prevents false completion, completed checkpoints skip on `--resume`, mismatched commit/config/data fingerprint refuses resume, report-only does not invoke workers, and signals leave a resumable manifest.

- [ ] **Step 4: Implement the master state machine**

Phases are `preflight`, `pass1`, `trajectory`, `selection`, `pass2`, `verify`, `report`, `cleanup`, `complete`. Before each subprocess, enforce budget and free reserve. After each sequence checkpoint, clean only marked temp. Exit nonzero for incomplete configurations but still build an incomplete report.

- [ ] **Step 5: Implement real config worker**

The worker loads the model once, then for each sequence loads sorted images, creates a metrics-only `StreamingWindowEngine`, runs full sliding-window inference, parses pose shards, evaluates GT, closes the sink, and cleans engine temp. Pass 2 receives selected intervals and captures only their union.

- [ ] **Step 6: Expose unified evaluation arguments**

Add `--segment_mode`, `--normal_method`, `--geometry_seg_profile`, `--model_ckpt`, and diagnostic arguments to `eval_launch.py`. Existing invocations without these arguments preserve current defaults and behavior.

- [ ] **Step 7: Run orchestrator tests and commit**

Run: `python -m pytest -q tests/test_diagnostic_orchestrator.py tests/test_eval_launch.py tests/test_demo.py tests/test_demo_lc.py`.

Commit:

```bash
git add inference_engine/diagnostics/orchestrator.py scripts/run_segmentation_diagnostics.py eval_launch.py tests/test_diagnostic_orchestrator.py tests/test_eval_launch.py
git commit -m "feat: orchestrate bounded two-pass diagnostics"
```

---

### Task 8: Synthetic end-to-end verifier and cloud documentation

**Files:**
- Create: `scripts/verify_segmentation_diagnostics.py`
- Create: `tests/test_segmentation_diagnostics_smoke.py`
- Create: `docs/segmentation-diagnostics-cloud.md`
- Modify: `README.md`

**Interfaces:**
- `python scripts/verify_segmentation_diagnostics.py` performs CPU-only Pass 1 -> selection -> Pass 2 -> artifacts -> report without weights.

- [ ] **Step 1: Write the failing smoke test**

Run the verifier in a subprocess and require exit 0, `[PASS]` lines for schema, parity, storage, selection, rendering, report, and a final readable `report/index.html`.

- [ ] **Step 2: Implement synthetic end-to-end verification**

Generate two-frame point maps and four synthetic profile records, run the real diagnostic metrics/selector/storage/render/report code, verify checksums and links, and print the hard/warning budget configuration. Do not mock the functions being verified.

- [ ] **Step 3: Run smoke and focused suite**

Run:

```bash
python scripts/verify_segmentation_diagnostics.py
python -m pytest -q tests/test_segmentation_diagnostics_smoke.py
```

- [ ] **Step 4: Write cloud execution documentation**

Document branch checkout, environment/build, CPU verifier, dry-run, KITTI 04 validation, full 00–10 run, resume, report-only, expected phase logs, 50 GiB behavior, output paths, and success criteria. Commands must use one checkpoint and the four fixed profiles.

- [ ] **Step 5: Update README and commit**

```bash
git add scripts/verify_segmentation_diagnostics.py tests/test_segmentation_diagnostics_smoke.py docs/segmentation-diagnostics-cloud.md README.md
git commit -m "docs: add cloud segmentation diagnostics workflow"
```

---

### Task 9: Full regression, parity audit, cleanup, and GitHub push

**Files:**
- Review all changes against `codex/unified-segmentation-methods`.

- [ ] **Step 1: Rebuild native extensions**

Run: `python setup.py build_ext --inplace`
Expected: both Cython extensions build successfully.

- [ ] **Step 2: Run complete verification**

```bash
python -m pytest -q
python scripts/verify_segmentation_modes.py
python scripts/verify_segmentation_diagnostics.py
python -m compileall -q inference_engine scripts eval_launch.py
```

Expected: all tests and both smoke scripts pass.

- [ ] **Step 3: Audit behavior preservation**

```bash
git diff --exit-code codex/unified-segmentation-methods -- \
    inference_engine/utils/depth.py \
    inference_engine/utils/layer_atomic_geometry.py \
    inference_engine/utils/geometry_segmentation.py
git diff --check codex/unified-segmentation-methods...HEAD
```

Expected: method implementation files have no differences and diff check is clean.

- [ ] **Step 4: Review resource and failure behavior**

Run storage tests with simulated warning/hard/free-space failures, interrupted resume, incomplete report, and sequential worker assertions. Confirm generated build `.cpp` files are not staged.

- [ ] **Step 5: Push the completed branch**

```bash
git push -u origin codex/segmentation-diagnostics
git ls-remote --heads origin codex/segmentation-diagnostics
```

Remote hash must match local HEAD before reporting completion.
