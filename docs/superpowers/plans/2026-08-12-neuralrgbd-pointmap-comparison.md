# NeuralRGBD Point-Map Method Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add evaluation-only NeuralRGBD profiles and an auditable comparison reader, then run Depth, Geometry, and Atomic over the exact nine LASER kf10 sequences on the approved cloud machine.

**Architecture:** Preserve the immutable two-dataset paper profile and extend the typed protocol with explicit `paper` and `comparison` modes. Three Hydra profiles select existing segmentation strategies while sharing the model, windows, dataset, alignment, and metrics; a comparison module enforces the Depth paper gate and rejects coverage, provenance, or ordinary-cache mismatches.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, frozen dataclasses, NumPy, Open3D, PyTorch, JSON/CSV, pytest, Git, SSH, and GNU screen.

## Global Constraints

- Working branch is `codex/laser-paper-pointmap-eval`; approved design commit is `3b66dca`.
- Modify only `configs/evaluation/`, `mv_recon/`, `tests/mv_recon/`, and `docs/`.
- Do not modify `pipeline/`, `inference_engine/`, Pi3, segmentation implementations, anchor propagation, loop closure, or ordinary prediction-cache code.
- Existing `evaluation=mv_recon_laser_paper` semantics and exact two-dataset locks remain unchanged.
- Comparison profiles select only `NRGBD-dense` and all nine entries of the fixed kf10 map when `max_sequences=null`.
- NeuralRGBD map SHA256 remains `f18f2143f8a373727aa4d7043b779b77639354fda80523cdc2a139164ddc33ba`.
- All methods use window `20/5`, both confidence ratios `0.5`, anchor IoU `0.4`, loop disabled/traditional, crop 224, Umeyama Sim(3), point-to-point ICP at `0.1 m`, Open3D default normals, and the GT validity mask.
- Depth uses Felzenszwalb `300/1.1/500`; Geometry uses cross normals at 20 degrees; Atomic uses conservative split at score `0.10`.
- Depth cache mode is `auto`; Geometry and Atomic are `readonly` against the same ordinary-cache root.
- Comparison runs serialize as `subset`, never complete two-dataset Table 4 runs.
- Geometry and Atomic cannot start unless all six Depth metrics match the published NeuralRGBD row at three displayed decimal places.
- All numerical JSON is finite and all result writes are atomic.
- Cloud audit established that all 1,101 NeuralRGBD inputs are valid; 7-Scenes is outside this run.
- Generated Cython sources `inference_engine/utils/_segmentation_cy.cpp` and
  `inference_engine/utils/fast_seg.cpp` remain untracked and must never be staged.

---

## File Map

- Modify `configs/evaluation/mv_recon_laser_paper.yaml`: add `mode: paper`.
- Create `configs/evaluation/mv_recon_laser_nrgbd_depth.yaml`.
- Create `configs/evaluation/mv_recon_laser_nrgbd_geometry.yaml`.
- Create `configs/evaluation/mv_recon_laser_nrgbd_atomic.yaml`.
- Modify `mv_recon/protocol.py`: comparison schema and locks.
- Modify `mv_recon/eval.py`: comparison subset state and manifest provenance.
- Create `mv_recon/compare_results.py`: Depth gate and strict comparison.
- Modify `mv_recon/README.md`: cloud workflow.
- Modify `tests/mv_recon/test_protocol.py` and `tests/mv_recon/test_eval_orchestration.py`.
- Create `tests/mv_recon/test_compare_results.py`.

---

### Task 1: Typed Comparison Mode and Method Locks

**Files:**
- Modify: `mv_recon/protocol.py:27-511`
- Modify: `configs/evaluation/mv_recon_laser_paper.yaml:26-55`
- Modify: `tests/mv_recon/test_protocol.py`

**Interfaces:**
- Consumes: existing typed protocol, `PipelineConfig`, `SegmentationMethod`, and paper constants.
- Produces: `LaserPaperProtocol.mode`, `COMPARISON_PROTOCOL_NAME`, paper-preserving validation, and method-specific comparison locks.

- [ ] **Step 1: Add failing mode and dataset tests**

Add these cases to `test_protocol.py`:

```python
def test_paper_profile_declares_paper_mode(tmp_path):
    resolved = resolve_evaluation_protocol(_root_config(tmp_path), ROOT)
    assert resolved.protocol.mode == "paper"
    assert resolved.datasets == ("7scenes-dense", "NRGBD-dense")


def test_paper_mode_still_rejects_nrgbd_only(tmp_path):
    config = _root_config(tmp_path)
    config.eval_datasets = ["NRGBD-dense"]
    with pytest.raises(ValueError, match="paper protocol.*datasets"):
        resolve_evaluation_protocol(config, ROOT)


def test_comparison_mode_accepts_nrgbd_only_depth(tmp_path):
    config = _comparison_config(tmp_path, method="depth")
    resolved = resolve_evaluation_protocol(config, ROOT)
    assert resolved.protocol.mode == "comparison"
    assert resolved.datasets == ("NRGBD-dense",)
    assert resolved.pipeline.config.segmentation.method.value == "depth"
```

Also test unknown mode; empty, duplicate, and unsupported comparison datasets;
changed paper references; Geometry other than `cross/20.0`; and Atomic other
than `conservative/0.10`.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/mv_recon/test_protocol.py -q`

Expected: new tests fail because `protocol.mode` does not exist.

- [ ] **Step 3: Extend the exact schema**

Add:

```python
COMPARISON_PROTOCOL_NAME = "laser_neuralrgbd_pointmap_comparison"
PROTOCOL_MODES = frozenset({"paper", "comparison"})
```

Add `"mode"` to `_PROTOCOL_KEYS` and this dataclass field:

```python
@dataclass(frozen=True)
class LaserPaperProtocol:
    name: str
    mode: str
    version: int
```

Parse only string members of `PROTOCOL_MODES`. Add `mode: paper` to the
existing paper YAML without changing any other value.

- [ ] **Step 4: Split common and mode-specific locks**

Replace `_validate_paper_locks()` with a dispatcher that always locks common
window, confidence, depth/temporal, all strategy defaults, anchors, loop,
geometry metrics, and paper references. Then apply:

```python
if protocol.mode == "paper":
    expected["protocol.name"] = (protocol.name, PAPER_PROTOCOL_NAME)
    expected["segmentation.method"] = (
        config.segmentation.method,
        SegmentationMethod.DEPTH,
    )
else:
    expected["protocol.name"] = (
        protocol.name,
        COMPARISON_PROTOCOL_NAME,
    )
    method = config.segmentation.method
    if method is SegmentationMethod.GEOMETRY:
        expected["segmentation.geometry.normal_method"] = (
            config.segmentation.geometry.normal_method, "cross"
        )
        expected["segmentation.geometry.normal_threshold_degrees"] = (
            config.segmentation.geometry.normal_threshold_degrees, 20.0
        )
    elif method is SegmentationMethod.ATOMIC:
        expected["segmentation.atomic.split_mode"] = (
            config.segmentation.atomic.split_mode.value, "conservative"
        )
        expected["segmentation.atomic.split_score_threshold"] = (
            config.segmentation.atomic.split_score_threshold, 0.10
        )
    elif method is not SegmentationMethod.DEPTH:
        raise ValueError(f"unsupported comparison segmentation: {method}")
```

- [ ] **Step 5: Validate datasets by mode**

Paper mode requires exact `PAPER_DATASETS`. Comparison mode requires a unique,
non-empty ordered subset whose members all belong to `PAPER_DATASETS`.

- [ ] **Step 6: Verify paper and comparison tests GREEN**

Run:

```bash
python -m pytest tests/mv_recon/test_protocol.py tests/test_eval_launch_lc.py -q
```

- [ ] **Step 7: Commit**

```bash
git add configs/evaluation/mv_recon_laser_paper.yaml \
        mv_recon/protocol.py tests/mv_recon/test_protocol.py
git commit -m "feat: add locked point-map comparison mode"
```

---

### Task 2: Explicit NeuralRGBD Method Profiles

**Files:**
- Create: `configs/evaluation/mv_recon_laser_nrgbd_depth.yaml`
- Create: `configs/evaluation/mv_recon_laser_nrgbd_geometry.yaml`
- Create: `configs/evaluation/mv_recon_laser_nrgbd_atomic.yaml`
- Modify: `tests/mv_recon/test_protocol.py`

**Interfaces:**
- Consumes: comparison mode from Task 1.
- Produces: three audited Hydra `evaluation=` profiles.

- [ ] **Step 1: Add failing composition tests**

```python
@pytest.mark.parametrize(
    ("profile", "method", "cache_mode"),
    (
        ("mv_recon_laser_nrgbd_depth", "depth", "auto"),
        ("mv_recon_laser_nrgbd_geometry", "geometry", "readonly"),
        ("mv_recon_laser_nrgbd_atomic", "atomic", "readonly"),
    ),
)
def test_nrgbd_profiles_are_locked(tmp_path, profile, method, cache_mode):
    resolved = resolve_evaluation_protocol(
        _compose_profile(tmp_path, profile), ROOT
    )
    assert resolved.protocol.mode == "comparison"
    assert resolved.datasets == ("NRGBD-dense",)
    assert resolved.pipeline.config.segmentation.method.value == method
    assert resolved.pipeline.config.prediction_cache.mode.value == cache_mode
```

Assert window `20/5`, both confidence ratios `0.5`, anchors, loop, and each
method's exact parameters.

- [ ] **Step 2: Run tests to verify missing-profile RED**

Run: `python -m pytest tests/mv_recon/test_protocol.py -q`

- [ ] **Step 3: Create Depth profile**

```yaml
# @package _global_
defaults:
  - mv_recon_laser_paper
  - _self_

name: mv_recon_laser_nrgbd_depth
eval_datasets: [NRGBD-dense]
protocol:
  name: laser_neuralrgbd_pointmap_comparison
  mode: comparison
  prediction_cache_mode: auto
```

- [ ] **Step 4: Create Geometry profile**

Inherit `mv_recon_laser_nrgbd_depth`, set its name, set cache mode `readonly`,
and replace the complete pipeline override list. Retain all common paper locks
and use:

```yaml
- segmentation.method=geometry
- segmentation.geometry.normal_method=cross
- segmentation.geometry.normal_threshold_degrees=20.0
```

- [ ] **Step 5: Create Atomic profile**

Inherit Depth, set its name, set cache mode `readonly`, retain the same common
locks, and use:

```yaml
- segmentation.method=atomic
- segmentation.atomic.split_mode=conservative
- segmentation.atomic.split_score_threshold=0.10
```

- [ ] **Step 6: Run profile tests GREEN**

Run: `python -m pytest tests/mv_recon/test_protocol.py -q`

- [ ] **Step 7: Commit**

```bash
git add configs/evaluation/mv_recon_laser_nrgbd_*.yaml \
        tests/mv_recon/test_protocol.py
git commit -m "feat: add NeuralRGBD point-map method profiles"
```

---

### Task 3: Comparison Subset State and Manifest Provenance

**Files:**
- Modify: `mv_recon/eval.py:362-470`
- Modify: `tests/mv_recon/test_eval_orchestration.py`

**Interfaces:**
- Consumes: `resolved.protocol.mode`, selected plans, and segmentation method.
- Produces: comparison `subset` state and manifest keys `evaluation_mode` and `segmentation_method`.

- [ ] **Step 1: Add failing NeuralRGBD-only orchestration tests**

```python
@pytest.mark.parametrize(
    ("profile", "method"),
    (
        ("mv_recon_laser_nrgbd_depth", "depth"),
        ("mv_recon_laser_nrgbd_geometry", "geometry"),
        ("mv_recon_laser_nrgbd_atomic", "atomic"),
    ),
)
def test_nrgbd_comparison_is_subset_with_method_manifest(
    tmp_path, profile, method
):
    result = run_evaluation(
        _config_for_profile(tmp_path, profile, max_sequences=1),
        dependencies=_dependencies(tmp_path, State()),
        repository_root=ROOT,
    )
    assert result.state == "subset"
    assert [(x.dataset, x.sequence) for x in result.sequences] == [
        ("NRGBD-dense", "breakfast_room")
    ]
    manifest = json.loads(
        (tmp_path / "results/protocol_manifest.json").read_text()
    )
    assert manifest["evaluation_mode"] == "comparison"
    assert manifest["segmentation_method"] == method
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/mv_recon/test_eval_orchestration.py -q`

- [ ] **Step 3: Add manifest fields**

In `_build_protocol_manifest()` add:

```python
"evaluation_mode": resolved.protocol.mode,
"segmentation_method": resolved.pipeline.config.segmentation.method.value,
```

- [ ] **Step 4: Mark comparison as subset**

Import `PAPER_DATASETS` and calculate:

```python
subset = (
    resolved.protocol.mode == "comparison"
    or resolved.datasets != PAPER_DATASETS
    or any(
        len(plan.sequences) != plan.expected_sequence_count
        for plan in plans
    )
)
```

- [ ] **Step 5: Run orchestration and result tests GREEN**

```bash
python -m pytest tests/mv_recon/test_eval_orchestration.py \
                     tests/mv_recon/test_results.py -q
```

- [ ] **Step 6: Commit**

```bash
git add mv_recon/eval.py tests/mv_recon/test_eval_orchestration.py
git commit -m "feat: mark point-map comparison provenance"
```

---

### Task 4: Depth Gate and Strict Comparison Reader

**Files:**
- Create: `mv_recon/compare_results.py`
- Create: `tests/mv_recon/test_compare_results.py`

**Interfaces:**
- Consumes: run directories containing `results.json` and `protocol_manifest.json`.
- Produces: `validate_depth_gate()`, `compare_run_directories()`, `depth_gate.json`, `comparison.json`, and `comparison.csv`.

- [ ] **Step 1: Create deterministic run fixtures**

Use fixture primary values that format to the paper row:

```python
PAPER_PRIMARY = {
    "accuracy_mean_m": 0.0201,
    "accuracy_median_m": 0.0101,
    "completion_mean_m": 0.0121,
    "completion_median_m": 0.0041,
    "normal_consistency_mean": 0.7131,
    "normal_consistency_median": 0.8561,
}
```

`_write_run()` must emit state `subset`, empty failures, one NRGBD summary,
sequence primary/diagnostics, input/GT hashes, ordinary keys, and a manifest
with mode, method, checkpoint hash, map hash, and run state.

- [ ] **Step 2: Add failing Depth gate tests**

```python
def test_depth_gate_accepts_three_decimal_paper_match(tmp_path):
    run = _write_run(tmp_path / "depth", "depth", EXPECTED_NAMES)
    values = validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)
    assert values["accuracy_mean_m"] == pytest.approx(0.0201)


def test_depth_gate_rejects_changed_metric(tmp_path):
    run = _write_run(
        tmp_path / "depth", "depth", EXPECTED_NAMES,
        primary={**PAPER_PRIMARY, "accuracy_mean_m": 0.021},
    )
    with pytest.raises(ValueError, match="reproduction gate"):
        validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)
```

Also reject wrong state, failures, missing/duplicate sequence coverage, wrong
method, results/manifest identity disagreement, and non-finite values.

- [ ] **Step 3: Add failing comparability tests**

Independently reject differences in checkpoint hash, map hash, sequence order,
input hash, GT hash, ordinary prediction key, metric schema, and manifest
method.

- [ ] **Step 4: Add failing successful projection test**

```python
report = compare_run_directories(depth, geometry, atomic, output)
assert [row["method"] for row in report["methods"]] == [
    "depth", "geometry", "atomic"
]
assert report["methods"][1]["delta_to_depth"]["accuracy_mean_m"] \
    == pytest.approx(-0.001)
assert (output / "comparison.json").is_file()
assert (output / "comparison.csv").is_file()
```

Assert Chamfer-L1 and 1/2/5 cm precision, recall, and F-score are sequence
macro averages.

- [ ] **Step 5: Run tests to verify module-missing RED**

Run: `python -m pytest tests/mv_recon/test_compare_results.py -q`

- [ ] **Step 6: Implement strict run loading**

```python
METHODS = ("depth", "geometry", "atomic")
PRIMARY_FIELDS = tuple(field.name for field in fields(PrimaryMetrics))


def _load_run(run_dir: Path, expected_method: str) -> dict[str, object]:
    results = _read_object(run_dir / "results.json")
    manifest = _read_object(run_dir / "protocol_manifest.json")
    if results["schema_version"] != METRIC_SCHEMA_VERSION:
        raise ValueError("metric schema mismatch")
    if results["state"] != "subset" or results["failures"]:
        raise ValueError("comparison run is not a successful subset")
    if manifest["evaluation_mode"] != "comparison":
        raise ValueError("run is not a comparison profile")
    if manifest["segmentation_method"] != expected_method:
        raise ValueError("comparison method mismatch")
    _require_finite(results)
    return {"results": results, "manifest": manifest}
```

Reject booleans as numeric metrics and reject missing or duplicate sequences.

- [ ] **Step 7: Implement the Depth gate**

```python
def validate_depth_gate(
    run_dir: Path,
    *,
    expected_sequences: Sequence[str] | None = None,
) -> dict[str, float]:
    run = _load_run(run_dir, "depth")
    expected = tuple(expected_sequences or _nrgbd_sequence_names())
    _validate_coverage(run, expected)
    primary = _dataset_primary(run)
    reference = PAPER_REFERENCE_VALUES["NRGBD-dense"]
    mismatches = {
        name: (format(primary[name], ".3f"), format(reference[name], ".3f"))
        for name in PRIMARY_FIELDS
        if format(primary[name], ".3f") != format(reference[name], ".3f")
    }
    if mismatches:
        raise ValueError(f"Depth reproduction gate failed: {mismatches}")
    return primary
```

- [ ] **Step 8: Implement comparison validation and macro diagnostics**

First require each run's result identity to agree with its manifest checkpoint,
map, and metric-schema fields. Then require identical dataset/sequence order,
checkpoint/map hashes, per-sequence input/GT hashes, ordinary keys, and metric
schema across methods. Average sequence diagnostics with `np.mean`; require
identical threshold lists.
Compute method minus Depth. Record `lower` direction for distance/Chamfer and
`higher` for NC/precision/recall/F-score.

- [ ] **Step 9: Implement atomic projections and CLI**

Use temporary sibling files, `fsync`, `Path.replace()`, and JSON
`allow_nan=False`. Support:

```bash
python mv_recon/compare_results.py \
  --depth-run outputs/pointmap/nrgbd_depth \
  --depth-gate-only \
  --output-dir outputs/pointmap/nrgbd_comparison

python mv_recon/compare_results.py \
  --depth-run outputs/pointmap/nrgbd_depth \
  --geometry-run outputs/pointmap/nrgbd_geometry \
  --atomic-run outputs/pointmap/nrgbd_atomic \
  --output-dir outputs/pointmap/nrgbd_comparison
```

- [ ] **Step 10: Run comparison tests GREEN**

Run: `python -m pytest tests/mv_recon/test_compare_results.py -q`

- [ ] **Step 11: Commit**

```bash
git add mv_recon/compare_results.py tests/mv_recon/test_compare_results.py
git commit -m "feat: compare NeuralRGBD point-map methods"
```

---

### Task 5: Documentation, Verification, and Push

**Files:**
- Modify: `mv_recon/README.md`
- Modify design document only if implementation names changed during TDD.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: documented commands, green suite, verified commit, and pushed branch.

- [ ] **Step 1: Document eligibility and profiles**

State that all nine NeuralRGBD sequences are used; run state is intentionally
`subset`; only Depth is checked against the paper; Geometry/Atomic are
controlled experiments; cache modes are `auto/readonly/readonly`; later
methods require a passing Depth gate.

- [ ] **Step 2: Document all commands and outputs**

Use exact roots:

```text
outputs/pointmap/nrgbd_depth_preflight
outputs/pointmap/nrgbd_depth_smoke
outputs/pointmap/nrgbd_depth
outputs/pointmap/nrgbd_geometry_smoke
outputs/pointmap/nrgbd_geometry
outputs/pointmap/nrgbd_atomic_smoke
outputs/pointmap/nrgbd_atomic
outputs/pointmap/nrgbd_comparison
```

Every resume repeats the same profile and output with `protocol.resume=true`.

- [ ] **Step 3: Run focused and full tests**

```bash
python -m pytest tests/mv_recon -q
python -m pytest -q
```

Expected: all pass; the existing Torch JIT deprecation warning may remain.

- [ ] **Step 4: Run static and scope checks**

```bash
python -m py_compile mv_recon/eval.py mv_recon/protocol.py \
  mv_recon/geometry_metrics.py mv_recon/results.py \
  mv_recon/compare_results.py
git diff --check ac2340cc90c874fd839141aa89b29587de32ed7e...HEAD
git diff --name-only ac2340cc90c874fd839141aa89b29587de32ed7e...HEAD
```

- [ ] **Step 5: Commit documentation**

```bash
git add mv_recon/README.md \
  docs/superpowers/specs/2026-08-12-neuralrgbd-pointmap-comparison-design.md
git commit -m "docs: add NeuralRGBD comparison workflow"
```

- [ ] **Step 6: Push and verify SHA**

```bash
git push origin codex/laser-paper-pointmap-eval
git ls-remote --heads origin codex/laser-paper-pointmap-eval
git rev-parse HEAD
```

Expected: local and remote SHA match.

---

### Task 6: Cloud Depth Reproduction Gate

**Files:**
- Do not modify tracked cloud files.
- Create only symlinks, screen logs, caches, and outputs under the checkout.

**Interfaces:**
- Consumes: pushed branch, audited NeuralRGBD data, Pi3 checkpoint, RTX 5090.
- Produces: passing all-nine Depth result and `depth_gate.json`, or preserved diagnostics.

- [ ] **Step 1: Update exact branch**

```bash
cd /root/autodl-tmp/LASER-pointmap-eval
git pull --ff-only origin codex/laser-paper-pointmap-eval
git rev-parse HEAD
```

- [ ] **Step 2: Create and verify links**

```bash
ln -sfn NeuralRGBD data/nrgbd
ln -sfn PI3/model.safetensors weights/model.safetensors
test -d data/nrgbd/breakfast_room
test -f weights/model.safetensors
readlink -f data/nrgbd
readlink -f weights/model.safetensors
```

Before each canonical run, require that its output directory is absent. If a
compatible partial output already exists, inspect its manifest and resume it
with `protocol.resume=true`; never silently mix or overwrite an unidentified
old run.

- [ ] **Step 3: Verify runtime and extensions**

```bash
conda activate vggt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python -c 'from inference_engine.utils.fast_seg import fast_graph_segmentation; from inference_engine.utils._segmentation_cy import merge_regions; print("extensions OK")'
python -c 'import torch, open3d; print(torch.__version__, torch.cuda.get_device_name(), open3d.__version__)'
```

- [ ] **Step 4: Run tests and preflight**

```bash
python -m pytest tests/mv_recon -q
python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth \
  protocol.preflight_only=true \
  output_dir=outputs/pointmap/nrgbd_depth_preflight
```

- [ ] **Step 5: Run one-sequence Depth smoke in screen**

```bash
mkdir -p outputs/pointmap/logs
screen -dmS nrgbd_depth_smoke bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt
cd /root/autodl-tmp/LASER-pointmap-eval
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python mv_recon/eval.py evaluation=mv_recon_laser_nrgbd_depth \
  protocol.max_sequences=1 \
  output_dir=outputs/pointmap/nrgbd_depth_smoke \
  > outputs/pointmap/logs/nrgbd_depth_smoke.log 2>&1
'
```

Require state `subset`, exactly `breakfast_room`, and no failures.

- [ ] **Step 6: Run all-nine Depth in screen**

```bash
screen -dmS nrgbd_depth bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt
cd /root/autodl-tmp/LASER-pointmap-eval
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python mv_recon/eval.py evaluation=mv_recon_laser_nrgbd_depth \
  output_dir=outputs/pointmap/nrgbd_depth \
  > outputs/pointmap/logs/nrgbd_depth.log 2>&1
'
```

Monitor with `screen -ls`, `tail -n 80` on the log, and `nvidia-smi`. Resume
the same output with `protocol.resume=true` after a recoverable interruption.

- [ ] **Step 7: Apply hard gate**

```bash
python mv_recon/compare_results.py \
  --depth-run outputs/pointmap/nrgbd_depth \
  --depth-gate-only \
  --output-dir outputs/pointmap/nrgbd_comparison
cat outputs/pointmap/nrgbd_comparison/depth_gate.json
```

If this exits nonzero, stop before Task 7 and diagnose Depth.

---

### Task 7: Cloud Geometry, Atomic, and Final Comparison

**Files:**
- Do not modify tracked files.
- Produce method outputs under `outputs/pointmap/`.

**Interfaces:**
- Consumes: passing Depth gate and complete ordinary cache.
- Produces: two more all-nine results plus strict comparison JSON/CSV.

- [ ] **Step 1: Run Geometry smoke and verify readonly hits**

Run `evaluation=mv_recon_laser_nrgbd_geometry` with
`protocol.max_sequences=1` and output `nrgbd_geometry_smoke` in screen. Require
zero ordinary misses and the same `breakfast_room` ordinary key as Depth.

- [ ] **Step 2: Run all-nine Geometry**

```bash
screen -dmS nrgbd_geometry bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt
cd /root/autodl-tmp/LASER-pointmap-eval
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python mv_recon/eval.py evaluation=mv_recon_laser_nrgbd_geometry \
  output_dir=outputs/pointmap/nrgbd_geometry \
  > outputs/pointmap/logs/nrgbd_geometry.log 2>&1
'
```

- [ ] **Step 3: Run Atomic smoke and verify readonly hits**

Run `evaluation=mv_recon_laser_nrgbd_atomic` with
`protocol.max_sequences=1` and output `nrgbd_atomic_smoke` in screen. Require
zero ordinary misses and the Depth ordinary key.

- [ ] **Step 4: Run all-nine Atomic**

```bash
screen -dmS nrgbd_atomic bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt
cd /root/autodl-tmp/LASER-pointmap-eval
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python mv_recon/eval.py evaluation=mv_recon_laser_nrgbd_atomic \
  output_dir=outputs/pointmap/nrgbd_atomic \
  > outputs/pointmap/logs/nrgbd_atomic.log 2>&1
'
```

- [ ] **Step 5: Generate strict final comparison**

```bash
python mv_recon/compare_results.py \
  --depth-run outputs/pointmap/nrgbd_depth \
  --geometry-run outputs/pointmap/nrgbd_geometry \
  --atomic-run outputs/pointmap/nrgbd_atomic \
  --output-dir outputs/pointmap/nrgbd_comparison
```

- [ ] **Step 6: Inspect and report**

```bash
cat outputs/pointmap/nrgbd_comparison/comparison.csv
python -m json.tool outputs/pointmap/nrgbd_comparison/comparison.json
```

Report six primary metrics, Chamfer-L1, 1/2/5 cm precision/recall/F-score,
method-minus-Depth deltas and metric direction, states/counts, identical
ordinary keys/readonly hits, code/runtime provenance, and the explicit
7-Scenes limitation.
