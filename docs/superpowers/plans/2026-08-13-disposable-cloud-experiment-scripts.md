# Disposable Cloud Experiment Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver five disposable cloud scripts that run the exact requested point-cloud and KITTI ATE matrices, retain only validated compact results and logs, safely clean large intermediates, and generate CSV/XLSX summaries.

**Architecture:** Keep LASER reconstruction and evaluation code unchanged. A Python utility owns experiment identities, dataset preparation, compact result validation, and cleanup guards; two small shell runners invoke existing CLIs one method at a time; a summary script creates the final tables; and one wrapper is the only user-facing entry point.

**Tech Stack:** Python 3.11, Bash, existing LASER typed configuration and evaluators, NumPy/Pillow, `@oai/artifact-tool` for XLSX authoring, pytest.

## Global Constraints

- Source baseline is `codex/laser-paper-pointmap-eval` at commit `cfe26f8b57341a9320eae2ef71532a353654d1a6` plus this disposable script change.
- Do not modify `pipeline/`, `reconstruction/`, `inference_engine/`, `loop_closure/`, or `evaluation/` algorithm behavior.
- Point cloud is all 18 7-Scenes kf10 sequences and all 9 NeuralRGBD kf10 scenes at `w20-o5` with five segmentation identities and `no_loop`.
- KITTI ATE is sequences `00` through `10` at `s1-w75-o30` with `depth-traditional` plus five `corrected` segmentation identities.
- `atomic-original=none`, `atomic-split-assisted=conservative`, and `atomic-split-no-assisted=normal_only`.
- Registration confidence keep ratio is `0.5` for all requested methods.
- Delete a method artifact only after its compact metric JSON validates; delete scene prediction cache only after every requested method validates.
- Never delete raw datasets, weights, retained metrics, logs, or summaries.
- `KEEP_INTERMEDIATES=1` disables all cleanup.
- Final summaries must reject incomplete coverage and non-finite metrics.

---

### Task 1: Experiment identity, data preparation, and cleanup utility

**Files:**
- Create: `tools/disposable_experiments/preflight_experiments.py`
- Test: `tests/disposable_experiments/test_preflight_experiments.py`

**Interfaces:**
- Produces CLI subcommands `preflight`, `plan`, `prepare-pointcloud`, `compact-pointcloud`, `compact-ate`, `result-ok`, `cleanup-artifact`, and `cleanup-scene-cache`.
- Produces importable constants `POINTCLOUD_METHODS`, `ATE_METHODS`, `KITTI_SEQUENCES`, and functions `load_valid_result(path, expected_identity)`, `guarded_remove(path, allowed_root)`, and `build_method_overrides(method, evaluation)`.
- Consumes existing kf10 JSON maps, reconstruction manifests, evaluator JSON output, raw dataset directories, and prediction-cache manifests.

- [ ] **Step 1: Write tests for exact matrices and overrides**

```python
def test_exact_requested_method_matrices():
    assert list(module.POINTCLOUD_METHODS) == [
        "depth", "geometry", "atomic-original",
        "atomic-split-assisted", "atomic-split-no-assisted",
    ]
    assert list(module.ATE_METHODS) == [
        "depth-traditional", "depth-corrected", "geometry-corrected",
        "atomic-original-corrected", "atomic-split-assisted-corrected",
        "atomic-split-no-assisted-corrected",
    ]
    assert module.KITTI_SEQUENCES == tuple(f"{index:02d}" for index in range(11))


def test_atomic_method_overrides_are_explicit():
    assert "segmentation.atomic.split_mode=none" in module.build_method_overrides(
        "atomic-original", "pointcloud"
    )
    assert "segmentation.atomic.split_mode=conservative" in module.build_method_overrides(
        "atomic-split-assisted", "pointcloud"
    )
    assert "segmentation.atomic.split_mode=normal_only" in module.build_method_overrides(
        "atomic-split-no-assisted", "pointcloud"
    )
```

- [ ] **Step 2: Write tests for strict compact-result validation**

```python
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_result_validation_rejects_non_finite_metrics(tmp_path, bad):
    path = write_pointcloud_result(tmp_path, accuracy_mean_m=bad)
    with pytest.raises(ValueError, match="finite"):
        module.load_valid_result(path, expected_pointcloud_identity())


def test_result_identity_must_match_split_mode(tmp_path):
    path = write_pointcloud_result(tmp_path, split_mode="conservative")
    with pytest.raises(ValueError, match="identity"):
        module.load_valid_result(
            path,
            {**expected_pointcloud_identity(), "split_mode": "none"},
        )
```

- [ ] **Step 3: Write cleanup guard and cache-ownership tests**

```python
def test_guarded_remove_rejects_raw_data(tmp_path):
    raw = tmp_path / "data" / "KITTI" / "00"
    raw.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside allowed root"):
        module.guarded_remove(raw, tmp_path / "outputs" / "work")


def test_scene_cache_cleanup_uses_prediction_key_only(tmp_path):
    cache = make_cache_entries(tmp_path, keys=("wanted", "other"))
    module.remove_prediction_keys(cache, ("wanted",))
    assert not (cache / "v2" / "wanted").exists()
    assert (cache / "v2" / "other").is_dir()
```

- [ ] **Step 4: Run the focused tests and verify failure**

Run: `pytest -q tests/disposable_experiments/test_preflight_experiments.py`

Expected: FAIL because the utility does not exist.

- [ ] **Step 5: Implement the utility and CLI**

Implementation requirements:

```python
POINTCLOUD_METHODS = {
    "depth": ("depth", "none", "no_loop"),
    "geometry": ("geometry", "none", "no_loop"),
    "atomic-original": ("atomic", "none", "no_loop"),
    "atomic-split-assisted": ("atomic", "conservative", "no_loop"),
    "atomic-split-no-assisted": ("atomic", "normal_only", "no_loop"),
}

ATE_METHODS = {
    "depth-traditional": ("depth", "none", "traditional"),
    "depth-corrected": ("depth", "none", "corrected"),
    "geometry-corrected": ("geometry", "none", "corrected"),
    "atomic-original-corrected": ("atomic", "none", "corrected"),
    "atomic-split-assisted-corrected": ("atomic", "conservative", "corrected"),
    "atomic-split-no-assisted-corrected": ("atomic", "normal_only", "corrected"),
}
```

`prepare-pointcloud` must use the existing calibration in
`datasets/preprocess/prepare_7scenes.py`, create benchmark images as symlinks,
and write `ground_truth.npz` plus `frame_ids.npy` for exactly the mapped frames.
It must not scan or materialize unselected frames.

`compact-pointcloud` must load the evaluator's primary JSON and the artifact
diagnostics/manifest, recompute or ingest the full point-cloud diagnostics,
and atomically write one self-contained metric JSON containing identity,
primary metrics, Chamfer, and 1/2/5 cm threshold metrics.

`compact-ate` must atomically combine `trajectory_metrics.json` with artifact
identity and prediction key.

`cleanup-scene-cache` must accept explicit prediction keys captured from
validated compact results; it must never glob by scene name.

- [ ] **Step 6: Run focused tests**

Run: `pytest -q tests/disposable_experiments/test_preflight_experiments.py`

Expected: PASS.

- [ ] **Step 7: Commit the utility**

```bash
git add tools/disposable_experiments/preflight_experiments.py \
  tests/disposable_experiments/test_preflight_experiments.py
git commit -m "feat: add disposable experiment utility"
```

### Task 2: Point-cloud runner

**Files:**
- Create: `tools/disposable_experiments/run_pointcloud_all.sh`
- Test: `tests/disposable_experiments/test_shell_runners.py`

**Interfaces:**
- Consumes `preflight_experiments.py`, `run_reconstruction.py`, and existing point-cloud evaluation modules.
- Produces compact JSON at `metrics/pointcloud/<dataset>/<safe-scene>/<method>.json` and logs under `logs/pointcloud/`.

- [ ] **Step 1: Write shell contract tests**

```python
def test_pointcloud_runner_has_exact_schedule_and_cleanup_gate():
    text = POINTCLOUD_RUNNER.read_text()
    assert "window.size=20" in text
    assert "window.overlap=5" in text
    assert "configs/reconstruction/pi3_laser_no_loop.yaml" in text
    assert "compact-pointcloud" in text
    assert text.index("compact-pointcloud") < text.index("cleanup-artifact")
```

- [ ] **Step 2: Run the contract test and verify failure**

Run: `pytest -q tests/disposable_experiments/test_shell_runners.py -k pointcloud`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the point-cloud runner**

The shell script must use `set -Eeuo pipefail`, quote all paths, execute data
sets and scenes serially, and for each method:

```bash
conda run --no-capture-output -n "${LASER_ENV}" \
  python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser_no_loop.yaml \
  --set "input.image_dir=${image_dir}" \
  --set "model.checkpoint=${PI3_CHECKPOINT}" \
  --set window.size=20 --set window.overlap=5 \
  --set "prediction_cache.root=${scene_cache}" \
  --set "output.result_dir=${artifact_root}" \
  "${method_overrides[@]}"
```

It then invokes artifact-only point-cloud evaluation, compacts/validates the
metric, deletes only that method artifact, and after all five methods deletes
the prepared benchmark directory and explicit prediction keys unless
`KEEP_INTERMEDIATES=1`.

- [ ] **Step 4: Run shell and syntax tests**

Run: `bash -n tools/disposable_experiments/run_pointcloud_all.sh && pytest -q tests/disposable_experiments/test_shell_runners.py -k pointcloud`

Expected: PASS.

- [ ] **Step 5: Commit the runner**

```bash
git add tools/disposable_experiments/run_pointcloud_all.sh \
  tests/disposable_experiments/test_shell_runners.py
git commit -m "feat: add disposable pointcloud runner"
```

### Task 3: KITTI ATE runner

**Files:**
- Create: `tools/disposable_experiments/run_kitti_ate_all.sh`
- Modify: `tests/disposable_experiments/test_shell_runners.py`

**Interfaces:**
- Consumes `preflight_experiments.py`, `run_reconstruction.py`, and `evaluate_ate.py`.
- Produces compact JSON at `metrics/ate/kitti/<sequence>/<method>.json` and logs under `logs/ate/kitti/`.

- [ ] **Step 1: Add exact KITTI matrix tests**

```python
def test_ate_runner_has_exact_schedule_matrix_and_cleanup_gate():
    text = ATE_RUNNER.read_text()
    assert "{00..10}" in text or "seq -w 0 10" in text
    assert "window.size=75" in text
    assert "window.overlap=30" in text
    assert "registration.confidence_keep_ratio=0.5" in text
    assert text.index("compact-ate") < text.index("cleanup-artifact")
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `pytest -q tests/disposable_experiments/test_shell_runners.py -k ate`

Expected: FAIL because the ATE runner does not exist.

- [ ] **Step 3: Implement the ATE runner**

Use `configs/reconstruction/pi3_laser.yaml`, `data/KITTI/<seq>/image_2`, and
`data/KITTI/<seq>/poses.txt` with `ground-truth-format=replica`. Apply explicit
method overrides from the utility, `window.size=75`, `window.overlap=30`, and
`registration.confidence_keep_ratio=0.5` for every run.

After `evaluate_ate.py`, compact and validate the metric before deleting its
artifact. After all six methods validate, delete explicit prediction keys for
the sequence unless `KEEP_INTERMEDIATES=1`.

- [ ] **Step 4: Run syntax and contract tests**

Run: `bash -n tools/disposable_experiments/run_kitti_ate_all.sh && pytest -q tests/disposable_experiments/test_shell_runners.py`

Expected: PASS.

- [ ] **Step 5: Commit the runner**

```bash
git add tools/disposable_experiments/run_kitti_ate_all.sh \
  tests/disposable_experiments/test_shell_runners.py
git commit -m "feat: add disposable KITTI ATE runner"
```

### Task 4: Strict result summarizer and workbooks

**Files:**
- Create: `tools/disposable_experiments/summarize_results.py`
- Test: `tests/disposable_experiments/test_summarize_results.py`

**Interfaces:**
- Consumes validated compact metric JSON files and imports method/sequence definitions from `preflight_experiments.py`.
- Produces `pointcloud_long.csv`, `kitti_ate_long.csv`, `pointcloud_summary.xlsx`, and `kitti_ate_summary.xlsx`.

- [ ] **Step 1: Write coverage and table-content tests**

```python
def test_summary_rejects_missing_method(tmp_path):
    write_complete_fixture(tmp_path)
    missing = next(tmp_path.rglob("atomic-split-no-assisted.json"))
    missing.unlink()
    with pytest.raises(ValueError, match="missing"):
        module.collect_results(tmp_path)


def test_long_rows_keep_numeric_metrics(tmp_path):
    rows = module.collect_pointcloud_rows(write_complete_pointcloud_fixture(tmp_path))
    assert len(rows) == (18 + 9) * 5
    assert isinstance(rows[0]["accuracy_mean_m"], float)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/disposable_experiments/test_summarize_results.py`

Expected: FAIL because the summarizer does not exist.

- [ ] **Step 3: Implement strict CSV collection**

CSV rows must include dataset, scene/sequence, method, segmentation method,
split mode, reconstruction mode, schedule fields, all numeric metrics, and
prediction key. Write to a temporary path and replace the final CSV atomically.

- [ ] **Step 4: Implement XLSX authoring with `@oai/artifact-tool`**

The summary script invokes the bundled disposable workbook runtime and writes:

- point-cloud presentation sheets `7-Scenes` and `NeuralRGBD`, plus `Raw Data`;
- ATE presentation sheet `KITTI 00-10`, plus `Raw Data`.

Presentation cells use wrapped metric text matching the supplied screenshots;
raw-data sheets contain typed numeric columns. Headers use light blue, method
labels use a pale blue fill, and comparison cells use restrained green/peach
fills. Freeze the method columns and hide gridlines.

- [ ] **Step 5: Run summarizer tests**

Run: `pytest -q tests/disposable_experiments/test_summarize_results.py`

Expected: PASS.

- [ ] **Step 6: Generate fixture workbooks and verify them**

Run the summarizer on complete synthetic fixtures, then use artifact-tool to:

1. inspect the key table ranges including values;
2. scan for `#REF!|#DIV/0!|#VALUE!|#NAME?|#N/A`;
3. render every sheet at least once;
4. visually inspect the rendered PNGs.

Expected: all sheets legible, no missing metrics, no formula errors, and no
clipped headers or summary cells.

- [ ] **Step 7: Commit the summarizer**

```bash
git add tools/disposable_experiments/summarize_results.py \
  tests/disposable_experiments/test_summarize_results.py
git commit -m "feat: add disposable experiment summaries"
```

### Task 5: Unified entry point, cloud dry-run deployment, and handoff

**Files:**
- Create: `tools/disposable_experiments/run_all.sh`
- Modify: `tests/disposable_experiments/test_shell_runners.py`
- Modify: `docs/superpowers/specs/2026-08-13-disposable-cloud-experiment-scripts-design.md`

**Interfaces:**
- Produces the user command `screen -dmS laser-experiments bash -lc '<absolute run_all.sh> >> <campaign.log> 2>&1'`.
- Consumes all four prior entry points.

- [ ] **Step 1: Test unified order and foreground semantics**

```python
def test_run_all_orders_preflight_experiments_and_summary():
    text = RUN_ALL.read_text()
    positions = [
        text.index("preflight_experiments.py preflight"),
        text.index("run_pointcloud_all.sh"),
        text.index("run_kitti_ate_all.sh"),
        text.index("summarize_results.py"),
    ]
    assert positions == sorted(positions)
    assert "screen" not in text
```

- [ ] **Step 2: Implement `run_all.sh`**

Use `set -Eeuo pipefail`, resolve the repository root from the script path,
export only task-specific defaults, and execute preflight, both runners, then
summary. Print final output paths only after the summarizer succeeds.

- [ ] **Step 3: Run the full local verification suite**

Run:

```bash
bash -n tools/disposable_experiments/*.sh
pytest -q tests/disposable_experiments
pytest -q
git diff --check
```

Expected: all commands pass.

- [ ] **Step 4: Commit the unified entry point**

```bash
git add tools/disposable_experiments/run_all.sh \
  tests/disposable_experiments/test_shell_runners.py \
  docs/superpowers/specs/2026-08-13-disposable-cloud-experiment-scripts-design.md
git commit -m "feat: add unified disposable experiment command"
```

- [ ] **Step 5: Deploy only the disposable directory to the cloud checkout**

Copy the verified `tools/disposable_experiments/` directory to:

```text
/root/autodl-tmp/LASER-Paper-Pointmap-Eval/tools/disposable_experiments/
```

Do not overwrite raw datasets, weights, existing outputs, or core LASER source.

- [ ] **Step 6: Run cloud preflight and dry-run planning**

Run:

```bash
conda run --no-capture-output -n vggt \
  python tools/disposable_experiments/preflight_experiments.py preflight
conda run --no-capture-output -n vggt \
  python tools/disposable_experiments/preflight_experiments.py plan
bash -n tools/disposable_experiments/*.sh
```

Expected: exact 27 point-cloud scenes, 11 KITTI sequences, 201 total method
runs, complete data/weights/runtime checks, and no experiment execution.

- [ ] **Step 7: Deliver the single background command**

Provide:

```bash
cd /root/autodl-tmp/LASER-Paper-Pointmap-Eval && \
screen -dmS laser-experiments bash -lc \
'tools/disposable_experiments/run_all.sh >> outputs/disposable_experiments/campaign.log 2>&1'
```

Also provide non-mutating status commands:

```bash
screen -ls
tail -n 100 outputs/disposable_experiments/campaign.log
```

Do not start the long campaign on the user's behalf.
