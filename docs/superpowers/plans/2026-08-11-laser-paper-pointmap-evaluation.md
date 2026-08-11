# LASER Paper-Compatible Point-Map Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation-only tool that reproduces the LASER CVPR 2026 Pi3 point-map protocol on 7-Scenes and NeuralRGBD and emits auditable Table 4 metrics for cloud execution.

**Architecture:** Keep `StreamingPipelineModel` and every reconstruction-core module unchanged. A strict evaluation profile is resolved into the existing typed `PipelineConfig`; focused protocol, geometry, and result modules validate the run, compute the official metrics, and persist resumable artifacts, while `mv_recon/eval.py` only orchestrates those boundaries.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, frozen dataclasses, PyTorch, NumPy, SciPy `cKDTree`, Open3D point-to-point ICP/default normals, pytest, JSON/CSV/YAML.

## Global Constraints

- Base implementation commit is `fcbe67981e905b13c4c1620b04a372760ecc87db` from `codex/pi3-ordinary-prediction-cache`.
- Modify only `configs/evaluation/`, `mv_recon/`, `tests/`, and `docs/`; do not modify `pipeline/`, `inference_engine/`, `pi3/`, or `loop_closure/`.
- The locked reconstruction values are window `20/5`, Depth segmentation, both confidence keep ratios `0.5`, depth merge `0.1`, temporal IoU `0.3`, Felzenszwalb `300/1.1/500`, anchor enabled at IoU `0.4`, loop disabled, and traditional aggregation.
- The locked geometry values are a `224 x 224` center crop, same-pixel Umeyama Sim(3), Open3D point-to-point ICP with identity initialization and threshold `0.1 m`, and Open3D default normal estimation.
- The fixed sequence maps are `datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json` (SHA256 `e9954bfcf4b4a3273224e8375d468638e1fe4d7b6d926ff32147367bb4574008`) and `datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json` (SHA256 `f18f2143f8a373727aa4d7043b779b77639354fda80523cdc2a139164ddc33ba`); expected sequence counts are 18 and 9 respectively, and every consecutive frame difference is 10.
- Primary metrics are Acc mean/median, Comp mean/median, and NC mean/median; NC mean and NC median each average their two directional statistics separately.
- The final point metrics use only the GT validity mask and never filter by predicted confidence.
- Dataset results are macro averages over exact sequence-map entries; missing sequences make the run non-complete and never reduce the paper denominator silently.
- `protocol.max_sequences=1` means one sequence from each selected dataset and always produces `subset` status.
- Strict paper mode has no alternate ICP, normal-estimation, or numerical fallback.
- `preflight_only=true` must validate and write protocol artifacts without constructing Pi3.
- Local Python 3.13 cannot install the pinned Open3D wheel; production Open3D imports must be lazy, and unit tests must inject a deterministic backend.
- Every JSON writer must set `allow_nan=False` and use temporary sibling files followed by atomic replacement.
- Preserve the existing ordinary prediction cache implementation and fingerprint semantics unchanged.

## File Map

- Create `configs/evaluation/mv_recon_laser_paper.yaml`: immutable Hydra paper profile and reference numbers.
- Create `mv_recon/protocol.py`: typed protocol schema, pipeline resolution, strict locks, sequence-map parsing, and preflight plans.
- Create `mv_recon/geometry_metrics.py`: crop, Sim(3), Open3D ICP/default normals, directional metrics, and extended diagnostics.
- Create `mv_recon/results.py`: typed sequence/run results, macro aggregation, identity-safe resume, and atomic JSON/CSV/YAML writers.
- Rewrite `mv_recon/eval.py`: dependency-injectable orchestration over the existing dataset and streaming interfaces.
- Modify `mv_recon/README.md`: paper-profile cloud workflow and artifact interpretation.
- Create `tests/mv_recon/test_protocol.py`: profile, lock, map, operational override, and preflight-only tests.
- Create `tests/mv_recon/test_geometry_metrics.py`: metric direction, NC semantics, crop, Sim(3), ICP, and failure tests.
- Create `tests/mv_recon/test_results.py`: aggregation, states, atomic output, resume, and non-finite serialization tests.
- Create `tests/mv_recon/test_eval_orchestration.py`: fake dataset/model end-to-end orchestration and model-construction gates.

---

### Task 1: Immutable Paper Profile and Typed Protocol Resolution

**Files:**
- Create: `configs/evaluation/mv_recon_laser_paper.yaml`
- Create: `mv_recon/protocol.py`
- Create: `tests/mv_recon/test_protocol.py`

**Interfaces:**
- Consumes: `pipeline.config.load_pipeline_config(path, overrides)` and the composed Hydra root `DictConfig`.
- Produces: `GeometryProtocol`, `DatasetReference`, `LaserPaperProtocol`, `SequenceSpec`, `DatasetPlan`, `ResolvedEvaluationProtocol`, `resolve_evaluation_protocol()`, `load_sequence_map()`, and `validate_dataset_plan()`.

- [ ] **Step 1: Write failing tests for the shipped profile and exact locks**

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

from mv_recon.protocol import (
    EXPECTED_DATASET_SEQUENCE_COUNTS,
    resolve_evaluation_protocol,
)


ROOT = Path(__file__).resolve().parents[2]


def _root_config(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"checkpoint")
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"), version_base="1.2"
    ):
        return compose(
            config_name="eval_mv_recon_dense",
            overrides=[
                "evaluation=mv_recon_laser_paper",
                "device=cpu",
                f"output_dir={tmp_path / 'results'}",
                f"pi3.checkpoint={tmp_path / 'model.safetensors'}",
            ],
        )


def test_paper_profile_resolves_all_locked_pipeline_values(tmp_path):
    resolved = resolve_evaluation_protocol(_root_config(tmp_path), ROOT)
    config = resolved.pipeline.config
    assert (config.window.size, config.window.overlap) == (20, 5)
    assert config.segmentation.method.value == "depth"
    assert config.segmentation.confidence_keep_ratio == pytest.approx(0.5)
    assert config.loop.registration.confidence_keep_ratio == pytest.approx(0.5)
    assert config.loop.enabled is False
    assert config.loop.method.value == "traditional"
    assert resolved.protocol.geometry.center_crop_size == 224
    assert EXPECTED_DATASET_SEQUENCE_COUNTS == {
        "7scenes-dense": 18,
        "NRGBD-dense": 9,
    }


@pytest.mark.parametrize(
    "override",
    (
        "window.size=19",
        "segmentation.method=atomic",
        "loop.registration.confidence_keep_ratio=0.3",
        "anchor_propagation.enabled=false",
    ),
)
def test_strict_profile_rejects_pipeline_drift_before_model_load(
    tmp_path, override
):
    config = _root_config(tmp_path)
    config.protocol.pipeline_overrides.append(override)
    with pytest.raises(ValueError, match="laser.*protocol.*drift"):
        resolve_evaluation_protocol(config, ROOT)


def test_resume_switch_changes_full_hash_but_not_compatibility_identity(tmp_path):
    initial = _root_config(tmp_path)
    resumed = _root_config(tmp_path)
    resumed.protocol.resume = True
    first = resolve_evaluation_protocol(initial, ROOT)
    second = resolve_evaluation_protocol(resumed, ROOT)
    assert first.sha256 != second.sha256
    assert first.identity_sha256 == second.identity_sha256
```

- [ ] **Step 2: Run protocol tests and verify the module/profile are missing**

Run: `python -m pytest tests/mv_recon/test_protocol.py -v`

Expected: collection fails because `mv_recon.protocol` does not exist.

- [ ] **Step 3: Add the exact Hydra paper profile**

```yaml
# @package _global_

defaults:
  - override /data: mv_recon_dense

name: mv_recon_laser_paper

pi3:
  pretrained_model_name_or_path: yyfz233/Pi3
  checkpoint: weights/model.safetensors

device: cuda
load_img_size: 518
verbose: false
dir_suffix: null
save_suffix: null

hydra:
  run:
    dir: ${work_dir}/outputs/.hydra/${name}/${now:%Y-%m-%d_%H-%M-%S}

eval_datasets: [7scenes-dense, NRGBD-dense]

protocol:
  name: laser_cvpr2026_table4_pi3
  version: 1
  strict: true
  preflight_only: false
  resume: false
  max_sequences: null
  pipeline_config: configs/pipeline/default.yaml
  pipeline_cache_dir: ${output_dir}/pipeline_cache
  prediction_cache_root: inference_cache/predictions
  prediction_cache_mode: auto
  process_device: cpu
  dtype: auto
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
    - anchor_propagation.enabled=true
    - anchor_propagation.correspondence_iou_threshold=0.4
    - loop.enabled=false
    - loop.method=traditional
    - loop.registration.confidence_keep_ratio=0.5
  geometry:
    center_crop_size: 224
    alignment: umeyama_sim3_then_icp
    icp_type: point_to_point
    icp_threshold_m: 0.1
    normal_estimation: open3d_default
    fscore_thresholds_m: [0.01, 0.02, 0.05]
  paper_reference:
    7scenes-dense:
      accuracy_mean_m: 0.013
      accuracy_median_m: 0.005
      completion_mean_m: 0.017
      completion_median_m: 0.006
      normal_consistency_mean: 0.607
      normal_consistency_median: 0.665
    NRGBD-dense:
      accuracy_mean_m: 0.020
      accuracy_median_m: 0.010
      completion_mean_m: 0.012
      completion_median_m: 0.004
      normal_consistency_mean: 0.713
      normal_consistency_median: 0.856
```

- [ ] **Step 4: Implement frozen schemas and exact-key parsing**

```python
@dataclass(frozen=True)
class GeometryProtocol:
    center_crop_size: int
    alignment: str
    icp_type: str
    icp_threshold_m: float
    normal_estimation: str
    fscore_thresholds_m: tuple[float, ...]


@dataclass(frozen=True)
class DatasetReference:
    accuracy_mean_m: float
    accuracy_median_m: float
    completion_mean_m: float
    completion_median_m: float
    normal_consistency_mean: float
    normal_consistency_median: float


@dataclass(frozen=True)
class LaserPaperProtocol:
    name: str
    version: int
    strict: bool
    preflight_only: bool
    resume: bool
    max_sequences: int | None
    pipeline_config: Path
    pipeline_cache_dir: Path
    prediction_cache_root: Path
    prediction_cache_mode: str
    process_device: str
    dtype: str
    pipeline_overrides: tuple[str, ...]
    geometry: GeometryProtocol
    paper_reference: Mapping[str, DatasetReference]


@dataclass(frozen=True)
class ResolvedEvaluationProtocol:
    protocol: LaserPaperProtocol
    pipeline: LoadedPipelineConfig
    datasets: tuple[str, ...]
    output_dir: Path
    resolved_yaml: str
    sha256: str
    identity_sha256: str
```

Convert only the `protocol` node with `OmegaConf.to_container(resolve=True)`, compare its key sets against explicit allowed/required sets, reject booleans where an integer is required, validate positive finite thresholds, require the two exact dataset-reference keys, and require every `DatasetReference` field by its explicit metric name.

- [ ] **Step 5: Resolve operational values through the canonical pipeline loader**

```python
def resolve_evaluation_protocol(
    hydra_cfg: DictConfig,
    repository_root: str | Path,
    *,
    cuda_capability: tuple[int, int] | None = None,
) -> ResolvedEvaluationProtocol:
    protocol = _parse_protocol(hydra_cfg.protocol, repository_root)
    dtype = _resolve_dtype(protocol.dtype, hydra_cfg.device, cuda_capability)
    operational = (
        f"model.checkpoint={Path(hydra_cfg.pi3.checkpoint)}",
        f"model.inference_device={hydra_cfg.device}",
        f"model.process_device={protocol.process_device}",
        f"model.dtype={dtype}",
        f"output.cache_dir={protocol.pipeline_cache_dir}",
        f"prediction_cache.root={protocol.prediction_cache_root}",
        f"prediction_cache.mode={protocol.prediction_cache_mode}",
    )
    loaded = load_pipeline_config(
        protocol.pipeline_config,
        (*protocol.pipeline_overrides, *operational),
    )
    _validate_paper_locks(protocol, loaded.config)
    datasets = tuple(str(name) for name in hydra_cfg.eval_datasets)
    resolved_payload = {
        **_protocol_payload(protocol),
        "eval_datasets": list(datasets),
    }
    resolved_yaml = OmegaConf.to_yaml(
        OmegaConf.create(resolved_payload), sort_keys=True
    )
    identity_payload = dict(resolved_payload)
    identity_payload.pop("resume")
    identity_payload.pop("preflight_only")
    identity_yaml = OmegaConf.to_yaml(
        OmegaConf.create(identity_payload), sort_keys=True
    )
    return ResolvedEvaluationProtocol(
        protocol=protocol,
        pipeline=loaded,
        datasets=datasets,
        output_dir=Path(hydra_cfg.output_dir),
        resolved_yaml=resolved_yaml,
        sha256=hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest(),
        identity_sha256=hashlib.sha256(
            identity_yaml.encode("utf-8")
        ).hexdigest(),
    )
```

For `dtype=auto`, select `bfloat16` only for CUDA capability major >= 8; otherwise select `float16`. CPU tests pass an explicit capability and use `model.inference_device=cpu` without touching CUDA.

- [ ] **Step 6: Run focused tests, then the pipeline-config regression tests**

Add a source-profile assertion that `hydra.run.dir` resolves outside
`output_dir`, then run:

Run: `python -m pytest tests/mv_recon/test_protocol.py tests/test_pipeline_config.py -v`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the profile and protocol resolver**

```bash
git add configs/evaluation/mv_recon_laser_paper.yaml mv_recon/protocol.py tests/mv_recon/test_protocol.py
git commit -m "feat: lock LASER paper point-map protocol"
```

---

### Task 2: Sequence-Map and Dataset Preflight Contract

**Files:**
- Modify: `mv_recon/protocol.py`
- Modify: `tests/mv_recon/test_protocol.py`

**Interfaces:**
- Consumes: `ResolvedEvaluationProtocol`, composed `hydra_cfg.data`, and dataset objects exposing `sequence_list` and `get_seq_framenum()`.
- Produces: `SequenceSpec`, `DatasetPlan`, `load_sequence_map()`, `build_dataset_plans()`, `validate_dataset_plan()`, and `manifest_digest_for_paths()`.

- [ ] **Step 1: Add failing tests for map structure, kf10, counts, and per-dataset limiting**

```python
def test_shipped_maps_have_exact_counts_and_kf10_order(tmp_path):
    resolved = resolve_evaluation_protocol(_root_config(tmp_path), ROOT)
    plans = build_dataset_plans(resolved, _root_config(tmp_path).data, ROOT)
    assert [len(plan.sequences) for plan in plans] == [18, 9]
    assert plans[0].sequences[0].name == "chess/seq-03"
    assert all(
        right - left == 10
        for plan in plans
        for sequence in plan.sequences
        for left, right in zip(sequence.frame_ids, sequence.frame_ids[1:])
    )


def test_max_sequences_limits_each_dataset_and_forces_subset(tmp_path):
    config = _root_config(tmp_path)
    config.protocol.max_sequences = 1
    resolved = resolve_evaluation_protocol(config, ROOT)
    plans = build_dataset_plans(resolved, config.data, ROOT)
    assert [len(plan.sequences) for plan in plans] == [1, 1]
    assert resolved.protocol.max_sequences == 1


def test_sequence_map_rejects_non_kf10_interval(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"scene": [0, 10, 21]}', encoding="utf-8")
    with pytest.raises(ValueError, match="interval 10"):
        load_sequence_map(path, expected_count=1)
```

- [ ] **Step 2: Run the new tests and verify the APIs are absent**

Run: `python -m pytest tests/mv_recon/test_protocol.py -v`

Expected: failures name `build_dataset_plans` and `load_sequence_map`.

- [ ] **Step 3: Implement immutable sequence and dataset plans**

```python
@dataclass(frozen=True)
class SequenceSpec:
    name: str
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class DatasetPlan:
    name: str
    sequence_map_path: Path
    sequence_map_sha256: str
    expected_sequence_count: int
    sequences: tuple[SequenceSpec, ...]


def load_sequence_map(path: Path, expected_count: int) -> tuple[SequenceSpec, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) != expected_count:
        raise ValueError(f"sequence map must contain exactly {expected_count} entries")
    sequences = []
    for name, ids in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("sequence names must be non-empty strings")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"sequence {name} must contain frame IDs")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
            raise ValueError(f"sequence {name} frame IDs must be integers")
        if any(right - left != 10 for left, right in zip(ids, ids[1:])):
            raise ValueError(f"sequence {name} must use interval 10")
        sequences.append(SequenceSpec(name, tuple(ids)))
    return tuple(sequences)
```

- [ ] **Step 4: Implement dataset-plan construction and dataset metadata validation**

`build_dataset_plans()` must accept only `7scenes-dense` and `NRGBD-dense`, require the fixed repository-relative map path and checked-in SHA256, retain JSON insertion order, then slice each plan with `max_sequences` only after full-map validation.

```python
def validate_dataset_plan(dataset: object, plan: DatasetPlan) -> None:
    available = set(dataset.sequence_list)
    for sequence in plan.sequences:
        if sequence.name not in available:
            raise ValueError(f"dataset is missing sequence {sequence.name}")
        frame_count = int(dataset.get_seq_framenum(sequence_name=sequence.name))
        if sequence.frame_ids[-1] >= frame_count:
            raise ValueError(
                f"sequence {sequence.name} requests frame "
                f"{sequence.frame_ids[-1]} but contains {frame_count} frames"
            )
```

Add `manifest_digest_for_paths(paths)` using `ImageManifest` plus the existing `digest_image_manifest()` so result resume and ordinary prediction cache identify the same image contents. Add `digest_ground_truth(point_maps, valid_mask)` so changed GT geometry or validity cannot reuse stale metric results.

- [ ] **Step 5: Run map/preflight tests and verify both shipped maps directly**

Run: `python -m pytest tests/mv_recon/test_protocol.py -v`

Run: `jq -e 'length == 18' datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json`

Run: `jq -e 'length == 9' datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json`

Expected: pytest and both `jq` commands exit zero.

- [ ] **Step 6: Commit dataset preflight support**

```bash
git add mv_recon/protocol.py tests/mv_recon/test_protocol.py
git commit -m "feat: validate paper sequence maps"
```

---

### Task 3: Official Geometry Alignment and Point-Map Metrics

**Files:**
- Create: `mv_recon/geometry_metrics.py`
- Create: `tests/mv_recon/test_geometry_metrics.py`
- Preserve unchanged: `mv_recon/eval_utils.py`

**Interfaces:**
- Consumes: cropped prediction/GT arrays, GT mask, `GeometryProtocol`, and optional `GeometryBackend`.
- Produces: `PrimaryMetrics`, `DirectionalNormalMetrics`, `ThresholdMetrics`, `GeometryDiagnostics`, `GeometryEvaluation`, `Open3DGeometryBackend`, `center_crop()`, `combine_normal_consistency()`, `compute_directional_metrics()`, and `evaluate_point_maps()`.

- [ ] **Step 1: Write failing direction and NC aggregation tests**

```python
def test_directional_metric_names_match_official_accuracy_and_completion():
    pred = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    gt = np.array([[0.0, 0.0, 0.0]])
    pred_normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gt_normals = np.array([[1.0, 0.0, 0.0]])
    result = compute_directional_metrics(
        pred, gt, pred_normals, gt_normals, (0.01, 0.02, 0.05)
    )
    assert result.primary.accuracy_mean_m == pytest.approx(1.0)
    assert result.primary.completion_mean_m == pytest.approx(0.0)
    assert result.primary.normal_consistency_mean == pytest.approx(0.75)
    assert result.primary.normal_consistency_median == pytest.approx(0.75)


def test_nc_median_averages_directional_medians_not_concatenated_samples():
    mean, median = combine_normal_consistency(
        nc1=np.array([0.0, 0.0, 1.0]),
        nc2=np.array([0.5]),
    )
    assert mean == pytest.approx(0.4166666667)
    assert median == pytest.approx(0.25)
```

- [ ] **Step 2: Write failing crop, mask, Sim(3), and backend invocation tests**

```python
class RecordingBackend:
    def __init__(self):
        self.thresholds = []

    def refine_and_estimate_normals(self, predicted, ground_truth, threshold_m):
        self.thresholds.append(threshold_m)
        return BackendResult(
            predicted_points=predicted,
            ground_truth_points=ground_truth,
            predicted_normals=np.tile([1.0, 0.0, 0.0], (len(predicted), 1)),
            ground_truth_normals=np.tile([1.0, 0.0, 0.0], (len(ground_truth), 1)),
            transformation=np.eye(4),
            fitness=1.0,
            inlier_rmse=0.0,
        )


def test_evaluate_point_maps_uses_224_crop_gt_mask_and_point_to_point_backend():
    gt = _grid_points(frames=1, height=226, width=228)
    pred = gt * 2.0 + np.array([3.0, -1.0, 0.5])
    mask = np.ones(gt.shape[:-1], dtype=bool)
    mask[:, 0, :] = False
    backend = RecordingBackend()
    result = evaluate_point_maps(pred, gt, mask, _geometry(), backend=backend)
    assert result.diagnostics.predicted_point_count == 224 * 224
    assert backend.thresholds == [0.1]
    assert result.primary.accuracy_mean_m == pytest.approx(0.0, abs=1e-8)
```

- [ ] **Step 3: Run geometry tests and verify the module is missing**

Run: `python -m pytest tests/mv_recon/test_geometry_metrics.py -v`

Expected: collection fails because `mv_recon.geometry_metrics` does not exist.

- [ ] **Step 4: Implement typed metric and backend result objects**

```python
@dataclass(frozen=True)
class PrimaryMetrics:
    accuracy_mean_m: float
    accuracy_median_m: float
    completion_mean_m: float
    completion_median_m: float
    normal_consistency_mean: float
    normal_consistency_median: float


@dataclass(frozen=True)
class BackendResult:
    predicted_points: np.ndarray
    ground_truth_points: np.ndarray
    predicted_normals: np.ndarray
    ground_truth_normals: np.ndarray
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float


class GeometryBackend(Protocol):
    def refine_and_estimate_normals(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        threshold_m: float,
    ) -> BackendResult: ...
```

Also define frozen directional/threshold/diagnostic/evaluation dataclasses. Validate every scalar and array is finite before returning.

- [ ] **Step 5: Implement center crop, validated Umeyama, and official directional math**

`center_crop()` must crop the last spatial dimensions for images and the two point-map spatial dimensions for NumPy point maps/masks, reject dimensions below 224, and use integer center bounds exactly as the legacy evaluator.

`combine_normal_consistency()` returns `((mean(nc1) + mean(nc2)) / 2, (median(nc1) + median(nc2)) / 2)`. `evaluate_point_maps()` must:

1. verify prediction shape equals GT shape and mask shape equals `GT.shape[:-1]`;
2. crop prediction, GT, and mask to `center_crop_size`;
3. require at least three finite masked correspondences;
4. call unchanged `mv_recon.eval_utils.umeyama(pred.T, gt.T)`;
5. reject zero/degenerate prediction variance and non-finite transform values;
6. transform the full cropped prediction with `scale * einsum(...) + translation`;
7. flatten both point maps with the same GT mask;
8. call the backend at exactly `icp_threshold_m`;
9. query `cKDTree(gt).query(pred)` for Accuracy/NC1 and `cKDTree(pred).query(gt)` for Completion/NC2;
10. compute precision from Accuracy distances, recall from Completion distances, and harmonic-mean F-score with zero-safe division.

- [ ] **Step 6: Implement the lazy official Open3D backend**

```python
class Open3DGeometryBackend:
    def refine_and_estimate_normals(self, predicted, ground_truth, threshold_m):
        import open3d as o3d

        predicted_cloud = o3d.geometry.PointCloud()
        predicted_cloud.points = o3d.utility.Vector3dVector(predicted)
        gt_cloud = o3d.geometry.PointCloud()
        gt_cloud.points = o3d.utility.Vector3dVector(ground_truth)
        registration = o3d.pipelines.registration.registration_icp(
            predicted_cloud,
            gt_cloud,
            threshold_m,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        predicted_cloud.transform(registration.transformation)
        predicted_cloud.estimate_normals()
        gt_cloud.estimate_normals()
        return BackendResult(
            predicted_points=np.asarray(predicted_cloud.points),
            ground_truth_points=np.asarray(gt_cloud.points),
            predicted_normals=np.asarray(predicted_cloud.normals),
            ground_truth_normals=np.asarray(gt_cloud.normals),
            transformation=np.asarray(registration.transformation),
            fitness=float(registration.fitness),
            inlier_rmse=float(registration.inlier_rmse),
        )
```

Do not catch `ImportError` to select another algorithm; surface an actionable message stating that strict paper evaluation requires Open3D.

- [ ] **Step 7: Verify parity with existing helpers and all geometry failures**

Add deterministic fixtures that compare Accuracy/Completion mean, median, NC1, and NC2 against `mv_recon.eval_utils.accuracy()` and `completion()`. Add rejection tests for non-finite masked points, fewer than three points, crop too large, degenerate variance, empty cloud, non-finite ICP transform, and invalid normal shapes.

Run: `python -m pytest tests/mv_recon/test_geometry_metrics.py -v`

Expected: all tests pass without importing Open3D unless the real backend test is explicitly selected in an environment where it is installed.

- [ ] **Step 8: Commit official geometry metrics**

```bash
git add mv_recon/geometry_metrics.py tests/mv_recon/test_geometry_metrics.py
git commit -m "feat: add LASER point-map geometry metrics"
```

---

### Task 4: Typed Results, Macro Aggregation, and Identity-Safe Resume

**Files:**
- Create: `mv_recon/results.py`
- Create: `tests/mv_recon/test_results.py`

**Interfaces:**
- Consumes: `PrimaryMetrics`, geometry diagnostics, `DatasetPlan`, resolved protocol/pipeline hashes, checkpoint hash, per-sequence input manifest hash, and per-sequence GT point-map/mask hash.
- Produces: `RunIdentity`, `SequenceResult`, `FailureRecord`, `DatasetSummary`, `RunResults`, `ResultStore`, `aggregate_dataset()`, and `delta_to_reference()`.

- [ ] **Step 1: Write failing tests for exact macro averages and run states**

```python
def test_dataset_summary_is_macro_average_over_sequence_results():
    first = _sequence("a", accuracy_mean_m=0.01, completion_mean_m=0.03)
    second = _sequence("b", accuracy_mean_m=0.03, completion_mean_m=0.01)
    summary = aggregate_dataset(
        "7scenes-dense", (first, second), expected_sequences=("a", "b")
    )
    assert summary.primary.accuracy_mean_m == pytest.approx(0.02)
    assert summary.primary.completion_mean_m == pytest.approx(0.02)
    assert summary.completed_sequences == 2


def test_missing_sequence_cannot_be_complete():
    with pytest.raises(ValueError, match="missing.*b"):
        aggregate_dataset(
            "7scenes-dense", (_sequence("a"),), expected_sequences=("a", "b")
        )


def test_max_sequence_run_serializes_subset_not_complete(tmp_path):
    store = ResultStore(tmp_path, _identity(), resume=False)
    store.initialize(
        expected_sequences={
            "7scenes-dense": ("chess/seq-03",),
            "NRGBD-dense": ("breakfast_room",),
        },
        full_sequence_counts={"7scenes-dense": 18, "NRGBD-dense": 9},
        subset=True,
    )
    store.record_sequence(_sequence("a"))
    result = store.finalize()
    assert result.state == "subset"
```

- [ ] **Step 2: Write failing atomic-output and resume-identity tests**

```python
def test_nonempty_output_requires_explicit_resume(tmp_path):
    (tmp_path / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="resume"):
        ResultStore(tmp_path, _identity(), resume=False)


@pytest.mark.parametrize(
    "field",
    (
        "protocol_identity_sha256",
        "pipeline_sha256",
        "checkpoint_sha256",
        "metric_version",
    ),
)
def test_resume_rejects_changed_run_identity(tmp_path, field):
    first = ResultStore(tmp_path, _identity(), resume=False)
    first.initialize(
        expected_sequences={"7scenes-dense": ("scene",)},
        full_sequence_counts={"7scenes-dense": 1},
        subset=False,
    )
    changed = replace(_identity(), **{field: "f" * 64})
    with pytest.raises(ValueError, match="resume identity mismatch"):
        ResultStore(tmp_path, changed, resume=True)


def test_resume_rejects_changed_sequence_map_identity(tmp_path):
    first = ResultStore(tmp_path, _identity(), resume=False)
    first.initialize(
        expected_sequences={"7scenes-dense": ("scene",)},
        full_sequence_counts={"7scenes-dense": 1},
        subset=False,
    )
    changed = replace(
        _identity(), sequence_map_sha256={"7scenes-dense": "f" * 64}
    )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        ResultStore(tmp_path, changed, resume=True)


def test_json_rejects_non_finite_metric(tmp_path):
    store = ResultStore(tmp_path, _identity(), resume=False)
    with pytest.raises(ValueError, match="finite"):
        store.record_sequence(_sequence("a", accuracy_mean_m=float("nan")))
```

- [ ] **Step 3: Run result tests and verify the module is missing**

Run: `python -m pytest tests/mv_recon/test_results.py -v`

Expected: collection fails because `mv_recon.results` does not exist.

- [ ] **Step 4: Implement result dataclasses and canonical schema**

```python
METRIC_SCHEMA_VERSION = "laser-pointmap-metrics-v2"


@dataclass(frozen=True)
class RunIdentity:
    protocol_identity_sha256: str
    pipeline_sha256: str
    checkpoint_sha256: str
    sequence_map_sha256: Mapping[str, str]
    metric_version: str = METRIC_SCHEMA_VERSION


@dataclass(frozen=True)
class SequenceResult:
    dataset: str
    sequence: str
    frame_count: int
    input_manifest_sha256: str
    ground_truth_sha256: str
    ordinary_prediction_key: str
    primary: PrimaryMetrics
    diagnostics: GeometryDiagnostics


@dataclass(frozen=True)
class FailureRecord:
    dataset: str
    sequence: str
    category: str
    message: str
```

Define dataset and run summary types with state restricted to `preflight`, `running`, `complete`, `subset`, `incomplete`, or `failed`. Convert dataclasses with a local `json_safe()` that validates all floats with `math.isfinite()`.

- [ ] **Step 5: Implement macro aggregation and paper deltas**

`aggregate_dataset(dataset, results, expected_sequences)` must reject duplicate sequence names, require exactly the expected sequence-name set for complete aggregation, and average each primary field across sequence objects. `delta_to_reference()` returns computed minus paper for all six fields and never applies a pass/fail tolerance.

- [ ] **Step 6: Implement the atomic ResultStore**

```python
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
```

`ResultStore` must write `results.json` after initialize, each successful sequence, each failure, and finalize. It derives `summary.csv` and `sequences.csv` only from in-memory canonical results; appends one JSON object per line to `failures.jsonl` using an atomic whole-file replacement; and never scans stale CSV files.

- [ ] **Step 7: Implement exact resume selection**

On `resume=True`, load `results.json`, reject non-finite stored values, compare every `RunIdentity` field, and build a lookup by `(dataset, sequence)`. Reuse a sequence only if its stored `input_manifest_sha256`, `ground_truth_sha256`, expected sequence-map identity, and metric version match exactly. Expose:

```python
def reusable_sequence(
    self,
    dataset: str,
    sequence: str,
    input_manifest_sha256: str,
    ground_truth_sha256: str,
) -> SequenceResult | None:
    ...
```

A corrupt or incomplete stored sequence returns no reusable result; an incompatible run identity raises before any new file is written.

- [ ] **Step 8: Verify all result tests**

Run: `python -m pytest tests/mv_recon/test_results.py -v`

Expected: all tests pass and temporary files are absent after success and injected write failure.

- [ ] **Step 9: Commit result persistence**

```bash
git add mv_recon/results.py tests/mv_recon/test_results.py
git commit -m "feat: add auditable point-map result store"
```

---

### Task 5: Evaluation Orchestration and Model-Construction Gate

**Files:**
- Rewrite: `mv_recon/eval.py`
- Create: `tests/mv_recon/test_eval_orchestration.py`
- Modify: `tests/test_eval_launch_lc.py`

**Interfaces:**
- Consumes: resolved protocol/plans, `StreamingPipelineModel`, evaluation-local `run_streaming_inference()`, `evaluate_point_maps`, and `ResultStore`.
- Produces: `EvaluationDependencies`, `run_evaluation()`, `run_sequence()`, `build_protocol_manifest()`, and the Hydra `main()` entry point.

- [ ] **Step 1: Write a failing preflight-only model gate test**

```python
def test_preflight_only_never_constructs_streaming_model(tmp_path):
    state = {"models": 0}
    dependencies = _dependencies(
        model_factory=lambda config: state.__setitem__("models", state["models"] + 1),
    )
    config = _fake_hydra_config(tmp_path, preflight_only=True)
    result = run_evaluation(config, dependencies=dependencies)
    assert result.state == "preflight"
    assert state["models"] == 0
```

- [ ] **Step 2: Write a failing fake end-to-end evaluation test**

```python
def test_fake_two_dataset_run_uses_fresh_sequences_and_six_primary_metrics(tmp_path):
    dependencies, state = _complete_fake_dependencies(tmp_path)
    config = _fake_hydra_config(tmp_path, max_sequences=1)
    result = run_evaluation(config, dependencies=dependencies)
    assert result.state == "subset"
    assert state.model_constructions == 1
    assert state.inference_sequences == [
        ("7scenes-dense", "chess/seq-03"),
        ("NRGBD-dense", "breakfast_room"),
    ]
    assert set(asdict(result.datasets[0].primary)) == {
        "accuracy_mean_m", "accuracy_median_m",
        "completion_mean_m", "completion_median_m",
        "normal_consistency_mean", "normal_consistency_median",
    }
```

- [ ] **Step 3: Write failure, cleanup, and resume orchestration tests**

Cover these exact behaviors:

- data length/shape mismatch is recorded as a failure and exits through `EvaluationFailed`;
- a failure after one successful sequence preserves that sequence and marks the canonical result `incomplete`;
- `torch.cuda.empty_cache()` is called in the per-sequence `finally` path;
- an exact resumed sequence bypasses both inference and geometry evaluation;
- dataset instantiation occurs before model construction, so invalid maps/metadata fail without Pi3;
- predicted confidence is received from inference but never passed to `evaluate_point_maps()`.

- [ ] **Step 4: Run orchestration tests and verify old evaluator cannot satisfy them**

Run: `python -m pytest tests/mv_recon/test_eval_orchestration.py -v`

Expected: collection or assertions fail because `run_evaluation()` and dependency injection are absent.

- [ ] **Step 5: Replace module-level Open3D/Pi3 imports with evaluation dependencies**

```python
@dataclass(frozen=True)
class EvaluationDependencies:
    instantiate_dataset: Callable[[DictConfig], object]
    model_factory: Callable[[PipelineConfig], object]
    infer_point_maps: Callable[[list[str], object, DictConfig, tuple[int, int]], InferenceOutput]
    geometry_backend_factory: Callable[[], GeometryBackend]
    checkpoint_digest: Callable[[Path], str]
    git_commit: Callable[[], str]
    runtime_metadata: Callable[[], Mapping[str, object]]
    empty_cuda_cache: Callable[[], None]


def default_dependencies() -> EvaluationDependencies:
    return EvaluationDependencies(
        instantiate_dataset=hydra.utils.instantiate,
        model_factory=lambda config: StreamingPipelineModel(config).eval(),
        infer_point_maps=run_streaming_inference,
        geometry_backend_factory=Open3DGeometryBackend,
        checkpoint_digest=sha256_file,
        git_commit=_read_git_commit,
        runtime_metadata=_collect_runtime_metadata,
        empty_cuda_cache=torch.cuda.empty_cache,
    )
```

Open3D remains lazy inside `Open3DGeometryBackend`; direct `Pi3` import and the unused non-streaming `create_pi3()` are removed from `mv_recon/eval.py`. Define `InferenceOutput(points, confidence, ordinary_prediction_key, cache_diagnostics)` and implement `run_streaming_inference()` inside `mv_recon/eval.py` using the same `model.prepare()`, `run_windows()`, constraint, optimization, aggregation, and bilinear-resize calls currently used by `utils.interfaces.infer_streaming_mv_pointclouds()`. This evaluation-local wrapper exposes the prepared engine's prediction key and store counters without changing `utils/interfaces.py` or any reconstruction/cache module. `_collect_runtime_metadata()` reads the installed Open3D, PyTorch, NumPy, and SciPy versions plus CUDA/GPU information; missing Open3D raises an actionable strict-preflight error instead of selecting a fallback. Tests inject deterministic runtime metadata.

- [ ] **Step 6: Implement preflight, manifest, and model construction ordering**

`run_evaluation()` must execute in this order:

1. resolve strict protocol and canonical pipeline;
2. build and validate both full sequence maps;
3. instantiate each selected dataset and validate sequence metadata;
4. validate checkpoint existence and calculate its SHA256;
5. validate output collision/resume identity;
6. write `resolved_protocol.yaml`, `resolved_pipeline.yaml`, and `protocol_manifest.json`;
7. return `preflight` immediately when requested;
8. construct exactly one `StreamingPipelineModel`;
9. process selected sequences in dataset/map insertion order.

The protocol manifest records Git commit, dependency versions obtained with `importlib.metadata.version()`, Torch/CUDA device information, dtype, checkpoint/map/config hashes, selected sequence names, expected/selected counts, and cache roots.

- [ ] **Step 7: Implement the exact sequence runtime**

```python
def run_sequence(
    *,
    dataset_name: str,
    dataset: object,
    sequence: SequenceSpec,
    model: object,
    hydra_cfg: DictConfig,
    geometry: GeometryProtocol,
    backend: GeometryBackend,
    infer_point_maps: InferPointMaps,
) -> SequenceResult:
    data = dataset.get_data(
        sequence_name=sequence.name,
        ids=list(sequence.frame_ids),
    )
    paths = [str(path) for path in data["image_paths"]]
    images = data["images"]
    gt_points = np.asarray(data["pointclouds"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    _validate_loaded_sequence(sequence, paths, images, gt_points, valid_mask)
    height, width = (int(value) for value in images.shape[-2:])
    inference = infer_point_maps(
        paths, model, hydra_cfg, (height, width)
    )
    geometry_result = evaluate_point_maps(
        inference.points, gt_points, valid_mask, geometry, backend=backend
    )
    return _sequence_result_from_geometry(
        dataset_name, sequence, paths, inference, geometry_result
    )
```

Copy the `InferenceOutput` ordinary prediction key and cache counters into the sequence result and protocol manifest. Rewrite `protocol_manifest.json` atomically after every sequence so an interrupted job retains the cache keys/counts already observed.

- [ ] **Step 8: Implement strict failure state and exit semantics**

Catch sequence errors only to classify and record them, finalize as `incomplete`/`failed`, clean CUDA state, then raise `EvaluationFailed` so the CLI exits non-zero. Do not continue to compute a complete summary with the failed sequence removed. A full success requires 18 and 9 sequence results; a limited success is always `subset`.

- [ ] **Step 9: Keep the Hydra CLI stable and update legacy source assertions**

```python
@hydra.main(
    version_base="1.2",
    config_path="../configs",
    config_name="eval_mv_recon_dense",
)
def main(hydra_cfg: DictConfig) -> None:
    run_evaluation(hydra_cfg)


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon_laser_paper")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    with torch.no_grad():
        main()
```

Change `tests/test_eval_launch_lc.py` only where its legacy source-string checks assume `cfg.pi3.checkpoint` is read directly in `mv_recon/eval.py`; replace that assertion with one requiring `resolve_evaluation_protocol` and forbidding `dataclasses.replace` in the indoor evaluator.

- [ ] **Step 10: Run orchestration and existing launch tests**

Run: `python -m pytest tests/mv_recon/test_eval_orchestration.py tests/test_eval_launch_lc.py -v`

Expected: all tests pass with no Pi3 or Open3D construction in preflight/fake tests.

- [ ] **Step 11: Commit evaluation orchestration**

```bash
git add mv_recon/eval.py tests/mv_recon/test_eval_orchestration.py tests/test_eval_launch_lc.py
git commit -m "feat: orchestrate paper point-map evaluation"
```

---

### Task 6: Cloud Documentation and Artifact Contract Tests

**Files:**
- Modify: `mv_recon/README.md`
- Modify: `mv_recon/results.py`
- Modify: `tests/mv_recon/test_results.py`
- Modify: `tests/mv_recon/test_eval_orchestration.py`

**Interfaces:**
- Consumes: final CLI and artifact schema.
- Produces: reproducible cloud commands and regression coverage for every required output file.

- [ ] **Step 1: Add failing artifact contract assertions**

Extend the end-to-end fake run to assert these exact files exist:

```python
expected = {
    "resolved_protocol.yaml",
    "resolved_pipeline.yaml",
    "protocol_manifest.json",
    "results.json",
    "summary.csv",
    "sequences.csv",
    "failures.jsonl",
}
output_root = Path(config.output_dir)
assert expected <= {path.name for path in output_root.iterdir()}
```

Parse `results.json` and assert schema version, state, six primary fields, paper reference, `delta_to_paper`, completion counts, and per-sequence input/cache identities. Parse both CSVs with `csv.DictReader` and assert their numeric values equal the canonical JSON projections.

- [ ] **Step 2: Run the artifact tests and observe any missing projections**

Run: `python -m pytest tests/mv_recon/test_results.py tests/mv_recon/test_eval_orchestration.py -v`

Expected: any missing output or field fails with its exact filename/key.

- [ ] **Step 3: Complete output projections without changing canonical math**

Ensure `summary.csv` has one row per dataset with exactly these leading columns:

```text
dataset,status,expected_sequences,completed_sequences,
accuracy_mean_m,accuracy_median_m,completion_mean_m,completion_median_m,
normal_consistency_mean,normal_consistency_median
```

Ensure `sequences.csv` has dataset/sequence/frame count/input manifest/GT content/cache key, all six primary values, directional NC values, Chamfer-L1, ICP fitness/RMSE, Umeyama scale, counts, and threshold precision/recall/F-score fields.

- [ ] **Step 4: Document cloud setup and all four execution modes**

Add a `LASER paper point-map evaluation` section to `mv_recon/README.md` containing:

```bash
python mv_recon/eval.py evaluation=mv_recon_laser_paper protocol.preflight_only=true output_dir=outputs/mv_recon_laser_paper_preflight

python mv_recon/eval.py evaluation=mv_recon_laser_paper protocol.max_sequences=1 output_dir=outputs/mv_recon_laser_paper_smoke

python mv_recon/eval.py evaluation=mv_recon_laser_paper output_dir=outputs/mv_recon_laser_paper

python mv_recon/eval.py evaluation=mv_recon_laser_paper protocol.resume=true output_dir=outputs/mv_recon_laser_paper
```

State that `max_sequences=1` evaluates one sequence per dataset and is not a Table 4 result. Document the two data roots, local checkpoint path, Python/Open3D compatibility, primary-versus-extended metrics, complete/subset/incomplete states, and reference values.

- [ ] **Step 5: Run documentation-related tests and focused evaluation suite**

Run: `python -m pytest tests/mv_recon -v`

Expected: all point-map evaluation tests pass.

- [ ] **Step 6: Commit documentation and final artifact projections**

```bash
git add mv_recon/README.md mv_recon/results.py tests/mv_recon/test_results.py tests/mv_recon/test_eval_orchestration.py
git commit -m "docs: add cloud point-map evaluation workflow"
```

---

### Task 7: Full Verification and Scope Gate

**Files:**
- Verify only; modify a file only when a failing test identifies a defect within the approved scope.

**Interfaces:**
- Consumes: all implementation commits.
- Produces: fresh verification evidence and a branch ready to push for cloud testing.

- [ ] **Step 1: Run the complete local test suite**

Run: `python -m pytest -q`

Expected: all baseline 274 tests plus all new evaluation tests pass; the existing Torch JIT deprecation warning is allowed.

- [ ] **Step 2: Run syntax compilation for changed Python modules**

Run: `python -m py_compile mv_recon/eval.py mv_recon/protocol.py mv_recon/geometry_metrics.py mv_recon/results.py`

Expected: exit zero with no output.

- [ ] **Step 3: Run configuration and whitespace checks**

Run: `python -c 'from omegaconf import OmegaConf; OmegaConf.load("configs/evaluation/mv_recon_laser_paper.yaml")'`

Run: `git diff --check fcbe67981e905b13c4c1620b04a372760ecc87db..HEAD`

Expected: both commands exit zero.

- [ ] **Step 4: Enforce the evaluation-only diff boundary**

Run: `git diff --name-only fcbe67981e905b13c4c1620b04a372760ecc87db..HEAD`

Expected: every path begins with `configs/evaluation/`, `mv_recon/`, `tests/`, or `docs/`; specifically no path begins with `pipeline/`, `inference_engine/`, `pi3/`, or `loop_closure/`.

- [ ] **Step 5: Inspect final state and record environment limitation**

Run: `git status --short --branch`

Expected: implementation files are committed. Locally generated `inference_engine/utils/_segmentation_cy.cpp` and `inference_engine/utils/fast_seg.cpp` may remain untracked but must not appear in any commit. Record that real Open3D/cloud numerical validation remains for the user's supported Python/GPU environment.

- [ ] **Step 6: Push the implementation branch and prepare exact cloud commands**

```bash
git push -u origin codex/laser-paper-pointmap-eval
```

After push succeeds, provide a clone command that fetches only the implementation branch, followed by dependency installation, checkpoint/data placement expectations, preflight, per-dataset smoke, full run, and resume commands.
