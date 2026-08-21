# Window-Reference Cloud Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh-clone-safe, resumable six-run window-reference ablation campaign that stages aligned RGB/pose/point-map-GT scenes, shares one ordinary PI3 prediction cache per scene, evaluates with the existing LASER evaluators, and emits finite atomic JSON/CSV results.

**Architecture:** A typed sidecar package owns campaign configuration, the fixed matrix, scene resolution/staging, compact records, diagnostics, evaluator adapters, execution/resume policy, bootstrap/preflight, and CLI dispatch while leaving PI3, reconstruction, prediction-cache, segmentation, artifact, and metric algorithms unchanged. The root CLI and package initializers stay import-light; `plan` and `bootstrap --dry-run` use no Torch path, real runs lazily adapt `PipelineRunner`, and synthetic runs use a deterministic CPU fixture through the same compact-record boundary. Each real scene is staged once, executed serially, summarized, and safely cleaned before the next scene.

**Tech Stack:** Python 3.11, standard library (`argparse`, `csv`, `dataclasses`, `hashlib`, `json`, `pathlib`, `shutil`, `subprocess`, `tempfile`), existing OmegaConf, NumPy, Pillow, Torch, SciPy/evo/Open3D evaluator dependencies, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-window-reference-cloud-campaign-design.md` at design commit `256c39352ef3acf52ebea8ca46bb7a8053f17557` (the spec's `977b261a8cd4809a69ef19b8d9738401b52c5a62` is the code baseline it records, not the design-document commit).

## Global Constraints

- Begin implementation from branch `codex/laser-paper-pointmap-eval` at design commit `256c39352ef3acf52ebea8ca46bb7a8053f17557`; preserve the approved spec unchanged.
- The default matrix is exactly `depth__wr-off`, `depth__wr-on`, `geometry__wr-off`, `geometry__wr-on`, `atomic__wr-off`, `atomic__wr-on`, in that order.
- Every run fixes `reconstruction.mode=no_loop`, `window.size=75`, `window.overlap=30`, and staged `input.sample_stride=1`.
- Every atomic run fixes `segmentation.atomic.split_mode=conservative`; this value remains in every run identity and is not a matrix axis.
- Enabled window-reference runs use exactly: `sampling_stride=4`, `max_keyframes=4`, `relative_depth_tolerance=0.05`, `min_reference_score=0.30`, `stop_coverage_ratio=0.90`, `min_coverage_gain=0.03`, `min_region_correspondences=8`, `min_region_coverage=0.10`, `min_region_purity=0.80`, and `merge_vote_threshold=0.80`.
- Disabled runs change only `segmentation.window_reference.enabled=false`; all ten remaining window-reference values are resolved and snapshotted unchanged.
- Unknown campaign YAML keys are errors. Refinement values must have Python type `bool`; strings such as `off`, `on`, `yes`, and `no` are errors even if a YAML parser would otherwise coerce them.
- Raw data, external prepared GT, weights, historical outputs, and any path outside the resolved campaign root are read-only and are never cleanup targets.
- Direct `.ply`, `.pcd`, KITTI Velodyne `.bin`, automatic KITTI/7-Scenes/NeuralRGBD downloads, credential storage, authenticated URLs, KITTI leaderboard submission, and official KITTI 100--800 m metric claims are out of scope.
- KITTI input is left-colour `image_2` plus finite 12-value pose rows; accept both normalized `{root}/{sequence}/{image_2,poses.txt}` and official `dataset/sequences/{sequence}/image_2` plus `dataset/poses/{sequence}.txt` layouts.
- The point-cloud preset is exactly `chess/seq-03` (100 mapped frames), `thin_geometry` (40 mapped frames), and `complete_kitchen` (122 mapped frames), with expected prepared shapes `(100,392,518,3)`, `(40,392,518,3)`, and `(122,392,518,3)` respectively.
- KITTI presets are exactly: `kitti-smoke=04[0:80]`, `kitti-small=04,03,07`, and `kitti-formal-subset=00,01,04,07`; no preset claims complete `00..10` coverage.
- One scene and one run execute at a time on one GPU. `runtime.jobs` is reserved but must equal `1` in version 1.
- The default cache policy is `auto`: the first pending run uses `auto`, a deliberately requested refresh uses `refresh` for that first pending run only, and a complete shared cache forces `readonly` for the remaining runs. Explicit `readonly` and `off` retain their literal meanings.
- Resume skips only a schema-valid successful compact record whose complete identity equals the expected identity seed completed with that record's SHA-256 prediction key and whose required metrics are finite.
- A failed run preserves its numbered attempt directory and log. Evaluation may resume from a validated artifact without repeating reconstruction.
- `--fail-fast` stops after the first failure; `--keep-going` continues independent runs/scenes. Either mode returns non-zero while any requested run is invalid.
- Disabled refinement aggregates use JSON `null` plus `refinement_state="not_applicable"`; they never masquerade as fallback, zero-keyframe performance, or successful refinement.
- Overlap-aware diagnostics always report both `unique_frame_count` and `window_frame_observation_count`; repeated observations in overlapping windows are not deduplicated silently.
- JSON uses `allow_nan=False`; CSV represents unavailable values as empty fields. Every JSON/CSV/NumPy result is written to a sibling temporary file, flushed/fsynced, and atomically replaced.
- Default cloud clone/output/staging/cache/artifact paths are under `/root/autodl-tmp`; documentation must not present `/root/autodl-fs` as a default. `storage.minimum_free_gb` is `20.0`, with an explicit low-space warning.
- No new dependency may be added. Use only the standard library and existing OmegaConf/NumPy/Pillow/Torch/SciPy/evo/Open3D dependencies; do not change `requirements.txt` or `setup.py`.
- `bootstrap` and `plan --dry-run` must not import Torch, inspect CUDA, read weights, or read datasets. Bootstrap may download only public model weights through existing `scripts/download_weights.sh` and only when no external checkpoint is supplied.
- `preflight --allow-no-gpu` records that GPU execution is unavailable and succeeds if all non-GPU checks pass; real `run` rejects unavailable CUDA whenever `model.inference_device=cuda`.
- Historical KITTI compact metrics using `corrected` reconstruction may be labelled contextual reference only; never pool them with the new `no_loop` off/on pairs.
- Local code completion does not claim any real GPU/KITTI/point-cloud campaign passed. Real campaign commands are handoff commands to run only after a GPU is opened and the protected data plus checkpoint are present.
- Do not write or print hostname, username, password, token, cookie, SSH secret, KITTI login, or URL userinfo. Persisted argv is passed through deterministic secret redaction.

---

## Locked File Structure and Responsibilities

```text
run_window_reference_campaign.py
  Standard-library-only root entry point; delegates lazily after parsing the command name.

configs/experiments/window_reference_campaign.yaml
  Auditable version-1 defaults, exact axes, scene catalog, preset membership/slices,
  runtime/storage values, and existing evaluator config paths.

experiments/__init__.py
  Lazy compatibility exports so importing a campaign submodule does not import Torch.

experiments/window_reference_campaign/
  __init__.py      Import-light public constants/types only.
  config.py        Typed config, strict key/type/value validation, CLI override resolution/hash.
  matrix.py        Canonical variants, logical plan, identity seed/completion, stable paths.
  scenes.py        Dataset/layout resolution and one positional frame-selection vector.
  staging.py       Owned symlink/pose/point-map staging, hashes, reuse validation, atomic files.
  diagnostics.py   Overlap-aware on/off refinement aggregation and finite distributions.
  results.py       Compact record schema/validation, atomic JSON/CSV, summaries/failures.
  runner.py        Attempts, pipeline adapter, cache transition, resume, policy, safe cleanup.
  evaluation.py    Existing point-cloud and internal trajectory evaluator adapters.
  preflight.py     Stdlib bootstrap actions plus lazy environment/data/checkpoint/CUDA checks.
  synthetic.py     Deterministic CPU point-map fixture and synthetic execution adapter.
  cli.py           Import-light parser and lazy handlers for bootstrap/plan/preflight/run/summarize.

docs/window-reference-cloud-campaign.md
  Fresh-clone, no-GPU inspection, screen/GPU run, resume, summary, cleanup, and truthfulness guide.

tests/experiments/
  test_window_reference_campaign_config.py
  test_window_reference_campaign_scenes.py
  test_window_reference_campaign_runner.py
  test_window_reference_campaign_evaluation.py
  test_window_reference_campaign_results.py
  test_window_reference_campaign_preflight.py
  test_window_reference_campaign_cli.py
```

`config.py` owns configuration but never discovers files. `scenes.py` resolves sources but never writes. `staging.py` is the only dataset preparation writer. `runner.py` owns mutable campaign execution but delegates all metrics to `evaluation.py` and all serialization to `results.py`. No campaign module copies PI3 inference, reconstruction, cache validation, or evaluator formulas.

### Task 1: Typed Configuration, Exact Matrix, Identities, and Import-Light `plan`

**Files:**
- Create: `configs/experiments/window_reference_campaign.yaml`
- Create: `experiments/window_reference_campaign/__init__.py`
- Create: `experiments/window_reference_campaign/config.py`
- Create: `experiments/window_reference_campaign/matrix.py`
- Create: `experiments/window_reference_campaign/cli.py`
- Create: `run_window_reference_campaign.py`
- Modify: `experiments/__init__.py:1-29`
- Create: `tests/experiments/test_window_reference_campaign_config.py`

**Interfaces:**
- Consumes: `pipeline.config.SegmentationMethod`, `pipeline.config.AtomicSplitMode`, `pipeline.config.ReconstructionMode`, `pipeline.config.PredictionCacheMode`, OmegaConf, and the checked-in YAML.
- Produces exact enums `DatasetKind`, `EvaluationKind`, `CachePolicy`, `FailurePolicy`; dataclasses `WindowReferenceValues`, `SceneConfig`, `PresetSceneConfig`, `PresetConfig`, `MatrixConfig`, `RuntimeConfig`, `StorageConfig`, `EvaluationConfig`, `CampaignConfig`, `CampaignOverrides`, and `LoadedCampaignConfig`.
- `SceneConfig` fields are `scene_id: str`, `dataset: DatasetKind`, `scene: str`, `source_root: str | None`, `frame_index_map: str | None`, `prepared_gt: str | None`, and `expected_gt_shape: tuple[int, int] | None`. `PresetSceneConfig` fields are `scene_id: str`, `start: int`, `stop: int | None`, and `stride: int`; `PresetConfig.scenes` is `tuple[PresetSceneConfig, ...]`.
- `MatrixConfig` fields are `methods: tuple[SegmentationMethod, ...]`, `refinement: tuple[bool, ...]`, `reconstruction_mode: ReconstructionMode`, `window_size: int`, `overlap: int`, and `atomic_split_mode: AtomicSplitMode`. `RuntimeConfig` fields are `model_name: ModelName`, `model_dtype: str`, `inference_device: str`, `process_device: str`, `jobs: int`, `gpu: int`, `cache_policy: CachePolicy`, and `failure_policy: FailurePolicy`.
- `StorageConfig` fields are `data_root: Path`, `checkpoint: Path`, `output_root: Path`, `minimum_free_gb: float`, and `keep_artifacts: bool`. `EvaluationConfig` fields are `pointcloud_config: Path` and `trajectory_config: Path`.
- `CampaignConfig` fields are exactly `version: int`, `campaign_id: str`, `repository_root: Path`, `pipeline_config: Path`, `matrix: MatrixConfig`, `window_reference: WindowReferenceValues`, `runtime: RuntimeConfig`, `storage: StorageConfig`, `evaluation: EvaluationConfig`, `scenes: Mapping[str, SceneConfig]`, `presets: Mapping[str, PresetConfig]`, `selected_preset: str`, and `selected_scenes: tuple[PresetSceneConfig, ...]`; its `campaign_root` property returns `storage.output_root / campaign_id`.
- Produces `load_campaign_config(path: str | Path, overrides: CampaignOverrides) -> LoadedCampaignConfig`; `LoadedCampaignConfig.sha256` is SHA-256 of sorted canonical JSON for the fully resolved typed config.
- Produces `RunVariant(segmentation_method: SegmentationMethod, window_reference_enabled: bool)` with `run_id: str`; `PlannedRun(scene_id: str, dataset: DatasetKind, scene: str, slice_id: str, variant: RunVariant, relative_run_dir: Path)`; `CampaignPlan(campaign_id: str, preset: str, runs: tuple[PlannedRun, ...])`.
- Produces `RunIdentitySeed` with every required identity field except `prediction_key`, `RunIdentity` with `prediction_key`, `complete_identity(seed: RunIdentitySeed, prediction_key: str) -> RunIdentity`, and `identity_sha256(identity: RunIdentity) -> str`.
- Produces `expand_matrix(methods: tuple[SegmentationMethod, ...] = (), refinements: tuple[bool, ...] = ()) -> tuple[RunVariant, ...]`, `build_plan(loaded: LoadedCampaignConfig) -> CampaignPlan`, and `plan_payload(plan: CampaignPlan, loaded: LoadedCampaignConfig) -> dict[str, object]`.
- Produces root `main(argv: Sequence[str] | None = None) -> int`; at this task boundary `plan` is functional and other declared commands fail with parser help until their lazy handlers are added in Tasks 6–7.

- [ ] **Step 1: Write RED tests for strict booleans, unknown keys, presets, and the fixed matrix**

```python
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.window_reference_campaign.config import (
    CampaignOverrides,
    load_campaign_config,
)
from experiments.window_reference_campaign.matrix import expand_matrix


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/window_reference_campaign.yaml"


def test_default_matrix_is_the_approved_six_in_order():
    assert [item.run_id for item in expand_matrix()] == [
        "depth__wr-off",
        "depth__wr-on",
        "geometry__wr-off",
        "geometry__wr-on",
        "atomic__wr-off",
        "atomic__wr-on",
    ]


def test_typed_config_locks_primary_protocol_and_presets():
    loaded = load_campaign_config(
        CONFIG,
        CampaignOverrides(preset="pointcloud-small"),
    )
    config = loaded.config
    assert config.matrix.reconstruction_mode.value == "no_loop"
    assert (config.matrix.window_size, config.matrix.overlap) == (75, 30)
    assert config.matrix.atomic_split_mode.value == "conservative"
    assert config.runtime.jobs == 1
    assert [item.scene_id for item in config.selected_scenes] == [
        "seven-chess-seq-03",
        "nrgbd-thin-geometry",
        "nrgbd-complete-kitchen",
    ]
    assert [item.stop for item in load_campaign_config(
        CONFIG, CampaignOverrides(preset="kitti-smoke")
    ).config.selected_scenes] == [80]


@pytest.mark.parametrize("raw", ["off", "on", "yes", "no"])
def test_refinement_axis_rejects_yaml_like_strings(tmp_path, raw):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "refinement: [false, true]",
        f"refinement: [false, {raw!r}]",
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="matrix.refinement.*boolean"):
        load_campaign_config(path, CampaignOverrides(preset="synthetic-smoke"))


def test_unknown_campaign_key_is_rejected(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8") + "surprise: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown campaign field: surprise"):
        load_campaign_config(path, CampaignOverrides(preset="synthetic-smoke"))
```

- [ ] **Step 2: Write RED tests for complete identity fields and plan isolation**

```python
import json
import os
import subprocess
import sys

from experiments.window_reference_campaign.config import CampaignOverrides
from experiments.window_reference_campaign.matrix import (
    RunIdentitySeed,
    build_plan,
    complete_identity,
)


def test_identity_completion_contains_every_audit_field():
    seed = RunIdentitySeed(
        schema_version=1,
        campaign_config_sha256="1" * 64,
        source_commit="2" * 40,
        source_dirty=False,
        dataset="kitti",
        scene="04",
        frame_start=0,
        frame_stop=80,
        frame_stride=1,
        staged_manifest_sha256="3" * 64,
        segmentation_method="atomic",
        atomic_split_mode="conservative",
        window_reference_enabled=True,
        window_reference_config={
            "sampling_stride": 4,
            "max_keyframes": 4,
            "relative_depth_tolerance": 0.05,
            "min_reference_score": 0.30,
            "stop_coverage_ratio": 0.90,
            "min_coverage_gain": 0.03,
            "min_region_correspondences": 8,
            "min_region_coverage": 0.10,
            "min_region_purity": 0.80,
            "merge_vote_threshold": 0.80,
        },
        reconstruction_mode="no_loop",
        window_size=75,
        overlap=30,
        model_name="pi3",
        model_dtype="bfloat16",
        checkpoint_sha256="4" * 64,
    )
    identity = complete_identity(seed, "5" * 64)
    assert set(identity.to_payload()) == {
        *seed.to_payload(),
        "prediction_key",
    }
    assert identity.prediction_key == "5" * 64


def test_plan_dry_run_does_not_import_torch_or_touch_inputs(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class Block:\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "    if fullname == 'torch' or fullname.startswith('torch.'):\n"
        "      raise RuntimeError('torch import forbidden')\n"
        "sys.meta_path.insert(0, Block())\n",
        encoding="utf-8",
    )
    missing = tmp_path / "does-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_window_reference_campaign.py"),
            "plan",
            "--config", str(CONFIG),
            "--preset", "kitti-small",
            "--data-root", str(missing),
            "--checkpoint", str(missing / "model.safetensors"),
            "--output-root", str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert [run["run_id"] for run in payload["runs"][:6]] == [
        "depth__wr-off", "depth__wr-on",
        "geometry__wr-off", "geometry__wr-on",
        "atomic__wr-off", "atomic__wr-on",
    ]
    assert not (tmp_path / "output").exists()
```

- [ ] **Step 3: Run the focused RED nodes**

Run:

```bash
python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_window_reference_campaign_config.py::test_default_matrix_is_the_approved_six_in_order \
  tests/experiments/test_window_reference_campaign_config.py::test_typed_config_locks_primary_protocol_and_presets \
  tests/experiments/test_window_reference_campaign_config.py::test_plan_dry_run_does_not_import_torch_or_touch_inputs
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'experiments.window_reference_campaign'`.

- [ ] **Step 4: Implement strict typed config and the complete checked-in YAML**

Use `OmegaConf.load()` only as a YAML reader, convert to a plain mapping, reject unknown keys at every nesting level, then construct frozen dataclasses. Do not rely on YAML scalar coercion for booleans.

```python
# experiments/window_reference_campaign/config.py
@dataclass(frozen=True)
class WindowReferenceValues:
    sampling_stride: int
    max_keyframes: int
    relative_depth_tolerance: float
    min_reference_score: float
    stop_coverage_ratio: float
    min_coverage_gain: float
    min_region_correspondences: int
    min_region_coverage: float
    min_region_purity: float
    merge_vote_threshold: float

    def to_payload(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignOverrides:
    preset: str
    scene_ids: tuple[str, ...] = ()
    methods: tuple[SegmentationMethod, ...] = ()
    refinements: tuple[bool, ...] = ()
    data_root: Path | None = None
    output_root: Path | None = None
    checkpoint: Path | None = None
    start_frame: int | None = None
    max_frames: int | None = None
    frame_stride: int | None = None
    gpu: int | None = None
    cache_policy: CachePolicy | None = None
    keep_artifacts: bool | None = None


@dataclass(frozen=True)
class LoadedCampaignConfig:
    config: CampaignConfig
    resolved_payload: Mapping[str, object]
    sha256: str


def _require_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def load_campaign_config(
    path: str | Path,
    overrides: CampaignOverrides,
) -> LoadedCampaignConfig:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("campaign config must be a mapping")
    _reject_unknown(payload, _ROOT_FIELDS, "campaign")
    config = _construct_and_validate(payload, overrides)
    canonical = _campaign_payload(config)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return LoadedCampaignConfig(config, canonical, hashlib.sha256(encoded).hexdigest())
```

The YAML must contain these literal protocol and preset values (write the complete nested runtime/storage/evaluation/scene records, not aliases or interpolated environment secrets):

```yaml
version: 1
campaign:
  id: window-reference-v1
pipeline_config: configs/reconstruction/pi3_laser_no_loop.yaml
matrix:
  methods: [depth, geometry, atomic]
  refinement: [false, true]
  reconstruction_mode: no_loop
  window_size: 75
  overlap: 30
  atomic_split_mode: conservative
window_reference:
  sampling_stride: 4
  max_keyframes: 4
  relative_depth_tolerance: 0.05
  min_reference_score: 0.30
  stop_coverage_ratio: 0.90
  min_coverage_gain: 0.03
  min_region_correspondences: 8
  min_region_coverage: 0.10
  min_region_purity: 0.80
  merge_vote_threshold: 0.80
runtime:
  model_name: pi3
  model_dtype: bfloat16
  inference_device: cuda
  process_device: cpu
  jobs: 1
  gpu: 0
  cache_policy: auto
  failure_policy: keep-going
storage:
  data_root: data
  checkpoint: weights/PI3/model.safetensors
  output_root: outputs/window_reference_campaign
  minimum_free_gb: 20.0
  keep_artifacts: false
evaluation:
  pointcloud_config: configs/evaluation/pointcloud.yaml
  trajectory_config: configs/evaluation/ate.yaml
scenes:
  synthetic-fixture:
    dataset: synthetic
    scene: deterministic-pointmaps
    source_root: null
    frame_index_map: null
    prepared_gt: null
    expected_gt_shape: null
  seven-chess-seq-03:
    dataset: 7scenes
    scene: chess/seq-03
    source_root: 7-Scenes
    frame_index_map: datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json
    prepared_gt: pointcloud_benchmark/7scenes/chess__seq-03/ground_truth.npz
    expected_gt_shape: [392, 518]
  nrgbd-thin-geometry:
    dataset: nrgbd
    scene: thin_geometry
    source_root: NeuralRGBD
    frame_index_map: datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json
    prepared_gt: pointcloud_benchmark/nrgbd/thin_geometry/ground_truth.npz
    expected_gt_shape: [392, 518]
  nrgbd-complete-kitchen:
    dataset: nrgbd
    scene: complete_kitchen
    source_root: NeuralRGBD
    frame_index_map: datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json
    prepared_gt: pointcloud_benchmark/nrgbd/complete_kitchen/ground_truth.npz
    expected_gt_shape: [392, 518]
  kitti-00: {dataset: kitti, scene: "00", source_root: KITTI, frame_index_map: null, prepared_gt: null, expected_gt_shape: null}
  kitti-01: {dataset: kitti, scene: "01", source_root: KITTI, frame_index_map: null, prepared_gt: null, expected_gt_shape: null}
  kitti-03: {dataset: kitti, scene: "03", source_root: KITTI, frame_index_map: null, prepared_gt: null, expected_gt_shape: null}
  kitti-04: {dataset: kitti, scene: "04", source_root: KITTI, frame_index_map: null, prepared_gt: null, expected_gt_shape: null}
  kitti-07: {dataset: kitti, scene: "07", source_root: KITTI, frame_index_map: null, prepared_gt: null, expected_gt_shape: null}
presets:
  synthetic-smoke:
    scenes: [{scene_id: synthetic-fixture, start: 0, stop: 4, stride: 1}]
  pointcloud-small:
    scenes:
      - {scene_id: seven-chess-seq-03, start: 0, stop: 100, stride: 1}
      - {scene_id: nrgbd-thin-geometry, start: 0, stop: 40, stride: 1}
      - {scene_id: nrgbd-complete-kitchen, start: 0, stop: 122, stride: 1}
  kitti-smoke:
    scenes: [{scene_id: kitti-04, start: 0, stop: 80, stride: 1}]
  kitti-small:
    scenes:
      - {scene_id: kitti-04, start: 0, stop: null, stride: 1}
      - {scene_id: kitti-03, start: 0, stop: null, stride: 1}
      - {scene_id: kitti-07, start: 0, stop: null, stride: 1}
  kitti-formal-subset:
    scenes:
      - {scene_id: kitti-00, start: 0, stop: null, stride: 1}
      - {scene_id: kitti-01, start: 0, stop: null, stride: 1}
      - {scene_id: kitti-04, start: 0, stop: null, stride: 1}
      - {scene_id: kitti-07, start: 0, stop: null, stride: 1}
```

Resolve every non-null `frame_index_map` relative to `CampaignConfig.repository_root`; it is checked-in source metadata, not data-root content. Resolve `source_root` and `prepared_gt` relative to `storage.data_root`. Never read any of these paths in `plan`.

- [ ] **Step 5: Implement canonical variants, logical plan, and two-stage identity**

```python
# experiments/window_reference_campaign/matrix.py
_CANONICAL_METHODS = (
    SegmentationMethod.DEPTH,
    SegmentationMethod.GEOMETRY,
    SegmentationMethod.ATOMIC,
)
_CANONICAL_REFINEMENTS = (False, True)


@dataclass(frozen=True)
class RunVariant:
    segmentation_method: SegmentationMethod
    window_reference_enabled: bool

    @property
    def run_id(self) -> str:
        state = "on" if self.window_reference_enabled else "off"
        return f"{self.segmentation_method.value}__wr-{state}"


def expand_matrix(
    methods: tuple[SegmentationMethod, ...] = (),
    refinements: tuple[bool, ...] = (),
) -> tuple[RunVariant, ...]:
    selected_methods = methods or _CANONICAL_METHODS
    selected_refinements = refinements or _CANONICAL_REFINEMENTS
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("matrix methods must be unique")
    if any(type(value) is not bool for value in selected_refinements):
        raise ValueError("matrix refinement values must be booleans")
    if any(value not in _CANONICAL_METHODS for value in selected_methods):
        raise ValueError("matrix method is not supported")
    return tuple(
        RunVariant(method, enabled)
        for method in _CANONICAL_METHODS if method in selected_methods
        for enabled in _CANONICAL_REFINEMENTS if enabled in selected_refinements
    )


def complete_identity(seed: RunIdentitySeed, prediction_key: str) -> RunIdentity:
    _require_sha256(prediction_key, "prediction_key")
    return RunIdentity(**seed.to_payload(), prediction_key=prediction_key)
```

`RunIdentitySeed.__post_init__()` must validate exact enum strings, all SHA/commit formats (`source_commit` may be the literal `unknown` only when Git is unavailable), `frame_start >= 0`, `frame_stop > frame_start`, `frame_stride >= 1`, fixed `no_loop/75/30/conservative/pi3`, and the complete ten-key window-reference mapping. `RunIdentity.to_payload()` preserves booleans and uses only JSON scalars/mappings.

- [ ] **Step 6: Make `experiments` and CLI startup lazy, then implement `plan`**

Replace eager imports in `experiments/__init__.py` with PEP 562 compatibility exports so existing `from experiments import ExperimentConfig` remains valid without importing `experiments.runner` at package import time:

```python
_EXPORTS = {
    "EvaluationKind": ("experiments.config", "EvaluationKind"),
    "ExperimentConfig": ("experiments.config", "ExperimentConfig"),
    "ExperimentDatasetConfig": ("experiments.config", "ExperimentDatasetConfig"),
    "load_experiment_config": ("experiments.config", "load_experiment_config"),
    "ArtifactRepository": ("experiments.matrix", "ArtifactRepository"),
    "MatrixEntry": ("experiments.matrix", "MatrixEntry"),
    "build_matrix": ("experiments.matrix", "build_matrix"),
    "reconstruction_identity": ("experiments.matrix", "reconstruction_identity"),
    "ExperimentRunRecord": ("experiments.runner", "ExperimentRunRecord"),
    "matrix_cache_mode": ("experiments.runner", "matrix_cache_mode"),
    "run_matrix": ("experiments.runner", "run_matrix"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
```

`experiments/window_reference_campaign/__init__.py` contains only its docstring and `__all__: tuple[str, ...] = ()`; it imports no sibling module. `run_window_reference_campaign.py` may import only `collections.abc`, then import `experiments.window_reference_campaign.cli.main` inside its own `main()`. `cli.py` may import only stdlib modules at module scope. Its `plan` handler lazily imports `config`/`matrix`; `--dry-run` prints one sorted JSON object and does not create `output_root`, while non-dry `plan` atomically writes `{output_root}/{campaign_id}/plan.json` and prints that path.

- [ ] **Step 7: Run GREEN and regression tests**

Run:

```bash
python -m pytest -p no:cacheprovider -q tests/experiments/test_window_reference_campaign_config.py
python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_matrix.py \
  tests/experiments/test_pointcloud_experiment.py \
  tests/experiments/test_ate_experiment.py
python run_window_reference_campaign.py plan --preset synthetic-smoke --dry-run
```

Expected: all pytest nodes PASS; the CLI prints one scene times six runs, `no_loop`, `75/30`, and no filesystem path is required to exist.

- [ ] **Step 8: REFACTOR serialization through one canonical helper and commit**

Move repeated sorted JSON encoding into private `_canonical_json_bytes(value: Mapping[str, object]) -> bytes`, rerun Step 7, then commit exactly:

```bash
git add configs/experiments/window_reference_campaign.yaml \
  experiments/__init__.py \
  experiments/window_reference_campaign/__init__.py \
  experiments/window_reference_campaign/config.py \
  experiments/window_reference_campaign/matrix.py \
  experiments/window_reference_campaign/cli.py \
  run_window_reference_campaign.py \
  tests/experiments/test_window_reference_campaign_config.py
git commit -m "feat: add window reference campaign planning"
```

### Task 2: Scene Resolution and Campaign-Owned Single-Index Staging

**Files:**
- Create: `experiments/window_reference_campaign/scenes.py`
- Create: `experiments/window_reference_campaign/staging.py`
- Create: `tests/experiments/test_window_reference_campaign_scenes.py`

**Interfaces:**
- Consumes: Task 1 `SceneConfig`, `PresetSceneConfig`, `DatasetKind`, `EvaluationKind`; `pipeline.manifest.natural_sort_key`; checked-in sequence-map JSON; raw datasets below `CampaignConfig.storage.data_root`.
- Produces `ResolvedFrameSelection(start: int, stop: int, stride: int, source_frame_ids: tuple[int, ...])` and `apply_frame_selection(base_frame_ids: Sequence[int], selection: PresetSceneConfig, *, start_override: int | None, max_frames: int | None, stride_override: int | None) -> ResolvedFrameSelection`.
- Produces `KittiLayout(image_dir: Path, poses_path: Path, layout_name: str)` and `resolve_kitti_layout(dataset_root: str | Path, sequence: str) -> KittiLayout`.
- Produces `ResolvedScene(scene_id: str, dataset: DatasetKind, scene: str, slice_id: str, approved_data_root: Path, source_images: tuple[Path, ...], selection: ResolvedFrameSelection, evaluation_kind: EvaluationKind, poses_path: Path | None, prepared_gt_path: Path | None, frame_index_map: Path | None, expected_gt_shape: tuple[int, int] | None)` and `resolve_scene(config: CampaignConfig, selected: PresetSceneConfig) -> ResolvedScene`.
- Produces `StagedScene(scene_id: str, dataset: DatasetKind, scene: str, slice_id: str, image_dir: Path, source_frame_ids: tuple[int, ...], selection: ResolvedFrameSelection, poses_path: Path | None, pointcloud_gt_path: Path | None, manifest_path: Path, manifest_sha256: str)`.
- Produces `preview_staging_manifest(scene: ResolvedScene) -> tuple[dict[str, object], str]`, `stage_scene(scene: ResolvedScene, campaign_root: str | Path) -> StagedScene`, `load_valid_staging(path: str | Path, expected_payload: Mapping[str, object]) -> StagedScene`, and `prepare_pointcloud_gt(scene: ResolvedScene, destination: Path) -> tuple[Path, tuple[int, int, int, int]]`.
- Produces safety helpers `require_descendant(path: str | Path, root: str | Path, *, allow_equal: bool = False) -> Path` and `guarded_remove(path: str | Path, owned_root: str | Path) -> None`; Task 4 reuses these for artifacts/cache cleanup.
- `source_frame_ids` is the only applied selection vector. The staged pipeline always receives `input.sample_stride=1`; no downstream function slices it again.

- [ ] **Step 1: Write RED tests for both KITTI layouts, natural ordering, and one positional slice**

```python
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.window_reference_campaign.config import PresetSceneConfig
from experiments.window_reference_campaign.scenes import (
    ResolvedFrameSelection,
    apply_frame_selection,
    resolve_kitti_layout,
)


def _png(path: Path, colour: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color=(colour, colour, colour)).save(path)
    return path


@pytest.mark.parametrize("official", [False, True])
def test_kitti_resolver_accepts_normalized_and_official_layout(tmp_path, official):
    if official:
        image_dir = tmp_path / "dataset/sequences/04/image_2"
        poses = tmp_path / "dataset/poses/04.txt"
    else:
        image_dir = tmp_path / "04/image_2"
        poses = tmp_path / "04/poses.txt"
    _png(image_dir / "10.png")
    _png(image_dir / "2.png")
    poses.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(poses, np.tile(np.arange(12, dtype=float), (2, 1)))

    layout = resolve_kitti_layout(tmp_path, "04")

    assert layout.image_dir == image_dir.resolve()
    assert layout.poses_path == poses.resolve()
    assert layout.layout_name == ("official" if official else "normalized")


def test_frame_selection_is_applied_once_to_the_explicit_source_vector():
    selected = apply_frame_selection(
        tuple(range(0, 100, 10)),
        PresetSceneConfig(scene_id="fixture", start=1, stop=9, stride=2),
        start_override=None,
        max_frames=3,
        stride_override=None,
    )
    assert selected == ResolvedFrameSelection(
        start=1,
        stop=7,
        stride=2,
        source_frame_ids=(10, 30, 50),
    )
```

The selection contract is positional: `start`, exclusive `stop`, `max_frames`, and `stride` index the preset's explicit base vector. Thus the point-cloud map `[0,10,20,...]` with stride `1` retains its mapped IDs; it does not reinterpret `10` as an additional sampling stride. If a CLI `max_frames` is present, effective stop is `min(preset_stop_or_len, start + max_frames * effective_stride)`.

- [ ] **Step 2: Write RED staging/alignment/reuse/symlink-safety tests**

```python
import json
import os

from experiments.window_reference_campaign.scenes import ResolvedFrameSelection, ResolvedScene
from experiments.window_reference_campaign.staging import stage_scene


def _kitti_scene(tmp_path: Path) -> ResolvedScene:
    root = tmp_path / "data"
    images = tuple(_png(root / "04/image_2" / f"{index:06d}.png", index) for index in range(6))
    np.savetxt(
        root / "04/poses.txt",
        np.arange(6 * 12, dtype=float).reshape(6, 12),
    )
    return ResolvedScene(
        scene_id="kitti-04",
        dataset=DatasetKind.KITTI,
        scene="04",
        slice_id="f000001-000006-s2",
        approved_data_root=root.resolve(),
        source_images=tuple(images[index] for index in (1, 3, 5)),
        selection=ResolvedFrameSelection(1, 6, 2, (1, 3, 5)),
        evaluation_kind=EvaluationKind.INTERNAL_TRAJECTORY,
        poses_path=(root / "04/poses.txt").resolve(),
        prepared_gt_path=None,
        frame_index_map=None,
        expected_gt_shape=None,
    )


def test_staging_uses_one_vector_for_images_pose_rows_and_manifest(tmp_path):
    staged = stage_scene(_kitti_scene(tmp_path), tmp_path / "campaign")

    assert [path.name for path in staged.image_dir.iterdir()] == [
        "000000.png", "000001.png", "000002.png"
    ]
    assert all(path.is_symlink() for path in staged.image_dir.iterdir())
    np.testing.assert_array_equal(
        np.load(Path(staged.manifest_path).parent / "source_frame_ids.npy"),
        [1, 3, 5],
    )
    poses = np.atleast_2d(np.loadtxt(staged.poses_path))
    np.testing.assert_array_equal(
        poses,
        np.arange(6 * 12, dtype=float).reshape(6, 12)[[1, 3, 5]],
    )
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_frame_ids"] == [1, 3, 5]
    assert manifest["pipeline_sample_stride"] == 1


def test_staging_reuses_only_matching_hashes(tmp_path):
    scene = _kitti_scene(tmp_path)
    first = stage_scene(scene, tmp_path / "campaign")
    second = stage_scene(scene, tmp_path / "campaign")
    assert second.manifest_sha256 == first.manifest_sha256
    (second.image_dir / "000001.png").unlink()
    with pytest.raises(ValueError, match="staging file hash mismatch"):
        stage_scene(scene, tmp_path / "campaign")


def test_staging_rejects_symlink_target_outside_approved_data_root(tmp_path):
    scene = _kitti_scene(tmp_path)
    escaped = tmp_path / "outside.png"
    _png(escaped)
    scene = replace(scene, source_images=(escaped, *scene.source_images[1:]))
    with pytest.raises(ValueError, match="approved data root"):
        stage_scene(scene, tmp_path / "campaign")
```

- [ ] **Step 3: Write RED point-map GT tests for external and raw preparation paths**

```python
def test_external_pointcloud_gt_is_selected_by_exact_frame_ids(tmp_path):
    scene = make_pointcloud_scene(tmp_path, ids=(0, 10, 20), requested=(10, 20))
    prepared = scene.prepared_gt_path
    assert prepared is not None
    prepared.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prepared,
        point_maps=np.stack([
            np.full((392, 518, 3), value, dtype=np.float32)
            for value in (0.0, 10.0, 20.0)
        ]),
        valid_mask=np.ones((3, 392, 518), dtype=bool),
        frame_ids=np.array([0, 10, 20], dtype=np.int64),
    )

    staged = stage_scene(scene, tmp_path / "campaign")
    with np.load(staged.pointcloud_gt_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_ids"], [10, 20])
        assert data["point_maps"].shape == (2, 392, 518, 3)
        assert np.all(data["point_maps"][0] == 10.0)


def test_pointcloud_gt_rejects_leading_dimension_or_spatial_mismatch(tmp_path):
    scene = make_pointcloud_scene(tmp_path, ids=(0, 10), requested=(0, 10))
    write_external_gt(
        scene.prepared_gt_path,
        point_maps=np.zeros((1, 391, 518, 3), dtype=np.float32),
        valid_mask=np.ones((1, 391, 518), dtype=bool),
        frame_ids=np.array([0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="point-map GT.*frames|spatial shape"):
        stage_scene(scene, tmp_path / "campaign")
```

`make_pointcloud_scene()` and `write_external_gt()` are test-local helpers defined immediately above these tests. They create literal 640x480 RGB files, a sequence-map JSON, and the exact `ResolvedScene` fields; they do not mock the selection or staging functions.

- [ ] **Step 4: Run the focused RED tests**

Run:

```bash
python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_window_reference_campaign_scenes.py::test_kitti_resolver_accepts_normalized_and_official_layout \
  tests/experiments/test_window_reference_campaign_scenes.py::test_staging_uses_one_vector_for_images_pose_rows_and_manifest \
  tests/experiments/test_window_reference_campaign_scenes.py::test_external_pointcloud_gt_is_selected_by_exact_frame_ids
```

Expected: FAIL during collection because `scenes.py` and `staging.py` do not exist.

- [ ] **Step 5: Implement read-only scene resolution and strict source validation**

```python
# experiments/window_reference_campaign/scenes.py
def resolve_kitti_layout(dataset_root: str | Path, sequence: str) -> KittiLayout:
    root = Path(dataset_root).resolve(strict=True)
    candidates = (
        KittiLayout(root / sequence / "image_2", root / sequence / "poses.txt", "normalized"),
        KittiLayout(
            root / "dataset/sequences" / sequence / "image_2",
            root / "dataset/poses" / f"{sequence}.txt",
            "official",
        ),
    )
    matches = tuple(item for item in candidates if item.image_dir.is_dir() and item.poses_path.is_file())
    if len(matches) != 1:
        expected = "; ".join(f"{item.image_dir} + {item.poses_path}" for item in candidates)
        raise FileNotFoundError(f"KITTI {sequence} requires exactly one supported layout: {expected}")
    return matches[0]


def apply_frame_selection(
    base_frame_ids: Sequence[int],
    selection: PresetSceneConfig,
    *,
    start_override: int | None,
    max_frames: int | None,
    stride_override: int | None,
) -> ResolvedFrameSelection:
    base = tuple(base_frame_ids)
    start = selection.start if start_override is None else start_override
    stride = selection.stride if stride_override is None else stride_override
    configured_stop = len(base) if selection.stop is None else min(selection.stop, len(base))
    stop = configured_stop
    if max_frames is not None:
        stop = min(stop, start + max_frames * stride)
    ids = base[start:stop:stride]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("resolved source frame vector must be non-empty and unique")
    return ResolvedFrameSelection(start, stop, stride, ids)
```

Resolution rules are exact:

- KITTI: natural-sort every supported image, parse the entire pose file as `(N,12)`, require equal counts and finite values, then select by the one vector.
- 7-Scenes: load only the named sequence-map entry, require sorted unique non-negative IDs, map each ID to `frame-%06d.color.png`, and require its `.depth.png` and `.pose.txt` partners.
- NeuralRGBD: load only the named sequence-map entry, map each ID to `images/img{source_frame_id}.png` plus `depth/depth{source_frame_id}.png`, parse `poses.txt` as finite `(-1,4,4)`, and bounds-check every ID.
- Natural sorting is used for discovered KITTI images; mapped point-cloud scenes preserve the checked-in ID vector order.
- A prepared GT candidate that does not exist becomes `None` so staging prepares from raw data. An existing but malformed candidate is an error, never a fallback to raw.

- [ ] **Step 6: Implement atomic, owned staging and exact GT preparation**

```python
# experiments/window_reference_campaign/staging.py
STAGING_SCHEMA_VERSION = 1


def require_descendant(
    path: str | Path,
    root: str | Path,
    *,
    allow_equal: bool = False,
) -> Path:
    target = Path(path).resolve(strict=False)
    owner = Path(root).resolve(strict=False)
    if target == owner and allow_equal:
        return target
    if target == owner or owner not in target.parents:
        raise ValueError(f"path is outside campaign-owned root: {target}")
    return target


def _relative_link(source: Path, target: Path, approved_root: Path) -> None:
    resolved = source.resolve(strict=True)
    require_descendant(resolved, approved_root)
    target.symlink_to(os.path.relpath(resolved, start=target.parent))


def preview_staging_manifest(scene: ResolvedScene) -> tuple[dict[str, object], str]:
    image_records = [
        {
            "ordinal": ordinal,
            "source_frame_id": frame_id,
            "source_path": str(path.resolve(strict=True)),
            "source_sha256": sha256_file(path),
        }
        for ordinal, (frame_id, path) in enumerate(
            zip(scene.selection.source_frame_ids, scene.source_images, strict=True)
        )
    ]
    payload = {
        "schema_version": STAGING_SCHEMA_VERSION,
        "scene_id": scene.scene_id,
        "dataset": scene.dataset.value,
        "scene": scene.scene,
        "slice_id": scene.slice_id,
        "selection": {
            "start": scene.selection.start,
            "stop": scene.selection.stop,
            "stride": scene.selection.stride,
        },
        "source_frame_ids": list(scene.selection.source_frame_ids),
        "pipeline_sample_stride": 1,
        "images": image_records,
    }
    return payload, canonical_sha256(payload)
```

Build staging in `{campaign_root}/prepared/{dataset}/{scene_id}/{slice_id}.tmp-{pid}-{nonce}`, then atomically rename it to the final directory. Write images as relative symlinks named `000000.png` onward; write `source_frame_ids.npy`, `poses.txt`, `ground_truth.npz`, and `staging.json` through temporary siblings. The staging manifest records SHA-256 for source files and staged pose/GT payloads, but records no credentials or external URL.

For external GT, accept frame IDs from NPZ key `frame_ids`, sibling `source_frame_ids.npy`, or sibling `frame_ids.npy` in that precedence order; reject disagreement when more than one is present. Require `point_maps.shape == (N,H,W,3)`, `valid_mask.shape == (N,H,W)`, finite selected point values where mask is true, unique integer IDs, and configured `(H,W) == (392,518)`. Create a source-ID-to-position map, select exactly `scene.selection.source_frame_ids`, then write campaign-owned NPZ containing all three arrays.

For raw 7-Scenes/NeuralRGBD preparation, port only the relevant behavior of `tools/disposable_experiments/preflight_experiments.py:310-473`: instantiate the existing dataset class with metadata limited to the one scene, request exactly the selected IDs, verify returned `data["ind"]`, `image_paths`, `pointclouds`, and `valid_mask`, then atomically save the same three-key NPZ. Do not reuse its historical SHA gate or method tables.

Existing staging is reusable only when `staging.json` equals `preview_staging_manifest()` plus generated-file hashes and every staged link resolves to its recorded source digest. A missing/mutated staging file raises before any cleanup; the runner may later remove only the whole proven-owned staging directory after a completed scene.

- [ ] **Step 7: Run GREEN plus manifest/pipeline regressions**

Run:

```bash
python -m pytest -p no:cacheprovider -q tests/experiments/test_window_reference_campaign_scenes.py
python -m pytest -p no:cacheprovider -q \
  tests/test_pipeline_manifest.py \
  tests/disposable_experiments/test_preflight_experiments.py::test_nrgbd_preparation_reads_only_requested_scene
```

Expected: all tests PASS; official and normalized layouts produce the same selected IDs, and the pipeline manifest regression retains natural ordering.

- [ ] **Step 8: REFACTOR atomic file operations and commit**

Consolidate JSON, NumPy `.npy`, and compressed `.npz` writes behind `_atomic_path(target: Path, writer: Callable[[Path], None]) -> Path`; ensure the temporary suffix still lets NumPy write to the exact temporary filename (open the temporary file object rather than letting NumPy append an extension). Rerun Step 7, then commit:

```bash
git add experiments/window_reference_campaign/scenes.py \
  experiments/window_reference_campaign/staging.py \
  tests/experiments/test_window_reference_campaign_scenes.py
git commit -m "feat: add aligned campaign staging"
```

### Task 3: Overlap-Aware Diagnostics and Finite Compact Result Schema

**Files:**
- Create: `experiments/window_reference_campaign/diagnostics.py`
- Create: `experiments/window_reference_campaign/results.py`
- Create: `tests/experiments/test_window_reference_campaign_results.py`

**Interfaces:**
- Consumes: Task 1 `RunIdentitySeed`, `RunIdentity`, `complete_identity()`; artifact diagnostics payloads produced by `ReconstructionDiagnostics.to_payload()`; existing `PredictionStoreStats` fields.
- Produces `DistributionSummary(count: int, minimum: float, maximum: float, mean: float, median: float)` and `finite_distribution(values: Sequence[int | float]) -> DistributionSummary | None`.
- Produces `DiagnosticsSummary(unique_frame_count: int, window_frame_observation_count: int, window_count: int, region_count: DistributionSummary, refinement_enabled: bool, refinement_state: str, keyframe_count: DistributionSummary | None, keyframe_index_histogram: Mapping[str, int] | None, keyframe_index_records: tuple[str, ...] | None, coverage_ratio: DistributionSummary | None, regions_before: DistributionSummary | None, regions_after: DistributionSummary | None, region_reduction_absolute_total: int | None, region_reduction_relative: DistributionSummary | None, candidate_edge_total: int | None, accepted_edge_total: int | None, conflict_edge_total: int | None, projected_sample_total: int | None, occluded_sample_total: int | None, depth_rejected_sample_total: int | None, applied_frame_count: int | None, applied_frame_rate: float | None, fallback_reason_histogram: Mapping[str, int] | None)`.
- Produces `aggregate_diagnostics(payload: Mapping[str, object], *, refinement_enabled: bool) -> DiagnosticsSummary`; it consumes `stage_timings_ms`, `segmentation_summaries`, and `mode_scalars.window_count` but never mutates artifact diagnostics.
- Produces enums `RunStatus(SUCCEEDED, FAILED)` and `FailureStage(PREFLIGHT, STAGING, RECONSTRUCTION, EVALUATION, COMPACTION, CLEANUP)`; dataclasses `CacheStats(ordinary_hits: int, ordinary_misses: int, corrupt_count: int, read_ms: float, write_ms: float, saved_window_count: int, stored_bytes: int, events: tuple[Mapping[str, object], ...])`, `RunTimings(reconstruction_s: float | None, evaluation_s: float | None)`, `RunError(error_type: str, message: str)`, and `RunRecord`. `CacheStats.from_store_stats(stats: PredictionStoreStats) -> CacheStats` copies and freezes all counters/events.
- `RunRecord` fields are exactly `schema_version`, `run_id`, `identity_seed`, `identity`, `status`, `attempt`, `failure_stage`, `started_at`, `finished_at`, `frame_count`, `window_count`, `cache_policy`, `cache_stats`, `timings`, `diagnostics`, `evaluation_kind`, `evaluation_metrics`, `artifact_manifest_sha256`, and `error`. A successful record requires a complete identity, diagnostics, finite required metrics for its evaluation kind, no failure/error, and positive frame/attempt counts.
- Produces `atomic_json(path: str | Path, payload: Mapping[str, object]) -> Path`, `write_run_record(path: str | Path, record: RunRecord) -> Path`, `read_run_record(path: str | Path) -> RunRecord`, `load_valid_completed_run(path: str | Path, expected_seed: RunIdentitySeed) -> RunRecord`, `write_campaign_metadata(path: str | Path, payload: Mapping[str, object]) -> Path`, and `redact_argv(argv: Sequence[str]) -> tuple[str, ...]`.
- Produces `SummaryPaths(runs_csv: Path, diagnostics_csv: Path, trajectory_csv: Path, pointcloud_csv: Path, summary_json: Path, failures_json: Path)` and `write_summaries(records: Sequence[RunRecord], expected_seeds: Mapping[tuple[str, str, str, str], RunIdentitySeed], output_dir: str | Path) -> SummaryPaths`; expected keys are `(dataset, scene, slice_id, run_id)`.

- [ ] **Step 1: Write RED tests for overlap semantics and disabled/refinement states**

```python
from experiments.window_reference_campaign.diagnostics import aggregate_diagnostics


def _artifact_diagnostics(observations):
    return {
        "stage_timings_ms": {"reconstruction": 250.0},
        "segmentation_summaries": observations,
        "candidate_count": 0,
        "constraint_count": 0,
        "mode_scalars": {"window_count": 2},
    }


def test_overlap_observations_are_not_silently_deduplicated():
    summary = aggregate_diagnostics(
        _artifact_diagnostics([
            {
                "window_index": 0, "frame_index": 0, "region_count": 3,
                "window_reference_applied": False,
                "window_reference_keyframes": "0",
                "window_reference_keyframe_count": 1,
                "window_reference_is_keyframe": True,
                "window_reference_coverage_ratio": 0.5,
                "window_reference_regions_before": 3,
                "window_reference_regions_after": 3,
                "window_reference_candidate_edges": 0,
                "window_reference_accepted_edges": 0,
                "window_reference_conflict_edges": 0,
                "window_reference_projected_samples": 8,
                "window_reference_occluded_samples": 2,
                "window_reference_depth_rejected_samples": 1,
                "window_reference_fallback": "none",
            },
            {
                "window_index": 0, "frame_index": 1, "region_count": 2,
                "window_reference_applied": True,
                "window_reference_keyframes": "0",
                "window_reference_keyframe_count": 1,
                "window_reference_is_keyframe": False,
                "window_reference_coverage_ratio": 1.0,
                "window_reference_regions_before": 3,
                "window_reference_regions_after": 2,
                "window_reference_candidate_edges": 1,
                "window_reference_accepted_edges": 1,
                "window_reference_conflict_edges": 0,
                "window_reference_projected_samples": 9,
                "window_reference_occluded_samples": 1,
                "window_reference_depth_rejected_samples": 0,
                "window_reference_fallback": "none",
            },
            {
                "window_index": 1, "frame_index": 1, "region_count": 3,
                "window_reference_applied": False,
                "window_reference_keyframes": "1",
                "window_reference_keyframe_count": 1,
                "window_reference_is_keyframe": True,
                "window_reference_coverage_ratio": 0.5,
                "window_reference_regions_before": 3,
                "window_reference_regions_after": 3,
                "window_reference_candidate_edges": 0,
                "window_reference_accepted_edges": 0,
                "window_reference_conflict_edges": 0,
                "window_reference_projected_samples": 7,
                "window_reference_occluded_samples": 3,
                "window_reference_depth_rejected_samples": 2,
                "window_reference_fallback": "none",
            },
        ]),
        refinement_enabled=True,
    )
    assert summary.unique_frame_count == 2
    assert summary.window_frame_observation_count == 3
    assert summary.window_count == 2
    assert summary.applied_frame_count == 1
    assert summary.applied_frame_rate == pytest.approx(1 / 3)
    assert summary.region_reduction_absolute_total == 1
    assert summary.projected_sample_total == 24
    assert summary.keyframe_index_histogram == {"0": 2, "1": 1}


def test_disabled_refinement_is_not_fallback_or_zero_keyframe_performance():
    summary = aggregate_diagnostics(
        _artifact_diagnostics([
            {"window_index": 0, "frame_index": 0, "region_count": 4},
            {"window_index": 0, "frame_index": 1, "region_count": 3},
        ]),
        refinement_enabled=False,
    )
    assert summary.refinement_state == "not_applicable"
    assert summary.keyframe_count is None
    assert summary.coverage_ratio is None
    assert summary.applied_frame_count is None
    assert summary.applied_frame_rate is None
    assert summary.fallback_reason_histogram is None
```

- [ ] **Step 2: Write RED tests for exact resume validation, finite metrics, atomic output, and summaries**

```python
import csv
import json
from dataclasses import replace

from experiments.window_reference_campaign.results import (
    load_valid_completed_run,
    write_run_record,
    write_summaries,
)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_completed_record_rejects_non_finite_required_metric(tmp_path, bad):
    record = make_success_record(
        evaluation_kind="internal_trajectory",
        evaluation_metrics={
            "internal_ate_rmse_m": bad,
            "internal_rpe_translation_rmse_m": 0.1,
            "internal_rpe_rotation_rmse_deg": 0.2,
            "internal_matched_frame_count": 80,
        },
    )
    with pytest.raises(ValueError, match="finite"):
        write_run_record(tmp_path / "run.json", record)
    assert not (tmp_path / "run.json").exists()


def test_resume_requires_exact_seed_and_complete_prediction_key(tmp_path):
    record = make_success_record()
    path = write_run_record(tmp_path / "run.json", record)
    assert load_valid_completed_run(path, record.identity_seed) == record
    wrong = replace(record.identity_seed, overlap=29)
    with pytest.raises(ValueError, match="identity"):
        load_valid_completed_run(path, wrong)


def test_summary_writes_required_atomic_csv_and_json(tmp_path):
    records = tuple(make_six_success_records())
    paths = write_summaries(records, make_six_expected_seeds(), tmp_path / "summary")
    assert all(path.is_file() for path in paths.__dict__.values())
    with paths.runs_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[0]["run_id"] == "depth__wr-off"
    assert rows[1]["run_id"] == "depth__wr-on"
    assert json.loads(paths.summary_json.read_text())["completed_runs"] == 6
    assert json.loads(paths.failures_json.read_text()) == {"failures": []}
```

Define `make_identity_seed()`, `make_success_record()`, `make_six_success_records()`, and `make_six_expected_seeds()` in this test module with literal valid Task 1 dataclasses. `make_six_expected_seeds()` maps `(dataset, scene, slice_id, run_id)` to each record's exact seed. `make_success_record()` completes its seed with prediction key `"a" * 64`, uses a two-observation `DiagnosticsSummary`, `CacheStats(1, 0, 0, 0.1, 0.2, 1, 1024, ())`, and defaults to `evaluation_kind="none"`, `evaluation_metrics=None`.

- [ ] **Step 3: Run the focused RED nodes**

Run:

```bash
python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_window_reference_campaign_results.py::test_overlap_observations_are_not_silently_deduplicated \
  tests/experiments/test_window_reference_campaign_results.py::test_disabled_refinement_is_not_fallback_or_zero_keyframe_performance \
  tests/experiments/test_window_reference_campaign_results.py::test_resume_requires_exact_seed_and_complete_prediction_key
```

Expected: FAIL during collection because `diagnostics.py` and `results.py` do not exist.

- [ ] **Step 4: Implement finite distributions and diagnostics aggregation**

```python
# experiments/window_reference_campaign/diagnostics.py
def finite_distribution(
    values: Sequence[int | float],
) -> DistributionSummary | None:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        return None
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("diagnostic distribution values must be finite")
    return DistributionSummary(
        count=len(normalized),
        minimum=min(normalized),
        maximum=max(normalized),
        mean=statistics.fmean(normalized),
        median=statistics.median(normalized),
    )


def aggregate_diagnostics(
    payload: Mapping[str, object],
    *,
    refinement_enabled: bool,
) -> DiagnosticsSummary:
    observations = _validated_observations(payload.get("segmentation_summaries"))
    frame_ids = tuple(_integer(item, "frame_index") for item in observations)
    windows = {_integer(item, "window_index") for item in observations}
    regions = tuple(_nonnegative_integer(item, "region_count") for item in observations)
    region_distribution = finite_distribution(regions)
    if region_distribution is None:
        raise ValueError("diagnostics require at least one region observation")
    common = {
        "unique_frame_count": len(set(frame_ids)),
        "window_frame_observation_count": len(observations),
        "window_count": _window_count(payload, windows),
        "region_count": region_distribution,
        "refinement_enabled": refinement_enabled,
    }
    if not refinement_enabled:
        return DiagnosticsSummary(
            **common,
            refinement_state="not_applicable",
            keyframe_count=None,
            keyframe_index_histogram=None,
            keyframe_index_records=None,
            coverage_ratio=None,
            regions_before=None,
            regions_after=None,
            region_reduction_absolute_total=None,
            region_reduction_relative=None,
            candidate_edge_total=None,
            accepted_edge_total=None,
            conflict_edge_total=None,
            projected_sample_total=None,
            occluded_sample_total=None,
            depth_rejected_sample_total=None,
            applied_frame_count=None,
            applied_frame_rate=None,
            fallback_reason_histogram=None,
        )
    return _aggregate_enabled(common, observations)
```

`_aggregate_enabled()` requires every shipped `window_reference_*` field on every observation and validates exact scalar types. Parse `window_reference_keyframes` as an empty string or comma-separated non-negative decimal indices; retain the original strings in observation order and count indices into a sorted string-key histogram. Calculate absolute reduction as `sum(before-after)` and per-observation relative reduction as `(before-after)/before`. `refinement_state` is `"applied"` if any observation applied a merge, otherwise `"fallback"`. Count fallback strings only for non-keyframe observations, including `"none"` for successfully supported observations; keyframes do not inflate the fallback histogram.

- [ ] **Step 5: Implement compact records, identity equality, argv redaction, and atomic writers**

```python
# experiments/window_reference_campaign/results.py
RUN_SCHEMA_VERSION = 1
POINTCLOUD_METRICS = (
    "accuracy_mean_m", "accuracy_median_m",
    "completion_mean_m", "completion_median_m",
    "normal_consistency_mean", "normal_consistency_median",
    "chamfer_l1_m",
    "precision_1cm", "recall_1cm", "fscore_1cm",
    "precision_2cm", "recall_2cm", "fscore_2cm",
    "precision_5cm", "recall_5cm", "fscore_5cm",
)
TRAJECTORY_METRICS = (
    "internal_ate_rmse_m",
    "internal_rpe_translation_rmse_m",
    "internal_rpe_rotation_rmse_deg",
    "internal_matched_frame_count",
)


def load_valid_completed_run(
    path: str | Path,
    expected_seed: RunIdentitySeed,
) -> RunRecord:
    record = read_run_record(path)
    if record.status is not RunStatus.SUCCEEDED or record.identity is None:
        raise ValueError("run is not a completed success")
    expected = complete_identity(expected_seed, record.identity.prediction_key)
    if record.identity_seed != expected_seed or record.identity != expected:
        raise ValueError("run identity does not exactly match expected identity")
    _validate_success(record)
    return record


def redact_argv(argv: Sequence[str]) -> tuple[str, ...]:
    redacted: list[str] = []
    hide_next = False
    for raw in argv:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        key = raw.split("=", 1)[0].lower()
        if any(word in key for word in ("password", "token", "cookie", "credential", "secret")):
            if "=" in raw:
                redacted.append(f"{raw.split('=', 1)[0]}=<redacted>")
            else:
                redacted.append(raw)
                hide_next = True
            continue
        redacted.append(_redact_url_userinfo(raw))
    return tuple(redacted)
```

`_validate_success()` rejects non-finite timings/cache counters/diagnostic distributions and applies these evaluation rules: `none` requires `evaluation_metrics is None`; `pointcloud` requires exactly `POINTCLOUD_METRICS` with all finite values and precision/recall/F-score in `[0,1]`; `internal_trajectory` requires exactly `TRAJECTORY_METRICS`, three finite non-negative floats, and integer `internal_matched_frame_count >= 2`. Failed records require `identity=None`, `diagnostics=None`, a non-null `failure_stage`/`RunError`, and may keep a valid artifact digest only when failure stage is evaluation or compaction.

Use one `_atomic_text(path: Path, text: str) -> Path` helper with a same-directory `NamedTemporaryFile`, `flush()`, `os.fsync()`, and `Path.replace()`. Every JSON call uses sorted keys, indentation, `allow_nan=False`, and a final newline. Validate the serialized temporary payload by reading it back before replacement.

- [ ] **Step 6: Implement strict summary coverage and exact CSV projections**

`write_summaries()` first indexes expected `(dataset, scene, slice_id, run_id)` tuples. Reject duplicate records, unexpected records, missing successful records from `summary.json` coverage, identity seed mismatch, and per-scene prediction-key disagreement (ignore `cache_policy=off` only for cache hit assertions, not identity-key equality). It writes:

- `runs.csv`: identity axes, status/attempt/failure, frame/window counts, cache policy/counters, reconstruction/evaluation seconds, per-frame and per-window seconds, prediction key, artifact digest.
- `diagnostics.csv`: all `DiagnosticsSummary` scalars/distribution fields, JSON-encoded sorted histograms/records, and explicit `unique_frame_count` plus `window_frame_observation_count`.
- `trajectory.csv`: identity columns plus the four `internal_*` metrics; no official KITTI column names.
- `pointcloud.csv`: identity columns plus all 16 point-cloud metrics.
- `summary.json`: schema, expected/completed/failed counts, dataset/scene/run coverage, shared prediction key per scene, generated filenames, and no historical pooling.
- `failures.json`: failed/missing run IDs, attempts, stages, and redacted error messages.

Write CSV through `io.StringIO(newline="")` and `_atomic_text`; map `None` to `""`, booleans to lowercase `true`/`false`, and mappings/tuples to sorted compact JSON. No CSV cell receives `nan`, `inf`, or `-inf`.

- [ ] **Step 7: Run GREEN and artifact/diagnostic regressions**

Run:

```bash
python -m pytest -p no:cacheprovider -q tests/experiments/test_window_reference_campaign_results.py
python -m pytest -p no:cacheprovider -q \
  tests/test_reconstruction_artifacts.py \
  tests/reconstruction/test_no_loop_mode.py \
  tests/test_window_reference_refinement.py
```

Expected: all tests PASS; existing artifact diagnostics remain byte-schema compatible and the campaign summary distinguishes 2 unique frames from 3 observations.

- [ ] **Step 8: REFACTOR scalar validation and commit**

Consolidate repeated finite/integer/range checks into `_finite_number(value, field, *, minimum=None, maximum=None, integer=False) -> int | float`, rerun Step 7, then commit:

```bash
git add experiments/window_reference_campaign/diagnostics.py \
  experiments/window_reference_campaign/results.py \
  tests/experiments/test_window_reference_campaign_results.py
git commit -m "feat: add compact campaign results"
```

### Task 4: Serial Runner, Shared Cache Transition, Attempts, Resume, and Safe Cleanup

**Files:**
- Modify: `experiments/window_reference_campaign/matrix.py`
- Create: `experiments/window_reference_campaign/runner.py`
- Create: `tests/experiments/test_window_reference_campaign_runner.py`

**Interfaces:**
- Consumes: Task 1 `CampaignPlan`, `PlannedRun`, `RunIdentitySeed`, `complete_identity()`; Task 2 `ResolvedScene`, `StagedScene`, `stage_scene()`, `guarded_remove()`; Task 3 `RunRecord`, `CacheStats`, `RunTimings`, `RunError`, `aggregate_diagnostics()`, `load_valid_completed_run()`, `write_run_record()`, `write_summaries()`; existing `load_pipeline_config()`, `PipelineRunner`, `PipelineDependencies`, `OrdinaryPredictionStore`, and `load_reconstruction_artifact()`.
- Adds Task 1's deferred runtime constructor to `matrix.py`: `build_identity_seed(*, loaded: LoadedCampaignConfig, planned: PlannedRun, frame_start: int, frame_stop: int, frame_stride: int, staged_manifest_sha256: str, source_commit: str, source_dirty: bool, checkpoint_sha256: str) -> RunIdentitySeed`; it copies all fixed axes and the complete ten-key window-reference mapping from `loaded`/`planned`.
- Produces `PipelineExecution(artifact_dir: Path, artifact_manifest_sha256: str, prediction_key: str, frame_count: int, diagnostics_payload: Mapping[str, object], cache_stats: CacheStats)`.
- Produces `EvaluationOutput(evaluation_kind: EvaluationKind, metrics: Mapping[str, int | float] | None, output_paths: tuple[Path, ...])`; Task 5's adapters return this exact type.
- Produces `RunRequest(planned: PlannedRun, staged: StagedScene, identity_seed: RunIdentitySeed, run_dir: Path, attempt_dir: Path, artifact_dir: Path, cache_root: Path, cache_mode: PredictionCacheMode, log_path: Path)`.
- Produces `RunnerDependencies(execute_pipeline: Callable[[RunRequest, LoadedCampaignConfig], PipelineExecution], evaluate_artifact: Callable[[RunRequest, PipelineExecution, LoadedCampaignConfig], EvaluationOutput], stage_scene: Callable[[ResolvedScene, Path], StagedScene], validate_artifact: Callable[[Path, RunIdentitySeed], tuple[str, str]], monotonic: Callable[[], float], utc_now: Callable[[], datetime])`.
- Produces `CampaignOutcome(records: tuple[RunRecord, ...], failures: tuple[RunRecord, ...], skipped_run_ids: tuple[str, ...], exit_code: int)` and `run_campaign(loaded: LoadedCampaignConfig, plan: CampaignPlan, resolved_scenes: Mapping[str, ResolvedScene], *, resume: bool, failure_policy: FailurePolicy, keep_artifacts: bool, dependencies: RunnerDependencies | None = None) -> CampaignOutcome`.
- Produces public helpers `select_cache_mode(requested: CachePolicy, *, first_pending: bool, known_prediction_key: str | None, cache_root: Path, window_count: int) -> PredictionCacheMode`, `cache_entry_complete(cache_root: Path, prediction_key: str, window_count: int) -> bool`, `next_attempt(run_dir: Path) -> tuple[int, Path]`, `validate_artifact_for_seed(artifact_dir: Path, seed: RunIdentitySeed) -> tuple[str, str]`, and `execute_pipeline(request: RunRequest, loaded: LoadedCampaignConfig) -> PipelineExecution`.

- [ ] **Step 1: Write RED cache-transition and resume-recovery tests**

```python
from collections import Counter

from experiments.window_reference_campaign.runner import (
    PipelineExecution,
    RunnerDependencies,
    run_campaign,
)


def _execution(request, *, prediction_key="a" * 64):
    request.artifact_dir.mkdir(parents=True)
    (request.artifact_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    if request.cache_mode.value != "off":
        entry = request.cache_root / "v2" / prediction_key
        (entry / "windows").mkdir(parents=True, exist_ok=True)
        (entry / "manifest.json").write_text("{}\n", encoding="utf-8")
        (entry / "sequence.json").write_text("{}\n", encoding="utf-8")
        (entry / "windows/000000.pt").write_bytes(b"fixture-window")
        (entry / "complete.json").write_text(
            json.dumps({"schema_version": 2, "key": prediction_key, "window_count": 1}),
            encoding="utf-8",
        )
    return PipelineExecution(
        artifact_dir=request.artifact_dir,
        artifact_manifest_sha256="b" * 64,
        prediction_key=prediction_key,
        frame_count=40,
        diagnostics_payload=enabled_diagnostics_payload(),
        cache_stats=CacheStats(0, 1, 0, 0.0, 1.0, 1, 1024, ()),
    )


def _no_gt_evaluation(request, execution, loaded):
    return EvaluationOutput(EvaluationKind.NONE, None, ())


def test_first_pending_uses_auto_then_all_remaining_use_readonly(tmp_path):
    loaded, plan, scenes = make_runner_fixture(tmp_path, preset="synthetic-smoke")
    modes = []

    def execute(request, loaded_config):
        modes.append(request.cache_mode.value)
        return _execution(request)

    outcome = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=RunnerDependencies.for_tests(execute, _no_gt_evaluation),
    )

    assert outcome.exit_code == 0
    assert modes == ["auto", "readonly", "readonly", "readonly", "readonly", "readonly"]
    assert {record.identity.prediction_key for record in outcome.records} == {"a" * 64}


def test_resume_with_cleaned_cache_returns_first_remaining_run_to_auto(tmp_path):
    loaded, plan, scenes = make_runner_fixture(tmp_path, preset="synthetic-smoke")
    write_valid_prior_records(loaded, plan.runs[:2], scenes, prediction_key="a" * 64)
    modes = []

    def execute(request, loaded_config):
        modes.append(request.cache_mode.value)
        return _execution(request)

    outcome = run_campaign(
        loaded,
        plan,
        scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=RunnerDependencies.for_tests(execute, _no_gt_evaluation),
    )

    assert outcome.skipped_run_ids == ("depth__wr-off", "depth__wr-on")
    assert modes == ["auto", "readonly", "readonly", "readonly"]


def test_fake_prediction_provider_builds_once_and_replays_same_identity_five_times(tmp_path):
    loaded, plan, scenes = make_runner_fixture(tmp_path, preset="synthetic-smoke")

    class FakePredictionProvider:
        key = "c" * 64
        inference_calls = 0
        replay_calls = 0

        def execute(self, request):
            if request.cache_mode is PredictionCacheMode.AUTO:
                self.inference_calls += 1
            elif request.cache_mode is PredictionCacheMode.READONLY:
                self.replay_calls += 1
            return _execution(request, prediction_key=self.key)

    provider = FakePredictionProvider()
    outcome = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=FailurePolicy.KEEP_GOING,
        keep_artifacts=True,
        dependencies=RunnerDependencies.for_tests(
            lambda request, loaded_config: provider.execute(request),
            _no_gt_evaluation,
        ),
    )

    assert outcome.exit_code == 0
    assert provider.inference_calls == 1
    assert provider.replay_calls == 5
    assert {record.identity.prediction_key for record in outcome.records} == {provider.key}
```

`make_runner_fixture()`, `enabled_diagnostics_payload()`, and `write_valid_prior_records()` are literal test-local builders using the Task 1–3 public dataclasses. `RunnerDependencies.for_tests()` is a real classmethod implemented with deterministic `monotonic=itertools.count(step=1).__next__`, fixed UTC timestamps, an identity staging function that returns the supplied `StagedScene`, and a default fixture-artifact validator that returns the execution's recorded key/digest; it does not bypass runner policy.

- [ ] **Step 2: Write RED tests for fail-fast, keep-going, attempts, and artifact-only evaluation resume**

```python
@pytest.mark.parametrize(
    ("policy", "expected_calls"),
    [(FailurePolicy.FAIL_FAST, 1), (FailurePolicy.KEEP_GOING, 6)],
)
def test_failure_policy_preserves_attempt_and_returns_nonzero(
    tmp_path, policy, expected_calls
):
    loaded, plan, scenes = make_runner_fixture(tmp_path, preset="synthetic-smoke")
    calls = []

    def execute(request, loaded_config):
        calls.append(request.planned.run_id)
        request.log_path.write_text("actionable failure\n", encoding="utf-8")
        if request.planned.run_id == "depth__wr-off":
            raise RuntimeError("fixture reconstruction failed")
        return _execution(request)

    outcome = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=policy,
        keep_artifacts=True,
        dependencies=RunnerDependencies.for_tests(execute, _no_gt_evaluation),
    )

    assert outcome.exit_code == 1
    assert len(calls) == expected_calls
    failed = outcome.failures[0]
    assert failed.attempt == 1
    assert failed.failure_stage.value == "reconstruction"
    attempt = run_path(loaded, plan.runs[0]) / "attempts/0001/stdout.log"
    assert attempt.read_text(encoding="utf-8") == "actionable failure\n"


def test_evaluation_failure_resumes_from_valid_artifact_without_reconstruction(tmp_path):
    loaded, plan, scenes = make_runner_fixture(tmp_path, preset="synthetic-smoke")
    counts = Counter()

    def execute(request, loaded_config):
        counts["reconstruct"] += 1
        return _execution(request)

    def fail_evaluation(request, execution, loaded_config):
        counts["evaluate"] += 1
        if request.planned.run_id == "depth__wr-off" and counts["evaluate"] == 1:
            raise RuntimeError("evaluation interrupted")
        return _no_gt_evaluation(request, execution, loaded_config)

    first = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=True,
        dependencies=RunnerDependencies.for_tests(execute, fail_evaluation),
    )
    assert first.exit_code == 1
    assert counts == Counter(reconstruct=1, evaluate=1)

    second = run_campaign(
        loaded, plan, scenes,
        resume=True,
        failure_policy=FailurePolicy.FAIL_FAST,
        keep_artifacts=True,
        dependencies=RunnerDependencies.for_tests(execute, fail_evaluation),
    )
    assert second.exit_code == 0
    assert counts["reconstruct"] == 6
    assert counts["evaluate"] == 7
```

For this test only, `RunnerDependencies.for_tests()` receives a `validate_artifact` callback that returns the fixture manifest's prediction key/digest. Production uses `validate_artifact_for_seed()` and the existing artifact loader.

- [ ] **Step 3: Write RED tests for refresh/readonly/off, cache corruption visibility, and cleanup guards**

```python
@pytest.mark.parametrize(
    ("requested", "complete", "expected"),
    [
        (CachePolicy.AUTO, False, "auto"),
        (CachePolicy.AUTO, True, "readonly"),
        (CachePolicy.REFRESH, True, "refresh"),
        (CachePolicy.READONLY, False, "readonly"),
        (CachePolicy.OFF, False, "off"),
    ],
)
def test_cache_policy_is_explicit(tmp_path, requested, complete, expected):
    key = "a" * 64
    cache = tmp_path / "cache"
    if complete:
        write_cache_completion(cache, key, window_count=2)
    actual = select_cache_mode(
        requested,
        first_pending=True,
        known_prediction_key=key,
        cache_root=cache,
        window_count=2,
    )
    assert actual.value == expected


def test_cleanup_rejects_external_data_weights_and_campaign_root(tmp_path):
    campaign = tmp_path / "campaign"
    work = campaign / "work"
    artifact = campaign / "runs/kitti/04/depth__wr-off/artifact"
    artifact.mkdir(parents=True)
    guarded_remove(artifact, campaign)
    assert not artifact.exists()
    for forbidden in (tmp_path / "data", tmp_path / "weights", campaign):
        forbidden.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="outside campaign-owned root"):
            guarded_remove(forbidden, campaign)
```

- [ ] **Step 4: Run focused RED nodes**

Run:

```bash
python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_window_reference_campaign_runner.py::test_first_pending_uses_auto_then_all_remaining_use_readonly \
  tests/experiments/test_window_reference_campaign_runner.py::test_resume_with_cleaned_cache_returns_first_remaining_run_to_auto \
  tests/experiments/test_window_reference_campaign_runner.py::test_failure_policy_preserves_attempt_and_returns_nonzero \
  tests/experiments/test_window_reference_campaign_runner.py::test_evaluation_failure_resumes_from_valid_artifact_without_reconstruction
```

Expected: FAIL during collection because `runner.py` does not exist.

- [ ] **Step 5: Implement deterministic paths, attempts, cache state, and artifact validation**

Use exactly this owned layout for each planned run:

```python
def run_directory(campaign_root: Path, planned: PlannedRun) -> Path:
    return (
        campaign_root
        / "runs"
        / planned.dataset.value
        / planned.slice_id
        / planned.variant.run_id
    )


def scene_cache_root(campaign_root: Path, planned: PlannedRun) -> Path:
    return campaign_root / "work/cache" / planned.dataset.value / planned.scene_id / planned.slice_id


def next_attempt(run_dir: Path) -> tuple[int, Path]:
    attempts = run_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name) for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()]
    number = max(existing, default=0) + 1
    target = attempts / f"{number:04d}"
    target.mkdir(exist_ok=False)
    return number, target
```

`cache_entry_complete()` reads only `{cache_root}/v2/{prediction_key}/complete.json`, requires `{schema_version: 2, key: prediction_key, window_count: expected_window_count}`, and requires `manifest.json`, `sequence.json`, and `windows/%06d.pt` for every integer in `range(expected_window_count)`. Malformed JSON returns `False`, not a hit. The existing store remains authoritative during execution and records corruptions; the campaign probe never repairs or deletes it.

`select_cache_mode()` rules are:

```python
if requested is CachePolicy.OFF:
    return PredictionCacheMode.OFF
if requested is CachePolicy.READONLY:
    return PredictionCacheMode.READONLY
if first_pending and requested is CachePolicy.REFRESH:
    return PredictionCacheMode.REFRESH
if known_prediction_key and cache_entry_complete(cache_root, known_prediction_key, window_count):
    return PredictionCacheMode.READONLY
return PredictionCacheMode.AUTO
```

`validate_artifact_for_seed()` calls `load_reconstruction_artifact()`, reads `manifest.json` and `resolved_reconstruction.yaml`, verifies digests through the loader, and checks: segmentation method, `no_loop`, prediction key, checkpoint SHA, Git commit, `input.sample_stride=1`, `75/30`, atomic conservative, refinement enabled flag, and all ten resolved window-reference values. It returns `(prediction_key, sha256(manifest.json))`. It never treats directory existence as validity.

- [ ] **Step 6: Implement the `PipelineRunner` adapter without copying its responsibilities**

```python
def execute_pipeline(
    request: RunRequest,
    loaded: LoadedCampaignConfig,
) -> PipelineExecution:
    stores: list[OrdinaryPredictionStore] = []

    def build_store(**kwargs):
        store = OrdinaryPredictionStore(**kwargs)
        stores.append(store)
        return store

    overrides = (
        f"input.image_dir={request.staged.image_dir}",
        "input.sample_stride=1",
        f"model.checkpoint={loaded.config.storage.checkpoint}",
        f"model.inference_device=cuda:{loaded.config.runtime.gpu}",
        f"model.process_device={loaded.config.runtime.process_device}",
        f"model.dtype={loaded.config.runtime.model_dtype}",
        "window.size=75",
        "window.overlap=30",
        f"segmentation.method={request.planned.variant.segmentation_method.value}",
        "segmentation.atomic.split_mode=conservative",
        f"segmentation.window_reference.enabled={str(request.planned.variant.window_reference_enabled).lower()}",
        *_window_reference_overrides(loaded.config.window_reference),
        "reconstruction.mode=no_loop",
        f"prediction_cache.root={request.cache_root}",
        f"prediction_cache.mode={request.cache_mode.value}",
        f"output.scene_name={request.planned.slice_id}",
        f"output.cache_dir={request.run_dir / 'legacy-cache-unused'}",
        f"output.result_dir={request.run_dir}",
    )
    pipeline_config = load_pipeline_config(loaded.config.pipeline_config, overrides)
    dependencies = PipelineDependencies(build_prediction_store=build_store)
    pipeline = PipelineRunner(
        pipeline_config,
        dependencies=dependencies,
        artifact_output_dir=request.artifact_dir,
    )
    with request.log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            artifact = pipeline.run()
    if len(stores) != 1:
        raise RuntimeError("pipeline must construct exactly one prediction store")
    store = stores[0]
    return PipelineExecution(
        artifact_dir=request.artifact_dir,
        artifact_manifest_sha256=sha256_file(request.artifact_dir / "manifest.json"),
        prediction_key=artifact.prediction_key,
        frame_count=len(artifact.frame_ids),
        diagnostics_payload=artifact.diagnostics.to_payload(),
        cache_stats=CacheStats.from_store_stats(store.stats),
    )
```

Do not catch `KeyboardInterrupt`, `SystemExit`, or other `BaseException`; attempt/log/cache state remains inspectable on SIGINT/SIGTERM. Catch `Exception` at the run boundary, flush the log, atomically write a failed `run.json`, and apply the selected failure policy.

- [ ] **Step 7: Implement resume ordering, evaluation-only continuation, and per-scene cleanup**

For each scene in plan order:

1. Stage once and build all six identity seeds from its manifest/checkpoint/source metadata.
2. Validate existing `run.json` files when `resume=True`; skip only exact completed records. With `--no-resume`, do not overwrite a valid immutable record—raise an actionable identity error requiring a new output root/campaign ID.
3. Reject multiple prediction keys among skipped records before executing anything.
4. For each pending run, create a new numbered attempt. If a valid matching artifact remains after evaluation failure, skip reconstruction and evaluate it; otherwise safely remove only an invalid stale campaign-owned artifact before reconstruction.
5. Select cache mode using the shared known key/completion state. A partial/corrupt auto cache is left for `OrdinaryPredictionStore` validation/quarantine, and its corruption counter/events enter `CacheStats`.
6. Evaluate through `dependencies.evaluate_artifact`, aggregate diagnostics, complete identity with the returned prediction key, require the same key as every other completed scene record, and atomically publish `run.json` only after all validation succeeds.
7. If `keep_artifacts=False`, delete a successful run's validated `artifact/` only after its compact record is readable. After all requested scene records validate, delete the exact shared cache directory and prepared scene directory through `guarded_remove()`.
8. Call `write_summaries()` after each scene and once at campaign end. Return `exit_code=1` when failures/missing records remain.

Reconstruction/evaluation timers use `monotonic()`. Store rates in CSV as `reconstruction_s / frame_count`, `evaluation_s / frame_count`, and `reconstruction_s / window_count`; do not store wall-clock deltas as performance measurements.

- [ ] **Step 8: Run GREEN and cache/resume regressions**

Run:

```bash
python -m pytest -p no:cacheprovider -q tests/experiments/test_window_reference_campaign_runner.py
python -m pytest -p no:cacheprovider -q \
  tests/test_pipeline_runner.py \
  tests/test_prediction_provider.py \
  tests/test_prediction_store.py \
  tests/test_prediction_cache_matrix.py \
  tests/test_prediction_cache_method_parity.py \
  tests/test_reconstruction_artifacts.py
```

Expected: all tests PASS; the fake campaign performs one `auto` fill plus five `readonly` replays and retains one prediction key.

- [ ] **Step 9: REFACTOR attempt failure publication and commit**

Extract one `_failed_record(request, seed, attempt, stage, started_at, exc) -> RunRecord` so all failure stages preserve the same redacted error/attempt fields. Rerun Step 8, then commit:

```bash
git add experiments/window_reference_campaign/runner.py \
  experiments/window_reference_campaign/matrix.py \
  tests/experiments/test_window_reference_campaign_runner.py
git commit -m "feat: add resumable campaign runner"
```

### Task 5: Point-Cloud and KITTI Internal Trajectory Evaluator Adapters

**Files:**
- Create: `experiments/window_reference_campaign/evaluation.py`
- Create: `tests/experiments/test_window_reference_campaign_evaluation.py`

**Interfaces:**
- Consumes: Task 2 `StagedScene`; Task 4 `RunRequest`, `PipelineExecution`, `EvaluationOutput`; existing `load_pointmap_estimate()`, `PointCloudGroundTruth`, `evaluate_point_maps()`, `load_pointcloud_evaluation_config()`, `load_trajectory_estimate()`, `load_ground_truth_trajectory()`, `evaluate_trajectory()`, and `load_trajectory_evaluation_config()`.
- Produces `EvaluationDependencies(load_pointmap_estimate: Callable[[str | Path], PointMapEstimate], load_trajectory_estimate: Callable[[str | Path], TrajectoryEstimate], load_ground_truth_trajectory: Callable[[str | Path, str], GroundTruthTrajectory], load_pointcloud_config: Callable[[str | Path], PointCloudEvaluationConfig], load_trajectory_config: Callable[[str | Path], TrajectoryEvaluationConfig], pointcloud_backend_factory: Callable[[], GeometryBackend], evaluate_point_maps: Callable[..., GeometryEvaluation], evaluate_trajectory: Callable[[TrajectoryEstimate, GroundTruthTrajectory, TrajectoryEvaluationConfig], TrajectoryMetrics])` with defaults bound lazily inside `default_evaluation_dependencies() -> EvaluationDependencies`.
- Produces `evaluate_pointcloud_artifact(artifact_dir: str | Path, staged: StagedScene, evaluation_config: str | Path, output_dir: str | Path, *, dependencies: EvaluationDependencies | None = None) -> EvaluationOutput`.
- Produces `evaluate_kitti_artifact(artifact_dir: str | Path, staged: StagedScene, evaluation_config: str | Path, output_dir: str | Path, *, dependencies: EvaluationDependencies | None = None) -> EvaluationOutput`.
- Produces `evaluate_artifact(request: RunRequest, execution: PipelineExecution, loaded: LoadedCampaignConfig) -> EvaluationOutput`; this is the Task 4 default dependency and dispatches only `none`, `pointcloud`, or `internal_trajectory` from `request.staged.evaluation_kind`.
- The point-cloud metrics mapping has exactly the 16 `POINTCLOUD_METRICS` names from Task 3. The trajectory mapping has exactly the four `TRAJECTORY_METRICS` names from Task 3 and never emits unprefixed `ate_*`/`rpe_*` campaign columns.

- [ ] **Step 1: Write a RED literal point-map adapter test against the existing evaluator**

```python
import numpy as np
import torch

from evaluation.pointcloud.config import PointCloudEvaluationConfig
from evaluation.pointcloud.geometry_metrics import BackendResult
from pipeline.artifacts import PointMapEstimate
from pipeline.config import ReconstructionMode

from experiments.window_reference_campaign.evaluation import (
    EvaluationDependencies,
    evaluate_pointcloud_artifact,
)


class LiteralBackend:
    def refine_and_estimate_normals(self, predicted, ground_truth, threshold_m):
        normals_pred = np.tile([0.0, 0.0, 1.0], (len(predicted), 1))
        normals_gt = np.tile([0.0, 0.0, 1.0], (len(ground_truth), 1))
        return BackendResult(
            predicted_points=predicted,
            ground_truth_points=ground_truth,
            predicted_normals=normals_pred,
            ground_truth_normals=normals_gt,
            transformation=np.eye(4),
            fitness=1.0,
            inlier_rmse=0.0,
        )


def test_pointcloud_adapter_retains_primary_chamfer_and_1_2_5cm(tmp_path):
    points = np.array(
        [[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
          [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]],
        dtype=np.float32,
    )
    estimate = PointMapEstimate(
        frame_ids=(0,),
        global_points=torch.from_numpy(points),
        confidence=torch.ones((1, 2, 2)),
        reconstruction_mode=ReconstructionMode.NO_LOOP,
    )
    gt = tmp_path / "ground_truth.npz"
    np.savez_compressed(
        gt,
        point_maps=points,
        valid_mask=np.ones((1, 2, 2), dtype=bool),
        frame_ids=np.array([0]),
    )
    staged = make_staged_scene(
        tmp_path,
        evaluation_kind=EvaluationKind.POINTCLOUD,
        pointcloud_gt_path=gt,
    )
    config = PointCloudEvaluationConfig(center_crop_size=2)
    dependencies = EvaluationDependencies.for_tests(
        load_pointmap_estimate=lambda _: estimate,
        pointcloud_config=config,
        pointcloud_backend_factory=LiteralBackend,
    )

    result = evaluate_pointcloud_artifact(
        tmp_path / "artifact",
        staged,
        tmp_path / "ignored.yaml",
        tmp_path / "evaluation",
        dependencies=dependencies,
    )

    assert result.evaluation_kind is EvaluationKind.POINTCLOUD
    assert set(result.metrics) == set(POINTCLOUD_METRICS)
    assert result.metrics["accuracy_mean_m"] == pytest.approx(0.0)
    assert result.metrics["completion_median_m"] == pytest.approx(0.0)
    assert result.metrics["normal_consistency_mean"] == pytest.approx(1.0)
    assert result.metrics["chamfer_l1_m"] == pytest.approx(0.0)
    assert result.metrics["fscore_1cm"] == pytest.approx(1.0)
    assert result.metrics["fscore_2cm"] == pytest.approx(1.0)
    assert result.metrics["fscore_5cm"] == pytest.approx(1.0)
```

`EvaluationDependencies.for_tests()` accepts an already constructed `PointCloudEvaluationConfig`/`TrajectoryEvaluationConfig` and returns it from its loader callbacks; all metric computation still calls the real existing evaluators.

- [ ] **Step 2: Write RED literal KITTI 12-column trajectory and no-GT tests**

```python
from evaluation.trajectory.config import TrajectoryEvaluationConfig
from pipeline.artifacts import TrajectoryEstimate

from experiments.window_reference_campaign.evaluation import (
    evaluate_artifact,
    evaluate_kitti_artifact,
)


def _poses(count: int) -> torch.Tensor:
    poses = torch.eye(4, dtype=torch.float64).repeat(count, 1, 1)
    poses[:, 0, 3] = torch.arange(count, dtype=torch.float64)
    return poses


def test_kitti_adapter_uses_12_value_loader_and_internal_metric_names(tmp_path):
    poses = _poses(4)
    poses_path = tmp_path / "poses.txt"
    np.savetxt(poses_path, poses[:, :3].reshape(4, 12).numpy())
    staged = make_staged_scene(
        tmp_path,
        evaluation_kind=EvaluationKind.INTERNAL_TRAJECTORY,
        poses_path=poses_path,
        source_frame_ids=(0, 1, 2, 3),
    )
    dependencies = EvaluationDependencies.for_tests(
        load_trajectory_estimate=lambda _: TrajectoryEstimate(
            (0, 1, 2, 3), poses
        ),
        trajectory_config=TrajectoryEvaluationConfig(),
    )

    result = evaluate_kitti_artifact(
        tmp_path / "artifact",
        staged,
        tmp_path / "ignored.yaml",
        tmp_path / "evaluation",
        dependencies=dependencies,
    )

    assert set(result.metrics) == set(TRAJECTORY_METRICS)
    assert result.metrics["internal_ate_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["internal_rpe_translation_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["internal_rpe_rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["internal_matched_frame_count"] == 4
    assert all(not name.startswith(("ate_", "rpe_")) for name in result.metrics)


def test_no_gt_dispatch_emits_no_fabricated_quality_metrics(tmp_path):
    request, execution, loaded = make_evaluation_request(
        tmp_path, evaluation_kind=EvaluationKind.NONE
    )
    result = evaluate_artifact(request, execution, loaded)
    assert result == EvaluationOutput(EvaluationKind.NONE, None, ())
```

- [ ] **Step 3: Write RED non-finite and frame-alignment rejection tests**

```python
def test_pointcloud_adapter_rejects_gt_frame_ids_different_from_staging(tmp_path):
    staged, dependencies = pointcloud_adapter_fixture(tmp_path)
    with np.load(staged.pointcloud_gt_path) as original:
        np.savez_compressed(
            staged.pointcloud_gt_path,
            point_maps=original["point_maps"],
            valid_mask=original["valid_mask"],
            frame_ids=np.array([99]),
        )
    with pytest.raises(ValueError, match="frame IDs"):
        evaluate_pointcloud_artifact(
            tmp_path / "artifact", staged, tmp_path / "config.yaml",
            tmp_path / "evaluation", dependencies=dependencies,
        )


def test_adapter_rejects_non_finite_evaluator_result(tmp_path):
    staged, dependencies = pointcloud_adapter_fixture(tmp_path)
    dependencies = replace(
        dependencies,
        evaluate_point_maps=lambda *args, **kwargs: geometry_evaluation(
            accuracy_mean_m=float("nan")
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        evaluate_pointcloud_artifact(
            tmp_path / "artifact", staged, tmp_path / "config.yaml",
            tmp_path / "evaluation", dependencies=dependencies,
        )
```

The test-local `geometry_evaluation()` returns literal existing `GeometryEvaluation`, `PrimaryMetrics`, `GeometryDiagnostics`, `DirectionalNormalMetrics`, and three `ThresholdMetrics` dataclasses; it does not introduce a campaign-only fake metric type.

- [ ] **Step 4: Run focused RED nodes**

Run:

```bash
python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_window_reference_campaign_evaluation.py::test_pointcloud_adapter_retains_primary_chamfer_and_1_2_5cm \
  tests/experiments/test_window_reference_campaign_evaluation.py::test_kitti_adapter_uses_12_value_loader_and_internal_metric_names \
  tests/experiments/test_window_reference_campaign_evaluation.py::test_no_gt_dispatch_emits_no_fabricated_quality_metrics
```

Expected: FAIL during collection because `evaluation.py` does not exist.

- [ ] **Step 5: Implement the point-cloud adapter using the existing authoritative evaluator**

```python
def evaluate_pointcloud_artifact(
    artifact_dir: str | Path,
    staged: StagedScene,
    evaluation_config: str | Path,
    output_dir: str | Path,
    *,
    dependencies: EvaluationDependencies | None = None,
) -> EvaluationOutput:
    deps = dependencies or default_evaluation_dependencies()
    if staged.evaluation_kind is not EvaluationKind.POINTCLOUD:
        raise ValueError("point-cloud adapter requires pointcloud staged scene")
    if staged.pointcloud_gt_path is None:
        raise ValueError("point-cloud staged scene has no GT")
    estimate = deps.load_pointmap_estimate(artifact_dir)
    with np.load(staged.pointcloud_gt_path, allow_pickle=False) as data:
        frame_ids = tuple(int(item) for item in data["frame_ids"])
        if frame_ids != staged.source_frame_ids:
            raise ValueError("point-cloud GT frame IDs do not match staging")
        truth = PointCloudGroundTruth(data["point_maps"], data["valid_mask"])
    config = deps.load_pointcloud_config(evaluation_config)
    evaluation = deps.evaluate_point_maps(
        estimate,
        truth,
        config,
        backend_factory=deps.pointcloud_backend_factory,
    )
    metrics = _pointcloud_metrics(evaluation)
    _require_exact_finite_metrics(metrics, POINTCLOUD_METRICS)
    target = atomic_json(Path(output_dir) / "pointcloud_metrics.json", metrics)
    return EvaluationOutput(EvaluationKind.POINTCLOUD, metrics, (target,))
```

`_pointcloud_metrics()` copies all six `PrimaryMetrics` fields, `diagnostics.chamfer_l1_m`, and maps thresholds `0.01/0.02/0.05` (checked with absolute tolerance `1e-12`) to `precision_1cm` through `fscore_5cm`. Reject missing, duplicate, reordered-to-wrong-value, or extra thresholds. Do not reimplement Umeyama, ICP, normals, Chamfer, or F-score.

- [ ] **Step 6: Implement the internal trajectory adapter and dispatch**

```python
def evaluate_kitti_artifact(
    artifact_dir: str | Path,
    staged: StagedScene,
    evaluation_config: str | Path,
    output_dir: str | Path,
    *,
    dependencies: EvaluationDependencies | None = None,
) -> EvaluationOutput:
    deps = dependencies or default_evaluation_dependencies()
    if staged.evaluation_kind is not EvaluationKind.INTERNAL_TRAJECTORY:
        raise ValueError("KITTI adapter requires internal trajectory staged scene")
    if staged.poses_path is None:
        raise ValueError("KITTI staged scene has no poses")
    estimate = deps.load_trajectory_estimate(artifact_dir)
    truth = deps.load_ground_truth_trajectory(staged.poses_path, "replica")
    if truth.frame_ids != tuple(range(len(staged.source_frame_ids))):
        raise ValueError("staged KITTI pose rows are not dense ordinals")
    metrics = deps.evaluate_trajectory(
        estimate,
        truth,
        deps.load_trajectory_config(evaluation_config),
    )
    payload = {
        "internal_ate_rmse_m": metrics.ate_rmse_m,
        "internal_rpe_translation_rmse_m": metrics.rpe_translation_rmse_m,
        "internal_rpe_rotation_rmse_deg": metrics.rpe_rotation_rmse_deg,
        "internal_matched_frame_count": metrics.matched_frame_count,
    }
    _require_exact_finite_metrics(payload, TRAJECTORY_METRICS)
    target = atomic_json(Path(output_dir) / "internal_trajectory_metrics.json", payload)
    return EvaluationOutput(EvaluationKind.INTERNAL_TRAJECTORY, payload, (target,))
```

The evaluator consumes staged dense ordinal frame IDs because the same source vector already sliced pose rows; the original KITTI IDs remain in staging and run identity. Documentation and CSV label these metrics internal Sim(3) ATE/RPE, not official KITTI devkit metrics.

- [ ] **Step 7: Run GREEN and evaluator regressions**

Run:

```bash
python -m pytest -p no:cacheprovider -q tests/experiments/test_window_reference_campaign_evaluation.py
python -m pytest -p no:cacheprovider -q \
  tests/evaluation/test_pointcloud_evaluator.py \
  tests/evaluation/test_pointcloud_results.py \
  tests/evaluation/test_trajectory_evaluator.py \
  tests/evaluation/test_evaluate_pointcloud_cli.py \
  tests/evaluation/test_evaluate_ate_cli.py
```

Expected: all tests PASS and existing evaluator CLIs retain their schemas.

- [ ] **Step 8: REFACTOR finite metric publication and commit**

Use Task 3's atomic JSON helper and metric constants instead of parallel copies, keep all heavyweight imports inside `default_evaluation_dependencies()`, rerun Step 7, then commit:

```bash
git add experiments/window_reference_campaign/evaluation.py \
  tests/experiments/test_window_reference_campaign_evaluation.py
git commit -m "feat: add campaign evaluator adapters"
```

### Task 6: Standard-Library Bootstrap and Strict/No-GPU Preflight

**Files:**
- Create: `experiments/window_reference_campaign/preflight.py`
- Modify: `experiments/window_reference_campaign/cli.py`
- Create: `tests/experiments/test_window_reference_campaign_preflight.py`

**Interfaces:**
- Consumes: Task 1 typed config/plan/identity seed builder and lazy CLI; Task 2 scene resolution plus staging preview; Task 3 atomic JSON/argv redaction; existing `scripts/download_weights.sh`, `setup.py`, compiled extensions, and runtime dependencies.
- `preflight.py` imports only standard-library modules at module scope. All OmegaConf, NumPy, Pillow, Torch, Open3D, SciPy, campaign config, scene, and staging imports occur inside the specific function that needs them.
- Produces `BootstrapAction(label: str, argv: tuple[str, ...], cwd: Path, mutates_environment: bool)` and `build_bootstrap_actions(repository: str | Path, *, python_executable: str, external_checkpoint: str | Path | None) -> tuple[BootstrapAction, ...]`.
- Produces `run_bootstrap(actions: Sequence[BootstrapAction], *, execute: bool, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> int`; print mode emits shell-quoted commands, execute mode runs sequentially with `check=True`, and neither path adds dataset commands.
- Produces `GitState(commit: str, dirty: bool)`, `CudaState(available: bool, device_count: int, selected_device: int | None, device_name: str | None, bfloat16_supported: bool | None)`, `DiskUsage(total: int, used: int, free: int)`, `PreflightCheck(name: str, status: str, detail: Mapping[str, object])`, and `PreflightReport(schema_version: int, status: str, allow_no_gpu: bool, checks: tuple[PreflightCheck, ...], warnings: tuple[str, ...], errors: tuple[str, ...], git: GitState, checkpoint_sha256: str, free_disk_bytes: int, identity_seed_sha256: tuple[str, ...])`.
- Produces injectable `PreflightDependencies(python_version: tuple[int, int, int], import_module: Callable[[str], object], git_state: Callable[[Path], GitState], cuda_state: Callable[[int], CudaState], disk_usage: Callable[[Path], DiskUsage])` and `default_preflight_dependencies() -> PreflightDependencies`; the default wraps `shutil.disk_usage()` into `DiskUsage`.
- Produces `preflight_campaign(loaded: LoadedCampaignConfig, plan: CampaignPlan, *, allow_no_gpu: bool, dependencies: PreflightDependencies | None = None) -> PreflightReport` and `write_preflight_report(report: PreflightReport, campaign_root: str | Path) -> Path`.
- Extends CLI with functional `bootstrap` and `preflight`; `preflight` writes only `{campaign_root}/preflight.json`, and neither command stages a scene or constructs a model.

- [ ] **Step 1: Write RED subprocess tests proving bootstrap import isolation and no dataset downloads**

```python
import os
import subprocess
import sys

from experiments.window_reference_campaign.preflight import build_bootstrap_actions


def test_bootstrap_dry_run_works_when_torch_and_omegaconf_imports_are_blocked(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class Block:\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "    if fullname in {'torch', 'omegaconf'} or fullname.startswith(('torch.', 'omegaconf.')):\n"
        "      raise RuntimeError('heavy import forbidden during bootstrap')\n"
        "sys.meta_path.insert(0, Block())\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_window_reference_campaign.py"),
            "bootstrap",
            "--repository", str(ROOT),
            "--dry-run",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "git submodule update --init --recursive" in completed.stdout
    assert "setup.py build_ext --inplace" in completed.stdout


def test_bootstrap_actions_never_download_protected_datasets(tmp_path):
    actions = build_bootstrap_actions(
        ROOT,
        python_executable=sys.executable,
        external_checkpoint=None,
    )
    commands = "\n".join(shlex.join(action.argv) for action in actions)
    assert "scripts/download_weights.sh" in commands
    lowered = commands.lower()
    for forbidden in ("kitti", "7-scenes", "7scenes", "neuralrgbd", "cookie", "password", "token"):
        assert forbidden not in lowered
```

`ROOT` is the repository root constant defined at the top of the test. The first test imports no campaign module in the subprocess before invoking the root script; collection-time imports in the parent pytest process do not weaken it.

- [ ] **Step 2: Write RED no-GPU/strict preflight and missing-input tests**

```python
from experiments.window_reference_campaign.preflight import (
    CudaState,
    DiskUsage,
    GitState,
    PreflightDependencies,
    preflight_campaign,
)


def _dependencies(*, cuda_available: bool) -> PreflightDependencies:
    return PreflightDependencies.for_tests(
        python_version=(3, 11, 15),
        git=GitState("1" * 40, False),
        cuda=CudaState(
            available=cuda_available,
            device_count=1 if cuda_available else 0,
            selected_device=0 if cuda_available else None,
            device_name="fixture-gpu" if cuda_available else None,
            bfloat16_supported=True if cuda_available else None,
        ),
        free_bytes=100 * 1024**3,
    )


def test_allow_no_gpu_succeeds_but_records_not_gpu_ready(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path, preset="kitti-smoke")
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "ok_with_warnings"
    assert report.errors == ()
    assert any("CUDA" in warning for warning in report.warnings)
    cuda = next(check for check in report.checks if check.name == "cuda")
    assert cuda.status == "warning"
    assert cuda.detail["gpu_ready"] is False


def test_strict_preflight_rejects_same_no_gpu_environment(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path, preset="kitti-smoke")
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=False,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "error"
    assert any("CUDA" in error for error in report.errors)


def test_allow_no_gpu_does_not_hide_missing_checkpoint_or_dataset(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path, preset="kitti-smoke")
    loaded = replace(
        loaded,
        config=replace(
            loaded.config,
            storage=replace(
                loaded.config.storage,
                checkpoint=tmp_path / "missing.safetensors",
                data_root=tmp_path / "missing-data",
            ),
        ),
    )
    report = preflight_campaign(
        loaded,
        plan,
        allow_no_gpu=True,
        dependencies=_dependencies(cuda_available=False),
    )
    assert report.status == "error"
    assert any("checkpoint" in error for error in report.errors)
    assert any("KITTI" in error or "data" in error for error in report.errors)
```

`make_preflight_fixture()` creates 80 natural-sortable KITTI PNGs, 80 finite 12-column rows, a non-empty checkpoint, the configured output root, and a loaded/plan pair through public Task 1 APIs. `PreflightDependencies.for_tests()` returns importable sentinel modules with version strings for required-package checks and does not import real Torch/Open3D.

- [ ] **Step 3: Write RED tests for report-only writes, low space, dirty source, identities, and secret redaction**

```python
def test_preflight_writes_only_atomic_report_and_warns_for_space_dirty_source(tmp_path):
    loaded, plan = make_preflight_fixture(tmp_path, preset="kitti-smoke")
    dependencies = replace(
        _dependencies(cuda_available=False),
        git_state=lambda _: GitState("2" * 40, True),
        disk_usage=lambda _: DiskUsage(
            100 * 1024**3, 90 * 1024**3, 10 * 1024**3
        ),
    )
    report = preflight_campaign(
        loaded, plan, allow_no_gpu=True, dependencies=dependencies
    )
    target = write_preflight_report(report, loaded.config.campaign_root)
    assert target == loaded.config.campaign_root / "preflight.json"
    created = sorted(
        path.relative_to(loaded.config.campaign_root).as_posix()
        for path in loaded.config.campaign_root.rglob("*") if path.is_file()
    )
    assert created == ["preflight.json"]
    assert any("20.0 GiB" in warning for warning in report.warnings)
    assert any("dirty" in warning for warning in report.warnings)
    assert len(report.identity_seed_sha256) == 6
    assert len(set(report.identity_seed_sha256)) == 6


def test_persisted_argv_redacts_options_and_url_userinfo():
    assert redact_argv((
        "--token", "abc",
        "--password=hunter2",
        "https://alice:secret@example.invalid/repo.git",
        "--preset", "kitti-small",
    )) == (
        "--token", "<redacted>",
        "--password=<redacted>",
        "https://<redacted>@example.invalid/repo.git",
        "--preset", "kitti-small",
    )
```

- [ ] **Step 4: Run focused RED nodes**

Run:

```bash
python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_window_reference_campaign_preflight.py::test_bootstrap_dry_run_works_when_torch_and_omegaconf_imports_are_blocked \
  tests/experiments/test_window_reference_campaign_preflight.py::test_allow_no_gpu_succeeds_but_records_not_gpu_ready \
  tests/experiments/test_window_reference_campaign_preflight.py::test_allow_no_gpu_does_not_hide_missing_checkpoint_or_dataset
```

Expected: FAIL during collection because `preflight.py` does not exist and CLI has no bootstrap/preflight handler.

- [ ] **Step 5: Implement exact bootstrap actions and execute/print behavior**

```python
# experiments/window_reference_campaign/preflight.py
def build_bootstrap_actions(
    repository: str | Path,
    *,
    python_executable: str,
    external_checkpoint: str | Path | None,
) -> tuple[BootstrapAction, ...]:
    root = Path(repository).resolve(strict=True)
    actions = [
        BootstrapAction(
            "submodules",
            ("git", "submodule", "update", "--init", "--recursive"),
            root,
            True,
        ),
        BootstrapAction(
            "python-3.11",
            (
                python_executable,
                "-c",
                "import sys; assert sys.version_info[:2] == (3, 11), sys.version",
            ),
            root,
            False,
        ),
        BootstrapAction(
            "requirements",
            (python_executable, "-m", "pip", "install", "-r", "requirements.txt"),
            root,
            True,
        ),
        BootstrapAction(
            "cython-extensions",
            (python_executable, "setup.py", "build_ext", "--inplace"),
            root,
            True,
        ),
    ]
    if external_checkpoint is None:
        actions.append(BootstrapAction(
            "public-pi3-weight",
            ("bash", "scripts/download_weights.sh"),
            root,
            True,
        ))
    return tuple(actions)
```

Before returning, assert every relative script/file resolves under `repository`, and assert no lowercased command token contains protected dataset names or secret option words. `run_bootstrap()` prints `cd {shlex.quote(str(action.cwd))} && {shlex.join(action.argv)}` for every action. `--dry-run` is the default safe path; `--execute` is mutually exclusive and calls `subprocess.run(action.argv, cwd=action.cwd, check=True)` one action at a time. It never creates a conda environment or invents a package manager command not specified here.

- [ ] **Step 6: Implement comprehensive preflight with `--allow-no-gpu` as the only relaxation**

Preflight checks, in this order, are exact:

1. Python version is exactly major/minor `3.11`.
2. Import `omegaconf`, `numpy`, `PIL`, `torch`, `scipy`, `open3d`, `evo`, `inference_engine.utils.fast_seg`, and `inference_engine.utils._segmentation_cy`; capture version strings where exposed.
3. Resolve Git `HEAD` and `git status --porcelain`; unknown commit is an error in a Git clone, dirty is a warning and becomes identity state.
4. Require the checkpoint to be a non-empty regular file; capture exact byte size and SHA-256.
5. Resolve every selected scene read-only. Validate selected images, map IDs, pose/GT counts, finite KITTI `(N,12)` rows, point-map leading/spatial shapes, and both supported KITTI layouts. Explain official KITTI registration/purpose declaration when KITTI is absent; never offer a download URL.
6. Call `preview_staging_manifest()` without creating staging. Build six identity seeds per scene using its digest, Git state, and checkpoint digest; require unique relative run directories and unique seed hashes.
7. Resolve campaign root, reject a file/symlink in its place, locate the nearest existing parent, inspect write bits/access, and inspect `shutil.disk_usage()` without creating anything. Below `20.0 GiB` is a warning; a non-writable root is an error. Warn when the resolved path is under `/root/autodl-fs`; recommend `/root/autodl-tmp`.
8. Query selected CUDA index. Strict mode errors on unavailable/out-of-range CUDA or unsupported bfloat16; `allow_no_gpu=True` changes only unavailable/out-of-range CUDA to a warning and sets `gpu_ready=false`. If CUDA is available, dtype incompatibility remains an error.

```python
def preflight_campaign(
    loaded: LoadedCampaignConfig,
    plan: CampaignPlan,
    *,
    allow_no_gpu: bool,
    dependencies: PreflightDependencies | None = None,
) -> PreflightReport:
    deps = dependencies or default_preflight_dependencies()
    checks: list[PreflightCheck] = []
    warnings: list[str] = []
    errors: list[str] = []
    _check_python(deps, checks, errors)
    _check_imports(deps, checks, errors)
    git = _check_git(loaded.config.repository_root, deps, checks, warnings, errors)
    checkpoint_sha = _check_checkpoint(loaded.config.storage.checkpoint, checks, errors)
    seed_hashes = _check_scenes_and_identities(
        loaded, plan, git, checkpoint_sha, checks, errors
    )
    free = _check_storage(loaded, deps, checks, warnings, errors)
    _check_cuda(loaded, deps, allow_no_gpu, checks, warnings, errors)
    status = "error" if errors else ("ok_with_warnings" if warnings else "ok")
    return PreflightReport(
        1, status, allow_no_gpu, tuple(checks), tuple(warnings), tuple(errors),
        git, checkpoint_sha, free, tuple(seed_hashes),
    )
```

Collect all independent failures into one report rather than stopping after the first missing path. If prerequisite failure makes a dependent check impossible, record that check as `status="blocked"` with the exact prerequisite name; do not use an exception traceback as the report.

- [ ] **Step 7: Wire CLI handlers without making startup heavy**

`cli.py` defines parser arguments using stdlib only. Handler bodies import lazily:

- `bootstrap --repository PATH [--checkpoint PATH] [--dry-run|--execute]`.
- `preflight` accepts all common selection/path/model flags plus `--allow-no-gpu`; it loads config/plan, runs preflight, atomically writes the report, prints its path/status, and returns `0` only for `ok`/`ok_with_warnings`.
- `plan` remains unchanged and never imports `preflight.py` unless selected.

Do not add a dataset credential, URL, hostname, SSH, or login option to the parser.

- [ ] **Step 8: Run GREEN and bootstrap regressions**

Run:

```bash
python -m pytest -p no:cacheprovider -q tests/experiments/test_window_reference_campaign_preflight.py
python -m pytest -p no:cacheprovider -q \
  tests/test_download_weights_script.py \
  tests/test_model_loader.py \
  tests/disposable_experiments/test_preflight_experiments.py
python run_window_reference_campaign.py bootstrap --dry-run
```

Expected: all tests PASS; bootstrap output contains only submodule/Python/pip/build/public-weight actions and imports neither Torch nor OmegaConf in the blocked subprocess.

- [ ] **Step 9: REFACTOR check assembly and commit**

Make each `_check_*` return one or more immutable `PreflightCheck` values and append messages through `_warning()`/`_error()` helpers so status calculation is single-source. Rerun Step 8, then commit:

```bash
git add experiments/window_reference_campaign/preflight.py \
  experiments/window_reference_campaign/cli.py \
  tests/experiments/test_window_reference_campaign_preflight.py
git commit -m "feat: add campaign bootstrap and preflight"
```

### Task 7: Synthetic Six-Run Campaign, Complete CLI, and Fresh-Clone Cloud Guide

**Files:**
- Create: `experiments/window_reference_campaign/synthetic.py`
- Modify: `experiments/window_reference_campaign/runner.py`
- Modify: `experiments/window_reference_campaign/preflight.py`
- Modify: `experiments/window_reference_campaign/cli.py`
- Modify: `run_window_reference_campaign.py`
- Create: `docs/window-reference-cloud-campaign.md`
- Create: `tests/experiments/test_window_reference_campaign_cli.py`

**Interfaces:**
- Consumes: all Task 1–6 public interfaces, existing `SegmentationResult`, `build_window_reference_refiner()`, `load_pipeline_config()`, and `ReconstructionDiagnostics` payload shape.
- Produces `SyntheticFixture(point_maps: torch.Tensor, camera_poses: torch.Tensor, confidence: torch.Tensor, reference_intrinsic: torch.Tensor, merge_results: tuple[SegmentationResult, ...], rejection_results: tuple[SegmentationResult, ...])`, `build_synthetic_fixture(method: SegmentationMethod) -> SyntheticFixture`, `execute_synthetic(request: RunRequest, loaded: LoadedCampaignConfig) -> PipelineExecution`, and `validate_synthetic_artifact(artifact_dir: Path, seed: RunIdentitySeed) -> tuple[str, str]`.
- Supplies `RunnerDependencies.validate_artifact` with a production dispatcher that selects the real artifact validator or synthetic validator by dataset.
- Produces full CLI commands `bootstrap`, `plan`, `preflight`, `run`, and `summarize`. Resume remains `run --resume`; there is no separate resume verb.
- Common CLI options are exactly `--config`, `--preset`, repeatable `--scene`, `--methods`, `--refinement`, `--output-root`, `--data-root`, `--checkpoint`, `--dry-run`, mutually exclusive `--resume/--no-resume`, mutually exclusive `--fail-fast/--keep-going`, `--start-frame`, `--max-frames`, `--frame-stride`, `--gpu`, `--cache-policy auto|refresh|readonly|off`, and `--keep-artifacts`.
- `--methods` accepts a comma-separated subset of `depth,geometry,atomic`; `--refinement` accepts exactly `off`, `on`, or `off,on` and maps tokens explicitly rather than through YAML parsing.

- [ ] **Step 1: Write RED synthetic integration tests for six identities and required outcomes**

```python
import json
import subprocess
import sys


def test_synthetic_cli_runs_exact_six_without_checkpoint_gpu_or_pi3(tmp_path):
    missing = tmp_path / "missing"
    output = tmp_path / "campaign-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_window_reference_campaign.py"),
            "run",
            "--preset", "synthetic-smoke",
            "--data-root", str(missing / "data"),
            "--checkpoint", str(missing / "model.safetensors"),
            "--output-root", str(output),
            "--resume", "--fail-fast",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    campaign = output / "window-reference-v1"
    rows = read_csv(campaign / "summary/runs.csv")
    assert [row["run_id"] for row in rows] == [
        "depth__wr-off", "depth__wr-on",
        "geometry__wr-off", "geometry__wr-on",
        "atomic__wr-off", "atomic__wr-on",
    ]
    records = [json.loads(path.read_text()) for path in campaign.glob("runs/synthetic/*/*/run.json")]
    assert {record["identity"]["prediction_key"] for record in records} == {
        records[0]["identity"]["prediction_key"]
    }
    assert all(record["identity"]["reconstruction_mode"] == "no_loop" for record in records)
    assert all(record["identity"]["window_size"] == 75 for record in records)
    assert all(record["identity"]["overlap"] == 30 for record in records)
    assert all(record["identity"]["atomic_split_mode"] == "conservative" for record in records)


def test_synthetic_diagnostics_cover_merge_fallback_occlusion_and_depth_rejection(tmp_path):
    campaign = run_synthetic_campaign(tmp_path)
    rows = {row["run_id"]: row for row in read_csv(campaign / "summary/diagnostics.csv")}
    for run_id in ("depth__wr-off", "geometry__wr-off", "atomic__wr-off"):
        assert rows[run_id]["refinement_state"] == "not_applicable"
        assert rows[run_id]["applied_frame_count"] == ""
    for run_id in ("depth__wr-on", "geometry__wr-on", "atomic__wr-on"):
        assert int(rows[run_id]["applied_frame_count"]) >= 1
        assert int(rows[run_id]["accepted_edge_total"]) >= 1
        assert int(rows[run_id]["occluded_sample_total"]) >= 1
        assert int(rows[run_id]["depth_rejected_sample_total"]) >= 1
        histogram = json.loads(rows[run_id]["fallback_reason_histogram"])
        assert histogram["insufficient_support"] >= 1
```

- [ ] **Step 2: Write RED CLI contract tests for filters, resume, summary, metadata, and docs**

```python
def test_cli_subset_preserves_canonical_order_and_resume_skips_exact_records(tmp_path):
    campaign = run_synthetic_campaign(
        tmp_path,
        extra=("--methods", "atomic,depth", "--refinement", "on"),
    )
    first = read_csv(campaign / "summary/runs.csv")
    assert [row["run_id"] for row in first] == ["depth__wr-on", "atomic__wr-on"]
    attempts_before = sorted(campaign.glob("runs/synthetic/*/*/attempts/*"))
    run_synthetic_campaign(
        tmp_path,
        extra=("--methods", "atomic,depth", "--refinement", "on", "--resume"),
    )
    assert sorted(campaign.glob("runs/synthetic/*/*/attempts/*")) == attempts_before


def test_campaign_metadata_redacts_secrets_and_has_runtime_provenance(tmp_path):
    campaign = run_synthetic_campaign(tmp_path)
    payload = json.loads((campaign / "campaign.json").read_text())
    assert payload["source_commit"]
    assert type(payload["source_dirty"]) is bool
    assert payload["config_sha256"]
    assert payload["status"] == "succeeded"
    encoded = json.dumps(payload).lower()
    assert "password" not in encoded
    assert "cookie" not in encoded
    assert "hunter2" not in encoded


def test_document_contains_exact_fresh_clone_no_gpu_gpu_resume_and_cleanup_commands():
    text = (ROOT / "docs/window-reference-cloud-campaign.md").read_text(encoding="utf-8")
    for required in (
        "git clone --recursive --branch codex/laser-paper-pointmap-eval",
        "plan --preset kitti-small",
        "preflight --allow-no-gpu",
        "run --preset kitti-smoke",
        "run --preset pointcloud-small",
        "run --preset kitti-small",
        "--resume --keep-going",
        "summarize --preset kitti-small",
        "screen -dmS window-reference-campaign",
    ):
        assert required in text
    assert "/root/autodl-tmp" in text
    assert "/root/autodl-fs" not in extract_default_commands(text)
    assert "official KITTI devkit" in text
```

- [ ] **Step 3: Run focused RED nodes**

Run:

```bash
python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_window_reference_campaign_cli.py::test_synthetic_cli_runs_exact_six_without_checkpoint_gpu_or_pi3 \
  tests/experiments/test_window_reference_campaign_cli.py::test_synthetic_diagnostics_cover_merge_fallback_occlusion_and_depth_rejection \
  tests/experiments/test_window_reference_campaign_cli.py::test_document_contains_exact_fresh_clone_no_gpu_gpu_resume_and_cleanup_commands
```

Expected: FAIL because `synthetic.py`, complete run/summarize handlers, and the cloud guide do not exist.

- [ ] **Step 4: Implement deterministic synthetic fixtures through the shipped refiner**

Build two literal 33x33, four-frame point-map cases from pixel coordinates and a finite pinhole intrinsic. The merge case uses a one-region reference and adjacent two-region target; the rejection case scales selected target rays (all XYZ coordinates) in depth while preserving projection, and offsets one camera along Z to produce many-to-one projected samples. Use the resolved shipped window-reference config; do not loosen `sampling_stride=4`, `min_region_correspondences=8`, or any other threshold.

```python
def _point_map(height: int, width: int, intrinsic: torch.Tensor) -> torch.Tensor:
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    z = torch.ones_like(rows)
    x = (columns - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (rows - intrinsic[1, 2]) / intrinsic[1, 1] * z
    return torch.stack((x, y, z), dim=-1)


def execute_synthetic(
    request: RunRequest,
    loaded: LoadedCampaignConfig,
) -> PipelineExecution:
    fixture = build_synthetic_fixture(request.planned.variant.segmentation_method)
    observations = _run_fixture_refinement(
        fixture,
        enabled=request.planned.variant.window_reference_enabled,
        segmentation=loaded.config.window_reference,
    )
    prediction_key = hashlib.sha256(
        f"synthetic-v1:{request.staged.manifest_sha256}:75:30:pi3:bfloat16".encode()
    ).hexdigest()
    request.artifact_dir.mkdir(parents=True, exist_ok=False)
    manifest = atomic_json(
        request.artifact_dir / "synthetic.json",
        {
            "schema_version": 1,
            "identity_seed": request.identity_seed.to_payload(),
            "prediction_key": prediction_key,
            "frame_count": 4,
            "diagnostics": {
                "stage_timings_ms": {"reconstruction": 0.0},
                "segmentation_summaries": observations,
                "candidate_count": 0,
                "constraint_count": 0,
                "mode_scalars": {"window_count": 2},
            },
        },
    )
    if request.cache_mode is not PredictionCacheMode.OFF:
        entry = request.cache_root / "v2" / prediction_key
        (entry / "windows").mkdir(parents=True, exist_ok=True)
        atomic_json(entry / "manifest.json", {"synthetic": True, "key": prediction_key})
        atomic_json(entry / "sequence.json", {"synthetic": True, "key": prediction_key})
        for index in range(2):
            (entry / "windows" / f"{index:06d}.pt").write_bytes(b"synthetic-window")
        atomic_json(entry / "complete.json", {
            "schema_version": 2,
            "key": prediction_key,
            "window_count": 2,
        })
    return PipelineExecution(
        request.artifact_dir,
        sha256_file(manifest),
        prediction_key,
        4,
        json.loads(manifest.read_text())["diagnostics"],
        _synthetic_cache_stats(request.cache_mode),
    )
```

Run the refiner twice inside the fixture: one merge scenario and one conservative `insufficient_support`/depth-rejection scenario. Prefix frame indices/window indices so the concatenated observation payload contains four unique frames and repeated overlap observations. Assert fixture invariants inside `build_synthetic_fixture()` before returning: at least one accepted edge, at least one `insufficient_support`, and positive occluded/depth-rejected counts. Disabled runs retain only initial `region_count` plus window/frame indices, allowing Task 3 to emit not-applicable nulls.

Synthetic cache stats emulate the policy boundary only: first `auto` execution writes one campaign-owned small `complete.json`; `readonly` executions count hits. They do not claim PI3 inference or quality. `synthetic.json` is validated by exact identity seed, finite diagnostics, digest, and prediction key.

- [ ] **Step 5: Complete `run`/`summarize` dispatch and campaign metadata**

`run` behavior is exact:

1. Load config and plan.
2. Synthetic-only plans bypass checkpoint/CUDA/data checks but still validate typed config, identities, output ownership, disk warning, and matrix uniqueness; real/mixed plans require strict preflight and abort before staging if it reports errors.
3. Atomically create/update `campaign.json` with redacted argv, Git state, resolved config hash, Python/Torch/CUDA/GPU details (`null` where synthetic/bootstrap does not import Torch), checkpoint digest (`null` for synthetic), timestamps, and status `running`.
4. Resolve scenes and call `run_campaign()` with real or synthetic dependencies.
5. Atomically set final status to `succeeded` or `failed`; return `CampaignOutcome.exit_code`.

`summarize` loads the same config/plan, validates all present compact records through their expected seeds, writes all six summary files, prints paths, and returns non-zero for incomplete/invalid requested coverage. It never reconstructs, evaluates, stages, or deletes.

Add `--dry-run` to `run` as a synonym for printing the resolved logical plan without writes; reject `--dry-run` combined with `--no-resume`, `--keep-artifacts`, or failure-policy flags because those would misleadingly imply execution behavior.

- [ ] **Step 6: Write the fresh-clone cloud guide with exact commands and truth boundaries**

The guide must include literal commands for:

```bash
cd /root/autodl-tmp
git clone --recursive --branch codex/laser-paper-pointmap-eval \
  https://github.com/Cjuicy/LASER.git LASER-Window-Reference
cd /root/autodl-tmp/LASER-Window-Reference

conda run -n vggt python run_window_reference_campaign.py plan \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign

conda run -n vggt python run_window_reference_campaign.py preflight \
  --allow-no-gpu --preset pointcloud-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign

conda run -n vggt python run_window_reference_campaign.py preflight \
  --allow-no-gpu --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign
```

Then document foreground and `screen` GPU commands, `kitti-smoke` first, `pointcloud-small` second, `kitti-small` third, `--resume --keep-going`, `summarize`, non-mutating status (`screen -ls`, `tail`), and safe cleanup by exact campaign root. State that KITTI must be obtained through official registration/purpose declaration; no command downloads it. State that internal Sim(3) ATE/RPE is not the official KITTI devkit metric, historical corrected results are contextual only, and no real GPU campaign is claimed until actually run.

The safe manual work-area cleanup command must call the campaign guard, not a recursive shell glob:

```bash
conda run -n vggt python -c \
  'from experiments.window_reference_campaign.staging import guarded_remove; guarded_remove("/root/autodl-tmp/window-reference-campaign/window-reference-v1/work", "/root/autodl-tmp/window-reference-campaign/window-reference-v1")'
```

Include a residual-risk section stating all five facts: 75-frame refinement CPU cost still needs measurement; `thin_geometry` is a single partial window; each scene's first new run must regenerate its missing ordinary cache; historical corrected KITTI rows are context only; and a no-card instance cannot validate PI3, CUDA memory, or empirical quality.

- [ ] **Step 7: Run GREEN, CLI help, synthetic, and focused warning checks**

Run:

```bash
python -m pytest -p no:cacheprovider -q -W error tests/experiments/test_window_reference_campaign_cli.py
python run_window_reference_campaign.py --help
python run_window_reference_campaign.py plan --preset synthetic-smoke
python run_window_reference_campaign.py run --preset synthetic-smoke --fail-fast
python run_window_reference_campaign.py summarize --preset synthetic-smoke
python -m pytest -p no:cacheprovider -q tests/experiments/test_window_reference_campaign_*.py
```

Expected: all commands PASS, six compact synthetic records share one prediction key, and the campaign-specific suite emits no warnings.

- [ ] **Step 8: REFACTOR CLI option translation and commit**

Centralize common parser-to-`CampaignOverrides` conversion in `_campaign_overrides(arguments: argparse.Namespace) -> CampaignOverrides`; keep handlers lazy. Rerun Step 7, then commit:

```bash
git add experiments/window_reference_campaign/synthetic.py \
  experiments/window_reference_campaign/runner.py \
  experiments/window_reference_campaign/preflight.py \
  experiments/window_reference_campaign/cli.py \
  run_window_reference_campaign.py \
  docs/window-reference-cloud-campaign.md \
  tests/experiments/test_window_reference_campaign_cli.py
git commit -m "feat: complete window reference campaign CLI"
```

### Task 8: Final Review, Full Verification, No-GPU Cloud Validation, Push #2, and GPU Handoff

**Files:**
- Review: every file listed in Tasks 1–7
- Modify only if a RED test exposes a defect: the owning production/test/doc file from Tasks 1–7

**Interfaces:**
- Consumes: the complete campaign, approved spec, all focused/regression tests, an existing no-card AutoDL shell with the observed data/weight paths, and remote branch `codex/laser-paper-pointmap-eval`.
- Produces: a clean, reviewed branch whose local SHA equals remote branch SHA; passing local focused/full verification; successful cloud `plan` and `preflight --allow-no-gpu` reports; exact GPU handoff commands with no claim that they have run.
- This task introduces no new behavior. RED→GREEN applies to any review defect: first add the smallest failing regression node in the owning test file, observe its specified failure, implement the minimal fix, rerun focused/full tests, and commit that fix before proceeding.

- [ ] **Step 1: Use the required review/verification skills and inspect the complete diff**

REQUIRED SUB-SKILL: Use `superpowers:requesting-code-review` for a fresh spec-compliance review, then `superpowers:verification-before-completion` before any passing/completion claim.

Run:

```bash
git status --short
git diff --check 256c39352ef3acf52ebea8ca46bb7a8053f17557..HEAD
git diff --stat 256c39352ef3acf52ebea8ca46bb7a8053f17557..HEAD
git diff 256c39352ef3acf52ebea8ca46bb7a8053f17557..HEAD -- \
  run_window_reference_campaign.py \
  configs/experiments/window_reference_campaign.yaml \
  experiments/window_reference_campaign \
  experiments/__init__.py \
  docs/window-reference-cloud-campaign.md \
  tests/experiments
```

Expected: no whitespace errors; changes are limited to the campaign/lazy-export surface; no algorithm, dependency, credential, dataset, or generated output change appears.

- [ ] **Step 2: Run spec/security/static scans**

Run:

```bash
rg -n "password|passwd|token|cookie|credential|ssh://|@.*:" \
  run_window_reference_campaign.py \
  configs/experiments/window_reference_campaign.yaml \
  experiments/window_reference_campaign \
  docs/window-reference-cloud-campaign.md
rg -n "corrected|traditional|window_size:|overlap:" \
  configs/experiments/window_reference_campaign.yaml \
  experiments/window_reference_campaign \
  docs/window-reference-cloud-campaign.md
python -m compileall -q experiments pipeline reconstruction inference_engine
```

Expected: the first scan finds only deliberate redaction/forbidden-word validation code and documentation warnings, never a value/secret; every schedule scan match is either fixed `75/30` or a historical-context warning, with no runnable non-`no_loop` campaign setting; compileall exits `0`.

- [ ] **Step 3: Run focused, synthetic, regression, and full local verification fresh**

Run in order:

```bash
python -m pytest -p no:cacheprovider -q -W error tests/experiments/test_window_reference_campaign_*.py
python run_window_reference_campaign.py plan --preset synthetic-smoke
python run_window_reference_campaign.py run --preset synthetic-smoke --fail-fast
python run_window_reference_campaign.py plan --preset kitti-small --dry-run
python -m pytest -p no:cacheprovider -q \
  tests/test_window_reference_refinement.py \
  tests/test_prediction_cache_method_parity.py \
  tests/test_prediction_cache_matrix.py \
  tests/test_prediction_store.py \
  tests/test_pipeline_runner.py \
  tests/test_reconstruction_artifacts.py \
  tests/evaluation/test_pointcloud_evaluator.py \
  tests/evaluation/test_trajectory_evaluator.py
python -m pytest -p no:cacheprovider -q
git diff --check
git status --short
```

Expected: campaign-specific tests pass with warnings promoted to errors; synthetic plan/run passes; focused regressions pass; full suite passes with no new warnings; Git shows only intentional tracked changes/commits and no generated campaign output.

- [ ] **Step 4: Commit a review fix only through a demonstrated RED→GREEN cycle**

If review found a defect, add the exact failing pytest node to the owning `tests/experiments/test_window_reference_campaign_*.py`, run that node and record the concrete failure, make the minimal fix, rerun Step 3, then commit only those files:

```bash
git add run_window_reference_campaign.py \
  configs/experiments/window_reference_campaign.yaml \
  experiments/__init__.py \
  experiments/window_reference_campaign \
  docs/window-reference-cloud-campaign.md \
  tests/experiments
git commit -m "fix: close campaign acceptance gap"
```

If review found no defect, do not run these two commands and do not create an empty commit.

- [ ] **Step 5: Push the verified implementation branch (remote push #2 after the design push)**

Run:

```bash
git push origin HEAD:codex/laser-paper-pointmap-eval
git rev-parse HEAD
git ls-remote --heads origin refs/heads/codex/laser-paper-pointmap-eval
```

Expected: push succeeds; the local 40-hex SHA printed by `git rev-parse` equals the remote SHA printed by `git ls-remote`. This is the implementation push following the already published design commit.

- [ ] **Step 6: On the existing no-card cloud shell, validate fresh-clone `plan` and `preflight --allow-no-gpu`**

Run only after Step 5, in the no-card shell; these commands contain no hostname, account, token, cookie, or credential:

```bash
cd /root/autodl-tmp
git clone --recursive --branch codex/laser-paper-pointmap-eval \
  https://github.com/Cjuicy/LASER.git LASER-Window-Reference
cd /root/autodl-tmp/LASER-Window-Reference

conda run -n vggt python -m pytest -p no:cacheprovider -q \
  tests/experiments/test_window_reference_campaign_config.py \
  tests/experiments/test_window_reference_campaign_preflight.py

conda run -n vggt python run_window_reference_campaign.py plan \
  --preset pointcloud-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign

conda run -n vggt python run_window_reference_campaign.py plan \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign

conda run -n vggt python run_window_reference_campaign.py preflight \
  --allow-no-gpu --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign
```

Expected: both plans show six runs per selected scene and `no_loop/75/30`; both preflights exit `0`, record `gpu_ready=false` as a warning, validate point-map/KITTI layouts, checkpoint, disk, and identities, and write only `preflight.json` below the campaign root. This does not validate PI3 inference.

- [ ] **Step 7: Hand off—but do not execute—the real GPU campaigns**

After the user opens a GPU, the first command is:

```bash
cd /root/autodl-tmp/LASER-Window-Reference
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n vggt \
  python run_window_reference_campaign.py run \
  --preset kitti-smoke \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign \
  --resume --fail-fast
```

Only after `kitti-smoke` validates, hand off `pointcloud-small`, then `kitti-small`, using the same paths and `--resume --keep-going`. The final report must state exactly: local tests/synthetic and cloud no-GPU plan/preflight passed; real GPU/KITTI/point-cloud quality and runtime remain unvalidated until those commands actually finish. Do not claim a GPU result, CUDA-memory bound, empirical improvement, or official KITTI score from code completion.

## Spec Coverage Map

| Spec section | Implemented and verified by |
|---|---|
| 1 Goal | Tasks 1, 4, 7: six identities, shared cache, three scene kinds, atomic summaries/resume/storage |
| 2 Approved decisions | Tasks 1, 4: fixed `no_loop/75/30`, conservative atomic, exact ten window-reference values |
| 3 Scope boundaries | Global Constraints; Tasks 1, 6, 7: sidecar/root CLI/bootstrap/commands; no algorithm/data download work |
| 4 Existing contracts | Tasks 2, 4, 5: `ImageManifest`, `PipelineRunner`, artifact diagnostics, prediction fingerprint, existing evaluators |
| 5 Cloud findings | Tasks 6–8 and docs: `/root/autodl-tmp`, PI3-only weight, no-card truth, space warning, no secrets |
| 6 Scene presets | Tasks 1, 2, 7: synthetic, exact three point-cloud scenes/shapes, exact KITTI presets/layouts |
| 7 Architecture/CLI/fresh clone | Locked file map; Tasks 1, 6, 7 and docs |
| 8 Typed config/identities | Task 1 plus Task 3 exact resume equality |
| 9 Staging/alignment | Task 2 single vector, relative links, poses/GT, hashes, sample stride 1 |
| 10 Cache/execution | Task 4 first-pending auto/refresh, readonly reuse, serial attempts, cleanup |
| 11 Evaluation | Task 5 point-cloud 16 metrics, internal KITTI names, no-GT omission |
| 12 Diagnostics | Task 3 on/off/null/fallback and unique-vs-observation semantics; Task 7 synthetic outcomes |
| 13 Result layout | Tasks 3–4 atomic run/campaign/CSV/JSON layout and redacted provenance |
| 14 Bootstrap/preflight | Task 6 import isolation, packages/extensions/checkpoint/data/space/CUDA checks |
| 15 Errors/resume | Tasks 3–4 fail-fast/keep-going, immutable records, artifact-only resume, signals, safe removal |
| 16 Test strategy | RED→GREEN→REFACTOR in Tasks 1–7; Task 8 focused/full/cloud verification |
| 17 Implementation tasks | This plan's eight sequential reviewer-sized tasks |
| 18 Acceptance criteria | Task 8 local/full/no-GPU gates and Task 7 fresh-clone docs |
| 19 Residual risks | Global truth boundary, docs, and Task 8 GPU-only handoff retain all five risks explicitly |

Plan execution ends after Task 8 handoff. It must not silently start a protected-data or paid-GPU campaign.
