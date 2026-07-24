# Modular Segmentation and Loop Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one strict, modular LASER pipeline with `depth`/`geometry`/`atomic` segmentation, unchanged anchor propagation, `traditional`/`corrected` loop closure, and all 10 unique method/split validation configurations.

**Architecture:** Start from the corrected-loop branch, inject segmentation and anchor services into window engines, and select a loop-closure strategy that owns its cache semantics, constraint construction, optimization application, and aggregation. A single strict OmegaConf-backed YAML schema and one `run_laser.py` entry point replace all legacy method flags and legacy loop configuration dictionaries.

**Tech Stack:** Python 3, PyTorch, NumPy 1.26.4, SciPy, scikit-image, OmegaConf from the existing `hydra-core` dependency, PyYAML, pytest, existing Cython segmentation extensions, SALAD/DINO, and the existing Sim(3) optimizer.

## Global Constraints

- Execute in an isolated worktree created from `codex/loop-closure-corrected-pipeline@6441b8b`.
- Bring the approved design commits `a80d786` and `a82e424` into the integration branch before implementation.
- Port atomic split behavior from `codex/auto-post-merge-split`; do not merge that branch wholesale.
- Port traditional loop behavior from `origin/codex/loop-closure-cloud-updated-20260714@eaedaae`; do not restore its percentile ambiguity.
- Public segmentation names are exactly `depth`, `geometry`, and `atomic`.
- `atomic.split_mode` is exactly one of `none`, `conservative`, and `normal_only`; the default is `conservative`.
- Public loop names are exactly `traditional` and `corrected`; the default is `corrected`.
- Every confidence field uses positive `keep_ratio` semantics: `0.30` means retain the highest approximately 30% of finite values.
- Felzenszwalb comparison defaults are exactly `scale=300`, `sigma=1.1`, and `min_size=500`.
- Anchor propagation retains the current formulas, order, and default correspondence IoU threshold `0.4`.
- Traditional and corrected loop strategies consume the same canonical image manifest and the same SALAD candidate objects.
- Unknown, missing, and legacy configuration fields fail before model construction.
- Do not add HART-AP propagation code, configuration, tests, or runtime branches.
- Do not add a new dependency.
- Use tests before implementation and make one focused commit per task.

## Execution Baseline

Before Task 1, use `superpowers:using-git-worktrees` to create the isolated workspace. The intended branch name is `codex/modular-segmentation-loop-integration`. In that worktree:

```bash
git cherry-pick a80d786 a82e424
python setup.py build_ext --inplace
pytest -q
```

Expected baseline: Cython extensions build successfully and the corrected-pipeline test suite passes. If the baseline fails, stop and report the exact failing tests before changing implementation.

## Final File Map

Create:

```text
pipeline/__init__.py
pipeline/config.py
pipeline/manifest.py
pipeline/preflight.py
pipeline/diagnostics.py
pipeline/runner.py
configs/pipeline/default.yaml
configs/pipeline/test.yaml
inference_engine/segmentation/__init__.py
inference_engine/segmentation/base.py
inference_engine/segmentation/confidence.py
inference_engine/segmentation/depth.py
inference_engine/segmentation/geometry.py
inference_engine/segmentation/atomic.py
inference_engine/segmentation/registry.py
inference_engine/utils/post_merge_split.py
inference_engine/anchor_propagation.py
loop_closure/methods/__init__.py
loop_closure/methods/base.py
loop_closure/methods/shared.py
loop_closure/methods/traditional.py
loop_closure/methods/corrected.py
loop_closure/methods/registry.py
run_laser.py
scripts/verify_pipeline_matrix.py
docs/pipeline-configuration.md
tests/test_pipeline_config.py
tests/test_pipeline_manifest.py
tests/test_segmentation_strategies.py
tests/test_atomic_split_modes.py
tests/test_anchor_propagation_contract.py
tests/test_loop_method_contracts.py
tests/test_traditional_loop_method.py
tests/test_corrected_loop_method.py
tests/test_pipeline_runner.py
tests/test_pipeline_matrix.py
```

Modify:

```text
inference_engine/streaming_window_engine.py
inference_engine/utils/depth.py
inference_engine/utils/geometry_segmentation.py
inference_engine/utils/layer_atomic_geometry.py
inference_engine/utils/lsa.py
inference_engine/utils/registration_confidence.py
inference_engine/__init__.py
loop_closure/loop_model.py
loop_closure/utils/sim3loop.py
loop_closure/__init__.py
eval_launch.py
README.md
tests/test_depth_segmentation_stages.py
tests/test_geometry_segmentation.py
tests/test_layer_atomic_geometry.py
tests/test_layer_atomic_integration.py
tests/test_registration_confidence.py
```

Delete after replacements pass:

```text
demo.py
demo_lc.py
configs/loop_config.yaml
inference_engine/streaming_window_engine_lc.py
loop_closure/loop_closure.py
loop_closure/utils/config_utils.py
utils/image_paths.py
tests/test_demo.py
tests/test_demo_lc.py
tests/test_eval_launch_lc.py
tests/test_loop_closure_pipeline.py
tests/test_loop_image_manifest.py
tests/test_segmentation_engine_modes.py
tests/test_segmentation_modes.py
tests/test_streaming_window_engine_lc_pipeline.py
```

---

### Task 1: Strict Configuration Schema and Defaults

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/config.py`
- Create: `configs/pipeline/default.yaml`
- Create: `configs/pipeline/test.yaml`
- Test: `tests/test_pipeline_config.py`

**Interfaces:**
- Produces: `PipelineConfig`, configuration enums, `LoadedPipelineConfig`, and `load_pipeline_config(path: str | Path, overrides: Sequence[str] = ()) -> LoadedPipelineConfig`.
- Produces: `LoadedPipelineConfig.resolved_yaml: str` and `LoadedPipelineConfig.sha256: str`.
- Consumes: only OmegaConf/Hydra already declared in `requirements.txt`.

- [ ] **Step 1: Write strict-schema tests**

Create `tests/test_pipeline_config.py` with these behaviors:

```python
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.config import (
    AtomicSplitMode,
    LoopMethod,
    SegmentationMethod,
    load_pipeline_config,
)


DEFAULT = Path("configs/pipeline/default.yaml")


def test_default_config_has_approved_methods_and_defaults():
    loaded = load_pipeline_config(DEFAULT)
    assert loaded.config.segmentation.method is SegmentationMethod.ATOMIC
    assert loaded.config.segmentation.atomic.split_mode is AtomicSplitMode.CONSERVATIVE
    assert loaded.config.loop.method is LoopMethod.CORRECTED
    assert loaded.config.segmentation.felzenszwalb.scale == 300
    assert loaded.config.segmentation.felzenszwalb.sigma == pytest.approx(1.1)
    assert loaded.config.segmentation.felzenszwalb.min_size == 500
    assert len(loaded.sha256) == 64


def test_dotlist_overrides_use_new_field_paths_only():
    loaded = load_pipeline_config(
        DEFAULT,
        (
            "segmentation.method=geometry",
            "loop.method=traditional",
            "loop.registration.confidence_keep_ratio=0.4",
        ),
    )
    assert loaded.config.segmentation.method is SegmentationMethod.GEOMETRY
    assert loaded.config.loop.method is LoopMethod.TRADITIONAL
    assert loaded.config.loop.registration.confidence_keep_ratio == pytest.approx(0.4)


@pytest.mark.parametrize(
    "text",
    (
        "segmentation:\n  segment_mode: depth\n",
        "segmentation:\n  geometry_seg_profile: legacy\n",
        "loop:\n  registration_top_confidence_ratio: 0.3\n",
        "anchor_propagation:\n  depth_refine: true\n",
    ),
)
def test_legacy_fields_are_rejected(tmp_path, text):
    path = tmp_path / "legacy.yaml"
    path.write_text("version: 1\n" + text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown|missing|legacy"):
        load_pipeline_config(path)


def test_missing_required_field_is_rejected(tmp_path):
    path = tmp_path / "missing.yaml"
    path.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_pipeline_config(path)


@pytest.mark.parametrize("ratio", (0.0, -0.1, 1.1))
def test_invalid_keep_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="keep_ratio"):
        load_pipeline_config(
            DEFAULT,
            (f"loop.registration.confidence_keep_ratio={ratio}",),
        )


def test_window_overlap_must_be_strictly_smaller_than_size():
    with pytest.raises(ValueError, match="window.size"):
        load_pipeline_config(
            DEFAULT,
            ("window.size=5", "window.overlap=5"),
        )
```

- [ ] **Step 2: Run the tests and verify collection fails**

Run:

```bash
pytest tests/test_pipeline_config.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 3: Implement the complete structured schema**

Use `str, Enum` values so YAML method strings are type checked:

```python
class SegmentationMethod(str, Enum):
    DEPTH = "depth"
    GEOMETRY = "geometry"
    ATOMIC = "atomic"


class AtomicSplitMode(str, Enum):
    NONE = "none"
    CONSERVATIVE = "conservative"
    NORMAL_ONLY = "normal_only"


class LoopMethod(str, Enum):
    TRADITIONAL = "traditional"
    CORRECTED = "corrected"
```

Define frozen dataclasses with `omegaconf.MISSING` for every YAML field:

```python
@dataclass(frozen=True)
class InputConfig:
    image_dir: str = MISSING
    sample_stride: int = MISSING


@dataclass(frozen=True)
class OutputConfig:
    scene_name: str = MISSING
    cache_dir: str = MISSING
    result_dir: str = MISSING
    save_diagnostics: bool = MISSING


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str = MISSING
    inference_device: str = MISSING
    process_device: str = MISSING
    dtype: str = MISSING


@dataclass(frozen=True)
class WindowConfig:
    size: int = MISSING
    overlap: int = MISSING


@dataclass(frozen=True)
class FelzenszwalbConfig:
    scale: float = MISSING
    sigma: float = MISSING
    min_size: int = MISSING


@dataclass(frozen=True)
class GeometryConfig:
    normal_method: str = MISSING
    normal_threshold_degrees: float = MISSING


@dataclass(frozen=True)
class AtomicConfig:
    split_mode: AtomicSplitMode = MISSING
    split_score_threshold: float = MISSING


@dataclass(frozen=True)
class SegmentationConfig:
    method: SegmentationMethod = MISSING
    confidence_keep_ratio: float = MISSING
    depth_merge_threshold: float = MISSING
    temporal_iou_threshold: float = MISSING
    felzenszwalb: FelzenszwalbConfig = field(default_factory=FelzenszwalbConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    atomic: AtomicConfig = field(default_factory=AtomicConfig)


@dataclass(frozen=True)
class AnchorPropagationConfig:
    enabled: bool = MISSING
    correspondence_iou_threshold: float = MISSING


@dataclass(frozen=True)
class RegistrationConfig:
    confidence_keep_ratio: float = MISSING


@dataclass(frozen=True)
class DetectionConfig:
    method: str = MISSING
    salad_checkpoint: str = MISSING
    dino_checkpoint: str = MISSING
    image_size: list[int] = MISSING
    batch_size: int = MISSING
    similarity_threshold: float = MISSING
    top_k: int = MISSING
    nms_enabled: bool = MISSING
    nms_frame_radius: int = MISSING


@dataclass(frozen=True)
class ConstraintConfig:
    chunk_size: int = MISSING


@dataclass(frozen=True)
class OptimizerConfig:
    implementation: str = MISSING
    use_sim3: bool = MISSING
    max_iterations: int = MISSING
    initial_damping: float = MISSING


@dataclass(frozen=True)
class LoopConfig:
    enabled: bool = MISSING
    method: LoopMethod = MISSING
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class PipelineConfig:
    version: int = MISSING
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    anchor_propagation: AnchorPropagationConfig = field(
        default_factory=AnchorPropagationConfig
    )
    loop: LoopConfig = field(default_factory=LoopConfig)


@dataclass(frozen=True)
class LoadedPipelineConfig:
    config: PipelineConfig
    resolved_yaml: str
    sha256: str
```

Implement strict loading with `OmegaConf.structured`, `OmegaConf.merge`, `OmegaConf.from_dotlist`, `OmegaConf.missing_keys`, and `OmegaConf.to_object`. Convert `ConfigKeyError`, `MissingMandatoryValue`, and `ValidationError` into `ValueError` whose message includes the offending field. Validate all exact ranges from the design, including:

```python
if config.version != 1:
    raise ValueError("version must be 1")
if not config.window.size > config.window.overlap >= 1:
    raise ValueError("window.size must be greater than window.overlap >= 1")
for path, ratio in (
    ("segmentation.confidence_keep_ratio", config.segmentation.confidence_keep_ratio),
    ("loop.registration.confidence_keep_ratio", config.loop.registration.confidence_keep_ratio),
):
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(f"{path} keep_ratio must be in (0, 1]")
if config.segmentation.atomic.split_score_threshold < 0:
    raise ValueError("atomic split_score_threshold must be non-negative")
if config.segmentation.geometry.normal_method not in {"cross", "sobel"}:
    raise ValueError("geometry.normal_method must be cross or sobel")
if config.loop.detection.method != "salad":
    raise ValueError("loop.detection.method must be salad")
```

Generate the digest from the exact UTF-8 bytes of `OmegaConf.to_yaml(merged, resolve=True, sort_keys=True)`.

Populate `configs/pipeline/default.yaml` exactly from Section 5 of the approved design. Populate `configs/pipeline/test.yaml` with the same schema but temporary fixture-style paths:

```yaml
input:
  image_dir: tests/fixtures/images
output:
  scene_name: pipeline_test
  cache_dir: .pytest_cache/laser/cache
  result_dir: .pytest_cache/laser/results
```

All remaining values must match `default.yaml`.

- [ ] **Step 4: Run strict-schema tests**

Run:

```bash
pytest tests/test_pipeline_config.py -q
```

Expected: all configuration tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/__init__.py pipeline/config.py configs/pipeline/default.yaml configs/pipeline/test.yaml tests/test_pipeline_config.py
git commit -m "feat: add strict pipeline configuration"
```

---

### Task 2: Canonical Image Manifest and Preflight Validation

**Files:**
- Create: `pipeline/manifest.py`
- Create: `pipeline/preflight.py`
- Test: `tests/test_pipeline_manifest.py`
- Source reference: `utils/image_paths.py` from `codex/loop-closure-corrected-pipeline`

**Interfaces:**
- Consumes: `PipelineConfig`.
- Produces: immutable `ImageManifest(paths: tuple[Path, ...])`.
- Produces: `discover_image_manifest(image_dir: str | Path, sample_stride: int) -> ImageManifest`.
- Produces: `validate_preflight(config: PipelineConfig, manifest: ImageManifest, cuda_available: bool) -> None`.

- [ ] **Step 1: Write manifest and preflight tests**

```python
from pathlib import Path

import pytest

from pipeline.config import load_pipeline_config
from pipeline.manifest import discover_image_manifest
from pipeline.preflight import validate_preflight


def test_manifest_filters_natural_sorts_then_samples(tmp_path):
    for name in ("frame10.JPG", "frame2.png", "frame1.jpeg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    manifest = discover_image_manifest(tmp_path, sample_stride=2)
    assert [path.name for path in manifest.paths] == ["frame1.jpeg", "frame10.JPG"]


def test_manifest_rejects_rescanning_mutation(tmp_path):
    (tmp_path / "1.png").write_bytes(b"x")
    manifest = discover_image_manifest(tmp_path, 1)
    (tmp_path / "2.png").write_bytes(b"x")
    assert [path.name for path in manifest.paths] == ["1.png"]


@pytest.mark.parametrize("stride", (0, -1))
def test_manifest_requires_positive_stride(tmp_path, stride):
    with pytest.raises(ValueError, match="sample_stride"):
        discover_image_manifest(tmp_path, stride)


def test_preflight_rejects_missing_checkpoint_before_model_load(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(6):
        (image_dir / f"{index}.png").write_bytes(b"x")
    loaded = load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            f"input.image_dir={image_dir}",
            f"model.checkpoint={tmp_path / 'missing.safetensors'}",
        ),
    )
    manifest = discover_image_manifest(image_dir, 1)
    with pytest.raises(FileNotFoundError, match="model.checkpoint"):
        validate_preflight(loaded.config, manifest, cuda_available=True)
```

Also cover empty directories, nonexistent directories, insufficient images for overlap, missing SALAD/DINO weights when loop is enabled, and CUDA requested while unavailable.

- [ ] **Step 2: Run the tests and verify imports fail**

Run:

```bash
pytest tests/test_pipeline_manifest.py -q
```

Expected: collection fails because `pipeline.manifest` and `pipeline.preflight` do not exist.

- [ ] **Step 3: Implement the immutable manifest**

Port natural sorting from the corrected branch, but return `Path` objects:

```python
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class ImageManifest:
    paths: tuple[Path, ...]

    def __len__(self) -> int:
        return len(self.paths)

    def as_strings(self) -> list[str]:
        return [str(path) for path in self.paths]


def natural_sort_key(path: str | Path) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", Path(path).name.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )
```

`discover_image_manifest` must validate the directory and stride, filter suffixes case-insensitively, sort once, sample once, reject an empty result, and store resolved paths in a tuple.

- [ ] **Step 4: Implement preflight checks**

`validate_preflight` must run without constructing Pi3, SALAD, or DINO. Check:

```python
required_files = [("model.checkpoint", config.model.checkpoint)]
if config.loop.enabled:
    required_files.extend(
        [
            ("loop.detection.salad_checkpoint", config.loop.detection.salad_checkpoint),
            ("loop.detection.dino_checkpoint", config.loop.detection.dino_checkpoint),
        ]
    )
```

Require `len(manifest) > window.overlap`, reject unsupported model dtype/device strings, and reject `inference_device == "cuda"` when `cuda_available` is false.

- [ ] **Step 5: Run manifest/preflight tests**

Run:

```bash
pytest tests/test_pipeline_manifest.py tests/test_pipeline_config.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/manifest.py pipeline/preflight.py tests/test_pipeline_manifest.py
git commit -m "feat: add canonical manifest and preflight checks"
```

---

### Task 3: Depth and Geometry Segmentation Strategies

**Files:**
- Create: `inference_engine/segmentation/__init__.py`
- Create: `inference_engine/segmentation/base.py`
- Create: `inference_engine/segmentation/confidence.py`
- Create: `inference_engine/segmentation/depth.py`
- Create: `inference_engine/segmentation/geometry.py`
- Create: `inference_engine/segmentation/registry.py`
- Modify: `inference_engine/utils/depth.py`
- Modify: `inference_engine/utils/geometry_segmentation.py`
- Test: `tests/test_segmentation_strategies.py`
- Modify tests: `tests/test_depth_segmentation_stages.py`
- Modify tests: `tests/test_geometry_segmentation.py`

**Interfaces:**
- Consumes: `SegmentationConfig`.
- Produces: `SegmentationResult`, `SegmentationStrategy`, `build_segmentation_strategy(config)`.
- Produces: `select_numpy_top_confidence_mask(confidence, keep_ratio)`.
- Produces: `build_temporal_graphs(results, temporal_iou_threshold)`.

- [ ] **Step 1: Write strategy contract tests**

```python
import numpy as np
import pytest

from pipeline.config import load_pipeline_config
from inference_engine.segmentation import (
    SegmentationResult,
    build_segmentation_strategy,
)


def _point_batch(frames=2, height=12, width=16):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    depth = np.broadcast_to(1.0 + xx / 100.0, (frames, height, width))
    x = np.broadcast_to(xx, depth.shape)
    y = np.broadcast_to(yy, depth.shape)
    return np.stack((x, y, depth), axis=-1).copy()


@pytest.mark.parametrize("method", ("depth", "geometry"))
def test_strategy_returns_compact_full_coverage_labels(method):
    config = load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            f"segmentation.method={method}",
            "segmentation.felzenszwalb.min_size=4",
        ),
    ).config.segmentation
    points = _point_batch()
    confidence = np.ones(points.shape[:-1], dtype=np.float32)
    strategy = build_segmentation_strategy(config)
    results = strategy.segment(points, confidence, images=None)
    assert len(results) == points.shape[0]
    for result in results:
        assert isinstance(result, SegmentationResult)
        assert result.labels.shape == points.shape[1:3]
        np.testing.assert_array_equal(
            np.unique(result.labels),
            np.arange(np.unique(result.labels).size),
        )


def test_geometry_receives_explicit_normal_threshold(monkeypatch):
    config = load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            "segmentation.method=geometry",
            "segmentation.geometry.normal_threshold_degrees=17",
        ),
    ).config.segmentation
    strategy = build_segmentation_strategy(config)
    assert strategy.normal_threshold_degrees == pytest.approx(17.0)


def test_numpy_confidence_keep_ratio_has_positive_semantics():
    confidence = np.arange(10, dtype=np.float32)
    mask = select_numpy_top_confidence_mask(confidence, 0.3)
    assert set(confidence[mask].tolist()) == {7.0, 8.0, 9.0}
```

- [ ] **Step 2: Run tests and verify module import failure**

Run:

```bash
pytest tests/test_segmentation_strategies.py -q
```

Expected: collection fails because `inference_engine.segmentation` does not exist.

- [ ] **Step 3: Implement shared result and strategy protocol**

In `base.py`:

```python
DiagnosticValue = float | int | bool | str


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    diagnostics: Mapping[str, DiagnosticValue]

    def __post_init__(self):
        labels = np.asarray(self.labels)
        if labels.ndim != 2:
            raise ValueError("segmentation labels must be two-dimensional")
        unique = np.unique(labels)
        np.testing.assert_array_equal(unique, np.arange(unique.size))


class SegmentationStrategy(Protocol):
    name: SegmentationMethod

    def segment(
        self,
        point_maps: np.ndarray,
        confidence: np.ndarray | None,
        images: np.ndarray | None,
    ) -> list[SegmentationResult]:
        raise NotImplementedError
```

Implement `build_temporal_graphs` by collecting
`labels = [result.labels for result in results]` and calling
`match_segmentation_seq(labels, iou_thresh=temporal_iou_threshold)` exactly
once.

- [ ] **Step 4: Replace percentile semantics inside low-level segmentation**

Rename low-level keyword parameters from `top_conf_percentile` to
`confidence_keep_ratio` in:

- `segment_depth_felzenszwalb_rag_stages`;
- `segment_depth_felzenszwalb_rag`;
- `segment_geometry_felzenszwalb_rag_stages`;
- `segment_geometry_felzenszwalb_rag`;
- baseline helper callers used by tests.

Use one helper:

```python
def select_numpy_top_confidence_mask(confidence, keep_ratio):
    values = np.asarray(confidence)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        raise ValueError("confidence contains no finite values")
    if not np.isfinite(keep_ratio) or not 0.0 < keep_ratio <= 1.0:
        raise ValueError("confidence keep_ratio must be in (0, 1]")
    threshold = np.quantile(
        finite_values,
        1.0 - keep_ratio,
        method="nearest",
    )
    return finite & (values >= threshold)
```

Remove `GEOMETRY_SEGMENTATION_PROFILES` and its legacy branch. Pass the explicit
Felzenszwalb and normal-threshold values from `SegmentationConfig`.

- [ ] **Step 5: Implement depth and geometry wrappers**

Each strategy must validate `(N,H,W,3)` point maps and optional `(N,H,W)`
confidence. It loops over frames, calls the fixed-source algorithm, compacts
labels, and emits diagnostics containing:

```python
{
    "method": self.name.value,
    "region_count": int(np.unique(labels).size),
}
```

Geometry additionally reports `normal_method` and
`normal_threshold_degrees`.

- [ ] **Step 6: Run focused algorithm and strategy tests**

Run:

```bash
pytest tests/test_segmentation_strategies.py tests/test_depth_segmentation_stages.py tests/test_geometry_segmentation.py -q
```

Expected: all tests pass; no `top_conf_percentile` keyword remains in these files.

- [ ] **Step 7: Commit**

```bash
git add inference_engine/segmentation inference_engine/utils/depth.py inference_engine/utils/geometry_segmentation.py tests/test_segmentation_strategies.py tests/test_depth_segmentation_stages.py tests/test_geometry_segmentation.py
git commit -m "refactor: add depth and geometry segmentation strategies"
```

---

### Task 4: Atomic Merge with Three Split Modes

**Files:**
- Create: `inference_engine/utils/post_merge_split.py`
- Create: `inference_engine/segmentation/atomic.py`
- Modify: `inference_engine/segmentation/registry.py`
- Modify: `inference_engine/utils/layer_atomic_geometry.py`
- Test: `tests/test_atomic_split_modes.py`
- Modify tests: `tests/test_layer_atomic_geometry.py`
- Modify tests: `tests/test_layer_atomic_integration.py`
- Source reference: `inference_engine/utils/post_merge_split.py` and `tests/test_post_merge_split.py` from `codex/auto-post-merge-split`

**Interfaces:**
- Consumes: `AtomicConfig.split_mode` and `split_score_threshold`.
- Produces: `AtomicSegmentationStrategy`.
- Produces: `refine_auto_regions(point_map, rgb_image, auto_labels, atom_labels, atom_scales, seg_min_size, normal_method, split_score_threshold, split_mode) -> tuple[np.ndarray, SplitDiagnostics]`.
- Keeps: `merge_layer_atoms(point_map, initial_labels, coarse_labels, depth_merge_threshold) -> np.ndarray` output unchanged when `split_mode == NONE`.

- [ ] **Step 1: Write the three-state acceptance tests**

Port the deterministic normal fixture from the atomic branch and express the
approved enum:

```python
@pytest.mark.parametrize(
    ("mode", "expected_regions"),
    (
        (AtomicSplitMode.NONE, 1),
        (AtomicSplitMode.CONSERVATIVE, 1),
        (AtomicSplitMode.NORMAL_ONLY, 2),
    ),
)
def test_atomic_split_modes_are_mutually_exclusive(
    monkeypatch,
    mode,
    expected_regions,
):
    points, labels, atoms, rgb = split_fixture()
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        post_merge_split,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )
    refined, diagnostics = post_merge_split.refine_auto_regions(
        points,
        rgb,
        labels,
        atoms,
        np.ones(1),
        seg_min_size=20,
        normal_method="cross",
        split_score_threshold=0.10,
        split_mode=mode,
    )
    assert np.unique(refined).size == expected_regions
    assert diagnostics.split_mode == mode.value
```

Add tests for two-to-four leaves, no recursion, small-child rejection, invalid
point coverage, compact labels, deterministic output, RGB confirmation,
normalized-gap scale invariance, and `normal_only` skipping RGB/gap computation.

- [ ] **Step 2: Run the tests and verify missing implementation**

Run:

```bash
pytest tests/test_atomic_split_modes.py -q
```

Expected: collection or import fails because the new split API is absent.

- [ ] **Step 3: Port the fixed split implementation**

Port constants and math from `codex/auto-post-merge-split`:

```python
NORMAL_BARRIER_RAD = np.deg2rad(30.0)
MAX_LEAVES = 4
MIN_CHILD_FRACTION = 0.02
```

Replace `split_aux_confirmation: bool` with `split_mode: AtomicSplitMode`.
Implement scoring exactly:

```python
if split_mode is AtomicSplitMode.NONE:
    return compact_labels(auto_labels), SplitDiagnostics(split_mode="none")
normal_gain = _normal_gain(normals, valid, parent_mask, candidate)
if split_mode is AtomicSplitMode.NORMAL_ONLY:
    score = normal_gain
else:
    confirmation = max(rgb_contrast, normalized_gap_contrast)
    score = normal_gain * confirmation
```

Both active split modes must still enforce marker count, child area, full
coverage, maximum four leaves, non-recursion, finite-value handling, compact
labels, and one shared threshold.

- [ ] **Step 4: Expose merge metadata without changing merge-only output**

Retain the branch's `AtomMergeResult` idea:

```python
@dataclass(frozen=True)
class AtomMergeResult:
    labels: np.ndarray
    atom_labels: np.ndarray
    atom_scales: np.ndarray
```

`merge_layer_atoms` returns only `result.labels`; the private
`_merge_layer_atoms_with_metadata` supplies atom scales to split scoring.
Rename the combined entry to `segment_point_map_atomic` and return
`(labels, SplitDiagnostics)` so the strategy can preserve diagnostics.

- [ ] **Step 5: Register `atomic` as one strategy**

`build_segmentation_strategy` must have exactly three registry keys:

```python
STRATEGY_FACTORIES = {
    SegmentationMethod.DEPTH: DepthSegmentationStrategy,
    SegmentationMethod.GEOMETRY: GeometrySegmentationStrategy,
    SegmentationMethod.ATOMIC: AtomicSegmentationStrategy,
}
```

There must be no `layer_atomic_split` public registry entry.

- [ ] **Step 6: Run atomic and all segmentation tests**

Run:

```bash
pytest tests/test_atomic_split_modes.py tests/test_layer_atomic_geometry.py tests/test_layer_atomic_integration.py tests/test_segmentation_strategies.py -q
```

Expected: all tests pass for `none`, `conservative`, and `normal_only`.

- [ ] **Step 7: Commit**

```bash
git add inference_engine/utils/post_merge_split.py inference_engine/utils/layer_atomic_geometry.py inference_engine/segmentation/atomic.py inference_engine/segmentation/registry.py tests/test_atomic_split_modes.py tests/test_layer_atomic_geometry.py tests/test_layer_atomic_integration.py
git commit -m "feat: add atomic three-mode splitting"
```

---

### Task 5: Single Anchor Propagator and Injectable Streaming Engine

**Files:**
- Create: `inference_engine/anchor_propagation.py`
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `inference_engine/utils/lsa.py`
- Test: `tests/test_anchor_propagation_contract.py`
- Modify: `inference_engine/__init__.py`

**Interfaces:**
- Consumes: a `SegmentationStrategy`, `AnchorPropagationConfig`, and positive registration keep ratio.
- Produces: `AnchorPropagator.propagate(source_points, target_points, source_graphs, target_graphs, overlap) -> torch.Tensor`.
- Produces: the exact `StreamingWindowEngine.__init__` signature specified in Step 4.

- [ ] **Step 1: Freeze anchor output before moving code**

Create a deterministic two-frame graph fixture using the existing graph vertex
objects and assert both shape and fixed values:

```python
def test_anchor_propagation_keeps_existing_identity_scale_result():
    source_points, target_points, source_graphs, target_graphs = identity_anchor_fixture()
    propagator = AnchorPropagator(correspondence_iou_threshold=0.4)
    scale = propagator.propagate(
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap=1,
    )
    assert tuple(scale.shape) == (*target_points.shape[:-1], 1)
    torch.testing.assert_close(scale, torch.ones_like(scale))
```

Add a spy test proving each window transition invokes one propagator instance
once, independent of the selected segmentation strategy.

- [ ] **Step 2: Run the test and verify the class is missing**

Run:

```bash
pytest tests/test_anchor_propagation_contract.py -q
```

Expected: collection fails because `AnchorPropagator` is missing.

- [ ] **Step 3: Move, do not rewrite, anchor propagation**

Move the body of `refine_depth_segments` and
`align_adjacent_windows_depth_segments` from `inference_engine/utils/lsa.py`
behind:

```python
@dataclass(frozen=True)
class AnchorPropagator:
    correspondence_iou_threshold: float = 0.4

    def propagate(
        self,
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap: int,
    ) -> torch.Tensor:
        scale = align_adjacent_windows_depth_segments(
            source_points[..., -1],
            target_points[..., -1],
            source_graphs,
            target_graphs,
            overlap,
            corr_iou_thresh=self.correspondence_iou_threshold,
        )
        return torch.from_numpy(scale[..., None])
```

Keep the nested IoU-weighted `_propagate_scale_cache` and `_get_scale_mask`
functions byte-for-byte equivalent. `lsa.py` should only retain graph matching
helpers still used by segmentation.

- [ ] **Step 4: Refactor the base streaming engine constructor**

Remove `top_conf_percentile`, `depth_refine`, `segment_mode`,
`normal_method`, and `geometry_seg_profile`. Add:

```python
def __init__(
    self,
    delegate: nn.Module,
    inference_device: str,
    dtype: torch.dtype,
    segmentation_strategy: SegmentationStrategy,
    anchor_propagator: AnchorPropagator,
    registration_confidence_keep_ratio: float,
    anchor_enabled: bool,
    temporal_iou_threshold: float,
    window_size: int,
    overlap: int,
    cache_root: str,
    intermediate_device: str = "cuda",
    process_device: str = "cpu",
    benchmark_latency: bool = True,
):
```

`_build_segment_graph` must call the injected strategy, then
`build_temporal_graphs` exactly once. `_registration_worker` must call
`anchor_propagator.propagate` only when `anchor_enabled` is true. Pass
`working_window.get("images")` to segmentation so conservative atomic can use
RGB without reloading images.

- [ ] **Step 5: Run anchor, base-engine, and segmentation tests**

Run:

```bash
pytest tests/test_anchor_propagation_contract.py tests/test_segmentation_strategies.py tests/test_registration_confidence.py -q
```

Expected: all pass; `rg "depth_refine|top_conf_percentile|geometry_seg_profile" inference_engine/streaming_window_engine.py` returns no matches.

- [ ] **Step 6: Commit**

```bash
git add inference_engine/anchor_propagation.py inference_engine/streaming_window_engine.py inference_engine/utils/lsa.py inference_engine/__init__.py tests/test_anchor_propagation_contract.py
git commit -m "refactor: inject segmentation and anchor services"
```

---

### Task 6: Loop Contracts, Typed Caches, Shared Detection, and Optimizer Configuration

**Files:**
- Create: `loop_closure/methods/__init__.py`
- Create: `loop_closure/methods/base.py`
- Create: `loop_closure/methods/shared.py`
- Modify: `loop_closure/loop_model.py`
- Modify: `loop_closure/utils/sim3loop.py`
- Modify: `inference_engine/utils/registration_confidence.py`
- Test: `tests/test_loop_method_contracts.py`

**Interfaces:**
- Consumes: `ImageManifest`, `DetectionConfig`, `OptimizerConfig`.
- Produces: `LoopCandidate`, `LoopConstraint`, `LoopSolution`, `ReconstructionResult`, `WindowCache`, and `LoopClosureStrategy`.
- Produces: `detect_loop_candidates(detection_config, image_manifest, output_path) -> tuple[LoopCandidate, ...]`.
- Produces: method-independent Sim(3) validation and metrics.

- [ ] **Step 1: Write cache and shared-candidate contract tests**

```python
def test_loop_candidate_is_immutable_and_scored():
    candidate = LoopCandidate(frame_a=90, frame_b=10, similarity=0.81)
    assert candidate.frame_a == 90
    with pytest.raises(FrozenInstanceError):
        candidate.similarity = 0.5


def test_cache_rejects_cross_method_loading():
    cache = window_cache_fixture(loop_method=LoopMethod.TRADITIONAL)
    with pytest.raises(ValueError, match="traditional.*corrected"):
        WindowCache.from_payload(
            cache.to_payload(),
            expected_method=LoopMethod.CORRECTED,
        )


def test_both_methods_receive_same_candidate_tuple():
    candidates = (
        LoopCandidate(frame_a=90, frame_b=10, similarity=0.81),
    )
    traditional = recording_strategy(LoopMethod.TRADITIONAL)
    corrected = recording_strategy(LoopMethod.CORRECTED)
    traditional.build_constraints([], candidates)
    corrected.build_constraints([], candidates)
    assert traditional.received_candidates is candidates
    assert corrected.received_candidates is candidates
```

Also test schema version, frame range, method-specific state tags, non-finite
Sim(3), and positive-scale validation.

- [ ] **Step 2: Run the tests and verify missing contracts**

Run:

```bash
pytest tests/test_loop_method_contracts.py -q
```

Expected: collection fails because `loop_closure.methods.base` is missing.

- [ ] **Step 3: Implement immutable contracts**

Use:

```python
WINDOW_CACHE_SCHEMA_VERSION = 1
Sim3 = tuple[torch.Tensor | float, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class LoopCandidate:
    frame_a: int
    frame_b: int
    similarity: float


@dataclass(frozen=True)
class LoopConstraint:
    window_a: int
    window_b: int
    measurement: Sim3
    candidate: LoopCandidate


@dataclass(frozen=True)
class LoopSolution:
    optimized_transforms: tuple[Sim3, ...]
    constraints: tuple[LoopConstraint, ...]
    used_no_loop_path: bool


@dataclass(frozen=True)
class ReconstructionResult:
    payload: Mapping[str, object]
    summary: Mapping[str, object]


@dataclass
class WindowCache:
    schema_version: int
    loop_method: LoopMethod
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    segmentation_labels: tuple[np.ndarray, ...]
    anchor_scale_mask: torch.Tensor | None
    loop_state: dict[str, object]
```

`to_payload` must emit plain dictionaries suitable for `torch.save`.
`from_payload` must validate exact schema version, loop method, indices, tensor
fields, and the method-specific state tag.

The protocol is:

```python
class LoopClosureStrategy(Protocol):
    name: LoopMethod

    def create_window_engine(self, **dependencies) -> StreamingWindowEngine:
        raise NotImplementedError

    def build_constraints(
        self,
        caches: Sequence[WindowCache],
        candidates: tuple[LoopCandidate, ...],
    ) -> list[LoopConstraint]:
        raise NotImplementedError

    def optimize(
        self,
        caches: Sequence[WindowCache],
        constraints: Sequence[LoopConstraint],
    ) -> LoopSolution:
        raise NotImplementedError

    def aggregate(
        self,
        caches: Sequence[WindowCache],
        solution: LoopSolution,
    ) -> ReconstructionResult:
        raise NotImplementedError
```

- [ ] **Step 4: Convert SALAD detector to typed config and explicit manifest**

`LoopDetector` must accept:

```python
def __init__(
    self,
    detection_config: DetectionConfig,
    image_manifest: ImageManifest,
    output_path: Path,
):
```

Delete directory rescanning and sampling. Preserve existing SALAD descriptor,
top-k, similarity, and NMS math. `run()` returns immutable scored
`LoopCandidate` objects. DINO path remains passed through the existing VPR
model configuration adapter, but no legacy YAML dictionary is accepted.

- [ ] **Step 5: Convert optimizer to `OptimizerConfig`**

Change:

```python
Sim3LoopOptimizer(
    implementation=config.implementation,
    max_iterations=config.max_iterations,
    initial_damping=config.initial_damping,
    device="cpu",
)
```

Remove dictionary indexing and `eval(lambda_init)`. Preserve the existing
residual, Jacobian, solver, damping, and convergence math.

- [ ] **Step 6: Run contract and existing confidence tests**

Run:

```bash
pytest tests/test_loop_method_contracts.py tests/test_registration_confidence.py -q
```

Expected: all pass; shared candidates are immutable and cache cross-loading is rejected.

- [ ] **Step 7: Commit**

```bash
git add loop_closure/methods/base.py loop_closure/methods/shared.py loop_closure/methods/__init__.py loop_closure/loop_model.py loop_closure/utils/sim3loop.py inference_engine/utils/registration_confidence.py tests/test_loop_method_contracts.py
git commit -m "refactor: add loop contracts and shared services"
```

---

### Task 7: Traditional Loop Strategy

**Files:**
- Create: `loop_closure/methods/traditional.py`
- Test: `tests/test_traditional_loop_method.py`
- Source reference: cloud-updated `inference_engine/streaming_window_engine_lc.py`
- Source reference: cloud-updated `loop_closure/loop_closure.py`

**Interfaces:**
- Consumes: shared manifest, candidates, confidence mask helper, optimizer, segmenter, and anchor propagator.
- Produces: `TraditionalWindowEngine` and `TraditionalLoopClosureStrategy`.
- Produces cache state tag `traditional` with `relative_sim3` and `anchor_scale_applied=False`.

- [ ] **Step 1: Write traditional semantic tests**

Cover the branch-specific behavior:

```python
def test_traditional_window_defers_sim3_and_anchor_scale_application(
    monkeypatch,
    tmp_path,
):
    engine = build_traditional_test_engine(monkeypatch, tmp_path)
    cache = run_second_window(engine)
    assert cache.loop_state["tag"] == "traditional"
    assert cache.loop_state["anchor_scale_applied"] is False
    torch.testing.assert_close(cache.local_points, expected_unscaled_points)


def test_traditional_aggregation_applies_delayed_transforms_once():
    strategy = traditional_strategy_fixture()
    result = strategy.aggregate(
        traditional_cache_fixture(),
        traditional_solution_fixture(),
    )
    torch.testing.assert_close(
        result.payload["local_points"],
        expected_delayed_points,
    )


def test_traditional_constraint_keeps_baseline_compute_sim3_ab():
    constraint = strategy.build_constraints(caches, candidates)[0]
    assert_sim3_close(constraint.measurement, expected_compute_sim3_ab)


def test_traditional_uses_positive_shared_confidence_ratio(monkeypatch):
    calls = []
    monkeypatch.setattr(
        shared,
        "select_top_confidence_mask",
        lambda confidence, keep_ratio: calls.append(keep_ratio)
        or torch.ones_like(confidence, dtype=torch.bool),
    )
    strategy.build_constraints(caches, candidates)
    assert calls and set(calls) == {0.3}
```

Add a no-loop test returning the original sequential transforms and a cache
schema test.

- [ ] **Step 2: Run the tests and verify strategy is absent**

Run:

```bash
pytest tests/test_traditional_loop_method.py -q
```

Expected: collection fails because the traditional module does not exist.

- [ ] **Step 3: Port the traditional window worker**

Port the delayed behavior exactly:

- register against the previous cached local points;
- store the returned transform as `relative_sim3`;
- compute segmentation and anchor scale mask;
- do not multiply current local points by the Sim(3) scale;
- do not apply Sim(3) to camera poses;
- do not multiply the anchor scale mask before saving.

Replace both direct `torch.quantile` calls with
`select_top_confidence_mask(confidence, keep_ratio)`.

- [ ] **Step 4: Port traditional constraints and aggregation**

Preserve `compute_sim3_ab((s_a,R_a,t_a),(s_b,R_b,t_b))` as the loop
measurement. Preserve traditional delayed transform accumulation and anchor
scale application order, but operate on copied cache payloads so aggregation
does not mutate input caches.

No loop candidates must return `LoopSolution` with the original sequential
transform tuple and `used_no_loop_path=True` without invoking the optimizer.

- [ ] **Step 5: Run traditional and shared tests**

Run:

```bash
pytest tests/test_traditional_loop_method.py tests/test_loop_method_contracts.py tests/test_registration_confidence.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add loop_closure/methods/traditional.py tests/test_traditional_loop_method.py
git commit -m "feat: add traditional loop strategy"
```

---

### Task 8: Corrected Loop Strategy and Method Registry

**Files:**
- Create: `loop_closure/methods/corrected.py`
- Create: `loop_closure/methods/registry.py`
- Modify: `loop_closure/methods/__init__.py`
- Test: `tests/test_corrected_loop_method.py`
- Source reference: corrected-base `inference_engine/streaming_window_engine_lc.py`
- Source reference: corrected-base `loop_closure/loop_closure.py`
- Source tests: corrected-base `tests/test_loop_closure_pipeline.py`
- Source tests: corrected-base `tests/test_streaming_window_engine_lc_pipeline.py`

**Interfaces:**
- Consumes: the same shared dependencies as traditional.
- Produces: `CorrectedWindowEngine`, `CorrectedLoopClosureStrategy`, and `build_loop_strategy`.
- Produces cache state tag `corrected` with `sim3_abs`, optional `sim3_edge`, and `anchor_scale_applied=True`.

- [ ] **Step 1: Port corrected tests to the new contract before moving code**

At the top of `tests/test_corrected_loop_method.py`, port the deterministic
tensor/cache fixtures from
`codex/loop-closure-corrected-pipeline:tests/test_streaming_window_engine_lc_pipeline.py`.
Rename its engine constructor helper to `build_corrected_test_engine` and its
second-window helper to `run_second_window`. Tests must include:

```python
def test_corrected_window_uses_corrected_previous_window_for_registration(
    monkeypatch,
    tmp_path,
):
    engine = build_corrected_test_engine(monkeypatch, tmp_path)
    run_second_window(engine)
    assert engine.registration_sources[-1] is engine.previous_corrected_points


def test_corrected_window_applies_anchor_scale_immediately(
    monkeypatch,
    tmp_path,
):
    cache = run_second_window(build_corrected_test_engine(monkeypatch, tmp_path))
    assert cache.loop_state["anchor_scale_applied"] is True
    torch.testing.assert_close(cache.local_points, expected_scaled_points)


def test_corrected_cache_has_absolute_and_relative_sim3(
    monkeypatch,
    tmp_path,
):
    cache = run_second_window(build_corrected_test_engine(monkeypatch, tmp_path))
    assert "sim3_abs" in cache.loop_state
    assert "sim3_edge" in cache.loop_state


def test_corrected_loop_measurement_is_local_coordinate_constraint(
    sim3_abs_a,
    sim3_abs_b,
    global_alignment_a,
    global_alignment_b,
):
    constraint = build_local_loop_constraint(
        sim3_abs_a,
        sim3_abs_b,
        global_alignment_a,
        global_alignment_b,
    )
    residual = compose(
        constraint,
        compose(inverse(sim3_abs_a), sim3_abs_b),
    )
    assert_sim3_identity(residual)


def test_corrected_aggregation_applies_only_optimization_delta_once():
    strategy, caches, solution, expected = corrected_delta_fixture()
    actual = strategy.aggregate(caches, solution)
    torch.testing.assert_close(
        actual.payload["local_points"],
        expected["local_points"],
    )
    torch.testing.assert_close(
        actual.payload["camera_poses"],
        expected["camera_poses"],
    )
```

Also port invalid Sim(3), disjoint confidence, cache-count, no-loop, optimizer
edge-count, and diagnostic-metric tests.

- [ ] **Step 2: Run corrected tests and verify missing module**

Run:

```bash
pytest tests/test_corrected_loop_method.py -q
```

Expected: collection fails because the corrected strategy module is absent.

- [ ] **Step 3: Move corrected window semantics into `CorrectedWindowEngine`**

Preserve the corrected branch order:

```text
register current raw overlap to previous corrected overlap
-> save sim3_abs
-> derive sim3_edge
-> apply absolute scale to local points
-> apply absolute Sim(3) to poses
-> segment aligned points
-> propagate anchor scale
-> apply anchor scale immediately
-> cache corrected window
```

Populate the corrected cache state tag and never store ambiguous `sim3`.

- [ ] **Step 4: Move corrected loop constraint and delta aggregation**

Preserve:

```python
global_correction = compose(
    global_alignment_b,
    inverse(global_alignment_a),
)
constraint_ab = compose(
    inverse(sim3_abs_b),
    compose(global_correction, sim3_abs_a),
)
```

Optimization consumes sequential `sim3_edge` values and local loop
constraints. Reaccumulate optimized edges from identity. Aggregate with:

```python
delta_i = compose(optimized_abs_i, inverse(original_abs_i))
```

Apply each `delta_i` once to local point scale and camera pose. Do not reapply
the anchor scale mask.

- [ ] **Step 5: Add the two-method registry**

```python
LOOP_STRATEGIES = {
    LoopMethod.TRADITIONAL: TraditionalLoopClosureStrategy,
    LoopMethod.CORRECTED: CorrectedLoopClosureStrategy,
}


def build_loop_strategy(method, **dependencies):
    return LOOP_STRATEGIES[method](**dependencies)
```

Assert the registry key set is exactly the two approved enum values.

- [ ] **Step 6: Run both loop strategy suites**

Run:

```bash
pytest tests/test_corrected_loop_method.py tests/test_traditional_loop_method.py tests/test_loop_method_contracts.py -q
```

Expected: all pass; both strategies consume the same typed candidates and
reject each other's caches.

- [ ] **Step 7: Commit**

```bash
git add loop_closure/methods/corrected.py loop_closure/methods/registry.py loop_closure/methods/__init__.py tests/test_corrected_loop_method.py
git commit -m "feat: add corrected loop strategy"
```

---

### Task 9: Unified Runner, Diagnostics, and Single CLI

**Files:**
- Create: `pipeline/diagnostics.py`
- Create: `pipeline/runner.py`
- Create: `run_laser.py`
- Test: `tests/test_pipeline_runner.py`
- Modify: `loop_closure/__init__.py`
- Modify: `eval_launch.py`

**Interfaces:**
- Consumes: `LoadedPipelineConfig`, model loader, manifest, segmenter, anchor propagator, loop strategy, and SALAD detector.
- Produces: `PipelineRunner.run() -> ReconstructionResult`.
- Produces: `run_from_config(path, overrides=())`.
- CLI accepts only `--config` and repeated `--set KEY=VALUE`.

- [ ] **Step 1: Write dependency-injected runner tests**

```python
@pytest.mark.parametrize(
    ("segmentation_method", "loop_method"),
    (
        ("depth", "traditional"),
        ("depth", "corrected"),
        ("geometry", "traditional"),
        ("geometry", "corrected"),
        ("atomic", "traditional"),
        ("atomic", "corrected"),
    ),
)
def test_runner_selects_requested_strategies(
    tmp_path,
    segmentation_method,
    loop_method,
):
    dependencies = recording_dependencies(tmp_path)
    result = run_test_pipeline(
        dependencies,
        overrides=(
            f"segmentation.method={segmentation_method}",
            f"loop.method={loop_method}",
        ),
    )
    assert dependencies.segmentation_calls == [segmentation_method]
    assert dependencies.loop_calls == [loop_method]
    assert result.summary["segmentation_method"] == segmentation_method
    assert result.summary["loop_method"] == loop_method
```

Add tests proving:

- preflight executes before model loader;
- the same manifest object reaches inference and SALAD;
- no-loop candidates do not invoke joint Pi3 or optimizer;
- output files use the resolved config hash;
- CLI parser rejects old flags such as `--segment_mode`;
- background worker exceptions propagate to the main thread instead of silently
  yielding incomplete caches.

- [ ] **Step 2: Run tests and verify runner is absent**

Run:

```bash
pytest tests/test_pipeline_runner.py -q
```

Expected: collection fails because `pipeline.runner` is absent.

- [ ] **Step 3: Implement diagnostics writers**

Use JSON-safe conversion for NumPy scalars, tensors, enums, and paths. Write
atomically through a temporary file followed by `Path.replace`. Required
files:

```text
resolved_config.yaml
run_summary.json
loop_candidates.json
loop_constraints.json
segmentation_diagnostics.json
```

`run_summary.json` contains config hash, Git commit, method names, split mode,
image count and endpoints, window count, candidate count, valid constraint
count, no-loop flag, split totals, and stage timings.

- [ ] **Step 4: Implement orchestration**

The runner order is fixed:

```python
loaded = load_pipeline_config(path, overrides)
config = loaded.config
manifest = discover_image_manifest(
    config.input.image_dir,
    config.input.sample_stride,
)
validate_preflight(loaded.config, manifest, torch.cuda.is_available())
output_root = Path(config.output.result_dir) / config.output.scene_name
write_resolved_config(output_root, loaded)
model = dependencies.load_pi3(loaded.config.model)
segmenter = build_segmentation_strategy(loaded.config.segmentation)
anchor = AnchorPropagator(
    config.anchor_propagation.correspondence_iou_threshold
)
loop_strategy = build_loop_strategy(
    config.loop.method,
    config=config,
    model=model,
    manifest=manifest,
)
window_engine = loop_strategy.create_window_engine(
    segmentation_strategy=segmenter,
    anchor_propagator=anchor,
    anchor_enabled=config.anchor_propagation.enabled,
    cache_root=config.output.cache_dir,
)
caches = run_windows(window_engine, manifest)
candidates = (
    detect_loop_candidates(
        config.loop.detection,
        manifest,
        output_root / "loop_candidates.json",
    )
    if config.loop.enabled
    else ()
)
constraints = loop_strategy.build_constraints(caches, candidates)
solution = loop_strategy.optimize(caches, constraints)
result = loop_strategy.aggregate(caches, solution)
write_diagnostics(output_root, loaded, manifest, candidates, solution, result)
save_for_viser(
    result.payload,
    config.output.scene_name,
    config.output.result_dir,
    inverse_extrinsic=False,
)
```

The manifest instance is created once. Cache count must equal the count derived
from `window.size` and `window.overlap`.

- [ ] **Step 5: Implement the only public CLI**

```python
def build_parser():
    parser = argparse.ArgumentParser("LASER modular pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser
```

`run_laser.py` imports `run_from_config`, parses only these fields, and returns
exit code zero on success. It must not define method-specific flags.

Update `eval_launch.py` imports to use the strategy registry and typed config
when it invokes streaming inference; retain unrelated evaluation flags.

- [ ] **Step 6: Run runner and six core combination tests**

Run:

```bash
pytest tests/test_pipeline_runner.py -q
```

Expected: all six method combinations pass with mocks.

- [ ] **Step 7: Commit**

```bash
git add pipeline/diagnostics.py pipeline/runner.py run_laser.py loop_closure/__init__.py eval_launch.py tests/test_pipeline_runner.py
git commit -m "feat: add unified modular pipeline runner"
```

---

### Task 10: Ten-Configuration Matrix Runner

**Files:**
- Create: `scripts/verify_pipeline_matrix.py`
- Test: `tests/test_pipeline_matrix.py`

**Interfaces:**
- Consumes: a base YAML path and optional forwarded `--set` values.
- Produces: exactly 10 unique resolved run configurations.
- Supports: `--dry-run` to print and validate configs without loading models.

- [ ] **Step 1: Write exact matrix tests**

```python
def test_matrix_contains_exactly_ten_unique_configurations():
    matrix = build_matrix()
    assert len(matrix) == 10
    assert len({entry.name for entry in matrix}) == 10
    assert {
        (entry.segmentation_method, entry.atomic_split_mode, entry.loop_method)
        for entry in matrix
    } == {
        ("depth", None, "traditional"),
        ("depth", None, "corrected"),
        ("geometry", None, "traditional"),
        ("geometry", None, "corrected"),
        ("atomic", "none", "traditional"),
        ("atomic", "none", "corrected"),
        ("atomic", "conservative", "traditional"),
        ("atomic", "conservative", "corrected"),
        ("atomic", "normal_only", "traditional"),
        ("atomic", "normal_only", "corrected"),
    }


def test_dry_run_validates_all_configs_without_loading_models(monkeypatch):
    monkeypatch.setattr(
        matrix_module,
        "run_from_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model runner called")
        ),
    )
    assert matrix_module.main(
        ["--config", "configs/pipeline/test.yaml", "--dry-run"]
    ) == 0
```

- [ ] **Step 2: Run tests and verify script module is absent**

Run:

```bash
pytest tests/test_pipeline_matrix.py -q
```

Expected: collection fails because the matrix script is missing.

- [ ] **Step 3: Implement explicit matrix entries**

Use a frozen `MatrixEntry` dataclass. Do not derive the matrix through implicit
Cartesian products; list the 10 approved entries so accidental public methods
cannot silently expand experiments.

For depth and geometry, omit the atomic split override. For atomic, pass the
exact `segmentation.atomic.split_mode` override. Always add a unique
`output.scene_name`, `output.cache_dir`, and `output.result_dir` suffix.

- [ ] **Step 4: Run dry-run and unit tests**

Run:

```bash
pytest tests/test_pipeline_matrix.py -q
python scripts/verify_pipeline_matrix.py --config configs/pipeline/test.yaml --dry-run
```

Expected: tests pass and the command prints 10 unique resolved configurations
without importing or loading Pi3/SALAD weights.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_pipeline_matrix.py tests/test_pipeline_matrix.py
git commit -m "test: add ten-configuration pipeline matrix"
```

---

### Task 11: Remove Legacy Surfaces, Document the New Workflow, and Verify

**Files:**
- Delete: all paths listed in the plan's delete map
- Create: `docs/pipeline-configuration.md`
- Modify: `README.md`
- Modify: imports in remaining tests and modules found by the legacy identifier scan

**Interfaces:**
- Consumes: all prior tasks.
- Produces: one documented entry point, no public legacy fields, and a verified full suite.

- [ ] **Step 1: Add a legacy-identifier guard test**

In `tests/test_pipeline_config.py`, add:

```python
def test_public_pipeline_source_contains_no_legacy_parameter_names():
    roots = [
        Path("pipeline"),
        Path("run_laser.py"),
        Path("inference_engine/streaming_window_engine.py"),
        Path("inference_engine/segmentation"),
        Path("loop_closure/methods"),
    ]
    forbidden = {
        "top_conf_percentile",
        "depth_refine",
        "segment_mode",
        "geometry_seg_profile",
        "split_aux_confirmation",
        "registration_top_confidence_ratio",
    }
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in ([root] if root.is_file() else root.rglob("*.py"))
    )
    for name in forbidden:
        assert name not in text
```

Add a second guard asserting `rg`-equivalent Python source search finds no
`HART`, `HART_AP`, or `hart_anchor` identifiers in new runtime modules.

- [ ] **Step 2: Run the guard and observe remaining legacy files**

Run:

```bash
pytest tests/test_pipeline_config.py::test_public_pipeline_source_contains_no_legacy_parameter_names -q
```

Expected: fails until all public legacy names are removed from the new runtime
surface.

- [ ] **Step 3: Delete replaced entry points and implementations**

Delete the exact files in the delete map. Update imports to:

- use `pipeline.manifest` instead of `utils.image_paths`;
- use `loop_closure.methods` instead of `loop_closure.loop_closure`;
- use `CorrectedWindowEngine` or `TraditionalWindowEngine` through the strategy
  registry instead of `StreamingWindowEngineLC`;
- use `pipeline.config` instead of `loop_closure.utils.config_utils`.

Do not add compatibility aliases.

- [ ] **Step 4: Write configuration and experiment documentation**

`docs/pipeline-configuration.md` must include:

- the full default YAML;
- exact valid enums;
- positive keep-ratio semantics;
- the three atomic split modes and their score formulas;
- traditional versus corrected cache/constraint semantics;
- the single CLI and `--set` examples;
- all 10 matrix configurations;
- diagnostic output files;
- AutoDL/KITTI example commands.

Update `README.md` to link this guide and replace all `demo.py`,
`demo_lc.py`, `--segment_mode`, `--depth_refine`, and old loop-config examples
with:

```bash
python run_laser.py --config configs/pipeline/default.yaml
python run_laser.py \
  --config configs/pipeline/default.yaml \
  --set segmentation.method=atomic \
  --set segmentation.atomic.split_mode=normal_only \
  --set loop.method=traditional
```

- [ ] **Step 5: Run source and whitespace checks**

Run:

```bash
git diff --check
rg -n "top_conf_percentile|depth_refine|segment_mode|geometry_seg_profile|split_aux_confirmation|registration_top_confidence_ratio" pipeline run_laser.py inference_engine/streaming_window_engine.py inference_engine/segmentation loop_closure/methods
rg -n "HART|HART_AP|hart_anchor" pipeline inference_engine/anchor_propagation.py loop_closure/methods
```

Expected: `git diff --check` succeeds; both `rg` commands return no matches.

- [ ] **Step 6: Build and run focused suites**

Run:

```bash
python setup.py build_ext --inplace
pytest tests/test_pipeline_config.py tests/test_pipeline_manifest.py tests/test_segmentation_strategies.py tests/test_atomic_split_modes.py tests/test_anchor_propagation_contract.py tests/test_loop_method_contracts.py tests/test_traditional_loop_method.py tests/test_corrected_loop_method.py tests/test_pipeline_runner.py tests/test_pipeline_matrix.py -q
```

Expected: extension build succeeds and all focused tests pass.

- [ ] **Step 7: Run the full CPU suite**

Run:

```bash
pytest -q
```

Expected: all tests pass with no collection errors, background worker errors,
or legacy-import failures.

- [ ] **Step 8: Run the 10-configuration dry matrix**

Run:

```bash
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/test.yaml \
  --dry-run
```

Expected: exactly 10 unique configurations validate and print.

- [ ] **Step 9: Record GPU validation commands**

When weights, CUDA, and KITTI data exist, run:

```bash
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/default.yaml \
  --set input.image_dir=data/00/image_2 \
  --set output.scene_name=kitti_00
```

Expected: every run writes `resolved_config.yaml`, `run_summary.json`,
`loop_candidates.json`, `loop_constraints.json`,
`segmentation_diagnostics.json`, and reconstruction artifacts. If the local
environment lacks weights, CUDA, or data, record the preflight result and the
exact deferred command rather than claiming GPU validation.

- [ ] **Step 10: Commit final cleanup and documentation**

```bash
git add -A
git commit -m "docs: finalize modular pipeline workflow"
```

## Plan Completion Check

Before declaring implementation complete:

1. Confirm each approved design requirement maps to a completed task above.
2. Confirm `SegmentationMethod` has three members and `LoopMethod` has two.
3. Confirm the atomic split enum has three members and the matrix has 10 unique entries.
4. Confirm only one anchor propagation implementation exists.
5. Confirm traditional and corrected receive the identical manifest and candidate tuple.
6. Confirm all legacy public fields are absent.
7. Confirm all focused tests, the full suite, and dry matrix passed in fresh command output.
8. Use `superpowers:verification-before-completion` before reporting success.
