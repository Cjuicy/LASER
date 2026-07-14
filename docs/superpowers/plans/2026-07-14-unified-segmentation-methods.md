# Unified LASER Segmentation Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the validated LASER depth, LASER-Geometry, and layer-atomic segmentation methods from one branch with a shared downstream graph and streaming pipeline.

**Architecture:** Keep all three method-specific segmentation functions intact and add a thin router at `make_sp_graph(...)`. The normal and loop-closure engines both call one parent helper, while both demos expose the same mode options and geometry defaults.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, scikit-image, Cython, pytest.

## Global Constraints

- Branch from `feature/layer-atomic-geometry` at `7dca624`.
- Port LASER-Geometry segmentation behavior from `Cjuicy/LASER-Geometry/main` at `340c599`.
- Do not behaviorally edit `inference_engine/utils/depth.py` or `inference_engine/utils/layer_atomic_geometry.py`.
- Formal comparisons use Felzenszwalb `scale=300`, `sigma=1.1`, and `min_size=500`.
- Geometry defaults to `baseline_params`; `legacy` remains explicit and uses `200 / 1.0 / 300`.
- Do not change segment matching, scale anchors, scale propagation, Sim(3), caching, or loop closure.
- Do not add comparison metrics or experiment reports.

---

### Task 1: Port the pinned LASER-Geometry segmentation core

**Files:**
- Create: `inference_engine/utils/segmentation_trace.py`
- Create: `inference_engine/utils/geometry_segmentation.py`
- Modify: `inference_engine/utils/geometry.py`
- Create: `tests/test_geometry_segmentation.py`

**Interfaces:**
- Consumes: one depth map `(H, W)`, optional confidence, point map, intrinsic, profile parameters, and normal method.
- Produces: `segment_geometry_felzenszwalb_rag(...) -> np.ndarray`, `segment_geometry_felzenszwalb_rag_baseline_params(...) -> np.ndarray`, and their stage-returning variants.

- [ ] **Step 1: Write failing characterization tests**

Add tests that pin source behavior without routing:

```python
import numpy as np

from inference_engine.utils.geometry import build_geometry_info_np
from inference_engine.utils.geometry_segmentation import (
    merge_regions_geometry,
    segment_geometry_felzenszwalb_rag,
    segment_geometry_felzenszwalb_rag_baseline_params_stages,
)


def test_geometry_merge_joins_similar_adjacent_regions():
    labels = np.tile(np.array([10, 10, 20, 20]), (4, 1))
    depth = np.full(labels.shape, 2.0, dtype=np.float32)
    normals = np.zeros((*labels.shape, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    merged = merge_regions_geometry(
        labels,
        depth,
        {"normal": normals, "valid_mask": np.ones(labels.shape, dtype=bool)},
        depth_thresh=0.2,
        normal_thresh_deg=5.0,
    )

    np.testing.assert_array_equal(merged, np.zeros_like(labels))


def test_geometry_merge_keeps_depth_discontinuity():
    labels = np.tile(np.array([3, 3, 9, 9]), (4, 1))
    depth = np.tile(np.array([1.0, 1.0, 3.0, 3.0]), (4, 1))
    normals = np.zeros((*labels.shape, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    merged = merge_regions_geometry(
        labels,
        depth,
        {"normal": normals, "valid_mask": np.ones(labels.shape, dtype=bool)},
        depth_thresh=0.2,
        normal_thresh_deg=5.0,
    )

    np.testing.assert_array_equal(np.unique(merged), np.array([0, 1]))


def test_geometry_supports_cross_and_sobel_normals():
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    intrinsic = np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=np.float32)

    for method in ("cross", "sobel"):
        info = build_geometry_info_np(depth, intrinsic=intrinsic, normal_method=method)
        assert info["normal"].shape == (8, 8, 3)
        assert np.isfinite(info["normal"]).all()


def test_geometry_baseline_profile_uses_aligned_parameters(monkeypatch):
    calls = {}

    def fake_stages(depth_map, **kwargs):
        calls.update(kwargs)
        return type("Stages", (), {"merged_labels": np.zeros_like(depth_map, dtype=np.intp)})()

    monkeypatch.setattr(
        "inference_engine.utils.geometry_segmentation.segment_geometry_felzenszwalb_rag_stages",
        fake_stages,
    )
    segment_geometry_felzenszwalb_rag_baseline_params_stages(np.ones((4, 4)))

    assert calls["seg_scale"] == 300
    assert calls["seg_sigma"] == 1.1
    assert calls["seg_min_size"] == 500
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run:

```bash
python -m pytest tests/test_geometry_segmentation.py -q
```

Expected: collection fails because `geometry_segmentation` and geometry feature helpers do not exist.

- [ ] **Step 3: Port the exact source implementations**

Use the definitions from `LASER-Geometry@340c599` without formula changes:

```python
@dataclass(frozen=True)
class SegmentationStages:
    initial_labels: np.ndarray
    merged_labels: np.ndarray
    confidence_threshold: float
    high_confidence_mask: np.ndarray


def confidence_selection(conf, quantile, method=None):
    conf = np.asarray(conf)
    if quantile is None:
        return float("nan"), np.ones(conf.shape, dtype=bool)
    kwargs = {} if method is None else {"method": method}
    threshold = float(np.quantile(conf.reshape(-1), quantile, **kwargs))
    return threshold, np.isfinite(conf) & (conf >= threshold)
```

Port these exact geometry helpers to `geometry.py`:

```python
depth_to_local_points_np
compute_normals_cross_np
compute_normals_sobel_np
compute_normals_pca_np
compute_depth_edge_np
compute_normal_edge_np
build_geometry_info_np
```

Port these exact segmentation definitions to `geometry_segmentation.py`:

```python
_select_batch_item
compute_region_geometry_descriptors
should_merge_geometry
merge_regions_geometry
segment_geometry_felzenszwalb_rag_stages
segment_geometry_felzenszwalb_rag
segment_geometry_felzenszwalb_rag_baseline_params_stages
segment_geometry_felzenszwalb_rag_baseline_params
```

The legacy defaults remain `200 / 1.0 / 300`; the baseline-profile defaults remain `300 / 1.1 / 500`.

- [ ] **Step 4: Run characterization tests**

Run:

```bash
python -m pytest tests/test_geometry_segmentation.py -q
```

Expected: all geometry characterization tests pass.

- [ ] **Step 5: Commit the geometry port**

```bash
git add inference_engine/utils/segmentation_trace.py \
  inference_engine/utils/geometry_segmentation.py \
  inference_engine/utils/geometry.py \
  tests/test_geometry_segmentation.py
git commit -m "feat: port validated geometry segmentation"
```

---

### Task 2: Add the thin three-mode graph router

**Files:**
- Modify: `inference_engine/utils/lsa.py`
- Create: `tests/test_segmentation_modes.py`

**Interfaces:**
- Consumes: `make_sp_graph(point_maps, ..., segment_mode, normal_method, geometry_seg_profile)`.
- Produces: the existing nested temporal `Vertex` graph returned by `match_segmentation_seq`.

- [ ] **Step 1: Write failing routing tests**

Use monkeypatches to pin exact routing and parameters:

```python
import numpy as np
import pytest

from inference_engine.utils import lsa


@pytest.mark.parametrize("mode", ["depth", "geometry", "layer_atomic"])
def test_router_uses_aligned_felzenszwalb_parameters(monkeypatch, mode):
    calls = []

    def fake_batch(data, op, **kwargs):
        calls.append((data, op, kwargs))
        return np.zeros(data.shape[:3], dtype=np.intp)

    monkeypatch.setattr(lsa, "batched_image_op_wrapper", fake_batch)
    monkeypatch.setattr(lsa, "match_segmentation_seq", lambda labels, iou_thresh: labels)
    points = np.ones((2, 8, 8, 3), dtype=np.float32)

    lsa.make_sp_graph(points, segment_mode=mode)

    _, _, kwargs = calls[0]
    assert kwargs["seg_scale"] == 300
    assert kwargs["seg_sigma"] == 1.1
    assert kwargs["seg_min_size"] == 500


def test_geometry_legacy_uses_historical_parameters(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lsa,
        "batched_image_op_wrapper",
        lambda data, op, **kwargs: calls.append((op, kwargs)) or np.zeros(data.shape[:3], dtype=np.intp),
    )
    monkeypatch.setattr(lsa, "match_segmentation_seq", lambda labels, iou_thresh: labels)

    lsa.make_sp_graph(
        np.ones((1, 8, 8, 3)),
        segment_mode="geometry",
        geometry_seg_profile="legacy",
    )

    assert calls[0][1]["seg_scale"] == 200
    assert calls[0][1]["seg_sigma"] == 1.0
    assert calls[0][1]["seg_min_size"] == 300


def test_router_rejects_unknown_mode():
    with pytest.raises(ValueError, match="segment_mode"):
        lsa.make_sp_graph(np.ones((1, 8, 8, 3)), segment_mode="unknown")
```

Also assert depth receives `point_maps[..., -1]`, geometry receives depth plus `point_map=point_maps`, layer-atomic receives full point maps, and all labels pass through one `match_segmentation_seq` call.

- [ ] **Step 2: Run routing tests and verify failure**

Run:

```bash
python -m pytest tests/test_segmentation_modes.py -q
```

Expected: tests fail because `make_sp_graph` has no `segment_mode` or geometry route.

- [ ] **Step 3: Implement the minimal router**

Add constants and profile metadata:

```python
SEGMENT_MODES = ("depth", "geometry", "layer_atomic")
FELZENSZWALB_BASELINE_PARAMS = {
    "seg_scale": 300,
    "seg_sigma": 1.1,
    "seg_min_size": 500,
}
GEOMETRY_SEGMENTATION_PROFILES = {
    "baseline_params": (
        segment_geometry_felzenszwalb_rag_baseline_params,
        FELZENSZWALB_BASELINE_PARAMS,
    ),
    "legacy": (
        segment_geometry_felzenszwalb_rag,
        {"seg_scale": 200, "seg_sigma": 1.0, "seg_min_size": 300},
    ),
}
```

Extend `make_sp_graph(...)` with an omitted-mode compatibility sentinel:

```python
segment_mode=None  # preserves the validated layer-atomic branch call shape
normal_method="cross"
geometry_seg_profile="baseline_params"
```

Normalize `None` to the historical layer-atomic route without injecting new
keyword arguments. Engines and CLIs always pass their explicit mode; their
user-facing default remains `depth`.

Branch only around `batched_image_op_wrapper(...)`, forward the exact profile parameters, and keep the existing single call to:

```python
return match_segmentation_seq(labels, iou_thresh=corr_iou_thresh)
```

- [ ] **Step 4: Run routing and existing algorithm tests**

Run:

```bash
python -m pytest tests/test_segmentation_modes.py \
  tests/test_depth_segmentation_stages.py \
  tests/test_layer_atomic_geometry.py \
  tests/test_layer_atomic_integration.py -q
```

Expected: all tests pass without changing existing layer-atomic expectations.

- [ ] **Step 5: Commit the router**

```bash
git add inference_engine/utils/lsa.py tests/test_segmentation_modes.py
git commit -m "feat: route three segmentation methods"
```

---

### Task 3: Wire the shared router into both streaming engines

**Files:**
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `inference_engine/streaming_window_engine_lc.py`
- Create: `tests/test_segmentation_engine_modes.py`

**Interfaces:**
- Consumes: engine constructor fields `segment_mode`, `normal_method`, and `geometry_seg_profile`.
- Produces: one parent `_build_segment_graph(local_points, conf)` helper used by both workers.

- [ ] **Step 1: Write failing engine configuration tests**

```python
import pytest
import torch

from inference_engine.streaming_window_engine import StreamingWindowEngine
from inference_engine.streaming_window_engine_lc import StreamingWindowEngineLC


def make_engine(tmp_path, **kwargs):
    return StreamingWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        benchmark_latency=False,
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["depth", "geometry", "layer_atomic"])
def test_engine_accepts_three_modes(tmp_path, mode):
    engine = make_engine(tmp_path, segment_mode=mode, depth_refine=True)
    assert engine.segment_mode == mode


@pytest.mark.parametrize("mode", ["geometry", "layer_atomic"])
def test_non_depth_mode_requires_refinement(tmp_path, mode):
    with pytest.raises(ValueError, match="depth_refine"):
        make_engine(tmp_path, segment_mode=mode, depth_refine=False)


def test_geometry_defaults_to_aligned_profile(tmp_path):
    engine = make_engine(tmp_path, segment_mode="geometry", depth_refine=True)
    assert engine.geometry_seg_profile == "baseline_params"
```

Monkeypatch `make_sp_graph` and assert `_build_segment_graph(...)` forwards all three fields. Instantiate `StreamingWindowEngineLC` and assert it inherits the same configuration.

- [ ] **Step 2: Run engine tests and verify failure**

```bash
python -m pytest tests/test_segmentation_engine_modes.py -q
```

Expected: constructor and helper tests fail because mode configuration is absent.

- [ ] **Step 3: Implement shared engine configuration**

Add constructor defaults and validation to the parent:

```python
segment_mode: str = "depth"
normal_method: str = "cross"
geometry_seg_profile: str = "baseline_params"
```

Reject unknown modes, reject non-depth modes without refinement, and validate geometry options when geometry is active. Add:

```python
def _build_segment_graph(self, local_points, conf):
    return make_sp_graph(
        local_points.cpu().numpy(),
        conf_map=conf.cpu().numpy(),
        top_conf_percentile=self.top_conf_percentile,
        segment_mode=self.segment_mode,
        normal_method=self.normal_method,
        geometry_seg_profile=self.geometry_seg_profile,
    )
```

Replace every direct worker call to `make_sp_graph(...)` in the parent and LC child with this helper. Extend the LC constructor to forward the same explicit fields to `super().__init__`.

Print one concise effective configuration line during construction, using the aligned triplet for all formal modes and the historical triplet only for explicit geometry legacy.

- [ ] **Step 4: Run engine and integration tests**

```bash
python -m pytest tests/test_segmentation_engine_modes.py \
  tests/test_layer_atomic_integration.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit engine integration**

```bash
git add inference_engine/streaming_window_engine.py \
  inference_engine/streaming_window_engine_lc.py \
  tests/test_segmentation_engine_modes.py
git commit -m "feat: configure segmentation modes in streaming engines"
```

---

### Task 4: Expose identical CLI choices and cloud smoke verification

**Files:**
- Modify: `demo.py`
- Modify: `demo_lc.py`
- Modify: `README.md`
- Create: `scripts/verify_segmentation_modes.py`
- Create: `docs/unified-segmentation-cloud-validation.md`
- Modify: `tests/test_demo.py`
- Modify: `tests/test_demo_lc.py`
- Create: `tests/test_segmentation_smoke_script.py`

**Interfaces:**
- Consumes: CLI values for mode, normal method, and geometry profile.
- Produces: configured engines and a deterministic CPU smoke command.

- [ ] **Step 1: Write failing CLI and smoke-script tests**

```python
def test_segmentation_cli_defaults(parser):
    args = parser.parse_args([])
    assert args.segment_mode == "depth"
    assert args.normal_method == "cross"
    assert args.geometry_seg_profile == "baseline_params"


def test_segmentation_cli_accepts_all_modes(parser):
    for mode in ("depth", "geometry", "layer_atomic"):
        assert parser.parse_args(["--segment_mode", mode]).segment_mode == mode
```

Monkeypatch each demo's engine constructor and verify `load_model(...)` forwards all fields. Test `scripts/verify_segmentation_modes.py` in a subprocess and require one `[PASS]` line per mode.

- [ ] **Step 2: Run CLI tests and verify failure**

```bash
python -m pytest tests/test_demo.py tests/test_demo_lc.py \
  tests/test_segmentation_smoke_script.py -q
```

Expected: parser assertions fail and the smoke script is missing.

- [ ] **Step 3: Add matching CLI flags and forwarding**

Add to both parsers:

```python
parser.add_argument(
    "--segment_mode",
    default="depth",
    choices=("depth", "geometry", "layer_atomic"),
)
parser.add_argument(
    "--normal_method",
    default="cross",
    choices=("cross", "sobel"),
)
parser.add_argument(
    "--geometry_seg_profile",
    default="baseline_params",
    choices=("baseline_params", "legacy"),
)
```

Forward the three values unchanged to the corresponding engine constructor.

- [ ] **Step 4: Add deterministic smoke verification and cloud instructions**

The smoke script creates a finite `(2, 32, 32, 3)` point-map sequence and confidence sequence, calls `make_sp_graph(...)` for each formal mode, asserts two graph layers with full pixel coverage, and prints:

```text
[PASS] mode=depth frames=2
[PASS] mode=geometry frames=2
[PASS] mode=layer_atomic frames=2
```

Document these cloud steps exactly:

```bash
git fetch origin
git switch codex/unified-segmentation-methods
git pull --ff-only
conda activate laser
python setup.py build_ext --inplace
python -m pytest -q
python scripts/verify_segmentation_modes.py
```

Document three separate `demo.py` runs with identical data, checkpoint, sample interval, window size, overlap, and confidence settings, while using separate cache/output directories and changing only `--segment_mode`.

- [ ] **Step 5: Run CLI, documentation, and smoke tests**

```bash
python -m pytest tests/test_demo.py tests/test_demo_lc.py \
  tests/test_segmentation_smoke_script.py -q
python scripts/verify_segmentation_modes.py
```

Expected: all tests pass and all three `[PASS]` lines appear.

- [ ] **Step 6: Commit CLI and cloud validation**

```bash
git add demo.py demo_lc.py README.md \
  scripts/verify_segmentation_modes.py \
  docs/unified-segmentation-cloud-validation.md \
  tests/test_demo.py tests/test_demo_lc.py \
  tests/test_segmentation_smoke_script.py
git commit -m "docs: add unified segmentation validation workflow"
```

---

### Task 5: Verify algorithm preservation, complete the branch, and publish

**Files:**
- Modify only files required by fixes discovered during verification.

**Interfaces:**
- Consumes: the full branch diff and all test commands.
- Produces: a pushed GitHub branch with reproducible validation commands.

- [ ] **Step 1: Rebuild native extensions from the completed tree**

```bash
python setup.py build_ext --inplace
```

Expected: both `_segmentation_cy` and `fast_seg` build successfully.

- [ ] **Step 2: Run focused preservation tests**

```bash
python -m pytest tests/test_depth_segmentation_stages.py \
  tests/test_layer_atomic_geometry.py \
  tests/test_layer_atomic_integration.py \
  tests/test_geometry_segmentation.py \
  tests/test_segmentation_modes.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run the complete suite and smoke script**

```bash
python -m pytest -q
python scripts/verify_segmentation_modes.py
```

Expected: zero pytest failures and three `[PASS]` lines.

- [ ] **Step 4: Prove depth and layer-atomic algorithm files are untouched**

```bash
git diff --exit-code feature/layer-atomic-geometry -- \
  inference_engine/utils/depth.py \
  inference_engine/utils/layer_atomic_geometry.py
```

Expected: no diff and exit status 0.

- [ ] **Step 5: Inspect final scope and environment**

```bash
git status --short --branch
git diff --check feature/layer-atomic-geometry...HEAD
git log --oneline feature/layer-atomic-geometry..HEAD
python -c "import os, torch; print('cuda=', torch.cuda.is_available()); print('weights=', os.path.isfile('weights/model.safetensors'))"
```

Expected: only known generated Cython sources may be untracked; no whitespace errors; commits are scoped. Run a local GPU short-sequence smoke only if CUDA and local weights are both present.

- [ ] **Step 6: Perform a full code review and fix Critical or Important findings**

Review the complete diff against the design, apply fixes with regression tests, and rerun Steps 1-5 after every behavioral fix.

- [ ] **Step 7: Push the completed branch**

```bash
git push -u origin codex/unified-segmentation-methods
```

Expected: GitHub contains the branch and reports it tracking `origin/codex/unified-segmentation-methods`.
